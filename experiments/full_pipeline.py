"""Complete HAD-MC Pipeline: All 5 Algorithms in Sequence"""

import sys
from hadmc.device_manager import DeviceManager
sys.path.insert(0, '/home/ubuntu/HAD-MC-Core-Algorithms')

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import json
from hadmc.quantization import LayerwisePrecisionAllocator
from hadmc.pruning import GradientSensitivityPruner
from hadmc.distillation import FeatureAlignedDistiller
from hadmc.fusion import OperatorFuser
from hadmc.incremental_update import HashBasedUpdater
from hadmc.utils import create_simple_cnn, calculate_model_size, calculate_flops, evaluate_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dataset(num_samples=1000):
    """Create dummy dataset for testing"""
    X_train = torch.randn(num_samples, 3, 32, 32)
    y_train = torch.randint(0, 10, (num_samples,))
    X_test = torch.randn(200, 3, 32, 32)
    y_test = torch.randint(0, 10, (200,))
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


def measure_latency(model, input_size=(1, 3, 32, 32), num_iterations=100):
    """Measure model inference latency"""
    model.eval()
    dummy_input = torch.randn(input_size)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_latency_ms = (end_time - start_time) / num_iterations * 1000
    return avg_latency_ms


def run_full_pipeline():
    """Run complete HAD-MC pipeline"""
    
    logger.info("="*80)
    logger.info("HAD-MC: Complete Pipeline Execution")
    logger.info("="*80)
    
    # Create datasets
    logger.info("\n[Step 0] Creating datasets...")
    train_loader, test_loader = create_dataset()
    
    # Create baseline FP32 model
    logger.info("\n[Step 1] Creating baseline FP32 model...")
    model_fp32 = create_simple_cnn(num_classes=10)
    
    # Train baseline model briefly
    logger.info("Training baseline model...")
    optimizer = torch.optim.Adam(model_fp32.parameters(), lr=0.001)
    model_fp32.train()
    for epoch in range(3):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model_fp32(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        logger.info(f"  Epoch {epoch+1}/3 complete")
    
    # Measure baseline metrics
    fp32_size = calculate_model_size(model_fp32)
    fp32_flops = calculate_flops(model_fp32)
    fp32_latency = measure_latency(model_fp32)
    fp32_accuracy = evaluate_model(model_fp32, test_loader)
    
    logger.info(f"\nBaseline FP32 Model:")
    logger.info(f"  Size: {fp32_size:.2f} MB")
    logger.info(f"  FLOPs: {fp32_flops/1e6:.2f} M")
    logger.info(f"  Latency: {fp32_latency:.2f} ms")
    logger.info(f"  Accuracy: {fp32_accuracy:.2f}%")
    
    results = {
        'baseline': {
            'size_mb': fp32_size,
            'flops': fp32_flops,
            'latency_ms': fp32_latency,
            'accuracy': fp32_accuracy
        }
    }
    
    # Algorithm 1: Layer-wise Precision Allocation
    logger.info("\n" + "="*80)
    logger.info("[Algorithm 1] Layer-wise Precision Allocation")
    logger.info("="*80)
    
    allocator = LayerwisePrecisionAllocator(
        model=model_fp32,
        calibration_loader=train_loader,
        tau_h=1e-3,
        tau_l=1e-5
    )
    model_quantized, precision_map = allocator.run()
    
    quant_size = calculate_model_size(model_quantized)
    quant_latency = measure_latency(model_quantized)
    quant_accuracy = evaluate_model(model_quantized, test_loader)
    
    logger.info(f"\nAfter Quantization:")
    logger.info(f"  Size: {quant_size:.2f} MB ({(1-quant_size/fp32_size)*100:.1f}% reduction)")
    logger.info(f"  Latency: {quant_latency:.2f} ms ({(1-quant_latency/fp32_latency)*100:.1f}% reduction)")
    logger.info(f"  Accuracy: {quant_accuracy:.2f}% ({quant_accuracy-fp32_accuracy:+.2f}%)")
    
    results['after_quantization'] = {
        'size_mb': quant_size,
        'latency_ms': quant_latency,
        'accuracy': quant_accuracy
    }
    
    # Algorithm 2: Gradient Sensitivity-Guided Pruning
    logger.info("\n" + "="*80)
    logger.info("[Algorithm 2] Gradient Sensitivity-Guided Pruning")
    logger.info("="*80)
    
    pruner = GradientSensitivityPruner(
        model=model_quantized,
        train_loader=train_loader,
        flops_target=fp32_flops * 0.5
    )
    model_pruned = pruner.run(prune_ratio=0.3)
    
    pruned_size = calculate_model_size(model_pruned)
    pruned_flops = calculate_flops(model_pruned)
    pruned_latency = measure_latency(model_pruned)
    pruned_accuracy = evaluate_model(model_pruned, test_loader)
    
    logger.info(f"\nAfter Pruning:")
    logger.info(f"  Size: {pruned_size:.2f} MB ({(1-pruned_size/fp32_size)*100:.1f}% reduction)")
    logger.info(f"  FLOPs: {pruned_flops/1e6:.2f} M ({(1-pruned_flops/fp32_flops)*100:.1f}% reduction)")
    logger.info(f"  Latency: {pruned_latency:.2f} ms ({(1-pruned_latency/fp32_latency)*100:.1f}% reduction)")
    logger.info(f"  Accuracy: {pruned_accuracy:.2f}% ({pruned_accuracy-fp32_accuracy:+.2f}%)")
    
    results['after_pruning'] = {
        'size_mb': pruned_size,
        'flops': pruned_flops,
        'latency_ms': pruned_latency,
        'accuracy': pruned_accuracy
    }
    
    # Algorithm 3: Feature-Aligned Knowledge Distillation
    logger.info("\n" + "="*80)
    logger.info("[Algorithm 3] Feature-Aligned Knowledge Distillation")
    logger.info("="*80)
    
    distiller = FeatureAlignedDistiller(
        teacher_model=model_fp32,
        student_model=model_pruned
    )
    model_distilled = distiller.run(train_loader, epochs=3)
    
    distilled_accuracy = evaluate_model(model_distilled, test_loader)
    
    logger.info(f"\nAfter Distillation:")
    logger.info(f"  Accuracy: {distilled_accuracy:.2f}% ({distilled_accuracy-pruned_accuracy:+.2f}%)")
    
    results['after_distillation'] = {
        'accuracy': distilled_accuracy
    }
    
    # Algorithm 4: Operator Fusion
    logger.info("\n" + "="*80)
    logger.info("[Algorithm 4] Operator Fusion")
    logger.info("="*80)
    
    fuser = OperatorFuser(model_distilled)
    model_fused = fuser.run()
    
    fused_latency = measure_latency(model_fused)
    
    logger.info(f"\nAfter Fusion:")
    logger.info(f"  Latency: {fused_latency:.2f} ms ({(1-fused_latency/fp32_latency)*100:.1f}% reduction)")
    
    results['after_fusion'] = {
        'latency_ms': fused_latency
    }
    
    # Algorithm 5: Hash-based Incremental Update
    logger.info("\n" + "="*80)
    logger.info("[Algorithm 5] Hash-based Incremental Update")
    logger.info("="*80)
    
    # Simulate model update
    model_v2 = create_simple_cnn(num_classes=10)
    
    updater = HashBasedUpdater(block_size=4096)
    update_package = updater.run(model_fused, model_v2)
    
    logger.info(f"\nIncremental Update:")
    logger.info(f"  Bandwidth reduction: {update_package['bandwidth_reduction']*100:.1f}%")
    logger.info(f"  Changed blocks: {len(update_package['changed_blocks'])}")
    
    results['incremental_update'] = {
        'bandwidth_reduction': update_package['bandwidth_reduction']
    }
    
    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"\nModel Size:")
    logger.info(f"  FP32:      {fp32_size:.2f} MB")
    logger.info(f"  HAD-MC:    {pruned_size:.2f} MB")
    logger.info(f"  Reduction: {(1-pruned_size/fp32_size)*100:.1f}%")
    
    logger.info(f"\nInference Latency:")
    logger.info(f"  FP32:      {fp32_latency:.2f} ms")
    logger.info(f"  HAD-MC:    {fused_latency:.2f} ms")
    logger.info(f"  Reduction: {(1-fused_latency/fp32_latency)*100:.1f}%")
    
    logger.info(f"\nAccuracy:")
    logger.info(f"  FP32:      {fp32_accuracy:.2f}%")
    logger.info(f"  HAD-MC:    {distilled_accuracy:.2f}%")
    logger.info(f"  Change:    {distilled_accuracy-fp32_accuracy:+.2f}%")
    
    # Save results
    with open('/home/ubuntu/HAD-MC-Core-Algorithms/results/pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n✓ Pipeline complete! Results saved to results/pipeline_results.json")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    run_full_pipeline()
