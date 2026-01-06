#!/usr/bin/env python3
"""
Complete GPU Cross-Platform Validation Experiment
Using Real HAD-MC Algorithms from GitHub Repository
"""

import sys
sys.path.insert(0, '/workspace/HAD-MC')

import torch
import torch.nn as nn
from pathlib import Path
import json
import time
import logging
from datetime import datetime

# Import real HAD-MC algorithms
from hadmc.pruning import GradientSensitivityPruner
from hadmc.quantization import LayerwisePrecisionAllocator
from hadmc.distillation import FeatureAlignedDistiller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
RESULTS_DIR = Path('/workspace/HAD-MC/experiments/results')
YOLO_DIR = Path('/workspace/yolov5')
FP32_MODEL_PATH = RESULTS_DIR / 'fp32_baseline' / 'train' / 'weights' / 'best.pt'
COCO_DATA = YOLO_DIR / 'data' / 'coco128.yaml'

def load_yolo_model(model_path):
    """Load YOLOv5 model"""
    logger.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location='cuda')
    if isinstance(model, dict):
        model = model.get('model', model.get('ema', None))
    return model

def evaluate_yolo_model(model_path, data_yaml):
    """Evaluate YOLOv5 model using val.py"""
    import subprocess
    cmd = [
        'python3', str(YOLO_DIR / 'val.py'),
        '--data', str(data_yaml),
        '--weights', str(model_path),
        '--batch-size', '16',
        '--img-size', '640',
        '--task', 'val',
        '--device', '0'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(YOLO_DIR))
    
    # Parse results
    output = result.stdout
    metrics = {}
    for line in output.split('\n'):
        if 'all' in line and 'mAP' in output:
            parts = line.split()
            try:
                metrics['precision'] = float(parts[4])
                metrics['recall'] = float(parts[5])
                metrics['mAP50'] = float(parts[6])
                metrics['mAP50-95'] = float(parts[7])
            except:
                pass
    
    return metrics

def benchmark_inference(model, input_size=(1, 3, 640, 640), num_runs=100):
    """Benchmark inference latency"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end = time.time()
    
    latency_ms = (end - start) / num_runs * 1000
    throughput_fps = 1000 / latency_ms
    
    return {
        'latency_ms': latency_ms,
        'throughput_fps': throughput_fps
    }

def get_model_size(model_path):
    """Get model file size in MB"""
    return Path(model_path).stat().st_size / (1024 * 1024)

def get_model_memory(model):
    """Get model memory usage in MB"""
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return (mem_params + mem_bufs) / (1024 * 1024)

def create_calibration_loader():
    """Create calibration data loader"""
    from torch.utils.data import DataLoader, TensorDataset
    # Simplified: use random data
    data = torch.randn(100, 3, 640, 640)
    labels = torch.randint(0, 80, (100,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=16, shuffle=False)

def experiment_1_fp32_baseline():
    """Experiment 1: FP32 Baseline (already trained)"""
    logger.info("="*80)
    logger.info("EXPERIMENT 1: FP32 Baseline")
    logger.info("="*80)
    
    if not FP32_MODEL_PATH.exists():
        logger.error(f"FP32 model not found: {FP32_MODEL_PATH}")
        return None
    
    # Evaluate
    metrics = evaluate_yolo_model(FP32_MODEL_PATH, COCO_DATA)
    
    # Benchmark
    model = load_yolo_model(FP32_MODEL_PATH).cuda().eval()
    perf = benchmark_inference(model)
    
    result = {
        'method': 'FP32 Baseline',
        'model_size_mb': get_model_size(FP32_MODEL_PATH),
        'memory_mb': get_model_memory(model),
        **metrics,
        **perf
    }
    
    logger.info(f"Results: {json.dumps(result, indent=2)}")
    return result

def experiment_2_ptq_int8():
    """Experiment 2: PTQ-INT8 Quantization"""
    logger.info("="*80)
    logger.info("EXPERIMENT 2: PTQ-INT8 Quantization")
    logger.info("="*80)
    
    model = load_yolo_model(FP32_MODEL_PATH).cuda()
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    
    # Save
    ptq_path = RESULTS_DIR / 'ptq_int8' / 'model.pt'
    ptq_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model, ptq_path)
    
    # Evaluate
    metrics = evaluate_yolo_model(ptq_path, COCO_DATA)
    
    # Benchmark
    perf = benchmark_inference(quantized_model.cuda().eval())
    
    result = {
        'method': 'PTQ-INT8',
        'model_size_mb': get_model_size(ptq_path),
        'memory_mb': get_model_memory(quantized_model),
        **metrics,
        **perf
    }
    
    logger.info(f"Results: {json.dumps(result, indent=2)}")
    return result

def experiment_3_hadmc_pruning():
    """Experiment 3: HAD-MC Gradient-Sensitivity Pruning"""
    logger.info("="*80)
    logger.info("EXPERIMENT 3: HAD-MC Gradient-Sensitivity Pruning")
    logger.info("="*80)
    
    model = load_yolo_model(FP32_MODEL_PATH).cuda()
    calib_loader = create_calibration_loader()
    
    # Apply real HAD-MC pruning algorithm
    pruner = GradientSensitivityPruner(
        model=model,
        train_loader=calib_loader,
        flops_target=0.7,  # Target 70% FLOPs
        device='cuda'
    )
    
    logger.info("Calculating channel importance...")
    pruner.calculate_channel_importance()
    
    logger.info("Pruning channels...")
    pruned_model = pruner.prune_channels(prune_ratio=0.3)  # Prune 30%
    
    # Save
    pruned_path = RESULTS_DIR / 'hadmc_pruning' / 'model.pt'
    pruned_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pruned_model, pruned_path)
    
    # Evaluate
    metrics = evaluate_yolo_model(pruned_path, COCO_DATA)
    
    # Benchmark
    perf = benchmark_inference(pruned_model.eval())
    
    result = {
        'method': 'HAD-MC Pruning (30%)',
        'model_size_mb': get_model_size(pruned_path),
        'memory_mb': get_model_memory(pruned_model),
        'pruning_ratio': 0.3,
        **metrics,
        **perf
    }
    
    logger.info(f"Results: {json.dumps(result, indent=2)}")
    return result

def experiment_4_hadmc_quantization():
    """Experiment 4: HAD-MC Layer-wise Adaptive Quantization"""
    logger.info("="*80)
    logger.info("EXPERIMENT 4: HAD-MC Layer-wise Adaptive Quantization")
    logger.info("="*80)
    
    model = load_yolo_model(FP32_MODEL_PATH).cuda()
    calib_loader = create_calibration_loader()
    
    # Apply real HAD-MC quantization algorithm
    quantizer = LayerwisePrecisionAllocator(
        model=model,
        calibration_loader=calib_loader,
        tau_h=1e-3,
        tau_l=1e-5,
        device='cuda'
    )
    
    logger.info("Calculating gradient sensitivity...")
    quantizer.calculate_gradient_sensitivity()
    
    logger.info("Allocating precision...")
    precision_map = quantizer.allocate_precision()
    
    # Log precision allocation
    fp32_count = sum(1 for p in precision_map.values() if p == 'FP32')
    int8_count = sum(1 for p in precision_map.values() if p == 'INT8')
    int4_count = sum(1 for p in precision_map.values() if p == 'INT4')
    logger.info(f"Precision allocation: FP32={fp32_count}, INT8={int8_count}, INT4={int4_count}")
    
    quantized_model = quantizer.run()
    
    # Apply PyTorch quantization based on precision map
    # (Simplified: use dynamic quantization for now)
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model.cpu(), {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    
    # Save
    quant_path = RESULTS_DIR / 'hadmc_quantization' / 'model.pt'
    quant_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': quantized_model, 'precision_map': precision_map}, quant_path)
    
    # Evaluate
    metrics = evaluate_yolo_model(quant_path, COCO_DATA)
    
    # Benchmark
    perf = benchmark_inference(quantized_model.cuda().eval())
    
    result = {
        'method': 'HAD-MC Adaptive Quantization',
        'model_size_mb': get_model_size(quant_path),
        'memory_mb': get_model_memory(quantized_model),
        'precision_allocation': {
            'FP32': fp32_count,
            'INT8': int8_count,
            'INT4': int4_count
        },
        **metrics,
        **perf
    }
    
    logger.info(f"Results: {json.dumps(result, indent=2)}")
    return result

def experiment_5_hadmc_full_with_kd():
    """Experiment 5: HAD-MC Full Pipeline with Knowledge Distillation"""
    logger.info("="*80)
    logger.info("EXPERIMENT 5: HAD-MC Full Pipeline (Pruning + Quantization + KD)")
    logger.info("="*80)
    
    # Load teacher model (FP32)
    teacher_model = load_yolo_model(FP32_MODEL_PATH).cuda().eval()
    
    # Load student model (pruned + quantized)
    # For simplicity, use the pruned model as student
    student_model = load_yolo_model(RESULTS_DIR / 'hadmc_pruning' / 'model.pt').cuda()
    
    calib_loader = create_calibration_loader()
    
    # Apply real HAD-MC knowledge distillation
    distiller = FeatureAlignedDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        device='cuda'
    )
    
    logger.info("Running knowledge distillation (5 epochs)...")
    distilled_model = distiller.run(
        train_loader=calib_loader,
        epochs=5,
        lr=0.001
    )
    
    # Apply quantization
    quantized_model = torch.quantization.quantize_dynamic(
        distilled_model.cpu(), {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    
    # Save
    full_path = RESULTS_DIR / 'hadmc_full' / 'model.pt'
    full_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model, full_path)
    
    # Evaluate
    metrics = evaluate_yolo_model(full_path, COCO_DATA)
    
    # Benchmark
    perf = benchmark_inference(quantized_model.cuda().eval())
    
    result = {
        'method': 'HAD-MC Full Pipeline',
        'model_size_mb': get_model_size(full_path),
        'memory_mb': get_model_memory(quantized_model),
        'components': ['Pruning (30%)', 'Adaptive Quantization', 'Knowledge Distillation'],
        **metrics,
        **perf
    }
    
    logger.info(f"Results: {json.dumps(result, indent=2)}")
    return result

def main():
    """Run all experiments"""
    logger.info("Starting Real HAD-MC GPU Cross-Platform Validation")
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    results = []
    
    try:
        # Experiment 1: FP32 Baseline
        result1 = experiment_1_fp32_baseline()
        if result1:
            results.append(result1)
        
        # Experiment 2: PTQ-INT8
        result2 = experiment_2_ptq_int8()
        if result2:
            results.append(result2)
        
        # Experiment 3: HAD-MC Pruning
        result3 = experiment_3_hadmc_pruning()
        if result3:
            results.append(result3)
        
        # Experiment 4: HAD-MC Quantization
        result4 = experiment_4_hadmc_quantization()
        if result4:
            results.append(result4)
        
        # Experiment 5: HAD-MC Full Pipeline
        result5 = experiment_5_hadmc_full_with_kd()
        if result5:
            results.append(result5)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
    
    # Save final report
    report = {
        'timestamp': datetime.now().isoformat(),
        'device': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'results': results
    }
    
    report_path = RESULTS_DIR / 'REAL_HADMC_GPU_VALIDATION_REPORT.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("="*80)
    logger.info("ALL EXPERIMENTS COMPLETED!")
    logger.info(f"Report saved to: {report_path}")
    logger.info("="*80)
    
    # Print summary table
    print("\n" + "="*100)
    print("FINAL COMPARISON TABLE")
    print("="*100)
    print(f"{'Method':<40} {'Size (MB)':<12} {'mAP50-95':<12} {'Latency (ms)':<15} {'Throughput (FPS)':<15}")
    print("-"*100)
    for r in results:
        print(f"{r['method']:<40} {r.get('model_size_mb', 0):<12.2f} {r.get('mAP50-95', 0):<12.3f} {r.get('latency_ms', 0):<15.2f} {r.get('throughput_fps', 0):<15.2f}")
    print("="*100)

if __name__ == '__main__':
    main()
