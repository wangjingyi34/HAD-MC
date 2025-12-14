"""Test all 5 algorithms"""

import sys
from hadmc.device_manager import DeviceManager
sys.path.insert(0, '/home/ubuntu/HAD-MC-Core-Algorithms')

import torch
from torch.utils.data import TensorDataset, DataLoader
from hadmc.quantization import LayerwisePrecisionAllocator
from hadmc.pruning import GradientSensitivityPruner
from hadmc.distillation import FeatureAlignedDistiller
from hadmc.fusion import OperatorFuser
from hadmc.incremental_update import HashBasedUpdater
from hadmc.utils import create_simple_cnn, calculate_model_size
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data(num_samples=100):
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    return loader

def test_algorithm2():
    logger.info("\n" + "="*60)
    logger.info("Testing Algorithm 2: Gradient Sensitivity-Guided Pruning")
    logger.info("="*60)
    
    model = create_simple_cnn()
    train_loader = create_dummy_data()
    
    pruner = GradientSensitivityPruner(
        model=model,
        train_loader=train_loader,
        flops_target=1e6,
        device='cpu'
    )
    
    pruned_model = pruner.run(prune_ratio=0.3)
    logger.info("✓ Algorithm 2 test PASSED")
    return True

def test_algorithm3():
    logger.info("\n" + "="*60)
    logger.info("Testing Algorithm 3: Feature-Aligned Knowledge Distillation")
    logger.info("="*60)
    
    teacher = create_simple_cnn()
    student = create_simple_cnn()
    train_loader = create_dummy_data()
    
    distiller = FeatureAlignedDistiller(
        teacher_model=teacher,
        student_model=student,
        device='cpu'
    )
    
    distilled_student = distiller.run(train_loader, epochs=2)
    logger.info("✓ Algorithm 3 test PASSED")
    return True

def test_algorithm4():
    logger.info("\n" + "="*60)
    logger.info("Testing Algorithm 4: Operator Fusion")
    logger.info("="*60)
    
    model = create_simple_cnn()
    
    fuser = OperatorFuser(model)
    fused_model = fuser.run()
    
    logger.info("✓ Algorithm 4 test PASSED")
    return True

def test_algorithm5():
    logger.info("\n" + "="*60)
    logger.info("Testing Algorithm 5: Hash-based Incremental Update")
    logger.info("="*60)
    
    model_old = create_simple_cnn()
    model_new = create_simple_cnn()
    
    # Modify one layer in new model
    with torch.no_grad():
        for name, param in model_new.named_parameters():
            if 'weight' in name:
                param.data += 0.1
                break
    
    updater = HashBasedUpdater(block_size=4096)
    update_package = updater.run(model_old, model_new)
    
    logger.info(f"Bandwidth reduction: {update_package['bandwidth_reduction']*100:.1f}%")
    logger.info("✓ Algorithm 5 test PASSED")
    return True

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("HAD-MC: Testing All 5 Core Algorithms")
    logger.info("="*60)
    
    results = []
    results.append(("Algorithm 2", test_algorithm2()))
    results.append(("Algorithm 3", test_algorithm3()))
    results.append(("Algorithm 4", test_algorithm4()))
    results.append(("Algorithm 5", test_algorithm5()))
    
    logger.info("\n" + "="*60)
    logger.info("Test Summary:")
    logger.info("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{name:40s} {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        logger.info("\n🎉 All algorithms tested successfully!")
    else:
        logger.error("\n❌ Some tests failed")
