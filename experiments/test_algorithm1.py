"""Test Algorithm 1: Layer-wise Precision Allocation"""

import sys
from hadmc.device_manager import DeviceManager
sys.path.insert(0, '/home/ubuntu/HAD-MC-Core-Algorithms')

import torch
from torch.utils.data import TensorDataset, DataLoader
from hadmc.quantization import LayerwisePrecisionAllocator
from hadmc.utils import create_simple_cnn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_algorithm1():
    logger.info("="*60)
    logger.info("Testing Algorithm 1: Layer-wise Precision Allocation")
    logger.info("="*60)
    
    # Create model
    model = create_simple_cnn(num_classes=10)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy calibration dataset
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    calibration_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Run Algorithm 1
    allocator = LayerwisePrecisionAllocator(
        model=model,
        calibration_loader=calibration_loader,
        tau_h=1e-3,
        tau_l=1e-5,
        device='cpu'
    )
    
    quantized_model, precision_map = allocator.run()
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("Precision Allocation Results:")
    logger.info("="*60)
    for name, precision in precision_map.items():
        grad_mag = allocator.gradient_sensitivity[name]
        logger.info(f"{name:40s} | {precision:5s} | Grad: {grad_mag:.2e}")
    
    # Summary
    fp32_count = sum(1 for p in precision_map.values() if p == 'FP32')
    int8_count = sum(1 for p in precision_map.values() if p == 'INT8')
    int4_count = sum(1 for p in precision_map.values() if p == 'INT4')
    
    logger.info("\n" + "="*60)
    logger.info(f"Summary: FP32={fp32_count}, INT8={int8_count}, INT4={int4_count}")
    logger.info("="*60)
    logger.info("✓ Algorithm 1 test PASSED")
    
    return True

if __name__ == "__main__":
    test_algorithm1()
