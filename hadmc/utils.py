"""Utility functions for HAD-MC framework"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def create_simple_cnn(num_classes=10):
    """Create a simple CNN for testing"""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, num_classes)
    )


def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """Calculate FLOPs (approximate)"""
    total_flops = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs = 2 * C_in * C_out * K * K * H * W
            c_in = module.in_channels
            c_out = module.out_channels
            k = module.kernel_size[0]
            # Approximate output size
            h_out = input_size[2] // (module.stride[0] if hasattr(module, 'stride') else 1)
            w_out = input_size[3] // (module.stride[0] if hasattr(module, 'stride') else 1)
            flops = 2 * c_in * c_out * k * k * h_out * w_out
            total_flops += flops
        elif isinstance(module, nn.Linear):
            flops = 2 * module.in_features * module.out_features
            total_flops += flops
    return total_flops


def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy
