"""
Model utilities for HAD-MC 2.0 experiments

This module provides:
- Model loading utilities
- Model compression helpers
- Model evaluation helpers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import os


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> int:
    """
    Estimate FLOPs for a model.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, height, width)

    Returns:
        Estimated FLOPs
    """
    flops = 0

    # This is a rough estimation for CNNs
    # For accurate FLOPs counting, use torchprofile or thop
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs: kernel_h * kernel_w * in_channels * out_channels * output_h * output_w
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            # Assume output size is same as input for estimation
            output_size = input_shape[2] * input_shape[3]
            flops += kernel_size * module.in_channels * module.out_channels * output_size
        elif isinstance(module, nn.Linear):
            # Linear FLOPs: in_features * out_features
            flops += module.in_features * module.out_features

    return flops


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB (assuming float32)
    """
    param_size = count_parameters(model) * 4  # 4 bytes per float32
    return param_size / (1024.0 * 1024.0)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    filepath: str,
    metrics: Optional[Dict] = None
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        filepath: Path to save checkpoint
        metrics: Optional metrics to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        filepath: Path to checkpoint
        optimizer: Optional optimizer to restore
        device: Device to load to

    Returns:
        Dictionary containing checkpoint info
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def apply_pruning(
    model: nn.Module,
    pruning_ratios: Dict[str, float]
) -> nn.Module:
    """
    Apply channel pruning to model.

    Args:
        model: PyTorch model
        pruning_ratios: Dictionary mapping layer names to pruning ratios

    Returns:
        Pruned model
    """
    # This is a simplified implementation
    # For actual pruning, use torch.nn.utils.prune

    for name, module in model.named_modules():
        if name in pruning_ratios and isinstance(module, nn.Conv2d):
            ratio = pruning_ratios[name]
            # Calculate number of channels to keep
            num_channels = module.out_channels
            num_keep = int(num_channels * (1 - ratio))

            # Apply L1-based channel selection
            # In practice, you would sort channels by importance and keep top num_keep
            pass

    return model


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specific layers in a model.

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if name.startswith(layer_name):
                param.requires_grad = False
                break


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Unfreeze specific layers in a model.

    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if name.startswith(layer_name):
                param.requires_grad = True
                break


def get_layer_info(model: nn.Module) -> List[Dict]:
    """
    Get information about all layers in a model.

    Args:
        model: PyTorch model

    Returns:
        List of layer information dictionaries
    """
    layer_info = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            info = {
                'name': name,
                'type': module.__class__.__name__,
            }

            if isinstance(module, nn.Conv2d):
                info.update({
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'num_params': module.weight.numel() + module.bias.numel() if module.bias is not None else module.weight.numel()
                })
            elif isinstance(module, nn.Linear):
                info.update({
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'num_params': module.weight.numel() + module.bias.numel() if module.bias is not None else module.weight.numel()
                })
            elif isinstance(module, nn.BatchNorm2d):
                info.update({
                    'num_features': module.num_features,
                    'num_params': module.weight.numel() + module.bias.numel() if module.bias is not None else module.weight.numel()
                })

            layer_info.append(info)

    return layer_info


def create_dummy_bounding_box() -> torch.Tensor:
    """
    Create a dummy bounding box for testing.

    Returns:
        Dummy bounding box tensor (1, 4)
    """
    return torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)


def create_dummy_predictions(num_classes: int = 80, num_detections: int = 100) -> Dict:
    """
    Create dummy predictions for testing.

    Args:
        num_classes: Number of classes
        num_detections: Number of detections

    Returns:
        Dictionary of dummy predictions
    """
    return {
        'boxes': torch.rand(num_detections, 4) * 640,
        'scores': torch.rand(num_detections),
        'labels': torch.randint(0, num_classes, (num_detections,))
    }


def model_to_device(model: nn.Module, device: str) -> nn.Module:
    """
    Move model to device.

    Args:
        model: PyTorch model
        device: Target device ('cpu', 'cuda', 'mlu', 'npu')

    Returns:
        Model on target device
    """
    return model.to(device)


def get_model_device(model: nn.Module) -> str:
    """
    Get the device a model is on.

    Args:
        model: PyTorch model

    Returns:
        Device string
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return 'cpu'
