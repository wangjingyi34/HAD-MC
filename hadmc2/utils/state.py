"""State representation for HAD-MC 2.0 MARL framework"""

import torch
import torch.nn as nn
from typing import List, Dict, Union
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)


class State:
    """
    State space representation for MARL.

    The state consists of three components:
    1. Model State: Architecture, parameters, FLOPs
    2. Hardware State: Compute capability, memory, power budget
    3. Compression State: Current pruning ratios, bit widths, etc.
    """

    def __init__(self):
        # Model State
        self.model_state = {
            'num_layers': 0,           # Total number of layers
            'layer_types': [],          # List of layer types
            'channel_counts': [],        # Output channels for each layer
            'param_counts': [],          # Parameter count per layer
            'flop_counts': [],           # FLOPs per layer
            'activation_sizes': [],      # Activation size per layer
        }

        # Hardware State
        self.hardware_state = {
            'compute_capability': 0.0,  # TFLOPS
            'memory_bandwidth': 0.0,    # GB/s
            'memory_capacity': 0.0,     # GB
            'power_budget': 0.0,        # W
            'supported_precisions': [],  # List of supported precision formats
            'has_tensor_core': False,    # Whether device has Tensor Core
            'has_sparsity_support': False,  # Whether device supports sparsity acceleration
        }

        # Compression State
        self.compression_state = {
            'pruning_ratios': [],      # Pruning ratio per layer [0, 1]
            'bit_widths': [],           # Bit width per layer [4, 8, 16, 32]
            'distillation_progress': 0.0,  # Distillation progress [0, 1]
            'fused_patterns': [],       # Applied fusion patterns
            'current_accuracy': 1.0,    # Current model accuracy
            'current_latency': 0.0,     # Current latency (ms)
            'current_energy': 0.0,      # Current energy (J)
            'current_size': 0.0,         # Current model size (MB)
        }

    def update_model_state(self, model: nn.Module):
        """Extract model state from PyTorch model"""
        layers = []
        param_counts = []
        flops = []
        activations = []
        layer_types = []
        channel_counts = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layers.append(name)

                if isinstance(module, nn.Conv2d):
                    layer_types.append('conv')
                    # Output channels
                    channel_counts.append(module.out_channels)
                    # Parameter count
                    weight_params = module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
                    bias_params = module.out_channels if module.bias is not None else 0
                    param_counts.append(weight_params + bias_params)
                    # FLOPs estimate
                    input_size = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                    flops.append(input_size * module.out_channels)  # Simplified
                    # Activation size (output feature map size)
                    activations.append(module.out_channels * 32 * 32)  # Assume 32x32 input
                elif isinstance(module, nn.Linear):
                    layer_types.append('fc')
                    channel_counts.append(module.out_features)
                    param_counts.append(module.in_features * module.out_features)
                    flops.append(module.in_features * module.out_features)
                    activations.append(module.out_features)
                elif isinstance(module, nn.BatchNorm2d):
                    layer_types.append('bn')
                    channel_counts.append(module.num_features)
                    param_counts.append(module.num_features * 2)  # scale and bias
                    flops.append(0)  # BN has negligible compute
                    activations.append(module.num_features * 32 * 32)

        self.model_state['num_layers'] = len(layers)
        self.model_state['layer_types'] = layer_types
        self.model_state['channel_counts'] = channel_counts
        self.model_state['param_counts'] = param_counts
        self.model_state['flop_counts'] = flops
        self.model_state['activation_sizes'] = activations

    def to_tensor(self) -> torch.Tensor:
        """
        Convert state to tensor for neural network input.

        Returns:
            torch.Tensor: Flattened and normalized state vector
        """
        features = []

        # ===== Model State Features =====
        # Number of layers (normalized to [0, 1])
        features.append(self.model_state['num_layers'] / 100.0)

        # Layer type one-hot encoding
        layer_type_map = {'conv': 0, 'bn': 1, 'fc': 2, 'relu': 3, 'pool': 4}
        num_layer_types = len(layer_type_map)

        # Count layer types
        type_counts = [0] * num_layer_types
        for lt in self.model_state['layer_types']:
            if lt in layer_type_map:
                type_counts[layer_type_map[lt]] += 1

        # Normalize layer type counts
        if self.model_state['num_layers'] > 0:
            type_counts = [c / self.model_state['num_layers'] for c in type_counts]
        features.extend(type_counts)

        # Channel counts (normalized)
        if self.model_state['channel_counts']:
            max_channels = max(self.model_state['channel_counts']) if self.model_state['channel_counts'] else 1
            channels_norm = [c / max_channels for c in self.model_state['channel_counts']]
            # Pad or truncate to fixed length (assume max 20 layers)
            channels_norm = channels_norm[:20] + [0.0] * max(0, 20 - len(channels_norm))
            features.extend(channels_norm)
        else:
            features.extend([0.0] * 20)

        # ===== Hardware State Features =====
        # Compute capability (normalized to [0, 1], assume max 100 TFLOPS)
        features.append(self.hardware_state['compute_capability'] / 100.0)

        # Memory bandwidth (normalized to [0, 1], assume max 3000 GB/s)
        features.append(self.hardware_state['memory_bandwidth'] / 3000.0)

        # Memory capacity (normalized to [0, 1], assume max 100 GB)
        features.append(self.hardware_state['memory_capacity'] / 100.0)

        # Power budget (normalized to [0, 1], assume max 500 W)
        features.append(self.hardware_state['power_budget'] / 500.0)

        # Supported precisions (multi-hot encoding)
        precision_map = {'FP32': 0, 'FP16': 1, 'INT8': 2, 'INT4': 3}
        precision_features = [0.0] * len(precision_map)
        for p in self.hardware_state['supported_precisions']:
            if p in precision_map:
                precision_features[precision_map[p]] = 1.0
        features.extend(precision_features)

        # Special capabilities
        features.append(1.0 if self.hardware_state['has_tensor_core'] else 0.0)
        features.append(1.0 if self.hardware_state['has_sparsity_support'] else 0.0)

        # ===== Compression State Features =====
        # Pruning ratios (assume max 20 layers)
        pruning_ratios = self.compression_state['pruning_ratios']
        pruning_ratios = pruning_ratios[:20] + [0.0] * max(0, 20 - len(pruning_ratios))
        features.extend(pruning_ratios)

        # Bit widths (normalized to [0, 1])
        bit_widths = self.compression_state['bit_widths']
        bit_widths_norm = [bw / 32.0 for bw in bit_widths]
        bit_widths_norm = bit_widths_norm[:20] + [0.0] * max(0, 20 - len(bit_widths_norm))
        features.extend(bit_widths_norm)

        # Distillation progress
        features.append(self.compression_state['distillation_progress'])

        # Current performance metrics
        features.append(self.compression_state['current_accuracy'])  # Already [0, 1]
        features.append(self.compression_state['current_latency'] / 100.0)  # Normalize to ms
        features.append(self.compression_state['current_energy'] / 10.0)  # Normalize to J
        features.append(self.compression_state['current_size'] / 100.0)  # Normalize to MB

        # Fused patterns count (normalized)
        fused_count = len(self.compression_state['fused_patterns'])
        features.append(fused_count / 20.0)

        return torch.tensor(features, dtype=torch.float32)

    def dim(self) -> int:
        """Return the dimension of the state vector"""
        return len(self.to_tensor())

    def copy(self) -> 'State':
        """Create a deep copy of the state"""
        import copy
        state = State()

        state.model_state = {
            'num_layers': self.model_state['num_layers'],
            'layer_types': self.model_state['layer_types'].copy(),
            'channel_counts': self.model_state['channel_counts'].copy(),
            'param_counts': self.model_state['param_counts'].copy(),
            'flop_counts': self.model_state['flop_counts'].copy(),
            'activation_sizes': self.model_state['activation_sizes'].copy(),
        }

        state.hardware_state = self.hardware_state.copy()
        state.hardware_state['supported_precisions'] = self.hardware_state['supported_precisions'].copy()

        state.compression_state = self.compression_state.copy()
        state.compression_state['pruning_ratios'] = self.compression_state['pruning_ratios'].copy()
        state.compression_state['bit_widths'] = self.compression_state['bit_widths'].copy()
        state.compression_state['fused_patterns'] = self.compression_state['fused_patterns'].copy()

        return state

    def __repr__(self) -> str:
        return (f"State(model_layers={self.model_state['num_layers']}, "
                f"hw_compute={self.hardware_state['compute_capability']:.1f}TF, "
                f"accuracy={self.compression_state['current_accuracy']:.4f}, "
                f"latency={self.compression_state['current_latency']:.2f}ms)")


def create_state_from_model_and_hardware(
    model: nn.Module,
    hardware_config: Union[Dict, 'HardwareConfig']
) -> State:
    """
    Convenience function to create a state from model and hardware config.

    Args:
        model: PyTorch model
        hardware_config: Dictionary or HardwareConfig object with hardware specifications

    Returns:
        State: Initialized state object
    """
    state = State()
    state.update_model_state(model)

    # Convert HardwareConfig to dict if necessary
    if hasattr(hardware_config, '__dataclass_fields__'):
        # It's a dataclass, convert to dict
        from hadmc2.hardware.hal import HardwareConfig
        if isinstance(hardware_config, HardwareConfig):
            hardware_config = asdict(hardware_config)

    state.hardware_state = hardware_config

    # Initialize compression state with no compression
    state.compression_state['pruning_ratios'] = [0.0] * state.model_state['num_layers']
    state.compression_state['bit_widths'] = [32] * state.model_state['num_layers']
    state.compression_state['current_size'] = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)  # Convert to MB

    return state
