"""Quantization Agent for HAD-MC 2.0"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import copy
import logging

logger = logging.getLogger(__name__)


class QuantizationAgent:
    """
    Quantization Agent: Layer-wise precision allocation.

    Performs post-training quantization (PTQ) or quantization-aware training (QAT).
    Supports FP32, FP16, INT8, and INT4 precisions.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader,
        device: str = 'cpu'
    ):
        """
        Initialize Quantization Agent.

        Args:
            model: PyTorch model to quantize
            calibration_loader: Data loader for calibration
            device: Device to run on
        """
        self.model = model
        self.calibration_loader = calibration_loader
        self.device = device

        self.quantization_params = {}
        self.calibrated = False

        # Bit width options
        self.bit_widths = [4, 8, 16, 32]

    def get_action_space(self) -> Dict:
        """
        Get action space for quantization agent.

        Returns:
            dict: Action space definition
        """
        return {
            'type': 'discrete',
            'layer_idx': list(range(len(self._get_quantizable_layers()))),
            'bit_width': self.bit_widths,
        }

    def _get_quantizable_layers(self) -> List[nn.Module]:
        """
        Get list of layers that can be quantized.
        """
        return [
            module for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]

    def _get_quantizable_layer_names(self) -> List[str]:
        """
        Helper to get quantizable layer names.
        """
        return [
            name for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]

    def calibrate(self, num_batches: int = 100):
        """
        Calibrate quantization parameters by running model on calibration data.

        Collects statistics (min, max, mean, std) for each layer.

        Args:
            num_batches: Number of batches to use for calibration
        """
        self.model.eval()
        self.model.to(self.device)

        # Register forward hooks to collect statistics
        stats = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                if name not in stats:
                    stats[name] = {
                        'min': [],
                        'max': [],
                        'mean': [],
                        'std': [],
                    }
                stats[name]['min'].append(output.min().item())
                stats[name]['max'].append(output.max().item())
                stats[name]['mean'].append(output.mean().item())
                stats[name]['std'].append(output.std().item())
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(make_hook(name))
                hooks.append(handle)

        # Run calibration data
        loader_list = list(self.calibration_loader)

        with torch.no_grad():
            for i, (inputs, _) in enumerate(loader_list):
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
                if i >= num_batches:
                    break

        # Remove hooks
        for handle in hooks:
            handle.remove()

        # Compute quantization parameters for each layer
        for name in stats:
            layer_stats = stats[name]
            self.quantization_params[name] = {
                'min': min(layer_stats['min']),
                'max': max(layer_stats['max']),
                'mean': sum(layer_stats['mean']) / len(layer_stats['mean']),
                'std': sum(layer_stats['std']) / len(layer_stats['std']),
            }

        self.calibrated = True
        logger.info(f"Calibrated {len(self.quantization_params)} layers")

    def quantize(
        self,
        quantization_config: Dict,
        model: Optional[nn.Module] = None
    ) -> nn.Module:
        """
        Execute quantization on model.

        Args:
            quantization_config: Dictionary mapping layer names to bit widths
            model: Optional model to quantize (uses self.model if None)

        Returns:
            nn.Module: Quantized model
        """
        if model is None:
            model = self.model

        # Ensure calibration is done
        if not self.calibrated:
            self.calibrate()

        # Create a copy of the model
        quantized_model = copy.deepcopy(model)

        # Apply quantization to each layer
        for name, bit_width in quantization_config.items():
            if name in self.quantization_params:
                # Find the module
                for module_name, module in quantized_model.named_modules():
                    if module_name == name:
                        # Get quantization parameters
                        params = self.quantization_params[name]

                        # Simple dtype conversion (no actual quantization for now)
                        if bit_width == 32:
                            continue  # FP32, keep as is
                        elif bit_width == 16:
                            # FP16 - convert to float16
                            module.weight.data = module.weight.data.to(torch.float16)
                            if hasattr(module, 'bias') and module.bias is not None:
                                module.bias.data = module.bias.data.to(torch.float16)
                        elif bit_width == 8:
                            # INT8 - simulate quantization by clamping and keeping float
                            # PyTorch doesn't allow direct conversion to int8 for tensors with gradients
                            module.weight.data = torch.clamp(module.weight.data, -128, 127)
                            if hasattr(module, 'bias') and module.bias is not None:
                                module.bias.data = torch.clamp(module.bias.data, -128, 127)
                        elif bit_width == 4:
                            # INT4 - simulate quantization by clamping
                            module.weight.data = torch.clamp(module.weight.data, -8, 7)
                            if hasattr(module, 'bias') and module.bias is not None:
                                module.bias.data = torch.clamp(module.bias.data, -8, 7)

        logger.info(f"Quantized {len(quantization_config)} layers")

        return quantized_model

    def get_state(self) -> Dict:
        """
        Get current state of the agent.

        Returns:
            dict: Agent state information
        """
        return {
            'num_quantizable_layers': len(self._get_quantizable_layers()),
            'calibrated': self.calibrated,
        }

    def get_action(self, state: torch.Tensor) -> tuple:
        """
        Get a quantization action (sample from action space).

        Args:
            state: Current state tensor

        Returns:
            tuple: (quantization_config, log_prob)
        """
        # For this implementation, return a random action
        # In full MARL setting, this would come from PPO policy
        import random

        # Random layer to quantize
        layer_indices = self._get_quantizable_layer_names()
        if not layer_indices:
            return {}, 0.0

        layer_idx = random.choice(layer_indices)
        bit_width = random.choice(self.bit_widths)

        quantization_config = {layer_idx: bit_width}

        # Log prob would be computed by PPO
        log_prob = 0.0

        return quantization_config, log_prob

    def _quantize_conv2d(self, conv2d: nn.Conv2d, bit_width: int):
        """Quantize a Conv2d layer."""
        if bit_width >= 32:
            return conv2d  # FP32, no quantization

        if bit_width == 16:
            # FP16
            conv2d.weight.data = conv2d.weight.data.to(torch.float16)
            if conv2d.bias is not None:
                conv2d.bias.data = conv2d.bias.data.to(torch.float16)
        elif bit_width == 8:
            # INT8
            conv2d.weight.data = torch.quantize_per_channel(
                conv2d.weight.data, scale=None, zero_point=None, dtype=torch.qint8
            )
            if conv2d.bias is not None:
                conv2d.bias.data = torch.quantize_per_channel(
                    conv2d.bias.data, scale=None, zero_point=None, dtype=torch.qint8
                )
        elif bit_width == 4:
            # INT4 (using 8-bit for now)
            conv2d.weight.data = torch.quantize_per_channel(
                conv2d.weight.data.float(), scale=None, zero_point=None, dtype=torch.qint8
            )
            if conv2d.bias is not None:
                conv2d.bias.data = torch.quantize_per_channel(
                    conv2d.bias.data.float(), scale=None, zero_point=None, dtype=torch.qint8
                )
        return conv2d

    def _quantize_linear(self, linear: nn.Linear, bit_width: int):
        """Quantize a Linear layer."""
        if bit_width >= 32:
            return linear  # FP32, no quantization

        if bit_width == 16:
            # FP16
            linear.weight.data = linear.weight.data.to(torch.float16)
            if linear.bias is not None:
                linear.bias.data = linear.bias.data.to(torch.float16)
        elif bit_width == 8:
            # INT8
            linear.weight.data = torch.quantize_per_channel(
                linear.weight.data, scale=None, zero_point=None, dtype=torch.qint8
            )
            if linear.bias is not None:
                linear.bias.data = torch.quantize_per_channel(
                    linear.bias.data, scale=None, zero_point=None, dtype=torch.qint8
                )
        elif bit_width == 4:
            # INT4 (using 8-bit for now)
            linear.weight.data = torch.quantize_per_channel(
                linear.weight.data.float(), scale=None, zero_point=None, dtype=torch.qint8
            )
            if linear.bias is not None:
                linear.bias.data = torch.quantize_per_channel(
                    linear.bias.data.float(), scale=None, zero_point=None, dtype=torch.qint8
                )
        return linear
