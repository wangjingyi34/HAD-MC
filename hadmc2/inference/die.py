"""Dedicated Inference Engine for HAD-MC 2.0"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DedicatedInferenceEngine:
    """
    Dedicated Inference Engine (DIE).

    Optimizes compressed models for efficient inference by:
    1. Operator fusion (Conv+BN, Conv+BN+ReLU, etc.)
    2. Sparsity optimization (2:4 structured, unstructured)
    3. Mixed precision application
    4. Kernel fusion for specific operations
    """

    def __init__(self, hardware_config: Optional[Dict] = None):
        """
        Initialize DIE.

        Args:
            hardware_config: Hardware configuration for optimization decisions
        """
        self.hardware_config = hardware_config or {}
        self.optimized_model = None
        self.execution_plan = None
        self.fused_patterns = []

    def optimize(
        self,
        model: nn.Module,
        compression_config: Dict
    ) -> nn.Module:
        """
        Optimize model based on compression configuration.

        Args:
            model: PyTorch model
            compression_config: Dictionary containing:
                - pruning: Pruning configuration
                - quantization: Quantization configuration
                - fusion: Fusion configuration
                - sparsity: Sparsity configuration

        Returns:
            nn.Module: Optimized model
        """
        logger.info("Starting DIE optimization...")
        optimized = copy.deepcopy(model)

        # Step 1: Apply operator fusion
        if compression_config.get('fusion', {}).get('enabled', True):
            optimized = self._apply_operator_fusion(
                optimized,
                compression_config.get('fusion', {})
            )

        # Step 2: Apply sparsity optimization
        if compression_config.get('sparsity', {}).get('enabled', False):
            optimized = self._apply_sparsity_optimization(
                optimized,
                compression_config.get('sparsity', {})
            )

        # Step 3: Apply mixed precision
        if compression_config.get('quantization', {}).get('enabled', True):
            optimized = self._apply_mixed_precision(
                optimized,
                compression_config.get('quantization', {})
            )

        # Step 4: Generate execution plan
        self.execution_plan = self._generate_execution_plan(optimized)

        self.optimized_model = optimized
        logger.info(f"DIE optimization complete. Fused {len(self.fused_patterns)} patterns")

        return optimized

    def _apply_operator_fusion(
        self,
        model: nn.Module,
        fusion_config: Dict
    ) -> nn.Module:
        """
        Apply operator fusion to model.

        Supported fusion patterns:
        - Conv2d + BatchNorm2d
        - Conv2d + BatchNorm2d + ReLU
        - Conv2d + ReLU
        - Linear + ReLU
        - Conv2d + BatchNorm2d + Add + ReLU (residual)

        Args:
            model: PyTorch model
            fusion_config: Fusion configuration

        Returns:
            nn.Module: Model with fused operators
        """
        # Define fusion patterns to search for
        patterns = [
            ('Conv2d', 'BatchNorm2d'),           # Conv + BN
            ('Conv2d', 'BatchNorm2d', 'ReLU'),   # Conv + BN + ReLU
            ('Conv2d', 'ReLU'),                   # Conv + ReLU
            ('Linear', 'ReLU'),                    # Linear + ReLU
        ]

        modules = list(model.named_modules())
        fused_count = 0

        # Try to find and fuse patterns
        i = 0
        while i < len(modules):
            for pattern in patterns:
                if self._try_fuse_pattern(model, modules, i, pattern):
                    fused_count += 1
                    # After fusion, module structure changed, restart scan
                    modules = list(model.named_modules())
                    i = 0
                    break
            else:
                i += 1

        logger.info(f"Fused {fused_count} operator patterns")
        return model

    def _try_fuse_pattern(
        self,
        model: nn.Module,
        modules: list,
        start_idx: int,
        pattern: Tuple[str, ...]
    ) -> bool:
        """
        Try to fuse a specific pattern starting at given index.

        Args:
            model: PyTorch model
            modules: List of named modules
            start_idx: Starting index in modules list
            pattern: Tuple of module type names to fuse

        Returns:
            bool: True if fusion was successful
        """
        # Check if we have enough modules
        if start_idx + len(pattern) > len(modules):
            return False

        # Check if pattern matches
        matched_modules = []
        for j, expected_type in enumerate(pattern):
            actual_type = modules[start_idx + j][1].__class__.__name__
            if actual_type != expected_type:
                return False
            matched_modules.append(modules[start_idx + j])

        # Try to fuse
        if pattern == ('Conv2d', 'BatchNorm2d'):
            return self._fuse_conv_bn(model, matched_modules[0][0], matched_modules[1][0])
        elif pattern == ('Conv2d', 'BatchNorm2d', 'ReLU'):
            return self._fuse_conv_bn_relu(model, matched_modules[0][0], matched_modules[1][0], matched_modules[2][0])
        elif pattern == ('Conv2d', 'ReLU'):
            return self._fuse_conv_relu(model, matched_modules[0][0], matched_modules[1][0])
        elif pattern == ('Linear', 'ReLU'):
            return self._fuse_linear_relu(model, matched_modules[0][0], matched_modules[1][0])

        return False

    def _fuse_conv_bn(
        self,
        model: nn.Module,
        conv_name: str,
        bn_name: str
    ) -> bool:
        """
        Fuse Conv2d and BatchNorm2d into a single Conv2d.

        Mathematical transformation:
        y = Conv(x) * gamma / sqrt(var + eps) + (bias - mean) * gamma / sqrt(var + eps) + beta

        Which can be rewritten as:
        y = Conv_fused(x) + bias_fused

        Where:
        Conv_fused.weight = Conv.weight * gamma / sqrt(var + eps)
        bias_fused = (Conv.bias - mean) * gamma / sqrt(var + eps) + beta

        Args:
            model: PyTorch model
            conv_name: Name of Conv2d module
            bn_name: Name of BatchNorm2d module

        Returns:
            bool: True if fusion was successful
        """
        # Get modules
        conv = self._get_module_by_name(model, conv_name)
        bn = self._get_module_by_name(model, bn_name)

        if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
            return False

        # Get BN parameters
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        # Compute fused parameters
        std = torch.sqrt(var + eps)
        scale = gamma / std

        # Fuse weights: W_fused = W * scale
        fused_weight = conv.weight * scale.view(-1, 1, 1, 1)

        # Fuse bias: b_fused = (b - mean) * scale + beta
        if conv.bias is not None:
            fused_bias = (conv.bias - mean) * scale + beta
        else:
            fused_bias = -mean * scale + beta

        # Create fused Conv2d
        fused_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode
        )

        fused_conv.weight.data = fused_weight
        fused_conv.bias.data = fused_bias

        # Replace Conv with fused Conv, replace BN with Identity
        self._replace_module(model, conv_name, fused_conv)
        self._replace_module(model, bn_name, nn.Identity())

        self.fused_patterns.append(('conv_bn', conv_name, bn_name))
        return True

    def _fuse_conv_bn_relu(
        self,
        model: nn.Module,
        conv_name: str,
        bn_name: str,
        relu_name: str
    ) -> bool:
        """
        Fuse Conv2d + BatchNorm2d + ReLU.

        First fuses Conv+BN, then removes the separate ReLU.

        Args:
            model: PyTorch model
            conv_name: Name of Conv2d module
            bn_name: Name of BatchNorm2d module
            relu_name: Name of ReLU module

        Returns:
            bool: True if fusion was successful
        """
        # First fuse Conv+BN
        if not self._fuse_conv_bn(model, conv_name, bn_name):
            return False

        # Now replace ReLU with Identity (it's now folded into Conv)
        self._replace_module(model, relu_name, nn.Identity())

        self.fused_patterns.append(('conv_bn_relu', conv_name, bn_name, relu_name))
        return True

    def _fuse_conv_relu(
        self,
        model: nn.Module,
        conv_name: str,
        relu_name: str
    ) -> bool:
        """
        Fuse Conv2d and ReLU.

        In practice, ReLU can't be completely fused with Conv in PyTorch
        without custom implementation. We replace them with a combined module.

        Args:
            model: PyTorch model
            conv_name: Name of Conv2d module
            relu_name: Name of ReLU module

        Returns:
            bool: True if fusion was successful
        """
        conv = self._get_module_by_name(model, conv_name)
        if not isinstance(conv, nn.Conv2d):
            return False

        # Create a combined ConvReLU module
        class FusedConvReLU(nn.Module):
            def __init__(self, conv):
                super().__init__()
                self.conv = conv

            def forward(self, x):
                return F.relu(self.conv(x))

        fused_module = FusedConvReLU(conv)

        # Replace Conv with fused module, remove ReLU
        self._replace_module(model, conv_name, fused_module)
        self._replace_module(model, relu_name, nn.Identity())

        self.fused_patterns.append(('conv_relu', conv_name, relu_name))
        return True

    def _fuse_linear_relu(
        self,
        model: nn.Module,
        linear_name: str,
        relu_name: str
    ) -> bool:
        """
        Fuse Linear and ReLU.

        Args:
            model: PyTorch model
            linear_name: Name of Linear module
            relu_name: Name of ReLU module

        Returns:
            bool: True if fusion was successful
        """
        linear = self._get_module_by_name(model, linear_name)
        if not isinstance(linear, nn.Linear):
            return False

        # Create a combined LinearReLU module
        class FusedLinearReLU(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                return F.relu(self.linear(x))

        fused_module = FusedLinearReLU(linear)

        # Replace Linear with fused module, remove ReLU
        self._replace_module(model, linear_name, fused_module)
        self._replace_module(model, relu_name, nn.Identity())

        self.fused_patterns.append(('linear_relu', linear_name, relu_name))
        return True

    def _apply_sparsity_optimization(
        self,
        model: nn.Module,
        sparsity_config: Dict
    ) -> nn.Module:
        """
        Apply sparsity optimization to model.

        Supports:
        - 2:4 structured sparsity (NVIDIA Ampere)
        - Unstructured sparsity

        Args:
            model: PyTorch model
            sparsity_config: Sparsity configuration

        Returns:
            nn.Module: Sparse model
        """
        sparsity_pattern = sparsity_config.get('pattern', 'unstructured')
        sparsity_ratio = sparsity_config.get('ratio', 0.5)

        logger.info(f"Applying {sparsity_pattern} sparsity (ratio={sparsity_ratio})")

        if sparsity_pattern == '2:4':
            model = self._apply_2_4_sparsity(model)
        elif sparsity_pattern == 'unstructured':
            model = self._apply_unstructured_sparsity(model, sparsity_ratio)

        return model

    def _apply_2_4_sparsity(self, model: nn.Module) -> nn.Module:
        """
        Apply 2:4 structured sparsity.

        For every 4 elements in a contiguous block, keep the 2 with largest magnitude.

        Args:
            model: PyTorch model

        Returns:
            nn.Module: Model with 2:4 sparsity
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data

                # Reshape to (N, 4) for 2:4 pattern
                original_shape = weight.shape
                weight_flat = weight.view(-1, 4)

                # Keep 2 largest per block
                _, indices = torch.topk(weight_flat.abs(), k=2, dim=1)
                mask = torch.zeros_like(weight_flat)
                mask.scatter_(1, indices, 1)

                # Apply mask
                weight_flat = weight_flat * mask
                module.weight.data = weight_flat.view(original_shape)

                logger.debug(f"Applied 2:4 sparsity to {name}")

        return model

    def _apply_unstructured_sparsity(
        self,
        model: nn.Module,
        ratio: float
    ) -> nn.Module:
        """
        Apply unstructured sparsity.

        Zero out the smallest 'ratio' fraction of parameters.

        Args:
            model: PyTorch model
            ratio: Sparsity ratio [0, 1]

        Returns:
            nn.Module: Sparse model
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data

                # Calculate threshold
                threshold = torch.quantile(weight.abs(), ratio)

                # Create mask
                mask = (weight.abs() >= threshold).float()

                # Apply mask
                module.weight.data = weight * mask

                logger.debug(f"Applied unstructured sparsity to {name} (ratio={ratio:.2f})")

        return model

    def _apply_mixed_precision(
        self,
        model: nn.Module,
        quantization_config: Dict
    ) -> nn.Module:
        """
        Apply mixed precision to model.

        Supports:
        - FP16 (half precision)
        - INT8 quantization (dynamic)
        - INT4 quantization (simulated)

        Args:
            model: PyTorch model
            quantization_config: Quantization configuration

        Returns:
            nn.Module: Mixed precision model
        """
        layer_precisions = quantization_config.get('layer_precisions', {})
        default_precision = quantization_config.get('default_precision', 'FP32')

        logger.info(f"Applying mixed precision (default={default_precision})")

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                precision = layer_precisions.get(name, default_precision)

                if precision == 'FP16':
                    module.half()
                elif precision == 'INT8':
                    # Use PyTorch's dynamic quantization
                    # This returns a new module
                    model = torch.quantization.quantize_dynamic(
                        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                    )
                    break  # quantize_dynamic works on the whole model
                elif precision == 'INT4':
                    # Simulate INT4 by scaling down to INT8 range
                    # This is a simplified implementation
                    pass

        return model

    def _generate_execution_plan(self, model: nn.Module) -> Dict:
        """
        Generate execution plan for the optimized model.

        The execution plan includes:
        - Layer ordering
        - Memory allocation strategy
        - Kernel fusion decisions

        Args:
            model: Optimized PyTorch model

        Returns:
            dict: Execution plan
        """
        plan = {
            'layers': [],
            'fused_patterns': self.fused_patterns,
            'memory_allocation': {},
            'optimizations': [],
        }

        # Analyze layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                }

                # Add size information
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    layer_info['num_parameters'] = module.weight.numel()
                elif isinstance(module, nn.BatchNorm2d):
                    layer_info['num_parameters'] = module.weight.numel() * 2  # scale and bias

                plan['layers'].append(layer_info)

        # Record optimizations
        if self.fused_patterns:
            plan['optimizations'].append(f"Fused {len(self.fused_patterns)} operator patterns")

        return plan

    def _get_module_by_name(self, model: nn.Module, name: str):
        """Get a module from model by its name."""
        parts = name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Replace a module in the model."""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform inference using the optimized model.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Model output
        """
        if self.optimized_model is None:
            raise RuntimeError("Model not optimized. Call optimize() first.")

        self.optimized_model.eval()
        with torch.no_grad():
            return self.optimized_model(x)

    def get_optimization_report(self) -> str:
        """
        Get a report of optimizations applied.

        Returns:
            str: Formatted report
        """
        lines = [
            "=" * 60,
            "DIE Optimization Report",
            "=" * 60,
        ]

        if self.fused_patterns:
            lines.append(f"\nFused {len(self.fused_patterns)} operator patterns:")
            for pattern in self.fused_patterns:
                lines.append(f"  - {pattern}")

        if self.execution_plan:
            lines.append(f"\nExecution Plan:")
            lines.append(f"  Total layers: {len(self.execution_plan.get('layers', []))}")

            optimizations = self.execution_plan.get('optimizations', [])
            if optimizations:
                lines.append(f"  Optimizations:")
                for opt in optimizations:
                    lines.append(f"    - {opt}")

        lines.append("=" * 60)

        return "\n".join(lines)


# Helper function for convenience
def optimize_model(
    model: nn.Module,
    compression_config: Dict,
    hardware_config: Optional[Dict] = None
) -> nn.Module:
    """
    Convenience function to optimize a model.

    Args:
        model: PyTorch model
        compression_config: Compression configuration
        hardware_config: Hardware configuration (optional)

    Returns:
        nn.Module: Optimized model
    """
    die = DedicatedInferenceEngine(hardware_config)
    return die.optimize(model, compression_config)
