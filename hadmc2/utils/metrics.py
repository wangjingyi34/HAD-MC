"""Metrics calculation utilities for HAD-MC 2.0"""

import torch
import torch.nn as nn
import time
from typing import Dict, Tuple
import numpy as np


class MetricsCalculator:
    """
    Calculator for various model metrics including accuracy, latency, energy, and size.
    """

    def __init__(self, device='cpu'):
        self.device = device

    def calculate_accuracy(self, model: nn.Module, dataloader) -> float:
        """
        Calculate model accuracy on a dataset.

        Args:
            model: PyTorch model
            dataloader: DataLoader for evaluation

        Returns:
            float: Accuracy value [0, 1]
        """
        model.eval()
        model.to(self.device)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)

                # Handle different output formats
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                # Get predictions
                if outputs.dim() > 1:
                    # Classification: take argmax
                    _, predicted = outputs.max(1)
                else:
                    # Binary classification
                    predicted = (outputs > 0).long()

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def calculate_latency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> float:
        """
        Measure model inference latency.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_warmup: Number of warmup iterations
            num_iterations: Number of timing iterations

        Returns:
            float: Average latency in milliseconds
        """
        model.eval()
        model.to(self.device)

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)

        # Synchronize for accurate timing
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mlu':
            # For Cambricon MLU
            import torch_mlu
            torch_mlu.synchronize()
        elif self.device == 'npu':
            # For Ascend NPU
            import torch_npu
            torch_npu.npu.synchronize()

        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)

        # Synchronize again
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mlu':
            import torch_mlu
            torch_mlu.synchronize()
        elif self.device == 'npu':
            import torch_npu
            torch_npu.npu.synchronize()

        end_time = time.time()

        avg_latency = ((end_time - start_time) / num_iterations) * 1000  # Convert to ms
        return avg_latency

    def calculate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """
        Calculate model FLOPs.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape

        Returns:
            int: Total FLOPs
        """
        def count_conv(module, input_shape):
            # Calculate FLOPs for Conv2d
            output_channels = module.out_channels
            input_channels = module.in_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            output_size = input_shape[2] // module.stride[0] * input_shape[3] // module.stride[1]

            flops = kernel_size * input_channels * output_channels * output_size * output_size
            return flops, (1, input_shape[1], output_size, output_size)

        def count_linear(module, input_shape):
            # Calculate FLOPs for Linear
            input_size = input_shape[2] * input_shape[3]  # H * W
            flops = module.in_features * module.out_features * input_size
            return flops, (1, input_shape[1], module.out_features)

        def count_bn(module, input_shape):
            # BatchNorm has negligible compute
            return 0, input_shape

        # Hook to count FLOPs
        hooks = []
        total_flops = 0
        layer_shapes = []

        def register_hook(module):
            def hook(module, input, output):
                nonlocal total_flops
                input_shape = input[0].shape

                if isinstance(module, nn.Conv2d):
                    flops, out_shape = count_conv(module, input_shape)
                elif isinstance(module, nn.Linear):
                    # Flatten input first
                    if len(input_shape) > 2:
                        input_shape = (input_shape[0], input_shape[1], -1)
                    flops, out_shape = count_linear(module, input_shape)
                elif isinstance(module, nn.BatchNorm2d):
                    flops, out_shape = count_bn(module, input_shape)
                else:
                    flops, out_shape = 0, input_shape

                total_flops += flops
                layer_shapes.append(out_shape)

            return hook

        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hook = module.register_forward_hook(register_hook(module))
                hooks.append(hook)

        # Run a forward pass to trigger hooks
        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            _ = model(dummy_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return total_flops

    def calculate_model_size(self, model: nn.Module) -> float:
        """
        Calculate model size in megabytes (MB).

        Args:
            model: PyTorch model

        Returns:
            float: Model size in MB
        """
        total_params = sum(p.numel() * p.element_size() for p in model.parameters())
        size_mb = total_params / (1024 * 1024)
        return size_mb

    def calculate_sparsity(self, model: nn.Module) -> float:
        """
        Calculate model sparsity (percentage of zero parameters).

        Args:
            model: PyTorch model

        Returns:
            float: Sparsity ratio [0, 1]
        """
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        sparsity = zero_params / total_params if total_params > 0 else 0.0
        return sparsity

    def calculate_compression_ratio(
        self,
        original_size: float,
        compressed_size: float
    ) -> float:
        """
        Calculate compression ratio.

        Args:
            original_size: Original model size
            compressed_size: Compressed model size

        Returns:
            float: Compression ratio
        """
        if compressed_size == 0:
            return float('inf')
        return original_size / compressed_size

    def evaluate_comprehensive(
        self,
        model: nn.Module,
        dataloader,
        input_shape: Tuple[int, ...]
    ) -> Dict:
        """
        Calculate all relevant metrics.

        Args:
            model: PyTorch model
            dataloader: DataLoader for accuracy
            input_shape: Input shape for latency

        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'accuracy': self.calculate_accuracy(model, dataloader),
            'latency_ms': self.calculate_latency(model, input_shape),
            'model_size_mb': self.calculate_model_size(model),
            'flops': self.calculate_flops(model, input_shape),
            'sparsity': self.calculate_sparsity(model),
        }

        # Calculate derived metrics
        metrics['throughput'] = 1000 / metrics['latency_ms']  # FPS
        metrics['params_millions'] = sum(p.numel() for p in model.parameters()) / 1e6

        return metrics

    def format_metrics(self, metrics: Dict) -> str:
        """
        Format metrics for pretty printing.

        Args:
            metrics: Dictionary of metrics

        Returns:
            str: Formatted string
        """
        lines = [
            "=" * 60,
            "Model Metrics",
            "=" * 60,
            f"Accuracy:         {metrics['accuracy']:.4f}",
            f"Latency:          {metrics['latency_ms']:.2f} ms",
            f"Throughput:       {metrics['throughput']:.2f} FPS",
            f"Model Size:       {metrics['model_size_mb']:.2f} MB",
            f"Parameters:       {metrics['params_millions']:.2f} M",
            f"FLOPs:            {metrics['flops']:,}",
            f"Sparsity:         {metrics['sparsity']:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)
