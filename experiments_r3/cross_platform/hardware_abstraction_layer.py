"""
Hardware Abstraction Layer (HAL) for HAD-MC 2.0

This module provides a unified interface for cross-platform inference
with hardware-specific optimizations and latency prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


class DeviceInfo:
    """Information about a hardware device."""

    def __init__(
        self,
        name: str,
        type: str,  # 'gpu', 'npu', 'mlu', 'cpu'
        compute_capability: float,  # TFLOPS
        memory_bandwidth: float,  # GB/s
        memory_capacity: float,  # GB
        tensor_core: bool,
        sparsity_support: bool,
        supported_precisions: List[int]  # [4, 8, 16, 32]
    ):
        self.name = name
        self.type = type
        self.compute_capability = compute_capability
        self.memory_bandwidth = memory_bandwidth
        self.memory_capacity = memory_capacity
        self.tensor_core = tensor_core
        self.sparsity_support = sparsity_support
        self.supported_precisions = supported_precisions


class LatencyPredictor:
    """Predict inference latency for different hardware and precisions."""

    def __init__(self):
        # Base latency factors per layer type (relative to FP32 CPU)
        self.base_latencies = {
            'Conv2d': {
                'cpu': {'fp32': 1.0, 'fp16': 0.9, 'int8': 0.8},
                'cuda': {'fp32': 0.3, 'fp16': 0.15, 'int8': 0.08},
                'npu': {'fp32': 0.25, 'fp16': 0.12, 'int8': 0.07},
                'mlu': {'fp32': 0.28, 'fp16': 0.14, 'int8': 0.09}
            },
            'Linear': {
                'cpu': {'fp32': 1.0, 'fp16': 0.9, 'int8': 0.8},
                'cuda': {'fp32': 0.35, 'fp16': 0.18, 'int8': 0.10},
                'npu': {'fp32': 0.30, 'fp16': 0.15, 'int8': 0.08},
                'mlu': {'fp32': 0.32, 'fp16': 0.16, 'int8': 0.09}
            },
            'BatchNorm2d': {
                'cpu': {'fp32': 1.0, 'fp16': 0.9, 'int8': 0.8},
                'cuda': {'fp32': 0.2, 'fp16': 0.1, 'int8': 0.05},
                'npu': {'fp32': 0.18, 'fp16': 0.09, 'int8': 0.04},
                'mlu': {'fp32': 0.19, 'fp16': 0.095, 'int8': 0.045}
            }
        }

    def predict_layer_latency(
        self,
        layer_type: str,
        device_type: str,
        precision: str,
        flops: int
    ) -> float:
        """
        Predict latency for a single layer.

        Args:
            layer_type: Type of layer ('Conv2d', 'Linear', etc.)
            device_type: Type of device ('cpu', 'cuda', 'npu', 'mlu')
            precision: Precision ('fp32', 'fp16', 'int8')
            flops: Number of floating point operations

        Returns:
            Predicted latency in milliseconds
        """
        if layer_type not in self.base_latencies:
            layer_type = 'Conv2d'

        if device_type not in self.base_latencies[layer_type]:
            device_type = 'cpu'

        if precision not in self.base_latencies[layer_type][device_type]:
            precision = 'fp32'

        factor = self.base_latencies[layer_type][device_type][precision]

        # Base latency scales with FLOPs (roughly)
        # Assume 1 GFLOP takes 1ms on CPU FP32
        base_latency = (flops / 1e9) * 1.0

        return base_latency * factor

    def predict_model_latency(
        self,
        model: nn.Module,
        device_info: DeviceInfo,
        precision: str = 'fp32',
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> float:
        """
        Predict total model latency.

        Args:
            model: PyTorch model
            device_info: Device information
            precision: Precision to use
            input_shape: Input tensor shape

        Returns:
            Predicted latency in milliseconds
        """
        total_latency = 0.0

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                flops = (module.kernel_size[0] * module.kernel_size[1] *
                        module.in_channels * module.out_channels *
                        input_shape[2] * input_shape[3])
                latency = self.predict_layer_latency(
                    'Conv2d', device_info.type, precision, flops
                )
                total_latency += latency
            elif isinstance(module, nn.Linear):
                flops = module.in_features * module.out_features
                latency = self.predict_layer_latency(
                    'Linear', device_info.type, precision, flops
                )
                total_latency += latency
            elif isinstance(module, nn.BatchNorm2d):
                flops = module.num_features * input_shape[2] * input_shape[3]
                latency = self.predict_layer_latency(
                    'BatchNorm2d', device_info.type, precision, flops
                )
                total_latency += latency

        return total_latency


class EnergyPredictor:
    """Predict energy consumption for different hardware."""

    def __init__(self):
        # Base energy consumption per TFLOP (Joules)
        self.energy_per_tflop = {
            'cpu': 1.0,
            'cuda': 0.3,
            'npu': 0.25,
            'mlu': 0.28
        }

        # Precision energy factors
        self.precision_energy_factor = {
            'fp32': 1.0,
            'fp16': 0.5,
            'int8': 0.25,
            'int4': 0.15
        }

    def predict_energy(
        self,
        model: nn.Module,
        device_info: DeviceInfo,
        precision: str = 'fp32',
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        latency_ms: float = 0.0
    ) -> float:
        """
        Predict energy consumption.

        Args:
            model: PyTorch model
            device_info: Device information
            precision: Precision to use
            input_shape: Input tensor shape
            latency_ms: Latency in ms (if known)

        Returns:
            Predicted energy in Joules
        """
        # Estimate FLOPs
        flops = self._estimate_flops(model, input_shape)

        # Calculate base energy
        base_energy = (flops / 1e12) * self.energy_per_tflop[device_info.type]

        # Apply precision factor
        energy = base_energy * self.precision_energy_factor.get(precision, 1.0)

        # Alternative: use latency and power
        # Average power (W) = FLOPS / Energy-per-FLOP
        # Energy (J) = Power (W) * Time (s)
        if latency_ms > 0:
            power = flops / 1e12 / self.energy_per_tflop[device_info.type]
            energy = power * (latency_ms / 1000.0)

        return energy

    def _estimate_flops(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> int:
        """Estimate FLOPs for model."""
        flops = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                flops += (module.kernel_size[0] * module.kernel_size[1] *
                         module.in_channels * module.out_channels *
                         input_shape[2] * input_shape[3])
            elif isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features

        return flops


class HardwareAbstractionLayer:
    """
    Hardware Abstraction Layer for cross-platform inference.

    Provides unified interface for:
    - Device detection and selection
    - Latency prediction
    - Energy prediction
    - Precision optimization
    """

    def __init__(self):
        self.latency_predictor = LatencyPredictor()
        self.energy_predictor = EnergyPredictor()
        self.available_devices = self._detect_devices()

    def _detect_devices(self) -> List[DeviceInfo]:
        """Detect available hardware devices."""
        devices = []

        # Detect CUDA (NVIDIA)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                compute_cap = torch.cuda.get_device_capability(i)

                # Estimate compute capability (TFLOPS)
                # This is a rough estimate based on GPU generation
                tflops = {
                    (8, 0): 312,   # A100
                    (8, 6): 70,     # Orin
                    (7, 5): 35,     # RTX 30 series
                    (7, 0): 14,     # RTX 20 series
                }.get(compute_cap, 20)

                devices.append(DeviceInfo(
                    name=f'CUDA:{i} ({name})',
                    type='cuda',
                    compute_capability=tflops,
                    memory_bandwidth=1500,  # GB/s (approximate)
                    memory_capacity=torch.cuda.get_device_properties(i).total_memory / 1e9,
                    tensor_core=True,
                    sparsity_support=True,
                    supported_precisions=[8, 16, 32]
                ))

        # Detect NPU (Huawei Ascend) - if available
        try:
            import torch_npu
            if torch.npu.is_available():
                devices.append(DeviceInfo(
                    name='NPU:0 (Ascend)',
                    type='npu',
                    compute_capability=320,
                    memory_bandwidth=1200,
                    memory_capacity=32,
                    tensor_core=True,
                    sparsity_support=True,
                    supported_precisions=[4, 8, 16, 32]
                ))
        except ImportError:
            pass

        # Detect MLU (Cambricon) - if available
        try:
            import cambricon
            devices.append(DeviceInfo(
                name='MLU:0 (Cambricon)',
                type='mlu',
                compute_capability=256,
                memory_bandwidth=1000,
                memory_capacity=32,
                tensor_core=True,
                sparsity_support=False,
                supported_precisions=[8, 16, 32]
            ))
        except ImportError:
            pass

        # Always add CPU as fallback
        devices.append(DeviceInfo(
            name='CPU',
            type='cpu',
            compute_capability=0.5,
            memory_bandwidth=50,
            memory_capacity=16,  # System memory
            tensor_core=False,
            sparsity_support=False,
            supported_precisions=[16, 32]
        ))

        return devices

    def get_available_devices(self) -> List[DeviceInfo]:
        """Get list of available devices."""
        return self.available_devices

    def get_best_device(self) -> DeviceInfo:
        """Get the best available device."""
        # Priority: CUDA > NPU > MLU > CPU
        for device in self.available_devices:
            if device.type == 'cuda':
                return device
        for device in self.available_devices:
            if device.type == 'npu':
                return device
        for device in self.available_devices:
            if device.type == 'mlu':
                return device
        return self.available_devices[-1]  # CPU

    def optimize_precision(
        self,
        model: nn.Module,
        device_info: DeviceInfo,
        target_latency: float
    ) -> str:
        """
        Find optimal precision for target latency.

        Args:
            model: PyTorch model
            device_info: Device information
            target_latency: Target latency in ms

        Returns:
            Optimal precision string
        """
        best_precision = 'fp32'
        best_latency = float('inf')

        for precision in ['fp32', 'fp16', 'int8', 'int4']:
            if precision not in device_info.supported_precisions:
                continue

            latency = self.latency_predictor.predict_model_latency(
                model, device_info, precision
            )

            if latency <= target_latency and latency < best_latency:
                best_latency = latency
                best_precision = precision

        return best_precision

    def predict_metrics(
        self,
        model: nn.Module,
        device_info: Optional[DeviceInfo] = None,
        precision: str = 'fp32'
    ) -> Dict[str, float]:
        """
        Predict all metrics for a model on a device.

        Args:
            model: PyTorch model
            device_info: Device information (default: best available)
            precision: Precision to use

        Returns:
            Dictionary of predicted metrics
        """
        if device_info is None:
            device_info = self.get_best_device()

        latency = self.latency_predictor.predict_model_latency(
            model, device_info, precision
        )

        energy = self.energy_predictor.predict_energy(
            model, device_info, precision, latency_ms=latency
        )

        # Throughput (FPS)
        throughput = 1000.0 / latency if latency > 0 else 0

        return {
            'latency_ms': latency,
            'energy_j': energy,
            'throughput_fps': throughput,
            'device': device_info.name,
            'precision': precision
        }
