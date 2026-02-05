"""Hardware Abstraction Layer for HAD-MC 2.0"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Hardware configuration dataclass."""
    name: str
    compute_capability: float  # TFLOPS
    memory_bandwidth: float    # GB/s
    memory_capacity: float     # GB
    power_budget: float        # W
    supported_precisions: List[str]
    has_tensor_core: bool
    has_sparsity_support: bool
    device: str = 'cuda:0'

    def to_tensor(self) -> torch.Tensor:
        """Convert hardware features to tensor for MARL."""
        features = [
            self.compute_capability / 100,  # Normalize to [0, 1], assume max 100 TFLOPS
            self.memory_bandwidth / 3000,  # Normalize to [0, 1], assume max 3000 GB/s
            self.memory_capacity / 100,  # Normalize to [0, 1], assume max 100 GB
            self.power_budget / 500,  # Normalize to [0, 1], assume max 500 W
            1.0 if self.has_tensor_core else 0.0,
            1.0 if self.has_sparsity_support else 0.0,
        ]

        # Precision support (multi-hot encoding)
        precision_map = {'FP32': 0, 'FP16': 1, 'INT8': 2, 'INT4': 3}
        precision_features = [0.0] * len(precision_map)
        for p in self.supported_precisions:
            if p in precision_map:
                precision_features[precision_map[p]] = 1.0
        features.extend(precision_features)

        return torch.tensor(features, dtype=torch.float32)


class HardwareAbstractionLayer(ABC):
    """Base class for hardware abstraction layer."""

    @abstractmethod
    def get_compute_capability(self) -> float:
        """Get compute capability (TFLOPS)."""
        pass

    @abstractmethod
    def get_memory_bandwidth(self) -> float:
        """Get memory bandwidth (GB/s)."""
        pass

    @abstractmethod
    def get_memory_capacity(self) -> float:
        """Get memory capacity (GB)."""
        pass

    @abstractmethod
    def get_power_budget(self) -> float:
        """Get power budget (W)."""
        pass

    @abstractmethod
    def get_supported_precisions(self) -> List[str]:
        """Get supported precision formats."""
        pass

    @abstractmethod
    def measure_latency(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """Measure actual inference latency (ms)."""
        pass

    @abstractmethod
    def measure_energy(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """Measure actual inference energy (J)."""
        pass

    @abstractmethod
    def estimate_latency(self, model_config: dict) -> float:
        """Estimate latency using LUT (ms)."""
        pass

    @abstractmethod
    def get_hardware_config(self) -> HardwareConfig:
        """Get hardware configuration."""
        pass

    @abstractmethod
    def get_hardware_features(self) -> torch.Tensor:
        """Get hardware features as tensor for MARL."""
        pass


class SimulatedHardwareAbstractionLayer(HardwareAbstractionLayer):
    """
    Simulated HAL for testing and development.

    Uses analytical models instead of actual hardware measurement.
    """

    def __init__(self, config_path: str = None):
        self.config = None
        self.latency_lut = {}
        self.hardware_config = None

        # Load or use default configuration
        if config_path and config_path.endswith('.yaml'):
            self._load_yaml_config(config_path)
        else:
            self._use_default_config()

        self._build_latency_lut()

    def _use_default_config(self):
        """Use default hardware configuration (NVIDIA A100)."""
        self.hardware_config = HardwareConfig(
            name='NVIDIA_A100',
            compute_capability=19.5,  # TFLOPS (FP16 Tensor Core)
            memory_bandwidth=2039,    # GB/s
            memory_capacity=80,        # GB
            power_budget=400,          # W
            supported_precisions=['FP32', 'FP16', 'BF16', 'INT8', 'INT4'],
            has_tensor_core=True,
            has_sparsity_support=True,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
        )
        logger.info(f"Using default hardware config: {self.hardware_config.name}")

    def _load_yaml_config(self, config_path: str):
        """Load hardware configuration from YAML file."""
        import yaml

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            hw_config = config.get('hardware', {})
            platform_name = list(hw_config.keys())[0]
            hw_data = hw_config[platform_name]

            self.hardware_config = HardwareConfig(
                name=platform_name,
                compute_capability=hw_data.get('compute_capability', 10.0),
                memory_bandwidth=hw_data.get('memory_bandwidth', 1000.0),
                memory_capacity=hw_data.get('memory_capacity', 16.0),
                power_budget=hw_data.get('power_budget', 250.0),
                supported_precisions=hw_data.get('supported_precisions', ['FP32', 'FP16', 'INT8']),
                has_tensor_core=hw_data.get('has_tensor_core', False),
                has_sparsity_support=hw_data.get('has_sparsity_support', False),
                device='cpu',  # Would be configured based on platform
            )

            logger.info(f"Loaded hardware config: {platform_name} from {config_path}")

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default hardware config")
            self._use_default_config()

    def _build_latency_lut(self):
        """
        Build latency lookup table.

        Uses analytical latency models based on FLOPs and memory access.
        """
        self.latency_lut = {}

        # Latency model parameters
        compute_tflops = self.hardware_config.compute_capability
        memory_bw_gbps = self.hardware_config.memory_bandwidth

        # Define latency functions for different layer types and precisions

        # Conv2d latency model
        # latency = alpha * FLOPs / compute + beta * memory_access / bandwidth
        self.latency_lut['conv2d'] = {
            'FP32': lambda flops, mem: (flops / (compute_tflops * 1e12) + mem / (memory_bw_gbps * 1e9)) * 1000,
            'FP16': lambda flops, mem: (flops / (compute_tflops * 2e12) + mem / (memory_bw_gbps * 1e9)) * 1000,
            'INT8': lambda flops, mem: (flops / (compute_tflops * 4e12) + mem / (memory_bw_gbps * 1e9)) * 1000,
            'INT4': lambda flops, mem: (flops / (compute_tflops * 8e12) + mem / (memory_bw_gbps * 1e9)) * 1000,
        }

        # Linear latency model
        self.latency_lut['linear'] = {
            'FP32': lambda flops, mem: (flops / (compute_tflops * 1e12) + mem / (memory_bw_gbps * 1e9)) * 1000,
            'FP16': lambda flops, mem: (flops / (compute_tflops * 2e12) + mem / (memory_bw_gbps * 1e9)) * 1000,
            'INT8': lambda flops, mem: (flops / (compute_tflops * 4e12) + mem / (memory_bw_gbps * 1e9)) * 1000,
        'INT4': lambda flops, mem: (flops / (compute_tflops * 8e12) + mem / (memory_bw_gbps * 1e9)) * 1000,
        }

        logger.info(f"Built latency LUT for {self.hardware_config.name}")

    def get_compute_capability(self) -> float:
        return self.hardware_config.compute_capability

    def get_memory_bandwidth(self) -> float:
        return self.hardware_config.memory_bandwidth

    def get_memory_capacity(self) -> float:
        return self.hardware_config.memory_capacity

    def get_power_budget(self) -> float:
        return self.hardware_config.power_budget

    def get_supported_precisions(self) -> List[str]:
        return self.hardware_config.supported_precisions

    def measure_latency(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> float:
        """
        Measure actual inference latency.

        Args:
            model: PyTorch model
            input_tensor: Input tensor
            num_warmup: Number of warmup iterations
            num_iterations: Number of timing iterations

        Returns:
            float: Average latency in milliseconds
        """
        device = input_tensor.device
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)

        # Synchronize for accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.time()

        avg_latency = ((end - start) / num_iterations) * 1000  # Convert to ms
        return avg_latency

    def measure_energy(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """
        Measure inference energy.

        Note: This is a simplified estimation based on latency.
        Actual energy measurement requires hardware-specific tools.

        Args:
            model: PyTorch model
            input_tensor: Input tensor

        Returns:
            float: Estimated energy in Joules
        """
        # Get latency
        latency_ms = self.measure_latency(model, input_tensor, num_iterations=10)

        # Simplified energy estimation
        # Energy = Power * Time
        power_w = self.hardware_config.power_budget * 0.5  # Assume 50% of TDP during inference
        energy_j = power_w * (latency_ms / 1000)  # W * s = J

        return energy_j

    def estimate_latency(self, model_config: dict) -> float:
        """
        Estimate latency using LUT.

        Args:
            model_config: Dictionary containing layer configurations
                {
                    'layers': [
                        {'type': 'conv2d', 'precision': 'INT8', 'flops': float, 'memory': float},
                        ...
                    ]
                }

        Returns:
            float: Estimated total latency in ms
        """
        total_latency = 0

        for layer_config in model_config.get('layers', []):
            layer_type = layer_config.get('type', 'conv2d')
            precision = layer_config.get('precision', 'FP32')
            flops = layer_config.get('flops', 0)
            memory = layer_config.get('memory', 0)

            # Get latency function from LUT
            if layer_type in self.latency_lut and precision in self.latency_lut[layer_type]:
                latency_fn = self.latency_lut[layer_type][precision]
                layer_latency = latency_fn(flops, memory)
            else:
                # Fallback: use FP32 if precision not found
                latency_fn = self.latency_lut.get(layer_type, {}).get('FP32',
                    lambda f, m: (f / 1e12 + m / 1e9) * 1000)
                layer_latency = latency_fn(flops, memory)

            total_latency += layer_latency

        return total_latency

    def get_hardware_config(self) -> HardwareConfig:
        return self.hardware_config

    def get_hardware_features(self) -> torch.Tensor:
        return self.hardware_config.to_tensor()

    def set_hardware(self, platform_name: str):
        """
        Set hardware platform by name.

        Args:
            platform_name: Name of the platform (e.g., 'nvidia_a100', 'ascend_310')
        """
        # This would load from config files
        # For now, just log it
        logger.info(f"Setting hardware to: {platform_name}")
