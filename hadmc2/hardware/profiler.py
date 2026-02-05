"""Hardware Profiler for HAD-MC 2.0"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HardwareProfiler:
    """
    Hardware profiler for measuring actual inference performance.

    Provides detailed profiling of:
    - Layer-wise latency
    - Memory usage
    - Power consumption (if available)
    - Throughput
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize hardware profiler.

        Args:
            device: Target device ('cuda', 'cpu', 'mlu', 'npu', etc.)
        """
        self.device = device
        self.profile_results = {}

        # Check if device is available
        self.device_available = self._check_device_available()

        if not self.device_available:
            logger.warning(f"Device {device} not available, profiler may not work correctly")

    def _check_device_available(self) -> bool:
        """Check if the specified device is available."""
        if self.device == 'cuda':
            return torch.cuda.is_available()
        elif self.device == 'mlu':
            try:
                import torch_mlu
                return torch_mlu.is_available()
            except ImportError:
                return False
        elif self.device == 'npu':
            try:
                import torch_npu
                return torch_npu.is_available()
            except ImportError:
                return False
        elif self.device == 'cpu':
            return True
        else:
            logger.warning(f"Unknown device: {self.device}")
            return False

    def profile_layer(
        self,
        layer: nn.Module,
        input_tensor: torch.Tensor,
        num_warmup: int = 10,
        num_iterations: int = 100,
        verbose: bool = False
    ) -> Dict:
        """
        Profile a single layer.

        Args:
            layer: PyTorch layer to profile
            input_tensor: Input tensor
            num_warmup: Number of warmup iterations
            num_iterations: Number of measurement iterations
            verbose: Print detailed information

        Returns:
            dict: Profiling results
        """
        layer.eval()

        # Move to target device
        input_tensor = input_tensor.to(self.device)
        layer = layer.to(self.device)

        result = {
            'layer_type': layer.__class__.__name__,
            'latency_ms': 0,
            'latency_std_ms': 0,
            'throughput_fps': 0,
            'device': self.device,
        }

        try:
            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = layer(input_tensor)

            # Synchronize for accurate timing
            self._synchronize()

            # Measure
            latencies = []
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = layer(input_tensor)

                self._synchronize()
                start = time.time()
                with torch.no_grad():
                    _ = layer(input_tensor)
                self._synchronize()
                end = time.time()

                latencies.append((end - start) * 1000)  # Convert to ms

            # Calculate statistics
            import numpy as np
            latencies = np.array(latencies)
            result['latency_ms'] = float(np.mean(latencies))
            result['latency_std_ms'] = float(np.std(latencies))
            result['throughput_fps'] = 1000 / result['latency_ms']

            if verbose:
                logger.info(f"Layer profiling: {layer.__class__.__name__}")
                logger.info(f"  Latency: {result['latency_ms']:.4f} ± {result['latency_std_ms']:.4f} ms")
                logger.info(f"  Throughput: {result['throughput_fps']:.2f} FPS")

        except Exception as e:
            logger.error(f"Error profiling layer: {e}")
            result['error'] = str(e)

        return result

    def profile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_warmup: int = 10,
        num_iterations: int = 100,
        profile_layers: bool = True
    ) -> Dict:
        """
        Profile entire model.

        Args:
            model: PyTorch model to profile
            input_shape: Input tensor shape
            num_warmup: Number of warmup iterations
            num_iterations: Number of measurement iterations
            profile_layers: Profile individual layers

        Returns:
            dict: Complete profiling results
        """
        model.eval()
        device = self.device

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)

        result = {
            'total_latency_ms': 0,
            'total_latency_std_ms': 0,
            'throughput_fps': 0,
            'layer_results': {},
            'device': device,
        }

        # Profile entire model
        try:
            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(dummy_input)

            self._synchronize()

            # Measure
            latencies = []
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model(dummy_input)

                self._synchronize()
                start = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                self._synchronize()
                end = time.time()

                latencies.append((end - start) * 1000)

            import numpy as np
            latencies = np.array(latencies)
            result['total_latency_ms'] = float(np.mean(latencies))
            result['total_latency_std_ms'] = float(np.std(latencies))
            result['throughput_fps'] = 1000 / result['total_latency_ms']

            # Profile individual layers
            if profile_layers:
                for name, layer in model.named_modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                        result['layer_results'][name] = self.profile_layer(
                            layer, dummy_input, num_warmup=20, num_iterations=50
                        )

            logger.info(f"Model profiling complete:")
            logger.info(f"  Total latency: {result['total_latency_ms']:.4f} ms")
            logger.info(f"  Throughput: {result['throughput_fps']:.2f} FPS")
            logger.info(f"  Profiled {len(result['layer_results'])} layers")

        except Exception as e:
            logger.error(f"Error profiling model: {e}")
            result['error'] = str(e)

        return result

    def _synchronize(self):
        """Synchronize device operations."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mlu':
            try:
                import torch_mlu
                torch_mlu.synchronize()
            except ImportError:
                pass
        elif self.device == 'npu':
            try:
                import torch_npu
                torch_npu.npu.synchronize()
            except ImportError:
                pass

    def measure_power(self, model: nn.Module, input_tensor: torch.Tensor) -> Optional[float]:
        """
        Measure power consumption (if supported).

        Args:
            model: PyTorch model
            input_tensor: Input tensor

        Returns:
            Optional[float]: Power in Watts, or None if not supported
        """
        # Power measurement is hardware-specific
        # For NVIDIA: use nvidia-smi
        # For other devices: vendor-specific tools

        if self.device == 'cuda':
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                power_str = result.stdout.strip()
                if power_str:
                    power_w = float(power_str)
                    return power_w
            except Exception as e:
                logger.debug(f"Could not measure power: {e}")

        return None

    def get_device_info(self) -> Dict:
        """
        Get information about the current device.

        Returns:
            dict: Device information
        """
        info = {
            'device': self.device,
            'available': self.device_available,
        }

        if self.device == 'cuda' and torch.cuda.is_available():
            info['name'] = torch.cuda.get_device_name(0)
            info['capability'] = torch.cuda.get_device_capability(0)
            info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        elif self.device == 'cpu':
            info['name'] = 'CPU'
            info['num_threads'] = torch.get_num_threads()

        return info

    def format_results(self, results: Dict) -> str:
        """
        Format profiling results for printing.

        Args:
            results: Profiling results dictionary

        Returns:
            str: Formatted string
        """
        lines = [
            "=" * 60,
            "Hardware Profiling Results",
            "=" * 60,
            f"Device: {results.get('device', 'unknown')}",
            f"Total Latency: {results.get('total_latency_ms', 0):.4f} ms",
            f"Throughput: {results.get('throughput_fps', 0):.2f} FPS",
        ]

        if 'layer_results' in results and results['layer_results']:
            lines.append("")
            lines.append("Layer-wise Results:")
            lines.append("-" * 60)

            for layer_name, layer_result in results['layer_results'].items():
                if 'error' not in layer_result:
                    lines.append(f"  {layer_name}:")
                    lines.append(f"    Type: {layer_result.get('layer_type', 'N/A')}")
                    lines.append(f"    Latency: {layer_result.get('latency_ms', 0):.4f} ms")

        lines.append("=" * 60)

        return "\n".join(lines)
