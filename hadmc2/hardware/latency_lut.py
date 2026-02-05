"""Latency Lookup Table for HAD-MC 2.0"""

import torch
import torch.nn as nn
import pickle
import os
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LatencyLookupTable:
    """
    Latency Lookup Table (LUT) for fast latency estimation.

    The LUT stores measured latencies for different layer configurations:
    - Layer type (conv2d, linear, etc.)
    - Precision (FP32, FP16, INT8, INT4)
    - Layer dimensions
    - Sparsity level

    This allows for fast latency estimation during MARL training.
    """

    def __init__(self, platform: str = 'default'):
        """
        Initialize LUT.

        Args:
            platform: Hardware platform identifier
        """
        self.platform = platform
        self.lut = {}  # Main lookup table
        self.measurements = {}  # Store raw measurements

    def build(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_samples: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Build latency LUT by measuring model layers.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_samples: Number of samples for averaging
            save_path: Optional path to save LUT
        """
        logger.info(f"Building latency LUT for model on {self.platform}...")

        device = next(model.parameters()).device
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)

        # Measure each layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layer_lut = self._measure_layer(
                    module, name, dummy_input, num_samples
                )
                self.lut[name] = layer_lut
                logger.info(f"  {name}: {layer_lut}")

        logger.info(f"LUT built with {len(self.lut)} entries")

        if save_path:
            self.save(save_path)

    def _measure_layer(
        self,
        module: nn.Module,
        name: str,
        dummy_input: torch.Tensor,
        num_samples: int
    ) -> Dict:
        """
        Measure latency for a single layer at different configurations.

        Args:
            module: PyTorch layer
            name: Layer name
            dummy_input: Input tensor
            num_samples: Number of measurements

        Returns:
            dict: Latency measurements for different configurations
        """
        device = dummy_input.device
        layer_data = {
            'type': module.__class__.__name__,
            'measurements': {},
        }

        # Get layer info
        if isinstance(module, nn.Conv2d):
            layer_info = {
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'groups': module.groups,
            }
        elif isinstance(module, nn.Linear):
            layer_info = {
                'in_features': module.in_features,
                'out_features': module.out_features,
            }
        elif isinstance(module, nn.BatchNorm2d):
            layer_info = {
                'num_features': module.num_features,
            }
        else:
            return layer_data

        layer_data['info'] = layer_info

        # Measure with different precisions
        for precision in ['FP32', 'FP16', 'INT8']:
            latency = self._measure_with_precision(
                module, dummy_input, precision, num_samples
            )
            layer_data['measurements'][precision] = latency

        # Calculate FLOPs and memory access
        flops = self._calculate_flops(module, layer_info)
        memory = self._calculate_memory_access(module, layer_info)

        layer_data['flops'] = flops
        layer_data['memory'] = memory

        return layer_data

    def _measure_with_precision(
        self,
        module: nn.Module,
        input_tensor: torch.Tensor,
        precision: str,
        num_samples: int
    ) -> float:
        """
        Measure layer latency with specific precision.

        Args:
            module: PyTorch module
            input_tensor: Input tensor
            precision: Precision (FP32, FP16, INT8)
            num_samples: Number of measurements

        Returns:
            float: Average latency in milliseconds
        """
        device = input_tensor.device
        original_dtype = input_tensor.dtype

        # Convert to target precision
        if precision == 'FP16' and device.type == 'cuda':
            module = module.half()
            input_tensor = input_tensor.half()
        elif precision == 'INT8':
            # INT8 requires quantization
            # For now, skip actual INT8 measurement
            return self._estimate_from_flops(module)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = module(input_tensor)

        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(num_samples):
                _ = module(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()

        # Restore original dtype
        if precision == 'FP16':
            module = module.float()

        avg_latency = ((end - start) / num_samples) * 1000  # Convert to ms
        return avg_latency

    def _calculate_flops(self, module: nn.Module, info: dict) -> int:
        """Calculate FLOPs for a layer."""
        if isinstance(module, nn.Conv2d):
            # Conv: in_ch * out_ch * k_h * k_w * out_h * out_w
            kernel = info['kernel_size'][0] * info['kernel_size'][1]
            flops = info['in_channels'] * info['out_channels'] * kernel * kernel
        elif isinstance(module, nn.Linear):
            flops = info['in_features'] * info['out_features']
        elif isinstance(module, nn.BatchNorm2d):
            flops = 0  # BN has negligible compute
        else:
            flops = 0
        return flops

    def _calculate_memory_access(self, module: nn.Module, info: dict) -> int:
        """Calculate memory access in bytes."""
        if isinstance(module, nn.Conv2d):
            # Read: input, Write: output
            # Simplified: weight size + output size
            weight_size = info['in_channels'] * info['out_channels'] * info['kernel_size'][0] * info['kernel_size'][1] * 4  # FP32
            # Output size (assuming 32x32 output)
            output_size = info['out_channels'] * 32 * 32 * 4
            return weight_size + output_size
        elif isinstance(module, nn.Linear):
            weight_size = info['in_features'] * info['out_features'] * 4
            output_size = info['out_features'] * 4
            return weight_size + output_size
        elif isinstance(module, nn.BatchNorm2d):
            # Read/write statistics
            return info['num_features'] * 4 * 2  # mean and var
        else:
            return 0

    def _estimate_from_flops(self, module: nn.Module) -> float:
        """Estimate latency from FLOPs (when measurement is not possible)."""
        # Fallback estimation based on FLOPs
        # Assume 10 TFLOPS compute capability
        flops = sum(p.numel() for p in module.parameters())
        estimated_latency = flops / (10e12 * 1e9) * 1000  # Convert to ms
        return max(estimated_latency, 0.01)  # Minimum 0.01ms

    def query(
        self,
        layer_name: str,
        precision: str = 'FP32',
        default_latency: Optional[float] = None
    ) -> float:
        """
        Query latency from LUT.

        Args:
            layer_name: Name of the layer
            precision: Precision format
            default_latency: Default value if not found in LUT

        Returns:
            float: Latency in milliseconds
        """
        if layer_name not in self.lut:
            logger.warning(f"Layer {layer_name} not found in LUT")
            return default_latency if default_latency is not None else 1.0

        measurements = self.lut[layer_name].get('measurements', {})
        latency = measurements.get(precision)

        if latency is None:
            # Try to interpolate or estimate
            if default_latency is not None:
                logger.warning(f"No {precision} measurement for {layer_name}, using default")
                return default_latency
            else:
                # Use FP32 as baseline
                baseline = measurements.get('FP32')
                if baseline:
                    # Simple scaling
                    if precision == 'FP16':
                        return baseline * 0.6
                    elif precision == 'INT8':
                        return baseline * 0.4
                    elif precision == 'INT4':
                        return baseline * 0.2
                return 1.0

        return latency

    def estimate_total_latency(
        self,
        model: nn.Module,
        precision_map: Optional[Dict[str, str]] = None
    ) -> float:
        """
        Estimate total model latency from LUT.

        Args:
            model: PyTorch model
            precision_map: Dictionary mapping layer names to precisions

        Returns:
            float: Total estimated latency in milliseconds
        """
        if precision_map is None:
            precision_map = {}

        total_latency = 0

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                precision = precision_map.get(name, 'FP32')
                latency = self.query(name, precision)
                total_latency += latency

        return total_latency

    def save(self, path: str):
        """
        Save LUT to file.

        Args:
            path: File path to save LUT
        """
        with open(path, 'wb') as f:
            pickle.dump(self.lut, f)
        logger.info(f"Saved latency LUT to {path}")

    def load(self, path: str):
        """
        Load LUT from file.

        Args:
            path: File path to load LUT from
        """
        try:
            with open(path, 'rb') as f:
                self.lut = pickle.load(f)
            logger.info(f"Loaded latency LUT from {path}")
            logger.info(f"LUT contains {len(self.lut)} entries")
        except FileNotFoundError:
            logger.warning(f"LUT file not found: {path}")
        except Exception as e:
            logger.error(f"Error loading LUT: {e}")

    def clear(self):
        """Clear all entries from LUT."""
        self.lut = {}
        logger.info("Cleared latency LUT")

    def get_statistics(self) -> Dict:
        """Get statistics about the LUT."""
        if not self.lut:
            return {}

        num_layers = len(self.lut)
        num_measurements = sum(
            len(layer.get('measurements', {}))
            for layer in self.lut.values()
        )

        return {
            'num_layers': num_layers,
            'num_measurements': num_measurements,
            'layer_names': list(self.lut.keys()),
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"LatencyLookupTable(platform={self.platform}, "
                f"layers={stats.get('num_layers', 0)}, "
                f"measurements={stats.get('num_measurements', 0)})")
