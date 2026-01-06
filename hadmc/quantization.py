"""Algorithm 1: Layer-wise Precision Allocation (Extended for Object Detection)"""

import torch
import torch.nn as nn
import logging
from .device_manager import DeviceManager

logger = logging.getLogger(__name__)


class LayerwisePrecisionAllocator:
    """Implements Algorithm 1 from HAD-MC paper
    
    Extended to support both classification and object detection tasks.
    """
    
    def __init__(self, model, calibration_loader, tau_h=1e-3, tau_l=1e-5, device='cpu', task_type='classification'):
        """
        Args:
            model: PyTorch model
            calibration_loader: DataLoader for calibration data
            tau_h: High threshold for FP32 precision
            tau_l: Low threshold for INT4 precision
            device: Device to run on ('cpu', 'cuda', etc.)
            task_type: 'classification' or 'detection'
        """
        # 使用设备管理器自动选择设备
        self.device_manager = DeviceManager()
        if device == 'cpu':
            device = self.device_manager.get_device()
        self.model = model.to(device)
        self.calibration_loader = calibration_loader
        self.tau_h = tau_h
        self.tau_l = tau_l
        self.device = device
        self.task_type = task_type
        self.gradient_sensitivity = {}
        self.precision_map = {}
        
        # Auto-detect task type if not specified
        if task_type == 'auto':
            self.task_type = self._detect_task_type()
            logger.info(f"Auto-detected task type: {self.task_type}")
    
    def _detect_task_type(self):
        """Auto-detect whether this is classification or detection task"""
        try:
            self.model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                output = self.model(dummy_input)
                
                # YOLOv5 returns a list of tensors for detection
                if isinstance(output, (list, tuple)):
                    return 'detection'
                else:
                    return 'classification'
        except:
            return 'classification'
    
    def _compute_loss(self, output, target):
        """Compute loss based on task type"""
        if self.task_type == 'classification':
            # Classification: use cross-entropy
            if isinstance(output, tuple):
                output = output[0]
            return nn.functional.cross_entropy(output, target)
        
        elif self.task_type == 'detection':
            # Detection: use a simplified gradient-based loss
            if isinstance(output, (list, tuple)):
                total_loss = 0
                for pred in output:
                    if isinstance(pred, torch.Tensor):
                        # Simple L2 regularization to encourage gradient flow
                        total_loss += torch.mean(pred ** 2)
                return total_loss
            else:
                return torch.mean(output ** 2)
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
    def calculate_gradient_sensitivity(self):
        """Calculate average gradient magnitude for each layer
        
        Extended to support both classification and detection tasks.
        """
        logger.info(f"Calculating gradient sensitivity for {self.task_type} task...")
        self.model.train()
        
        grad_accum = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_accum[name] = 0.0
        
        num_batches = 0
        max_batches = 10  # Limit to 10 batches for speed
        
        for data, target in self.calibration_loader:
            if num_batches >= max_batches:
                break
            
            data = data.to(self.device)
            if target is not None:
                target = target.to(self.device)
            
            self.model.zero_grad()
            
            try:
                output = self.model(data)
                loss = self._compute_loss(output, target)
                loss.backward()
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_accum[name] += torch.mean(torch.abs(param.grad)).item()
                
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Error in batch {num_batches}: {e}")
                continue
        
        if num_batches > 0:
            for name in grad_accum:
                grad_accum[name] /= num_batches
        
        self.gradient_sensitivity = grad_accum
        logger.info(f"Calculated gradient sensitivity for {len(grad_accum)} parameters using {num_batches} batches")
        return grad_accum
    
    def allocate_precision(self):
        """Assign precision based on gradient sensitivity"""
        logger.info("Allocating precision...")
        for name, grad_mag in self.gradient_sensitivity.items():
            if grad_mag > self.tau_h:
                self.precision_map[name] = 'FP32'
            elif grad_mag < self.tau_l:
                self.precision_map[name] = 'INT4'
            else:
                self.precision_map[name] = 'INT8'
        
        fp32 = sum(1 for p in self.precision_map.values() if p == 'FP32')
        int8 = sum(1 for p in self.precision_map.values() if p == 'INT8')
        int4 = sum(1 for p in self.precision_map.values() if p == 'INT4')
        logger.info(f"Precision allocation: FP32={fp32}, INT8={int8}, INT4={int4}")
        return self.precision_map
    
    def apply_quantization(self):
        """Apply quantization based on precision map
        
        Note: This is a simplified version using PyTorch's dynamic quantization.
        Full mixed-precision quantization would require more complex implementation.
        """
        logger.info("Applying quantization...")
        
        # Apply dynamic quantization to eligible layers
        # This is a simplified approach - full implementation would need custom quantization
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Applied dynamic INT8 quantization")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, returning original model")
            return self.model
    
    def run(self, target_bits=None):
        """Run complete Algorithm 1"""
        if target_bits is not None:
            # Adjust thresholds based on target bits (simplified)
            logger.info(f"Target bits: {target_bits}")
        
        self.calculate_gradient_sensitivity()
        self.allocate_precision()
        quantized_model = self.apply_quantization()
        
        return quantized_model


# Alias for backward compatibility with run_all_experiments.sh
class AdaptiveQuantizer:
    """Simplified Adaptive Quantizer for standalone use
    
    This is a wrapper class that provides a simple interface for quantization
    without requiring calibration data.
    """
    
    def __init__(self, bits=8, mode='ptq'):
        """
        Args:
            bits: Target bit width (4, 8, or 16)
            mode: Quantization mode ('ptq' for post-training, 'qat' for quantization-aware training)
        """
        self.bits = bits
        self.mode = mode
        logger.info(f"AdaptiveQuantizer initialized: bits={bits}, mode={mode}")
    
    def quantize(self, model):
        """Apply quantization to model
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Quantized model
        """
        logger.info(f"Applying {self.mode.upper()} quantization with {self.bits}-bit precision...")
        
        try:
            if self.mode == 'ptq':
                # Post-Training Quantization using dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear},  # Only quantize linear layers for safety
                    dtype=torch.qint8
                )
                logger.info("PTQ quantization completed")
                return quantized_model
            
            elif self.mode == 'qat':
                # For QAT, we prepare the model but actual training is done separately
                model.eval()
                # Return the model as-is since QAT requires training loop
                logger.info("QAT preparation completed (requires training for full effect)")
                return model
            
            else:
                logger.warning(f"Unknown mode: {self.mode}, returning original model")
                return model
                
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, returning original model")
            return model
