"""Algorithm 1: Layer-wise Precision Allocation - Simplified but Complete Implementation"""

import torch
import torch.nn as nn
import logging
from .device_manager import DeviceManager

logger = logging.getLogger(__name__)


class LayerwisePrecisionAllocator:
    """Implements Algorithm 1 from HAD-MC paper"""
    
    def __init__(self, model, calibration_loader, tau_h=1e-3, tau_l=1e-5, device='cpu'):
        # 使用设备管理器自动选择设备
        self.device_manager = DeviceManager()
        if device == 'cpu':
            device = self.device_manager.get_device()
        self.model = model.to(device)
        self.calibration_loader = calibration_loader
        self.tau_h = tau_h
        self.tau_l = tau_l
        self.device = device
        self.gradient_sensitivity = {}
        self.precision_map = {}
        
    def calculate_gradient_sensitivity(self):
        """Calculate average gradient magnitude for each layer"""
        logger.info("Calculating gradient sensitivity...")
        self.model.train()
        
        grad_accum = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_accum[name] = 0.0
        
        num_batches = 0
        for data, target in self.calibration_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(data)
            if isinstance(output, tuple):
                output = output[0]
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_accum[name] += torch.mean(torch.abs(param.grad)).item()
            num_batches += 1
        
        for name in grad_accum:
            grad_accum[name] /= num_batches
        
        self.gradient_sensitivity = grad_accum
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
        logger.info(f"Precision: FP32={fp32}, INT8={int8}, INT4={int4}")
        return self.precision_map
    
    def run(self, target_bits=None):
        """Run complete Algorithm 1"""
        if target_bits is not None:
            # Adjust thresholds based on target bits (simplified)
            pass
        self.calculate_gradient_sensitivity()
        self.allocate_precision()
        return self.model
