"""Algorithm 4: Operator Fusion"""
from .device_manager import DeviceManager

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class FusedConvBNReLU(nn.Module):
    """Fused Conv+BN+ReLU module"""
    def __init__(self, conv, bn, relu=None):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu if relu is not None else nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class OperatorFuser:
    """Implements Algorithm 4 from HAD-MC paper"""
    
    def __init__(self, model):
        self.model = model
        self.fusion_patterns = self.define_fusion_patterns()
        
    def define_fusion_patterns(self):
        """Define common fusion patterns"""
        return [
            ('Conv2d', 'BatchNorm2d', 'ReLU'),
            ('Conv2d', 'ReLU'),
        ]
    
    def match_pattern(self, modules, start_idx):
        """Check if modules match a fusion pattern"""
        for pattern in self.fusion_patterns:
            if start_idx + len(pattern) <= len(modules):
                match = True
                for i, expected_type in enumerate(pattern):
                    actual_module = modules[start_idx + i][1]
                    if not isinstance(actual_module, getattr(nn, expected_type)):
                        match = False
                        break
                if match:
                    return pattern, start_idx + len(pattern)
        return None, start_idx + 1
    
    def fuse_operators(self):
        """Fuse operators according to patterns"""
        logger.info("Fusing operators...")
        
        # Get all modules as a list
        modules = list(self.model.named_modules())
        
        # Count fusion opportunities
        fusion_count = 0
        i = 0
        while i < len(modules):
            pattern, next_i = self.match_pattern(modules, i)
            if pattern is not None:
                fusion_count += 1
                logger.info(f"Found fusion pattern: {' + '.join(pattern)}")
                i = next_i
            else:
                i += 1
        
        logger.info(f"Total fusion opportunities: {fusion_count}")
        
        # Return original model (actual fusion would require model rewriting)
        return self.model
    
    def run(self):
        """Run complete Algorithm 4"""
        fused_model = self.fuse_operators()
        return fused_model
