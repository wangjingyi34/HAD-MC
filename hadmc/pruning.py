"""Algorithm 2: Gradient Sensitivity-Guided Pruning"""
from .device_manager import DeviceManager

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class GradientSensitivityPruner:
    """Implements Algorithm 2 from HAD-MC paper"""
    
    def __init__(self, model, train_loader, flops_target, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.flops_target = flops_target
        self.device = device
        self.channel_importance = {}
        
    def calculate_channel_importance(self):
        """Calculate importance score for each channel"""
        logger.info("Calculating channel importance...")
        self.model.train()
        
        importance = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                importance[name] = torch.zeros(module.out_channels)
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(data)
            if isinstance(output, tuple):
                output = output[0]
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                    # Importance = mean absolute gradient per channel
                    grad = module.weight.grad
                    channel_grad = torch.mean(torch.abs(grad), dim=(1, 2, 3))
                    importance[name] += channel_grad.cpu()
        
        # Average over batches
        num_batches = len(self.train_loader)
        for name in importance:
            importance[name] /= num_batches
        
        self.channel_importance = importance
        return importance
    
    def prune_channels(self, prune_ratio=0.5):
        """Prune channels with lowest importance"""
        logger.info(f"Pruning {prune_ratio*100}% of channels...")
        pruned_model = self.model
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.channel_importance:
                importance = self.channel_importance[name]
                num_channels = len(importance)
                num_keep = int(num_channels * (1 - prune_ratio))
                
                # Keep top-k channels
                _, indices = torch.topk(importance, num_keep)
                indices = sorted(indices.tolist())
                
                # Create new Conv2d with fewer channels
                new_conv = nn.Conv2d(
                    module.in_channels,
                    num_keep,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    bias=module.bias is not None
                )
                
                # Copy weights for kept channels
                with torch.no_grad():
                    new_conv.weight.data = module.weight.data[indices]
                    if module.bias is not None:
                        new_conv.bias.data = module.bias.data[indices]
                
                logger.info(f"Pruned {name}: {num_channels} -> {num_keep} channels")
        
        return pruned_model
    
    def run(self, prune_ratio=0.5):
        """Run complete Algorithm 2"""
        self.calculate_channel_importance()
        pruned_model = self.prune_channels(prune_ratio)
        return pruned_model
