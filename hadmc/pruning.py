"""Algorithm 2: Gradient Sensitivity-Guided Pruning (Extended for Object Detection)"""
from .device_manager import DeviceManager

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class GradientSensitivityPruner:
    """Implements Algorithm 2 from HAD-MC paper
    
    Extended to support both classification and object detection tasks.
    """
    
    def __init__(self, model, train_loader, flops_target, device='cpu', task_type='classification'):
        """
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            flops_target: Target FLOPs after pruning
            device: Device to run on ('cpu', 'cuda', etc.)
            task_type: 'classification' or 'detection'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.flops_target = flops_target
        self.device = device
        self.task_type = task_type
        self.channel_importance = {}
        
        # Auto-detect task type if not specified
        if task_type == 'auto':
            self.task_type = self._detect_task_type()
            logger.info(f"Auto-detected task type: {self.task_type}")
    
    def _detect_task_type(self):
        """Auto-detect whether this is classification or detection task"""
        # Try a forward pass to see the output format
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
            # Default to classification
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
            # For YOLOv5, output is a list of [small, medium, large] detection layers
            if isinstance(output, (list, tuple)):
                # Use L2 loss on predictions as a proxy for gradient computation
                # This allows us to compute gradients without needing ground truth format
                total_loss = 0
                for pred in output:
                    if isinstance(pred, torch.Tensor):
                        # Simple L2 regularization to encourage gradient flow
                        total_loss += torch.mean(pred ** 2)
                return total_loss
            else:
                # Fallback: use mean squared error
                return torch.mean(output ** 2)
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
    def calculate_channel_importance(self):
        """Calculate importance score for each channel
        
        Extended to support both classification and detection tasks.
        """
        logger.info(f"Calculating channel importance for {self.task_type} task...")
        self.model.train()
        
        importance = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                importance[name] = torch.zeros(module.out_channels)
        
        num_batches = 0
        max_batches = 10  # Limit to 10 batches for speed
        
        for data, target in self.train_loader:
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
                
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                        # Importance = mean absolute gradient per channel
                        grad = module.weight.grad
                        channel_grad = torch.mean(torch.abs(grad), dim=(1, 2, 3))
                        importance[name] += channel_grad.cpu()
                
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Error in batch {num_batches}: {e}")
                continue
        
        # Average over batches
        if num_batches > 0:
            for name in importance:
                importance[name] /= num_batches
        
        self.channel_importance = importance
        logger.info(f"Calculated importance for {len(importance)} Conv2d layers using {num_batches} batches")
        return importance
    
    def prune_channels(self, prune_ratio=0.5):
        """Prune channels with lowest importance
        
        Note: This is a simplified version that marks channels for pruning
        but doesn't actually modify the model structure (which is complex for YOLOv5).
        Instead, it zeros out the weights of pruned channels.
        """
        logger.info(f"Pruning {prune_ratio*100}% of channels...")
        
        total_channels = 0
        pruned_channels = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.channel_importance:
                importance = self.channel_importance[name]
                num_channels = len(importance)
                num_prune = int(num_channels * prune_ratio)
                
                if num_prune > 0 and num_prune < num_channels:
                    # Find channels with lowest importance
                    _, indices = torch.topk(importance, num_prune, largest=False)
                    
                    # Zero out weights of pruned channels (structured pruning)
                    with torch.no_grad():
                        for idx in indices:
                            module.weight.data[idx] = 0
                            if module.bias is not None:
                                module.bias.data[idx] = 0
                    
                    total_channels += num_channels
                    pruned_channels += num_prune
                    
                    logger.info(f"Pruned {name}: {num_prune}/{num_channels} channels ({num_prune/num_channels*100:.1f}%)")
        
        if total_channels > 0:
            overall_ratio = pruned_channels / total_channels
            logger.info(f"Overall pruning: {pruned_channels}/{total_channels} channels ({overall_ratio*100:.1f}%)")
        
        return self.model
    
    def run(self, prune_ratio=0.5):
        """Run complete Algorithm 2"""
        self.calculate_channel_importance()
        pruned_model = self.prune_channels(prune_ratio)
        return pruned_model
