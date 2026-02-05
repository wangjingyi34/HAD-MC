"""Pruning Agent for HAD-MC 2.0"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy
import logging

logger = logging.getLogger(__name__)


class PruningAgent:
    """
    Pruning Agent: Gradient-based structured pruning.

    Uses gradient sensitivity to determine which channels are most important.
    Performs L1-norm based channel pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        device: str = 'cpu'
    ):
        """
        Initialize Pruning Agent.

        Args:
            model: PyTorch model to prune
            train_loader: Training data loader for computing gradients
            device: Device to run on
        """
        self.model = model
        self.train_loader = train_loader
        self.device = device

        self.importance_scores = {}
        self.gradients = {}

        # Pruning ratio options
        self.pruning_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def get_action_space(self) -> Dict:
        """
        Get the action space for the pruning agent.

        Returns:
            dict: Action space definition
        """
        return {
            'type': 'discrete',
            'layer_idx': list(range(len(self._get_prunable_layers()))),
            'pruning_ratio': self.pruning_ratios,
        }

    def compute_importance(self) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores for each channel using gradient sensitivity.

        Importance is based on Taylor expansion:
        I(j) = |W(j) * G(j)|

        where W(j) is the weight and G(j) is the gradient.

        Returns:
            dict: Importance scores per layer and channel
        """
        self.model.train()
        self.model.to(self.device)

        # Forward and backward pass
        inputs, targets = next(iter(self.train_loader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)

        # Handle different output formats
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        # Compute loss
        if outputs.dim() > 1 and outputs.shape[1] > 1:
            loss = F.cross_entropy(outputs, targets)
        else:
            loss = F.mse_loss(outputs.squeeze(), targets.float())

        # Backward - this populates module.weight.grad
        self.model.zero_grad()
        loss.backward()

        # Compute importance scores using weight.grad
        self.importance_scores = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data

                # Check if gradient is available
                if module.weight.grad is not None:
                    grad = module.weight.grad.data

                    # Compute Taylor importance
                    importance = (weight * grad).abs()

                    # Sum over spatial dimensions for Conv2d
                    if isinstance(module, nn.Conv2d):
                        # For Conv2d, sum over kernel dimensions
                        importance = importance.sum(dim=(1, 2, 3))
                    # For Linear, sum over output features
                    elif isinstance(module, nn.Linear):
                        importance = importance.sum(dim=1)

                    self.importance_scores[name] = importance
                else:
                    # No gradient available, use L1 norm of weights
                    if isinstance(module, nn.Conv2d):
                        importance = weight.abs().sum(dim=(1, 2, 3))
                    else:
                        importance = weight.abs().sum(dim=1)
                    self.importance_scores[name] = importance

        logger.info(f"Computed importance for {len(self.importance_scores)} layers")
        return self.importance_scores

    def prune(
        self,
        pruning_config: Dict,
        model: Optional[nn.Module] = None
    ) -> nn.Module:
        """
        Execute pruning on the model.

        Args:
            pruning_config: Dictionary mapping layer names to pruning ratios
            model: Optional model to prune (uses self.model if None)

        Returns:
            nn.Module: Pruned model
        """
        if model is None:
            model = self.model

        # Ensure we have importance scores
        if not self.importance_scores:
            self.compute_importance()

        pruned_model = copy.deepcopy(model)
        total_params_before = sum(p.numel() for p in pruned_model.parameters())
        total_params_after = 0

        for name, module in pruned_model.named_modules():
            if name in pruning_config and isinstance(module, (nn.Conv2d, nn.Linear)):
                ratio = pruning_config[name]

                if ratio > 0 and name in self.importance_scores:
                    importance = self.importance_scores[name]

                    # Determine number of channels to prune
                    num_channels = importance.numel()
                    num_prune = int(num_channels * ratio)

                    if num_prune > 0:
                        # Find least important channels
                        _, prune_indices = torch.topk(
                            importance, num_prune, largest=False
                        )

                        # Create mask
                        mask = torch.ones(num_channels, dtype=torch.bool)
                        mask[prune_indices] = False

                        # Apply mask to weights
                        weight = module.weight.data

                        if isinstance(module, nn.Conv2d):
                            # For Conv2d, mask along output channel dimension
                            weight[mask] = 0

                            # Also update out_channels attribute
                            module.out_channels = mask.sum().item()

                            # Zero out corresponding bias
                            if module.bias is not None:
                                module.bias.data[mask] = 0

                        elif isinstance(module, nn.Linear):
                            # For Linear, mask along output dimension
                            weight[mask] = 0

                            # Update out_features
                            module.out_features = mask.sum().item()

                            # Zero out corresponding bias
                            if module.bias is not None:
                                module.bias.data[mask] = 0

                        logger.debug(f"Pruned {name}: {num_prune}/{num_channels} channels ({ratio:.1%})")

        total_params_after = sum(p.numel() for p in pruned_model.parameters())
        actual_pruning_ratio = 1 - (total_params_after / total_params_before)

        logger.info(f"Pruning complete: {actual_pruning_ratio:.2%} parameters removed")

        return pruned_model

    def get_state(self) -> Dict:
        """
        Get the current state of the pruning agent.

        Returns:
            dict: Current state information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        nonzero_params = sum((p != 0).sum().item() for p in self.model.parameters())

        return {
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'sparsity': 1 - (nonzero_params / total_params) if total_params > 0 else 0,
            'num_pruned_layers': len(self.importance_scores),
        }

    def get_action(
        self,
        state: torch.Tensor
    ) -> tuple:
        """
        Get a pruning action (sample from action space).

        Args:
            state: Current state tensor

        Returns:
            tuple: (pruning_config, log_prob)
        """
        prunable_layers = self._get_prunable_layers()

        # For this implementation, return a random action
        # In the full MARL setting, this would come from the PPO policy
        import random

        layer_idx = random.choice(range(len(prunable_layers)))
        pruning_ratio = random.choice(self.pruning_ratios)

        pruning_config = {prunable_layers[layer_idx]: pruning_ratio}

        # Log prob would be computed by PPO
        log_prob = 0.0

        return pruning_config, log_prob

    def apply_action(
        self,
        model: nn.Module,
        action: Dict
    ) -> nn.Module:
        """
        Apply a pruning action to the model.

        Args:
            model: PyTorch model
            action: Action dictionary {'layer_idx': int, 'pruning_ratio': float}

        Returns:
            nn.Module: Pruned model
        """
        prunable_layers = self._get_prunable_layers()
        layer_idx = action['layer_idx']
        pruning_ratio = action['pruning_ratio']

        layer_name = prunable_layers[layer_idx]
        pruning_config = {layer_name: pruning_ratio}

        return self.prune(pruning_config, model)

    def _get_prunable_layers(self) -> List[str]:
        """Get list of layer names that can be pruned."""
        return [
            name for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]

    def __repr__(self) -> str:
        num_prunable = len(self._get_prunable_layers())
        return (f"PruningAgent(num_prunable_layers={num_prunable}, "
                f"ratios={self.pruning_ratios})")
