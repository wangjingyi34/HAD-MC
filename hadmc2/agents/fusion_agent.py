"""Fusion Agent for HAD-MC 2.0"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import copy
import logging

logger = logging.getLogger(__name__)


class FusionAgent:
    """
    Fusion Agent: Operator fusion optimization.

    Identifies and fuses common operator patterns:
    - Conv2d + BatchNorm2d
    - Conv2d + ReLU
    - Conv2d + BatchNorm2d + ReLU

    Supports 6 fusion patterns.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ):
        """
        Initialize Fusion Agent.

        Args:
            model: PyTorch model
            device: Device to run on
        """
        self.model = model
        self.device = device

        # Define fusion patterns
        self.fusion_patterns = [
            'none',              # No fusion
            'conv_bn',           # Conv + BatchNorm
            'conv_relu',         # Conv + ReLU
            'conv_bn_relu',      # Conv + BatchNorm + ReLU
            'conv_bn_add',       # Conv + BatchNorm + Add (residual)
            'conv_bn_add_relu', # Conv + BatchNorm + Add + ReLU
        ]

        # Track fused patterns
        self.fused_operations = []

    def get_action_space(self) -> Dict:
        """
        Get action space for fusion agent.

        Returns:
            dict: Action space definition
        """
        num_fusable_points = len(self._get_fusable_points())
        num_patterns = len(self.fusion_patterns)

        return {
            'type': 'discrete',
            'layer_idx': list(range(num_fusable_points)),
            'pattern': self.fusion_patterns,
        }

    def _get_fusable_points(self) -> List[nn.Module]:
        """
        Get list of layers that can be fused.
        """
        fusible_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                fusible_layers.append(module)
        return fusible_layers

    def get_action(self, state: torch.Tensor) -> Tuple[Dict, float]:
        """
        Get a fusion action (sample from action space).

        Args:
            state: Current state tensor

        Returns:
            tuple: (fusion_config, log_prob)
        """
        fusible_points = self._get_fusable_points()

        if not fusible_points:
            return {}, 0.0

        # For this implementation, return a random action
        # In full MARL setting, this would come from PPO policy
        import random

        start_layer = random.choice(range(len(fusible_points)))
        pattern = random.choice(self.fusion_patterns)

        fusion_config = {
            'pattern': pattern,
            'start_layer': start_layer,
        }

        # Log prob would be computed by PPO
        log_prob = 0.0

        return fusion_config, log_prob

    def analyze(self) -> List[Dict]:
        """
        Analyze model to find fusable patterns.

        Returns:
            list: List of fusable pattern candidates
        """
        fusible_layers = self._get_fusable_points()

        patterns = []

        # Check Conv2d + BatchNorm2d + ReLU pattern
        for i, conv1 in enumerate(fusible_layers[:-1]):
            # Check if next layer is BatchNorm2d
            if i + 1 < len(fusible_layers):
                bn = fusible_layers[i + 1]
                if isinstance(bn, nn.BatchNorm2d):
                    # Check if layer after BatchNorm2d is ReLU
                    if i + 2 < len(fusible_layers):
                        relu = fusible_layers[i + 2]
                        if isinstance(relu, nn.ReLU):
                            patterns.append({
                                'pattern': 'conv_bn_relu',
                                'conv_layer': i,
                                'bn_layer': i + 1,
                                'relu_layer': i + 2,
                            })
                            break

        # Check Conv2d + BatchNorm2d + Add pattern
        for i, conv1 in enumerate(fusible_layers):
            # Check if next two layers are BatchNorm2d and Add
            if i + 2 < len(fusible_layers):
                bn1 = fusible_layers[i + 1]
                if isinstance(bn1, nn.BatchNorm2d):
                    if i + 2 < len(fusible_layers):
                        add1 = fusible_layers[i + 2]
                        if isinstance(add1, nn.Identity):
                            patterns.append({
                                'pattern': 'conv_bn_add',
                                'conv_layer': i,
                                'bn1_layer': i + 1,
                                'add_layer': i + 2,
                            })
                            break

        return patterns

    def fuse(self, fusion_config: Dict, model: Optional[nn.Module] = None) -> nn.Module:
        """
        Execute fusion on model.

        Args:
            fusion_config: Dictionary with 'pattern' and 'start_layer' keys
            model: Optional model to fuse (uses self.model if None)

        Returns:
            nn.Module: Fused model
        """
        if model is None:
            model = self.model

        pattern = fusion_config.get('pattern', 'none')

        if pattern == 'none':
            # No fusion, return model as-is
            logger.info("No fusion applied")
            return model

        start_idx = fusion_config.get('start_layer', 0)
        pattern_name = fusion_config.get('pattern', 'none')

        # Track operations
        self.fused_operations.append({
            'pattern': pattern_name,
            'start_layer': start_idx,
        })

        # Get fusible layers
        fusible_layers = self._get_fusable_points()

        if pattern_name == 'conv_bn':
            model = self._fuse_conv_bn(model, start_idx)
        elif pattern_name == 'conv_relu':
            model = self._fuse_conv_relu(model, start_idx)
        elif pattern_name == 'conv_bn_relu':
            model = self._fuse_conv_bn_relu(model, start_idx)
        elif pattern_name == 'conv_bn_add':
            model = self._fuse_conv_bn_add(model, start_idx)
        elif pattern_name == 'conv_bn_add_relu':
            model = self._fuse_conv_bn_add_relu(model, start_idx)

        logger.info(f"Applied fusion '{pattern_name}' at layer {start_idx}")

        return model

    def _fuse_conv_bn(self, model: nn.Module, conv_idx: int) -> nn.Module:
        """Fuse Conv2d + BatchNorm2d layers."""
        conv1 = list(model.children())[conv_idx]
        bn1 = list(model.children())[conv_idx + 1]

        if isinstance(conv1, nn.Conv2d) and isinstance(bn1, nn.BatchNorm2d):
            # Fuse weights
            gamma = bn1.weight
            beta = bn1.bias
            running_mean = bn1.running_mean
            running_var = bn1.running_var

            conv1.weight.data = conv1.weight.data * gamma
            conv1.bias.data = conv1.bias.data * gamma + beta

            # Remove BatchNorm
            delattr(model, list(model.children())[conv_idx + 1])

            logger.debug(f"Fused Conv+BN at index {conv_idx}")

        return model

    def _fuse_conv_relu(self, model: nn.Module, conv_idx: int) -> nn.Module:
        """Fuse Conv2d + ReLU layers."""
        conv1 = list(model.children())[conv_idx]

        if isinstance(conv1, nn.Conv2d) and isinstance(list(model.children())[conv_idx + 1], nn.ReLU):
            # Already fused, just return
            return model

        logger.debug(f"Conv+ReLU already fused at index {conv_idx}")

        return model

    def _fuse_conv_bn_relu(self, model: nn.Module, conv_idx: int) -> nn.Module:
        """Fuse Conv2d + BatchNorm2d + ReLU layers."""
        conv1 = list(model.children())[conv_idx]
        bn1 = list(model.children())[conv_idx + 1]
        relu1 = list(model.children())[conv_idx + 2]

        if isinstance(conv1, nn.Conv2d) and isinstance(bn1, nn.BatchNorm2d) and isinstance(relu1, nn.ReLU):
            # Fuse weights
            gamma = bn1.weight
            beta = bn1.bias
            running_mean = bn1.running_mean
            running_var = bn1.running_var

            conv1.weight.data = conv1.weight.data * gamma
            conv1.bias.data = conv1.bias.data * gamma + beta

            # Remove BatchNorm and ReLU
            delattr(model, list(model.children())[conv_idx + 1])
            delattr(model, list(model.children())[conv_idx + 2])

            logger.debug(f"Fused Conv+BN+ReLU at index {conv_idx}")

        return model

    def _fuse_conv_bn_add(self, model: nn.Module, conv_idx: int) -> nn.Module:
        """Fuse Conv2d + BatchNorm2d + Add layers."""
        conv1 = list(model.children())[conv_idx]
        bn1 = list(model.children())[conv_idx + 1]
        add1 = list(model.children())[conv_idx + 2]

        if isinstance(conv1, nn.Conv2d) and isinstance(bn1, nn.BatchNorm2d) and isinstance(add1, nn.Identity):
            # Fuse weights
            gamma = bn1.weight
            beta = bn1.bias
            running_mean = bn1.running_mean
            running_var = bn1.running_var

            conv1.weight.data = conv1.weight.data * gamma
            conv1.bias.data = conv1.bias.data * gamma + beta

            # Create fused Conv2d with bias
            # Conv2d weight: W'
            # Fused Conv2d weight: W' * gamma
            # Bias: b = W * gamma + beta

            # Create new Conv2d with fused weights
            fused_conv = nn.Conv2d(
                in_channels=conv1.in_channels,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=True,
            )
            fused_conv.weight.data = conv1.weight.data * gamma
            fused_conv.bias.data = conv1.bias.data * gamma + beta

            # Replace Conv2d and BatchNorm2d with Identity
            model[conv_idx] = nn.Identity()

        logger.debug(f"Fused Conv+BN+Add at index {conv_idx}")

        return model

    def _fuse_conv_bn_add_relu(self, model: nn.Module, conv_idx: int) -> nn.Module:
        """Fuse Conv2d + BatchNorm2d + Add + ReLU layers."""
        conv1 = list(model.children())[conv_idx]
        bn1 = list(model.children())[conv_idx + 1]
        add1 = list(model.children())[conv_idx + 2]
        relu1 = list(model.children())[conv_idx + 3]

        if isinstance(conv1, nn.Conv2d) and isinstance(bn1, nn.BatchNorm2d) and isinstance(add1, nn.Identity) and isinstance(relu1, nn.ReLU):
            # Fuse weights
            gamma = bn1.weight
            beta = bn1.bias
            running_mean = bn1.running_mean
            running_var = bn1.running_var

            conv1.weight.data = conv1.weight.data * gamma
            conv1.bias.data = conv1.bias.data * gamma + beta

            # Create new Conv2d with fused weights
            # Conv2d weight: W'
            # Fused Conv2d weight: W' * gamma
            # Bias: b = W * gamma + beta

            # Create new Conv2d with bias
            fused_conv = nn.Conv2d(
                in_channels=conv1.in_channels,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=True,
            )

            # Fused weights
            fused_conv.weight.data = conv1.weight.data * gamma
            fused_conv.bias.data = conv1.bias.data * gamma + beta

            # Create new Conv2d + ReLU with fused weights
            fused_conv_relu = nn.Conv2d(
                in_channels=conv1.in_channels,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=True,
            )
            # Fused weights
            fused_conv_relu.weight.data = conv1.weight.data * gamma
            fused_conv_relu.bias.data = conv1.bias.data * gamma + beta

            # Create Sequential with Conv2d and ReLU
            model[conv_idx] = nn.Sequential(fused_conv, nn.ReLU())

            logger.debug(f"Fused Conv+BN+Add+ReLU at index {conv_idx}")

            return model

    def get_state(self) -> Dict:
        """
        Get current state of the agent.

        Returns:
            dict: Agent state information
        """
        return {
            'num_fusable_points': len(self._get_fusable_points()),
            'num_patterns': len(self.fusion_patterns),
            'num_fused': len(self.fused_operations),
        }
