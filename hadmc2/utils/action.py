"""Action space definition for HAD-MC 2.0 MARL framework"""

import torch
from typing import List, Dict
import random


class ActionSpace:
    """
    Action space definition for all five agents.

    Each agent has its own action space, which can be discrete or continuous.
    """

    def __init__(self, num_layers: int, num_fusion_points: int = 10):
        self.num_layers = num_layers
        self.num_fusion_points = num_fusion_points

        # ===== Pruning Agent Action Space (Discrete) =====
        self.pruning_actions = {
            'layer_idx': list(range(num_layers)),
            'pruning_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }

        # ===== Quantization Agent Action Space (Discrete) =====
        self.quantization_actions = {
            'layer_idx': list(range(num_layers)),
            'bit_width': [4, 8, 16, 32],
        }

        # ===== Distillation Agent Action Space (Continuous) =====
        self.distillation_actions = {
            'temperature': (1.0, 20.0),  # Temperature for soft labels
            'alpha': (0.0, 1.0),         # Weight for distillation loss
        }

        # ===== Fusion Agent Action Space (Discrete) =====
        self.fusion_actions = {
            'pattern': [
                'none',            # No fusion
                'conv_bn',         # Conv + BatchNorm
                'conv_relu',       # Conv + ReLU
                'conv_bn_relu',    # Conv + BatchNorm + ReLU
                'conv_bn_add',     # Conv + BatchNorm + Add (residual)
                'conv_bn_add_relu', # Conv + BatchNorm + Add + ReLU (residual)
            ],
            'start_layer': list(range(num_fusion_points)),
        }

        # ===== Update Agent Action Space (Discrete) =====
        self.update_actions = {
            'strategy': ['full', 'incremental', 'hash_based'],
            'update_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }

    def sample_pruning_action(self) -> Dict:
        """
        Sample a pruning action randomly.

        Returns:
            dict: {'layer_idx': int, 'pruning_ratio': float}
        """
        return {
            'layer_idx': random.choice(self.pruning_actions['layer_idx']),
            'pruning_ratio': random.choice(self.pruning_actions['pruning_ratio']),
        }

    def sample_quantization_action(self) -> Dict:
        """
        Sample a quantization action randomly.

        Returns:
            dict: {'layer_idx': int, 'bit_width': int}
        """
        return {
            'layer_idx': random.choice(self.quantization_actions['layer_idx']),
            'bit_width': random.choice(self.quantization_actions['bit_width']),
        }

    def sample_distillation_action(self) -> Dict:
        """
        Sample a distillation action randomly.

        Returns:
            dict: {'temperature': float, 'alpha': float}
        """
        temp_range = self.distillation_actions['temperature']
        alpha_range = self.distillation_actions['alpha']
        return {
            'temperature': random.uniform(temp_range[0], temp_range[1]),
            'alpha': random.uniform(alpha_range[0], alpha_range[1]),
        }

    def sample_fusion_action(self) -> Dict:
        """
        Sample a fusion action randomly.

        Returns:
            dict: {'pattern': str, 'start_layer': int}
        """
        return {
            'pattern': random.choice(self.fusion_actions['pattern']),
            'start_layer': random.choice(self.fusion_actions['start_layer']),
        }

    def sample_update_action(self) -> Dict:
        """
        Sample an update action randomly.

        Returns:
            dict: {'strategy': str, 'update_ratio': float}
        """
        return {
            'strategy': random.choice(self.update_actions['strategy']),
            'update_ratio': random.choice(self.update_actions['update_ratio']),
        }

    def sample_all_actions(self) -> Dict:
        """
        Sample actions for all agents at once.

        Returns:
            dict: Actions for all five agents
        """
        return {
            'pruning': self.sample_pruning_action(),
            'quantization': self.sample_quantization_action(),
            'distillation': self.sample_distillation_action(),
            'fusion': self.sample_fusion_action(),
            'update': self.sample_update_action(),
        }

    def get_action_dim(self, agent_name: str) -> int:
        """
        Get the action dimension for a specific agent.

        Args:
            agent_name: One of 'pruning', 'quantization', 'distillation', 'fusion', 'update'

        Returns:
            int: Number of possible actions for this agent
        """
        if agent_name == 'pruning':
            return len(self.pruning_actions['layer_idx']) * len(self.pruning_actions['pruning_ratio'])
        elif agent_name == 'quantization':
            return len(self.quantization_actions['layer_idx']) * len(self.quantization_actions['bit_width'])
        elif agent_name == 'distillation':
            return 2  # temperature and alpha
        elif agent_name == 'fusion':
            return len(self.fusion_actions['pattern']) * len(self.fusion_actions['start_layer'])
        elif agent_name == 'update':
            return len(self.update_actions['strategy']) * len(self.update_actions['update_ratio'])
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    def get_all_action_dims(self) -> Dict[str, int]:
        """
        Get action dimensions for all agents.

        Returns:
            dict: Action dimensions for each agent
        """
        return {
            'pruning': self.get_action_dim('pruning'),
            'quantization': self.get_action_dim('quantization'),
            'distillation': self.get_action_dim('distillation'),
            'fusion': self.get_action_dim('fusion'),
            'update': self.get_action_dim('update'),
        }

    def encode_action(self, agent_name: str, action: Dict) -> torch.Tensor:
        """
        Encode an action to a tensor index.

        Args:
            agent_name: Name of the agent
            action: Action dictionary

        Returns:
            torch.Tensor: Encoded action index
        """
        if agent_name == 'pruning':
            # Encode as: layer_idx * 10 + pruning_ratio_index
            layer_idx = action['layer_idx']
            pruning_ratio_index = self.pruning_actions['pruning_ratio'].index(action['pruning_ratio'])
            return torch.tensor(layer_idx * 10 + pruning_ratio_index, dtype=torch.long)

        elif agent_name == 'quantization':
            # Encode as: layer_idx * 4 + bit_width_index
            layer_idx = action['layer_idx']
            bit_width_index = self.quantization_actions['bit_width'].index(action['bit_width'])
            return torch.tensor(layer_idx * 4 + bit_width_index, dtype=torch.long)

        elif agent_name == 'distillation':
            # Return as tensor of 2 values
            return torch.tensor([action['temperature'], action['alpha']], dtype=torch.float32)

        elif agent_name == 'fusion':
            # Encode as: start_layer * 6 + pattern_index
            start_layer = action['start_layer']
            pattern_index = self.fusion_actions['pattern'].index(action['pattern'])
            return torch.tensor(start_layer * 6 + pattern_index, dtype=torch.long)

        elif agent_name == 'update':
            # Encode as: strategy_index * 10 + update_ratio_index
            strategy_index = self.update_actions['strategy'].index(action['strategy'])
            update_ratio_index = self.update_actions['update_ratio'].index(action['update_ratio'])
            return torch.tensor(strategy_index * 10 + update_ratio_index, dtype=torch.long)

        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    def decode_action(self, agent_name: str, action_idx: torch.Tensor) -> Dict:
        """
        Decode a tensor index back to an action.

        Args:
            agent_name: Name of the agent
            action_idx: Encoded action index

        Returns:
            dict: Decoded action
        """
        idx = action_idx.item() if torch.is_tensor(action_idx) else int(action_idx)

        if agent_name == 'pruning':
            layer_idx = idx // 10
            pruning_ratio_index = idx % 10
            return {
                'layer_idx': layer_idx,
                'pruning_ratio': self.pruning_actions['pruning_ratio'][pruning_ratio_index],
            }

        elif agent_name == 'quantization':
            layer_idx = idx // 4
            bit_width_index = idx % 4
            return {
                'layer_idx': layer_idx,
                'bit_width': self.quantization_actions['bit_width'][bit_width_index],
            }

        elif agent_name == 'distillation':
            if torch.is_tensor(action_idx) and action_idx.dim() > 0:
                return {
                    'temperature': float(action_idx[0]),
                    'alpha': float(action_idx[1]),
                }
            else:
                # Return default values
                return {
                    'temperature': 4.0,
                    'alpha': 0.5,
                }

        elif agent_name == 'fusion':
            start_layer = idx // 6
            pattern_index = idx % 6
            return {
                'pattern': self.fusion_actions['pattern'][pattern_index],
                'start_layer': start_layer,
            }

        elif agent_name == 'update':
            strategy_index = idx // 10
            update_ratio_index = idx % 10
            return {
                'strategy': self.update_actions['strategy'][strategy_index],
                'update_ratio': self.update_actions['update_ratio'][update_ratio_index],
            }

        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    def __repr__(self) -> str:
        return (f"ActionSpace(num_layers={self.num_layers}, "
                f"pruning_dim={self.get_action_dim('pruning')}, "
                f"quant_dim={self.get_action_dim('quantization')}, "
                f"distill_dim={self.get_action_dim('distillation')}, "
                f"fusion_dim={self.get_action_dim('fusion')}, "
                f"update_dim={self.get_action_dim('update')})")
