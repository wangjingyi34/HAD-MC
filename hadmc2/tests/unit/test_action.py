"""Unit tests for ActionSpace module"""

import unittest
import torch
from hadmc2.utils.action import ActionSpace


class TestActionSpace(unittest.TestCase):
    """Test ActionSpace class functionality."""

    def test_initialization(self):
        """Test ActionSpace initialization."""
        action_space = ActionSpace(num_layers=10, num_fusion_points=5)

        self.assertEqual(len(action_space.pruning_actions['layer_idx']), 10)
        self.assertEqual(len(action_space.quantization_actions['layer_idx']), 10)
        self.assertEqual(action_space.pruning_actions['pruning_ratio'], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def test_sample_pruning_action(self):
        """Test pruning action sampling."""
        action_space = ActionSpace(num_layers=10)

        action = action_space.sample_pruning_action()

        self.assertIn('layer_idx', action)
        self.assertIn('pruning_ratio', action)
        self.assertGreaterEqual(action['layer_idx'], 0)
        self.assertLess(action['layer_idx'], 10)
        self.assertIn(action['pruning_ratio'], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def test_sample_quantization_action(self):
        """Test quantization action sampling."""
        action_space = ActionSpace(num_layers=10)

        action = action_space.sample_quantization_action()

        self.assertIn('layer_idx', action)
        self.assertIn('bit_width', action)
        self.assertIn(action['bit_width'], [4, 8, 16, 32])

    def test_sample_distillation_action(self):
        """Test distillation action sampling."""
        action_space = ActionSpace(num_layers=10)

        action = action_space.sample_distillation_action()

        self.assertIn('temperature', action)
        self.assertIn('alpha', action)
        self.assertGreaterEqual(action['temperature'], 1.0)
        self.assertLessEqual(action['temperature'], 20.0)
        self.assertGreaterEqual(action['alpha'], 0.0)
        self.assertLessEqual(action['alpha'], 1.0)

    def test_sample_all_actions(self):
        """Test sampling all actions at once."""
        action_space = ActionSpace(num_layers=10)

        all_actions = action_space.sample_all_actions()

        self.assertIn('pruning', all_actions)
        self.assertIn('quantization', all_actions)
        self.assertIn('distillation', all_actions)
        self.assertIn('fusion', all_actions)
        self.assertIn('update', all_actions)

    def test_get_action_dim(self):
        """Test action dimension calculation."""
        action_space = ActionSpace(num_layers=10, num_fusion_points=5)

        pruning_dim = action_space.get_action_dim('pruning')
        quantization_dim = action_space.get_action_dim('quantization')
        distillation_dim = action_space.get_action_dim('distillation')

        self.assertEqual(pruning_dim, 100)  # 10 layers * 10 ratios
        self.assertEqual(quantization_dim, 40)  # 10 layers * 4 bit widths
        self.assertEqual(distillation_dim, 2)    # temperature and alpha

    def test_get_all_action_dims(self):
        """Test getting all action dimensions."""
        action_space = ActionSpace(num_layers=10, num_fusion_points=5)

        dims = action_space.get_all_action_dims()

        self.assertIn('pruning', dims)
        self.assertIn('quantization', dims)
        self.assertIn('distillation', dims)
        self.assertIn('fusion', dims)
        self.assertIn('update', dims)

    def test_encode_decode_action(self):
        """Test action encoding and decoding."""
        action_space = ActionSpace(num_layers=10)

        # Test pruning action
        original_action = {'layer_idx': 3, 'pruning_ratio': 0.5}
        encoded = action_space.encode_action('pruning', original_action)
        decoded = action_space.decode_action('pruning', encoded)

        self.assertEqual(decoded['layer_idx'], original_action['layer_idx'])
        self.assertEqual(decoded['pruning_ratio'], original_action['pruning_ratio'])

        # Test quantization action
        original_action = {'layer_idx': 2, 'bit_width': 8}
        encoded = action_space.encode_action('quantization', original_action)
        decoded = action_space.decode_action('quantization', encoded)

        self.assertEqual(decoded['layer_idx'], original_action['layer_idx'])
        self.assertEqual(decoded['bit_width'], original_action['bit_width'])


if __name__ == '__main__':
    unittest.main()
