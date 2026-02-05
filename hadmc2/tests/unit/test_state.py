"""Unit tests for State module"""

import unittest
import torch
import torch.nn as nn
from hadmc2.utils.state import State, create_state_from_model_and_hardware


class TestState(unittest.TestCase):
    """Test State class functionality."""

    def test_state_initialization(self):
        """Test that State initializes correctly."""
        state = State()

        self.assertEqual(state.model_state['num_layers'], 0)
        self.assertEqual(len(state.model_state['layer_types']), 0)
        self.assertEqual(state.compression_state['current_accuracy'], 1.0)
        self.assertEqual(state.compression_state['current_latency'], 0.0)

    def test_update_model_state(self):
        """Test updating model state from a model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

        state = State()
        state.update_model_state(model)

        self.assertEqual(state.model_state['num_layers'], 4)  # 2 Conv2d + 1 BN + 1 Linear
        self.assertEqual(len(state.model_state['layer_types']), 4)
        self.assertIn('conv', state.model_state['layer_types'])

    def test_to_tensor(self):
        """Test state to tensor conversion."""
        state = State()

        # Set some values
        state.model_state['num_layers'] = 5
        state.hardware_state['compute_capability'] = 50.0
        state.compression_state['current_accuracy'] = 0.95

        tensor = state.to_tensor()

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.shape[0], state.dim())

    def test_state_dim(self):
        """Test state dimension calculation."""
        state = State()
        self.assertGreater(state.dim(), 0)

    def test_state_copy(self):
        """Test state copy functionality."""
        state = State()
        state.model_state['num_layers'] = 5

        state_copy = state.copy()

        self.assertEqual(state_copy.model_state['num_layers'], 5)
        # Modify copy
        state_copy.model_state['num_layers'] = 10
        # Original should be unchanged
        self.assertEqual(state.model_state['num_layers'], 5)


class TestCreateStateFromModel(unittest.TestCase):
    """Test create_state_from_model_and_hardware function."""

    def test_create_state(self):
        """Test creating state from model and hardware config."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Linear(256, 10),
        )

        hardware_config = {
            'compute_capability': 10.0,
            'memory_bandwidth': 1000.0,
            'memory_capacity': 16.0,
            'power_budget': 250.0,
            'supported_precisions': ['FP32', 'FP16', 'INT8'],
            'has_tensor_core': True,
            'has_sparsity_support': True,
        }

        state = create_state_from_model_and_hardware(model, hardware_config)

        self.assertIsInstance(state, State)
        self.assertEqual(state.model_state['num_layers'], 2)
        self.assertEqual(state.hardware_state['compute_capability'], 10.0)
        self.assertEqual(state.hardware_state['supported_precisions'], ['FP32', 'FP16', 'INT8'])


if __name__ == '__main__':
    unittest.main()
