"""Unit tests for Agents"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from hadmc2.agents.pruning_agent import PruningAgent
from hadmc2.agents.quantization_agent import QuantizationAgent
from hadmc2.agents.distillation_agent import DistillationAgent
from hadmc2.agents.fusion_agent import FusionAgent
from hadmc2.agents.update_agent import UpdateAgent


class TestPruningAgent(unittest.TestCase):
    """Test PruningAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

        # Create dummy data loader
        dataset = TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 10, (100,))
        )
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        self.agent = PruningAgent(self.model, self.dataloader, device='cpu')

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(len(self.agent.pruning_ratios), 10)
        self.assertEqual(self.agent.device, 'cpu')

    def test_get_action_space(self):
        """Test getting action space."""
        action_space = self.agent.get_action_space()

        self.assertIn('type', action_space)
        self.assertIn('layer_idx', action_space)
        self.assertIn('pruning_ratio', action_space)

    def test_compute_importance(self):
        """Test computing importance scores."""
        importance = self.agent.compute_importance()

        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)

    def test_prune(self):
        """Test pruning."""
        pruning_config = {
            self._get_prunable_layer_names()[0]: 0.5,
            self._get_prunable_layer_names()[1]: 0.3,
        }

        pruned_model = self.agent.prune(pruning_config)

        # Check that model is modified
        self.assertIsNotNone(pruned_model)

    def test_get_state(self):
        """Test getting agent state."""
        state = self.agent.get_state()

        self.assertIn('total_params', state)
        self.assertIn('sparsity', state)

    def test_get_action(self):
        """Test getting action."""
        state = torch.randn(64)
        action, log_prob = self.agent.get_action(state)

        # Action should be a dict with layer names as keys
        self.assertIsInstance(action, dict)
        self.assertEqual(len(action), 1)  # One layer pruned
        # Check that action has valid values
        for layer_name, ratio in action.items():
            self.assertIsInstance(ratio, float)
            self.assertIn(ratio, self.agent.pruning_ratios)

    def _get_prunable_layer_names(self):
        """Helper to get prunable layer names."""
        return [
            name for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]


class TestQuantizationAgent(unittest.TestCase):
    """Test QuantizationAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

        dataset = TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 10, (100,))
        )
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        self.agent = QuantizationAgent(self.model, self.dataloader, device='cpu')

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(len(self.agent.bit_widths), 4)
        self.assertFalse(self.agent.calibrated)

    def test_get_action_space(self):
        """Test getting action space."""
        action_space = self.agent.get_action_space()

        self.assertIn('type', action_space)
        self.assertIn('bit_width', action_space)

    def test_calibrate(self):
        """Test calibration."""
        self.agent.calibrate(num_batches=10)

        self.assertTrue(self.agent.calibrated)
        self.assertGreater(len(self.agent.quantization_params), 0)

    def test_quantize(self):
        """Test quantization."""
        self.agent.calibrate(num_batches=5)

        quantization_config = {
            self._get_quantizable_layer_names()[0]: 8,
            self._get_quantizable_layer_names()[1]: 16,
        }

        quantized_model = self.agent.quantize(quantization_config)

        self.assertIsNotNone(quantized_model)

    def test_get_state(self):
        """Test getting agent state."""
        state = self.agent.get_state()

        self.assertIn('num_quantizable_layers', state)
        self.assertIn('calibrated', state)

    def _get_quantizable_layer_names(self):
        """Helper to get quantizable layer names."""
        return [
            name for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]

    def _get_quantizable_layers(self):
        """Helper to get quantizable layers."""
        return [
            module for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]


class TestDistillationAgent(unittest.TestCase):
    """Test DistillationAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        # Teacher model
        self.teacher = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

        # Student model (smaller)
        self.student = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

        dataset = TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 10, (100,))
        )
        self.train_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        self.val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        self.agent = DistillationAgent(self.teacher, self.student, device='cpu')

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.temperature, 4.0)
        self.assertEqual(self.agent.alpha, 0.5)

    def test_get_action_space(self):
        """Test getting action space."""
        action_space = self.agent.get_action_space()

        self.assertIn('type', action_space)
        self.assertIn('temperature', action_space)
        self.assertIn('alpha', action_space)

    def test_get_action(self):
        """Test getting action."""
        state = torch.randn(64)
        action, log_prob = self.agent.get_action(state)

        self.assertIn('temperature', action)
        self.assertIn('alpha', action)

    def test_get_state(self):
        """Test getting agent state."""
        state = self.agent.get_state()

        self.assertIn('temperature', state)
        self.assertIn('alpha', state)


class TestFusionAgent(unittest.TestCase):
    """Test FusionAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.agent = FusionAgent(self.model, device='cpu')

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(len(self.agent.fusion_patterns), 6)
        self.assertEqual(len(self.agent.fused_operations), 0)

    def test_get_action_space(self):
        """Test getting action space."""
        action_space = self.agent.get_action_space()

        self.assertIn('type', action_space)
        self.assertIn('pattern', action_space)

    def test_get_action(self):
        """Test getting action."""
        state = torch.randn(64)
        action, log_prob = self.agent.get_action(state)

        self.assertIn('pattern', action)
        self.assertIn('start_layer', action)

    def test_get_state(self):
        """Test getting agent state."""
        state = self.agent.get_state()

        self.assertIn('num_fusable_points', state)
        self.assertIn('num_fused', state)


class TestUpdateAgent(unittest.TestCase):
    """Test UpdateAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

        dataset = TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 10, (100,))
        )
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        self.agent = UpdateAgent(self.model, device='cpu')

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(len(self.agent.strategies), 3)
        self.assertEqual(len(self.agent.update_ratios), 10)

    def test_get_action_space(self):
        """Test getting action space."""
        action_space = self.agent.get_action_space()

        self.assertIn('type', action_space)
        self.assertIn('strategy', action_space)

    def test_get_action(self):
        """Test getting action."""
        state = torch.randn(64)
        action, log_prob = self.agent.get_action(state)

        self.assertIn('strategy', action)
        self.assertIn('update_ratio', action)

    def test_get_state(self):
        """Test getting agent state."""
        state = self.agent.get_state()

        self.assertIn('total_params', state)
        self.assertIn('model_size_mb', state)

    def test_build_hash_tables(self):
        """Test building hash tables."""
        self.agent.build_hash_tables(num_clusters=16)

        self.assertGreater(len(self.agent.hash_tables), 0)

    def test_compute_hash(self):
        """Test hash computation."""
        tensor = torch.randn(10)
        hash_val = self.agent._compute_hash(tensor)

        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)  # SHA256 hash length


if __name__ == '__main__':
    unittest.main()
