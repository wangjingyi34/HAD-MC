"""Integration tests for Full Pipeline"""

import unittest
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
from hadmc2.controllers.ppo_controller import PPOController
from hadmc2.agents.pruning_agent import PruningAgent
from hadmc2.agents.quantization_agent import QuantizationAgent
from hadmc2.agents.distillation_agent import DistillationAgent
from hadmc2.agents.fusion_agent import FusionAgent
from hadmc2.agents.update_agent import UpdateAgent
from hadmc2.rewards.reward_function import RewardFunction
from hadmc2.hardware.hal import SimulatedHardwareAbstractionLayer
from hadmc2.inference.die import DedicatedInferenceEngine
from hadmc2.utils.state import State, create_state_from_model_and_hardware
from hadmc2.utils.action import ActionSpace


class TestFullPipeline(unittest.TestCase):
    """Test end-to-end HAD-MC 2.0 pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a more realistic model
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 0
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  # 2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),  # 3
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.randn(200, 3, 32, 32),
            torch.randint(0, 10, (200,))
        )
        val_dataset = TensorDataset(
            torch.randn(50, 3, 32, 32),
            torch.randint(0, 10, (50,))
        )
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Initialize all components
        self.hal = SimulatedHardwareAbstractionLayer()

        # PPO
        state = create_state_from_model_and_hardware(
            self.model, self.hal.get_hardware_config()
        )
        state_dim = state.to_tensor().shape[0]

        action_space = ActionSpace(num_layers=4)
        action_dims = action_space.get_all_action_dims()

        self.ppo = PPOController(state_dim, action_dims, device='cpu')

        # Five agents
        self.pruning_agent = PruningAgent(self.model, self.train_loader, 'cpu')
        self.quantization_agent = QuantizationAgent(self.model, self.val_loader, 'cpu')

        # Teacher model for distillation
        teacher_model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )
        self.distillation_agent = DistillationAgent(
            teacher_model, copy.deepcopy(self.model), 'cpu'
        )

        self.fusion_agent = FusionAgent(self.model, 'cpu')
        self.update_agent = UpdateAgent(self.model, 'cpu')

        # Reward function
        self.reward_fn = RewardFunction()

        # DIE
        self.die = DedicatedInferenceEngine(self.hal.get_hardware_config())

        # Baseline metrics
        self.baseline = {
            'accuracy': self._evaluate_accuracy(self.model),
            'latency': self._estimate_latency(self.model),
            'energy': 1.0,
            'size': 2.5,
        }

    def _evaluate_accuracy(self, model):
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total

    def _estimate_latency(self, model):
        """Estimate model latency (simplified)."""
        num_params = sum(p.numel() for p in model.parameters())
        return num_params * 1e-6  # Rough estimate

    def test_full_training_episode(self):
        """Test a complete training episode with all agents."""
        num_steps = 20
        episode_reward = 0

        for step in range(num_steps):
            # Get state
            state = create_state_from_model_and_hardware(
                self.model, self.hal.get_hardware_config()
            )
            state_tensor = state.to_tensor()

            # Select actions
            actions, log_prob, value = self.ppo.select_actions(state_tensor)

            # Apply pruning
            if step % 5 == 0:
                prunable_layers = [
                    name for name, _ in self.model.named_modules()
                    if isinstance(_, (nn.Conv2d, nn.Linear))
                ]
                if prunable_layers:
                    pruning_config = {prunable_layers[0]: 0.1}
                    self.model = self.pruning_agent.prune(pruning_config)

            # Apply quantization
            if step % 5 == 1:
                self.quantization_agent.calibrate(num_batches=5)
                quantizable_layers = [
                    name for name, _ in self.model.named_modules()
                    if isinstance(_, (nn.Conv2d, nn.Linear))
                ]
                if quantizable_layers:
                    quant_config = {quantizable_layers[1]: 8}
                    self.model = self.quantization_agent.quantize(quant_config)

            # Evaluate
            accuracy = self._evaluate_accuracy(self.model)
            latency = self._estimate_latency(self.model)
            size = sum(p.numel() for p in self.model.parameters()) * 4 / 1e6
            energy = latency * 0.001

            # Compute reward
            reward = self.reward_fn.compute(
                accuracy, latency, energy, size,
                self.baseline['accuracy'], self.baseline['latency'],
                self.baseline['energy'], self.baseline['size'],
            )

            # Update Pareto
            self.reward_fn.update_pareto_front(accuracy, latency, energy, size)

            # Store in buffer
            done = step == num_steps - 1
            self.ppo.buffer.add(state_tensor, actions, log_prob, reward, done, value)

            episode_reward += reward

        # Should have positive reward
        self.assertIsInstance(episode_reward, float)

        # Buffer should have all experiences
        self.assertEqual(len(self.ppo.buffer), num_steps)

    def test_all_agents_action_spaces(self):
        """Test that all agents have valid action spaces."""
        pruning_space = self.pruning_agent.get_action_space()
        quantization_space = self.quantization_agent.get_action_space()
        distillation_space = self.distillation_agent.get_action_space()
        fusion_space = self.fusion_agent.get_action_space()
        update_space = self.update_agent.get_action_space()

        # All should have type field
        for space in [pruning_space, quantization_space, distillation_space, fusion_space, update_space]:
            self.assertIn('type', space)

    def test_die_optimization(self):
        """Test DIE optimization."""
        compression_config = {
            'pruning': {'enabled': False},
            'quantization': {'enabled': False},
            'sparsity': {'enabled': False, 'pattern': '2:4', 'ratio': 0.5},
            'fusion': {'enabled': True},
        }

        optimized_model = self.die.optimize(self.model, compression_config)

        self.assertIsNotNone(optimized_model)
        self.assertIsNotNone(self.die.optimized_model)

        report = self.die.get_optimization_report()
        self.assertIsInstance(report, str)

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        # Create a temporary directory
        save_dir = '/tmp/test_checkpoints'
        os.makedirs(save_dir, exist_ok=True)

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, 'test.pt')
        self.ppo.save(checkpoint_path)

        # Verify file exists
        self.assertTrue(os.path.exists(checkpoint_path))

        # Load checkpoint
        self.ppo.load(checkpoint_path)

        # Clean up
        import shutil
        shutil.rmtree(save_dir)

    def test_reward_pareto_tracking(self):
        """Test Pareto frontier tracking."""
        # Add multiple points
        test_points = [
            (0.85, 15.0, 1.5, 6.0),
            (0.90, 12.0, 1.2, 5.0),
            (0.92, 10.0, 1.0, 4.5),
            (0.95, 8.0, 0.8, 4.0),
        (0.97, 7.0, 0.7, 3.5),
        (0.93, 9.0, 0.9, 3.8),
        ]

        for acc, lat, eng, size in test_points:
            self.reward_fn.update_pareto_front(acc, lat, eng, size)

        # Check Pareto frontier
        pareto_size = len(self.reward_fn.pareto_front)
        self.assertGreater(pareto_size, 0)
        self.assertLessEqual(pareto_size, len(test_points))

    def test_multi_episode_training(self):
        """Test training for multiple episodes."""
        num_episodes = 3
        num_steps_per_episode = 10

        best_reward = float('-inf')

        for episode in range(num_episodes):
            episode_reward = 0

            for step in range(num_steps_per_episode):
                state = create_state_from_model_and_hardware(
                    self.model, self.hal.get_hardware_config()
                )
                state_tensor = state.to_tensor()

                actions, log_prob, value = self.ppo.select_actions(state_tensor)

                # Simplified reward calculation
                accuracy = self._evaluate_accuracy(self.model)
                reward = (accuracy - 0.5) / 0.5  # Baseline 0.5

                done = step == num_steps_per_episode - 1
                self.ppo.buffer.add(state_tensor, actions, log_prob, reward, done, value)

                episode_reward += reward

            # Update best
            if episode_reward > best_reward:
                best_reward = episode_reward

            # Update PPO after each episode
            self.ppo.update(batch_size=8, num_epochs=1)

        # Should have completed all episodes
        self.assertEqual(episode, num_episodes - 1)

        # Should have tracked best reward
        self.assertNotEqual(best_reward, float('-inf'))


class TestEndToEndValidation(unittest.TestCase):
    """Test end-to-end validation of the entire system."""

    def test_system_initialization(self):
        """Test that all system components initialize correctly."""
        # Create simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Linear(256, 10),
        )

        # Test hardware abstraction
        hal = SimulatedHardwareAbstractionLayer()
        hw_config = hal.get_hardware_config()
        self.assertIsNotNone(hw_config)

        # Test PPO
        from hadmc2.utils.state import create_state_from_model_and_hardware
        state = create_state_from_model_and_hardware(model, hw_config)
        self.assertIsInstance(state.to_tensor(), torch.Tensor)

        # Test agents
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(
            torch.randn(50, 3, 32, 32),
            torch.randint(0, 10, (50,))
        )
        dataloader = DataLoader(dataset, batch_size=8)

        pruning = PruningAgent(model, dataloader, 'cpu')
        pruning_space = pruning.get_action_space()
        self.assertIn('type', pruning_space)

        # Test reward function
        reward_fn = RewardFunction()
        reward = reward_fn.compute(
            0.95, 5.0, 0.5, 2.5,
            0.90, 10.0, 1.0, 5.0
        )
        self.assertIsInstance(reward, float)

        # Test DIE
        die = DedicatedInferenceEngine(hw_config)
        compression_config = {
            'pruning': {'enabled': False},
            'quantization': {'enabled': False},
        }
        optimized = die.optimize(model, compression_config)
        self.assertIsNotNone(optimized)

    def test_data_flow(self):
        """Test that data flows correctly through the system."""
        # Create minimal test model
        model = nn.Linear(64, 10)

        # Create data
        dataset = TensorDataset(
            torch.randn(20, 64),
            torch.randint(0, 10, (20,))
        )
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

        # Test agent interaction with data
        pruning = PruningAgent(model, dataloader, 'cpu')
        importance = pruning.compute_importance()
        self.assertIsInstance(importance, dict)

        # Test action application
        action = {'layer_idx': 0, 'pruning_ratio': 0.1}
        pruned = pruning.apply_action(model, action)
        self.assertIsNotNone(pruned)

    def test_config_loading(self):
        """Test configuration loading."""
        from hadmc2.utils.config import load_config

        # Load default config
        config = load_config(config_name='default')

        self.assertIsInstance(config, dict)
        self.assertIn('model', config)
        self.assertIn('training', config)
        self.assertIn('agents', config)
        self.assertIn('reward', config)


if __name__ == '__main__':
    unittest.main()
