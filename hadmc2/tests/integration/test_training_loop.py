"""Integration tests for Training Loop"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from hadmc2.controllers.ppo_controller import PPOController
from hadmc2.agents.pruning_agent import PruningAgent
from hadmc2.agents.quantization_agent import QuantizationAgent
from hadmc2.rewards.reward_function import RewardFunction
from hadmc2.hardware.hal import SimulatedHardwareAbstractionLayer
from hadmc2.utils.state import create_state_from_model_and_hardware
from hadmc2.utils.action import ActionSpace


class TestTrainingLoop(unittest.TestCase):
    """Test end-to-end training loop."""

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

        # Create dummy data
        train_dataset = TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 10, (100,))
        )
        val_dataset = TensorDataset(
            torch.randn(20, 3, 32, 32),
            torch.randint(0, 10, (20,))
        )
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # Initialize components
        self.hal = SimulatedHardwareAbstractionLayer()

        # PPO Controller
        state = create_state_from_model_and_hardware(
            self.model, self.hal.get_hardware_config()
        )
        self.state_dim = state.to_tensor().shape[0]

        self.action_dims = {
            'pruning': 2 * 10,   # 2 layers * 10 ratios
            'quantization': 2 * 4,  # 2 layers * 4 bit widths
            'distillation': 2,       # temperature and alpha
            'fusion': 2 * 6,         # 2 layers * 6 patterns
            'update': 3 * 5,         # 3 strategies * 5 ratios
        }

        self.ppo = PPOController(self.state_dim, self.action_dims, device='cpu')

        # Agents
        self.pruning_agent = PruningAgent(self.model, self.train_loader, 'cpu')
        self.quantization_agent = QuantizationAgent(self.model, self.val_loader, 'cpu')

        # Reward function
        self.reward_fn = RewardFunction()

        # Evaluate baseline
        self.baseline_accuracy = self._evaluate_accuracy(self.model)
        self.baseline_latency = self._estimate_latency(self.model)

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
        """Estimate model latency."""
        # Simplified latency estimation
        return 10.0

    def test_single_episode(self):
        """Test running a single training episode."""
        num_steps = 10
        episode_reward = 0

        # Initial state
        state = create_state_from_model_and_hardware(
            self.model, self.hal.get_hardware_config()
        )
        state_tensor = state.to_tensor()

        for step in range(num_steps):
            # Select actions
            actions, log_prob, value = self.ppo.select_actions(state_tensor)

            # Apply pruning (simplified)
            if step % 3 == 0:
                action_config = {'layer_idx': 0, 'pruning_ratio': 0.1}
                pruned = self.pruning_agent.prune({self._get_layer_names()[0]: 0.1})

            # Apply quantization (simplified)
            if step % 3 == 1:
                action_config = {'layer_idx': 1, 'bit_width': 8}
                # Would call quantization here

            # Evaluate
            accuracy = self._evaluate_accuracy(self.model)
            latency = self._estimate_latency(self.model)
            size = 2.5  # Simplified
            energy = latency * 0.1

            # Compute reward
            reward = self.reward_fn.compute(
                accuracy, latency, energy, size,
                self.baseline_accuracy, self.baseline_latency,
                1.0, 5.0
            )

            # Store in buffer
            done = step == num_steps - 1
            self.ppo.buffer.add(state_tensor, actions, log_prob, reward, done, value)

            episode_reward += reward

        # Should have collected experiences
        self.assertEqual(len(self.ppo.buffer), num_steps)

        # Total reward should be computed
        self.assertIsInstance(episode_reward, float)

    def test_ppo_update(self):
        """Test PPO update with collected experiences."""
        # Collect some experiences
        num_steps = 32

        for _ in range(num_steps):
            state_tensor = torch.randn(self.state_dim)

            actions = {
                'pruning': torch.randint(0, 20, (1,)),
                'quantization': torch.randint(0, 8, (1,)),
                'distillation': torch.randn(1, 2),
                'fusion': torch.randint(0, 12, (1,)),
                'update': torch.randint(0, 15, (1,)),
            }
            log_prob = torch.randn(1)
            reward = 1.0
            done = False
            value = torch.randn(1)

            self.ppo.buffer.add(state_tensor, actions, log_prob, reward, done, value)

        # Update PPO
        update_info = self.ppo.update(batch_size=16, num_epochs=2)

        self.assertIn('policy_loss', update_info)
        self.assertIn('value_loss', update_info)
        self.assertIn('entropy', update_info)

        # Buffer should be cleared after update
        self.assertEqual(len(self.ppo.buffer), 0)

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        num_episodes = 5
        num_steps_per_episode = 10

        all_rewards = []

        for episode in range(num_episodes):
            episode_reward = 0

            for step in range(num_steps_per_episode):
                state = create_state_from_model_and_hardware(
                    self.model, self.hal.get_hardware_config()
                )
                state_tensor = state.to_tensor()

                actions, log_prob, value = self.ppo.select_actions(state_tensor)

                # Simplified evaluation
                accuracy = self._evaluate_accuracy(self.model)
                reward = (accuracy - self.baseline_accuracy) / self.baseline_accuracy

                done = step == num_steps_per_episode - 1
                self.ppo.buffer.add(state_tensor, actions, log_prob, reward, done, value)

                episode_reward += reward

            all_rewards.append(episode_reward)

            # Update PPO every 2 episodes
            if episode % 2 == 1 and len(self.ppo.buffer) >= 16:
                self.ppo.update(batch_size=16, num_epochs=1)

        # Should have collected rewards for all episodes
        self.assertEqual(len(all_rewards), num_episodes)

        # All rewards should be computed
        for reward in all_rewards:
            self.assertIsInstance(reward, float)

    def test_agent_coordination(self):
        """Test coordination between multiple agents."""
        # This test verifies agents can work together

        # Compute importance for pruning
        importance = self.pruning_agent.compute_importance()
        self.assertIsInstance(importance, dict)

        # Calibrate for quantization
        self.quantization_agent.calibrate(num_batches=5)
        self.assertTrue(self.quantization_agent.calibrated)

        # Get action spaces
        pruning_space = self.pruning_agent.get_action_space()
        quantization_space = self.quantization_agent.get_action_space()

        self.assertIn('type', pruning_space)
        self.assertIn('type', quantization_space)

    def test_reward_with_pareto(self):
        """Test reward function with Pareto frontier."""
        # Build Pareto frontier
        self.reward_fn.update_pareto_front(0.90, 10.0, 1.0, 5.0)
        self.reward_fn.update_pareto_front(0.92, 9.0, 0.9, 4.5)
        self.reward_fn.update_pareto_front(0.95, 8.0, 0.8, 4.0)

        # Test Pareto-aware reward
        reward = self.reward_fn.compute_pareto_reward(
            accuracy=0.93,  # On Pareto front
            latency=8.5,    # On Pareto front
            energy=0.85,    # On Pareto front
            size=4.2,       # On Pareto front
            baseline_accuracy=0.90,
            baseline_latency=10.0,
            baseline_energy=1.0,
            baseline_size=5.0,
            use_pareto_bonus=True,
        )

        self.assertGreater(reward, 0)

    def test_hardware_abstraction(self):
        """Test hardware abstraction layer."""
        # Get hardware config
        hw_config = self.hal.get_hardware_config()

        self.assertIsNotNone(hw_config)
        self.assertEqual(hw_config.name, 'NVIDIA_A100')

        # Estimate latency
        model_config = {
            'layers': [
                {'type': 'conv2d', 'precision': 'FP32', 'flops': 1e6, 'memory': 1e5},
                {'type': 'conv2d', 'precision': 'FP32', 'flops': 5e5, 'memory': 5e4},
            ]
        }

        latency = self.hal.estimate_latency(model_config)

        self.assertIsInstance(latency, float)
        self.assertGreater(latency, 0)

    def _get_layer_names(self):
        """Helper to get prunable layer names."""
        return [
            name for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]


if __name__ == '__main__':
    unittest.main()
