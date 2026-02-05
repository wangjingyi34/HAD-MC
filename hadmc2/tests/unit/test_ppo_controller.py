"""Unit tests for PPO Controller"""

import unittest
import torch
import torch.nn as nn
from hadmc2.controllers.ppo_controller import PPOController, PolicyNetwork, ValueNetwork, ExperienceBuffer


class TestPolicyNetwork(unittest.TestCase):
    """Test PolicyNetwork class."""

    def test_initialization(self):
        """Test PolicyNetwork initialization."""
        state_dim = 64
        action_dims = {
            'pruning': 100,
            'quantization': 40,
            'distillation': 2,
            'fusion': 60,
            'update': 15,
        }

        policy = PolicyNetwork(state_dim, action_dims)

        self.assertIsInstance(policy, nn.Module)
        self.assertEqual(policy.state_dim, state_dim)
        self.assertEqual(policy.action_dims, action_dims)

    def test_forward(self):
        """Test forward pass through PolicyNetwork."""
        state_dim = 64
        action_dims = {
            'pruning': 10,
            'quantization': 8,
            'distillation': 2,
            'fusion': 12,
            'update': 10,
        }

        policy = PolicyNetwork(state_dim, action_dims)
        state = torch.randn(1, state_dim)

        distributions = policy(state)

        self.assertIn('pruning', distributions)
        self.assertIn('quantization', distributions)
        self.assertIn('distillation_mean', distributions)
        self.assertIn('fusion', distributions)
        self.assertIn('update', distributions)

        # Check softmax probabilities sum to ~1
        pruning_probs = distributions['pruning']
        self.assertTrue(torch.allclose(pruning_probs.sum(dim=1), torch.ones(1), atol=1e-5))

    def test_sample_actions(self):
        """Test action sampling from PolicyNetwork."""
        state_dim = 64
        action_dims = {
            'pruning': 10,
            'quantization': 4,
            'distillation': 2,
            'fusion': 6,
            'update': 5,
        }

        policy = PolicyNetwork(state_dim, action_dims)
        state = torch.randn(1, state_dim)

        actions, log_prob = policy.sample_actions(state)

        self.assertIn('pruning', actions)
        self.assertIn('quantization', actions)
        self.assertIsInstance(log_prob, torch.Tensor)


class TestValueNetwork(unittest.TestCase):
    """Test ValueNetwork class."""

    def test_initialization(self):
        """Test ValueNetwork initialization."""
        state_dim = 64

        value_net = ValueNetwork(state_dim)

        self.assertIsInstance(value_net, nn.Module)

    def test_forward(self):
        """Test forward pass through ValueNetwork."""
        state_dim = 64
        value_net = ValueNetwork(state_dim)
        state = torch.randn(1, state_dim)

        value = value_net(state)

        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(value.shape, (1, 1))


class TestExperienceBuffer(unittest.TestCase):
    """Test ExperienceBuffer class."""

    def test_initialization(self):
        """Test ExperienceBuffer initialization."""
        buffer = ExperienceBuffer()

        self.assertEqual(len(buffer.states), 0)
        self.assertEqual(len(buffer.rewards), 0)
        self.assertEqual(buffer.size, 0)

    def test_add(self):
        """Test adding experience to buffer."""
        buffer = ExperienceBuffer()

        state = torch.randn(64)
        actions = {
            'pruning': torch.tensor(5),
            'quantization': torch.tensor(2),
            'distillation': torch.tensor([4.0, 0.5]),
            'fusion': torch.tensor(3),
            'update': torch.tensor(4),
        }
        log_prob = torch.tensor(0.5)
        reward = 1.0
        done = False
        value = torch.tensor(0.5)

        buffer.add(state, actions, log_prob, reward, done, value)

        self.assertEqual(len(buffer.states), 1)
        self.assertEqual(len(buffer.rewards), 1)
        self.assertEqual(buffer.size, 1)

    def test_get_all(self):
        """Test getting all experiences."""
        buffer = ExperienceBuffer()

        # Add multiple experiences
        for i in range(10):
            state = torch.randn(64)
            actions = {
                'pruning': torch.tensor(i % 10),
                'quantization': torch.tensor(i % 4),
                'distillation': torch.tensor([4.0, 0.5]),
                'fusion': torch.tensor(i % 6),
                'update': torch.tensor(i % 5),
            }
            log_prob = torch.tensor(0.5)
            reward = float(i)
            done = i == 9
            value = torch.tensor(0.5)

            buffer.add(state, actions, log_prob, reward, done, value)

        states, actions, log_probs, rewards, dones, values = buffer.get_all()

        self.assertEqual(states.shape[0], 10)
        self.assertEqual(rewards.shape[0], 10)

    def test_clear(self):
        """Test clearing buffer."""
        buffer = ExperienceBuffer()

        state = torch.randn(64)
        actions = {
            'pruning': torch.tensor(5),
            'quantization': torch.tensor(2),
            'distillation': torch.tensor([4.0, 0.5]),
            'fusion': torch.tensor(3),
            'update': torch.tensor(4),
        }
        buffer.add(state, actions, torch.tensor(0.5), 1.0, False, torch.tensor(0.5))

        buffer.clear()

        self.assertEqual(len(buffer.states), 0)
        self.assertEqual(buffer.size, 0)


class TestPPOController(unittest.TestCase):
    """Test PPOController class."""

    def test_initialization(self):
        """Test PPOController initialization."""
        state_dim = 64
        action_dims = {
            'pruning': 10,
            'quantization': 4,
            'distillation': 2,
            'fusion': 6,
            'update': 5,
        }

        ppo = PPOController(state_dim, action_dims, device='cpu')

        self.assertIsInstance(ppo.policy_network, PolicyNetwork)
        self.assertIsInstance(ppo.value_network, ValueNetwork)
        self.assertIsInstance(ppo.buffer, ExperienceBuffer)
        self.assertEqual(ppo.state_dim, state_dim)

    def test_select_actions(self):
        """Test action selection."""
        state_dim = 64
        action_dims = {
            'pruning': 10,
            'quantization': 4,
            'distillation': 2,
            'fusion': 6,
            'update': 5,
        }

        ppo = PPOController(state_dim, action_dims, device='cpu')
        state = torch.randn(1, state_dim)

        actions, log_prob, value = ppo.select_actions(state)

        self.assertIn('pruning', actions)
        self.assertIn('quantization', actions)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)

    def test_compute_gae(self):
        """Test GAE computation."""
        state_dim = 64
        action_dims = {
            'pruning': 10,
            'quantization': 4,
            'distillation': 2,
            'fusion': 6,
            'update': 5,
        }

        ppo = PPOController(state_dim, action_dims, device='cpu')

        rewards = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5])
        values = torch.tensor([0.5, 1.0, 0.75, 1.5, 1.25])
        dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        next_value = torch.tensor(2.0)

        advantages, returns = ppo.compute_gae(rewards, values, dones, next_value)

        self.assertEqual(advantages.shape, (5,))
        self.assertEqual(returns.shape, (5,))

    def test_update(self):
        """Test PPO update with small buffer."""
        state_dim = 64
        action_dims = {
            'pruning': 10,
            'quantization': 4,
            'distillation': 2,
            'fusion': 6,
            'update': 5,
        }

        ppo = PPOController(state_dim, action_dims, device='cpu')

        # Add some experiences
        for _ in range(100):
            state = torch.randn(state_dim)
            actions = {
                'pruning': torch.tensor(5),
                'quantization': torch.tensor(2),
                'distillation': torch.tensor([4.0, 0.5]),
                'fusion': torch.tensor(3),
                'update': torch.tensor(4),
            }
            log_prob = torch.tensor(0.5)
            reward = 1.0
            done = False
            value = torch.tensor(0.5)

            ppo.buffer.add(state, actions, log_prob, reward, done, value)

        update_info = ppo.update(batch_size=32, num_epochs=2)

        self.assertIn('policy_loss', update_info)
        self.assertIn('value_loss', update_info)
        self.assertIn('entropy', update_info)


if __name__ == '__main__':
    unittest.main()
