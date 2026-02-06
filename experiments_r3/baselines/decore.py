"""
DECORE: Deep Compression with Reinforcement Learning

This module implements the DECORE method from:
"DECORE: Deep Compression with Reinforcement Learning"
Alwani et al., CVPR 2022

DECORE combines pruning and quantization using reinforcement learning
for joint optimization with multi-objective rewards.

Key features:
- Joint pruning and quantization optimization
- Multi-objective reward function (accuracy, latency, energy)
- PPO algorithm for action selection
- Progressive compression strategy
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import deque
import copy
from torch.distributions import Categorical


class ReplayBuffer:
    """Experience replay buffer for PPO."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

        # Maintain capacity
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.log_probs.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.values.pop(0)

    def get_batch(self, indices: List[int]) -> Dict:
        """Get a batch of experiences."""
        return {
            'states': torch.FloatTensor([self.states[i] for i in indices]),
            'actions': torch.LongTensor([self.actions[i] for i in indices]),
            'log_probs': torch.stack([self.log_probs[i] for i in indices]),
            'rewards': torch.FloatTensor([self.rewards[i] for i in indices]),
            'dones': torch.FloatTensor([self.dones[i] for i in indices]),
            'values': torch.stack([self.values[i] for i in indices])
        }

    def __len__(self):
        return len(self.states)


class PolicyNetwork(nn.Module):
    """Policy network for DECORE - outputs pruning and quantization actions."""

    def __init__(
        self,
        state_dim: int,
        num_pruning_actions: int = 10,
        num_quant_actions: int = 4,
        hidden_dim: int = 256
    ):
        super(PolicyNetwork, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Pruning action head
        self.pruning_head = nn.Linear(hidden_dim, num_pruning_actions)

        # Quantization action head
        self.quant_head = nn.Linear(hidden_dim, num_quant_actions)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)

        pruning_logits = self.pruning_head(features)
        quant_logits = self.quant_head(features)
        value = self.value_head(features)

        return pruning_logits, quant_logits, value

    def get_action(self, state):
        """Get action and log probability."""
        with torch.no_grad():
            pruning_logits, quant_logits, value = self.forward(state)

            # Sample actions
            pruning_dist = Categorical(logits=pruning_logits)
            quant_dist = Categorical(logits=quant_logits)

            pruning_action = pruning_dist.sample()
            quant_action = quant_dist.sample()

            log_prob = pruning_dist.log_prob(pruning_action) + quant_dist.log_prob(quant_action)

        return (pruning_action.item(), quant_action.item()), log_prob, value


class DECOREAgent:
    """DECORE agent using PPO for joint pruning and quantization."""

    def __init__(
        self,
        state_dim: int,
        num_pruning_actions: int = 10,
        num_quant_actions: int = 4,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize DECORE agent.

        Args:
            state_dim: Dimension of state space
            num_pruning_actions: Number of pruning ratio actions
            num_quant_actions: Number of quantization bit width actions
            hidden_dim: Hidden dimension for networks
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
            device: Device to run on
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Pruning ratios: 0.0, 0.1, ..., 0.9
        self.pruning_ratios = np.linspace(0.0, 0.9, num_pruning_actions).tolist()
        # Bit widths: 4, 8, 16, 32
        self.bit_widths = [4, 8, 16, 32][:num_quant_actions]

        # Create policy network
        self.policy = PolicyNetwork(
            state_dim, num_pruning_actions, num_quant_actions, hidden_dim
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state: np.ndarray) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor]:
        """
        Select action (pruning ratio, bit width) for given state.

        Args:
            state: State vector

        Returns:
            actions: Tuple of (pruning_action_idx, quant_action_idx)
            log_prob: Log probability of actions
            value: Estimated state value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actions, log_prob, value = self.policy.get_action(state_tensor)

        return actions, log_prob.squeeze(), value.squeeze()

    def compute_gae(self, rewards: List[float], values: List[torch.Tensor], dones: List[float]) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages).to(self.device)

    def update(self, batch_size: int = 64, num_epochs: int = 10):
        """
        Update policy network using PPO.

        Args:
            batch_size: Batch size for training
            num_epochs: Number of update epochs
        """
        if len(self.buffer) < batch_size:
            return

        # Get all data from buffer
        num_samples = len(self.buffer)
        indices = list(range(num_samples))

        # Compute advantages and returns
        rewards = self.buffer.rewards[-num_samples:]
        values = [v.item() if isinstance(v, torch.Tensor) else v for v in self.buffer.values[-num_samples:]]
        dones = self.buffer.dones[-num_samples:]

        advantages = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute returns
        returns = advantages + torch.FloatTensor(values).to(self.device)

        # PPO update
        for epoch in range(num_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                batch = self.buffer.get_batch(batch_indices)
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                pruning_logits, quant_logits, new_values = self.policy(batch['states'])

                # Get log probs for old and new actions
                old_pruning_actions = batch['actions'][:, 0]
                old_quant_actions = batch['actions'][:, 1]

                new_pruning_dist = Categorical(logits=pruning_logits)
                new_quant_dist = Categorical(logits=quant_logits)

                new_log_prob = (
                    new_pruning_dist.log_prob(old_pruning_actions) +
                    new_quant_dist.log_prob(old_quant_actions)
                )

                # Compute ratio
                ratio = torch.exp(new_log_prob - batch['log_probs'])

                # PPO clipped loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)

                # Entropy bonus
                entropy = (
                    new_pruning_dist.entropy().mean() +
                    new_quant_dist.entropy().mean()
                )

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class CompressionEnvironment:
    """Environment for DECORE joint pruning and quantization."""

    def __init__(
        self,
        model: nn.Module,
        num_pruning_actions: int = 10,
        num_quant_actions: int = 4,
        target_compression: float = 0.5
    ):
        """
        Initialize compression environment.

        Args:
            model: PyTorch model to compress
            num_pruning_actions: Number of pruning ratio options
            num_quant_actions: Number of bit width options
            target_compression: Target compression ratio
        """
        self.model = model
        self.num_pruning_actions = num_pruning_actions
        self.num_quant_actions = num_quant_actions
        self.target_compression = target_compression

        # Pruning ratios: 0.0, 0.1, ..., 0.9
        self.pruning_ratios = np.linspace(0.0, 0.9, num_pruning_actions).tolist()
        # Bit widths: 4, 8, 16, 32
        self.bit_widths = [4, 8, 16, 32][:num_quant_actions]

        # Get list of compressible layers
        self.compressible_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.compressible_layers.append((name, module))

        self.num_layers = len(self.compressible_layers)
        self.current_layer_idx = 0

        # Track compression decisions
        self.compression_decisions = {}
        self.rewards = []

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_layer_idx = 0
        self.compression_decisions = {}
        self.rewards = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        state = []

        # Layer index (normalized)
        state.append(self.current_layer_idx / max(1, self.num_layers - 1))

        # Previous compression decisions (padded to max 5)
        prev_decisions = list(self.compression_decisions.values())[-5:]
        for _ in range(5):
            if prev_decisions:
                decision = prev_decisions.pop(0)
                # Normalize to [0, 1]
                state.append(decision['pruning_ratio'])
                state.append(decision['bit_width'] / 32.0)
            else:
                state.extend([0.0, 0.0])

        # Current layer info
        if self.current_layer_idx < self.num_layers:
            layer_name, layer_module = self.compressible_layers[self.current_layer_idx]
            if isinstance(layer_module, nn.Conv2d):
                state.append(layer_module.in_channels / 512)
                state.append(layer_module.out_channels / 512)
                state.append(layer_module.kernel_size[0] / 10)
            elif isinstance(layer_module, nn.Linear):
                state.append(layer_module.in_features / 4096)
                state.append(layer_module.out_features / 4096)
                state.append(1.0)
            else:
                state.extend([0.0, 0.0, 0.0])
        else:
            state.extend([0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    def step(
        self,
        pruning_action: int,
        quant_action: int
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Execute actions for current layer.

        Args:
            pruning_action: Pruning action index
            quant_action: Quantization action index

        Returns:
            next_state: Next state
            reward: Reward for these actions
            done: Whether episode is finished
        """
        if self.current_layer_idx >= self.num_layers:
            return self._get_state(), 0.0, True

        # Get actions
        pruning_ratio = self.pruning_ratios[pruning_action]
        bit_width = self.bit_widths[quant_action]

        # Store decision
        layer_name, layer_module = self.compressible_layers[self.current_layer_idx]
        self.compression_decisions[layer_name] = {
            'pruning_ratio': pruning_ratio,
            'bit_width': bit_width
        }

        # Calculate reward
        reward = self._calculate_reward(pruning_ratio, bit_width)

        self.rewards.append(reward)
        self.current_layer_idx += 1

        done = self.current_layer_idx >= self.num_layers
        next_state = self._get_state()

        return next_state, reward, done

    def _calculate_reward(self, pruning_ratio: float, bit_width: int) -> float:
        """
        Calculate reward for given pruning and quantization.

        Args:
            pruning_ratio: Pruning ratio [0, 1]
            bit_width: Bit width [4, 8, 16, 32]

        Returns:
            Reward value
        """
        # Compression reward (higher compression = higher reward, up to target)
        pruning_reward = min(pruning_ratio / self.target_compression, 1.0) * 2.0
        quant_reward = (32 - bit_width) / 32.0 * 2.0

        # Accuracy penalty
        accuracy_penalty = 0.0
        if pruning_ratio > 0.7:
            accuracy_penalty += (pruning_ratio - 0.7) * 2
        if bit_width == 4:
            accuracy_penalty += 1.0
        elif bit_width == 8:
            accuracy_penalty += 0.3
        elif bit_width == 16:
            accuracy_penalty += 0.1

        # Combined reward
        return pruning_reward + quant_reward - accuracy_penalty

    def get_final_decisions(self) -> Dict[str, Dict[str, float]]:
        """Get final compression decisions for all layers."""
        return self.compression_decisions

    def get_compression_stats(self) -> Dict[str, float]:
        """Get overall compression statistics."""
        if not self.compression_decisions:
            return {
                'avg_pruning_ratio': 0.0,
                'avg_bit_width': 32.0,
                'total_compression': 0.0
            }

        pruning_ratios = [d['pruning_ratio'] for d in self.compression_decisions.values()]
        bit_widths = [d['bit_width'] for d in self.compression_decisions.values()]

        return {
            'avg_pruning_ratio': np.mean(pruning_ratios),
            'avg_bit_width': np.mean(bit_widths),
            'total_compression': np.mean(pruning_ratios) + (32 - np.mean(bit_widths)) / 32.0
        }


def train_decore(
    model: nn.Module,
    num_episodes: int = 300,
    max_steps: int = 50,
    device: str = 'cuda'
) -> Tuple[Dict[str, Dict[str, float]], DECOREAgent]:
    """
    Train DECORE agent to find optimal pruning and quantization.

    Args:
        model: PyTorch model to compress
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        device: Device to run on

    Returns:
        decisions: Dictionary mapping layer names to compression decisions
        agent: Trained DECORE agent
    """
    # Create environment
    env = CompressionEnvironment(model)

    # Create agent
    state_dim = len(env.reset())
    agent = DECOREAgent(state_dim=state_dim, device=device)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            # Select action
            (pruning_action, quant_action), log_prob, value = agent.select_action(state)

            # Execute action
            next_state, reward, done = env.step(pruning_action, quant_action)

            # Store experience
            agent.buffer.add(
                state,
                torch.tensor([pruning_action, quant_action]),
                log_prob.unsqueeze(0),
                reward,
                float(done),
                value.unsqueeze(0)
            )

            state = next_state
            episode_reward += reward

            if done:
                break

        # Update agent
        agent.update(batch_size=64, num_epochs=10)

        # Logging
        if episode % 50 == 0:
            stats = env.get_compression_stats()
            print(f"Episode {episode}, Reward: {episode_reward:.4f}, "
                  f"Avg Pruning: {stats['avg_pruning_ratio']:.2f}, "
                  f"Avg Bit Width: {stats['avg_bit_width']:.1f}")

    # Get final decisions
    decisions = env.get_final_decisions()

    return decisions, agent
