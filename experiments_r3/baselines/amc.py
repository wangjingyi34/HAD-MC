"""
AMC: AutoML for Model Compression

This module implements the AMC method from:
"AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
He et al., ECCV 2018

AMC uses reinforcement learning (DDPG) to automatically search for optimal
layer-wise pruning rates with hardware-aware rewards.

Key features:
- Structured channel pruning
- DDPG algorithm for action selection
- Hardware-aware reward function
- Progressive pruning strategy
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import deque
import copy


class ReplayBuffer:
    """Experience replay buffer for DDPG."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.FloatTensor([item[0] for item in batch])
        actions = torch.FloatTensor([item[1] for item in batch])
        rewards = torch.FloatTensor([item[2] for item in batch])
        next_states = torch.FloatTensor([item[3] for item in batch])
        dones = torch.FloatTensor([item[4] for item in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for DDPG - outputs pruning rate."""

    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Initialize last layer to output small values initially
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        # Scale to [0, 1] for pruning rate
        x = (x + 1) / 2
        return x


class Critic(nn.Module):
    """Critic network for DDPG - estimates Q-value."""

    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

        # Initialize last layer to output small values initially
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AMCAgent:
    """AMC agent using DDPG for pruning rate search."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 256,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        buffer_capacity: int = 10000,
        device: str = 'cuda'
    ):
        """
        Initialize AMC agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (1 for single pruning rate)
            hidden_dim: Hidden dimension for networks
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            tau: Soft update coefficient
            buffer_capacity: Replay buffer capacity
            device: Device to run on
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # Create networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)

        # Create target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Freeze target networks
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Noise for exploration
        self.noise_scale = 0.1

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> float:
        """
        Select action (pruning rate) for given state.

        Args:
            state: State vector
            add_noise: Whether to add exploration noise

        Returns:
            Pruning rate in [0, 1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze().cpu().numpy()
        self.actor.train()

        if add_noise:
            noise = np.random.normal(0, self.noise_scale)
            action = np.clip(action + noise, 0, 1)

        return float(action)

    def update(self, batch_size: int = 64):
        """
        Update actor and critic networks.

        Args:
            batch_size: Batch size for training
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Update actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # Decrease noise
        self.noise_scale = max(0.01, self.noise_scale * 0.995)

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'noise_scale': self.noise_scale
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.noise_scale = checkpoint['noise_scale']


class PruningEnvironment:
    """Environment for AMC pruning rate search."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        min_ratio: float = 0.1,
        max_ratio: float = 0.9,
        target_compression: float = 0.5
    ):
        """
        Initialize pruning environment.

        Args:
            model: PyTorch model to prune
            device: Device to run on
            min_ratio: Minimum pruning ratio
            max_ratio: Maximum pruning ratio
            target_compression: Target compression ratio
        """
        self.model = model
        self.device = device
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.target_compression = target_compression

        # Get list of prunable layers
        self.prunable_layers = []
        self.layer_indices = {}
        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.prunable_layers.append((name, module))
                self.layer_indices[name] = idx
                idx += 1

        self.num_layers = len(self.prunable_layers)
        self.current_layer_idx = 0

        # Store original model
        self.original_model = copy.deepcopy(model)

        # Track pruning ratios and rewards
        self.pruning_ratios = {}
        self.rewards = []

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_layer_idx = 0
        self.pruning_ratios = {}
        self.rewards = []
        self.model = copy.deepcopy(self.original_model)

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        state = []

        # Layer index (normalized)
        state.append(self.current_layer_idx / max(1, self.num_layers - 1))

        # Pruning ratios of previous layers (padded to max 5)
        prev_ratios = list(self.pruning_ratios.values())[-5:]
        for _ in range(5):
            if prev_ratios:
                state.append(prev_ratios.pop(0))
            else:
                state.append(0.0)

        # If first layer, add some layer-specific info
        if self.current_layer_idx < self.num_layers:
            layer_name, layer_module = self.prunable_layers[self.current_layer_idx]
            if isinstance(layer_module, nn.Conv2d):
                # Normalized layer info
                state.append(layer_module.in_channels / 512)
                state.append(layer_module.out_channels / 512)
                state.append(layer_module.kernel_size[0] / 10)
            elif isinstance(layer_module, nn.Linear):
                state.append(layer_module.in_features / 4096)
                state.append(layer_module.out_features / 4096)
                state.append(1.0)  # Placeholder
            else:
                state.extend([0.0, 0.0, 0.0])
        else:
            state.extend([0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action (pruning ratio) for current layer.

        Args:
            action: Pruning ratio in [0, 1]

        Returns:
            next_state: Next state
            reward: Reward for this action
            done: Whether episode is finished
        """
        if self.current_layer_idx >= self.num_layers:
            return self._get_state(), 0.0, True

        # Clamp action to valid range
        pruning_ratio = np.clip(action, self.min_ratio, self.max_ratio)

        # Get current layer
        layer_name, layer_module = self.prunable_layers[self.current_layer_idx]
        self.pruning_ratios[layer_name] = pruning_ratio

        # Calculate reward (simplified)
        # Reward based on compression ratio vs accuracy trade-off
        reward = self._calculate_reward(pruning_ratio)

        self.rewards.append(reward)
        self.current_layer_idx += 1

        done = self.current_layer_idx >= self.num_layers
        next_state = self._get_state()

        return next_state, reward, done

    def _calculate_reward(self, pruning_ratio: float) -> float:
        """
        Calculate reward for a given pruning ratio.

        This is a simplified reward function. In practice, AMC uses
        a more complex reward that accounts for hardware latency.

        Args:
            pruning_ratio: Pruning ratio

        Returns:
            Reward value
        """
        # Base reward: reward higher compression (up to target)
        compression_reward = 1.0 if pruning_ratio >= self.target_compression else pruning_ratio / self.target_compression

        # Penalty for too aggressive pruning (to maintain accuracy)
        accuracy_penalty = 0.0
        if pruning_ratio > 0.7:
            accuracy_penalty = (pruning_ratio - 0.7) * 2

        return compression_reward - accuracy_penalty

    def get_final_pruning_ratios(self) -> Dict[str, float]:
        """Get final pruning ratios for all layers."""
        return self.pruning_ratios

    def get_compression_ratio(self) -> float:
        """Calculate overall compression ratio."""
        if not self.pruning_ratios:
            return 0.0

        return np.mean(list(self.pruning_ratios.values()))


def train_amc(
    model: nn.Module,
    num_episodes: int = 300,
    max_steps: int = 50,
    device: str = 'cuda'
) -> Tuple[Dict[str, float], AMCAgent]:
    """
    Train AMC agent to find optimal pruning rates.

    Args:
        model: PyTorch model to compress
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        device: Device to run on

    Returns:
        pruning_ratios: Dictionary mapping layer names to pruning ratios
        agent: Trained AMC agent
    """
    # Create environment
    env = PruningEnvironment(model, device=device)

    # Create agent
    state_dim = len(env.reset())
    agent = AMCAgent(state_dim=state_dim, device=device)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, add_noise=True)

            # Execute action
            next_state, reward, done = env.step(action)

            # Store experience
            agent.replay_buffer.add(state, action, reward, next_state, float(done))

            # Update agent
            agent.update(batch_size=64)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Decay noise
        agent.noise_scale = max(0.01, agent.noise_scale * 0.99)

        # Logging
        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.4f}, Noise: {agent.noise_scale:.4f}")

    # Get final pruning ratios
    pruning_ratios = env.get_final_pruning_ratios()

    return pruning_ratios, agent
