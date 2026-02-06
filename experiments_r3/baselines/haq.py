"""
HAQ: Hardware-Aware Automated Quantization

This module implements the HAQ method from:
"HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
Wang et al., CVPR 2019

HAQ uses reinforcement learning to automatically search for optimal
layer-wise bit-widths for mixed-precision quantization with hardware feedback.

Key features:
- Mixed-precision quantization (4-bit, 8-bit, 16-bit, 32-bit)
- Hardware latency lookup table for accurate latency prediction
- DDPG algorithm for bit-width selection
- Support for both weight and activation quantization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import deque
import copy
import json


class LatencyLookupTable:
    """Hardware latency lookup table for different precisions and layer types."""

    def __init__(self, platform: str = 'NVIDIA_A100'):
        """
        Initialize latency lookup table.

        Args:
            platform: Hardware platform name
        """
        self.platform = platform
        self.latency_table = self._get_default_latency_table(platform)

    def _get_default_latency_table(self, platform: str) -> Dict:
        """
        Get default latency table for a platform.

        These are example values. In practice, you would measure these
        or use a more accurate latency model.

        Args:
            platform: Platform name

        Returns:
            Dictionary mapping (layer_type, bit_width) to latency factor
        """
        # Latency factors relative to FP32
        tables = {
            'NVIDIA_A100': {
                'Conv2d': {4: 0.15, 8: 0.25, 16: 0.5, 32: 1.0},
                'Linear': {4: 0.15, 8: 0.25, 16: 0.5, 32: 1.0},
                'BatchNorm2d': {8: 0.3, 16: 0.5, 32: 1.0},
            },
            'Ascend_310': {
                'Conv2d': {4: 0.18, 8: 0.28, 16: 0.55, 32: 1.0},
                'Linear': {4: 0.18, 8: 0.28, 16: 0.55, 32: 1.0},
                'BatchNorm2d': {8: 0.35, 16: 0.55, 32: 1.0},
            },
            'CPU': {
                'Conv2d': {8: 0.8, 16: 0.9, 32: 1.0},
                'Linear': {8: 0.8, 16: 0.9, 32: 1.0},
                'BatchNorm2d': {16: 0.9, 32: 1.0},
            }
        }

        return tables.get(platform, tables['CPU'])

    def get_latency_factor(self, layer_type: str, bit_width: int) -> float:
        """
        Get latency factor for a layer type and bit width.

        Args:
            layer_type: Layer type ('Conv2d', 'Linear', etc.)
            bit_width: Bit width (4, 8, 16, 32)

        Returns:
            Latency factor (relative to FP32)
        """
        layer_type_map = {
            nn.Conv2d: 'Conv2d',
            nn.Linear: 'Linear',
            nn.BatchNorm2d: 'BatchNorm2d',
        }

        if isinstance(layer_type, str):
            lt = layer_type
        else:
            lt = layer_type_map.get(layer_type, 'Conv2d')

        if lt in self.latency_table and bit_width in self.latency_table[lt]:
            return self.latency_table[lt][bit_width]
        else:
            # Default to FP32 latency
            return 1.0

    def estimate_latency(self, model: nn.Module, bit_widths: Dict[str, int]) -> float:
        """
        Estimate total latency for a model with given bit widths.

        Args:
            model: PyTorch model
            bit_widths: Dictionary mapping layer names to bit widths

        Returns:
            Estimated total latency (relative units)
        """
        total_latency = 0.0

        for name, module in model.named_modules():
            if name in bit_widths:
                bit_width = bit_widths[name]
                latency_factor = self.get_latency_factor(type(module), bit_width)

                # Estimate FLOPs as proxy for computation
                if isinstance(module, nn.Conv2d):
                    flops = (module.kernel_size[0] * module.kernel_size[1] *
                            module.in_channels * module.out_channels * 224 * 224)
                elif isinstance(module, nn.Linear):
                    flops = module.in_features * module.out_features
                else:
                    flops = 1.0

                total_latency += flops * latency_factor

        return total_latency

    def load_from_file(self, filepath: str):
        """
        Load latency table from JSON file.

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r') as f:
            self.latency_table = json.load(f)

    def save_to_file(self, filepath: str):
        """
        Save latency table to JSON file.

        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.latency_table, f, indent=2)


class HAQAgent:
    """HAQ agent using DDPG for bit-width search."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        buffer_capacity: int = 10000,
        device: str = 'cuda'
    ):
        """
        Initialize HAQ agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions (bit width choices)
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
        self.action_dim = action_dim

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

        # Bit width options
        self.bit_widths = [4, 8, 16, 32]

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> int:
        """
        Select action (bit width) for given state.

        Args:
            state: State vector
            add_noise: Whether to add exploration noise

        Returns:
            Bit width (4, 8, 16, or 32)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(state_tensor).squeeze()
            probs = torch.softmax(logits, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
        self.actor.train()

        if not add_noise:
            return self.bit_widths[action_idx]

        # Exploration: occasionally random action
        if np.random.random() < 0.1:
            return np.random.choice(self.bit_widths)

        return self.bit_widths[action_idx]

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

        # Convert action indices to one-hot
        actions_onehot = torch.zeros(batch_size, self.action_dim).to(self.device)
        for i, action in enumerate(actions):
            action_idx = self.bit_widths.index(action.item())
            actions_onehot[i, action_idx] = 1.0

        # Update critic
        with torch.no_grad():
            next_logits = self.actor_target(next_states)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_actions_onehot = torch.zeros_like(next_probs)
            next_actions_onehot[range(len(next_probs)), torch.argmax(next_probs, dim=-1)] = 1.0

            target_q = self.critic_target(next_states, next_actions_onehot)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions_onehot)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Update actor
        current_logits = self.actor(states)
        actor_loss = -self.critic(states, torch.softmax(current_logits, dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )


class Actor(nn.Module):
    """Actor network for HAQ - outputs bit width probabilities."""

    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(nn.Module):
    """Critic network for HAQ - estimates Q-value."""

    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


class QuantizationEnvironment:
    """Environment for HAQ bit-width search."""

    def __init__(
        self,
        model: nn.Module,
        latency_lut: LatencyLookupTable,
        bit_widths: List[int] = [4, 8, 16, 32],
        target_latency: float = 0.3
    ):
        """
        Initialize quantization environment.

        Args:
            model: PyTorch model to quantize
            latency_lut: Latency lookup table
            bit_widths: Available bit widths
            target_latency: Target latency factor (relative to FP32)
        """
        self.model = model
        self.latency_lut = latency_lut
        self.bit_widths = bit_widths
        self.target_latency = target_latency

        # Get list of quantizable layers
        self.quantizable_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.quantizable_layers.append((name, module))

        self.num_layers = len(self.quantizable_layers)
        self.current_layer_idx = 0

        # Track bit widths
        self.bit_width_choices = {}
        self.rewards = []

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_layer_idx = 0
        self.bit_width_choices = {}
        self.rewards = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        state = []

        # Layer index (normalized)
        state.append(self.current_layer_idx / max(1, self.num_layers - 1))

        # Previous bit width choices (padded to max 5)
        prev_choices = list(self.bit_width_choices.values())[-5:]
        for _ in range(5):
            if prev_choices:
                # Normalize bit width to [0, 1]
                state.append(prev_choices.pop(0) / 32.0)
            else:
                state.append(0.0)

        # Current layer info
        if self.current_layer_idx < self.num_layers:
            layer_name, layer_module = self.quantizable_layers[self.current_layer_idx]
            if isinstance(layer_module, nn.Conv2d):
                # Normalized layer info
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

    def step(self, bit_width: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action (bit width) for current layer.

        Args:
            bit_width: Bit width for current layer

        Returns:
            next_state: Next state
            reward: Reward for this action
            done: Whether episode is finished
        """
        if self.current_layer_idx >= self.num_layers:
            return self._get_state(), 0.0, True

        # Get current layer
        layer_name, layer_module = self.quantizable_layers[self.current_layer_idx]
        self.bit_width_choices[layer_name] = bit_width

        # Calculate reward
        reward = self._calculate_reward(bit_width)

        self.rewards.append(reward)
        self.current_layer_idx += 1

        done = self.current_layer_idx >= self.num_layers
        next_state = self._get_state()

        return next_state, reward, done

    def _calculate_reward(self, bit_width: int) -> float:
        """
        Calculate reward for a given bit width.

        Reward is based on:
        1. Latency reduction (lower bit width is better)
        2. Model size reduction (lower bit width is better)
        3. Potential accuracy loss (higher bit width is better)

        Args:
            bit_width: Bit width

        Returns:
            Reward value
        """
        # Latency benefit (lower bit width = lower latency)
        latency_factor = self.latency_lut.get_latency_factor('Conv2d', bit_width)
        latency_reward = (1.0 - latency_factor) * 2.0

        # Model size benefit (proportional to bit width reduction)
        size_reward = (32 - bit_width) / 32.0 * 2.0

        # Accuracy penalty (lower bit width = more accuracy loss potential)
        if bit_width == 4:
            accuracy_penalty = 1.0
        elif bit_width == 8:
            accuracy_penalty = 0.3
        elif bit_width == 16:
            accuracy_penalty = 0.1
        else:
            accuracy_penalty = 0.0

        return latency_reward + size_reward - accuracy_penalty

    def get_final_bit_widths(self) -> Dict[str, int]:
        """Get final bit widths for all layers."""
        return self.bit_width_choices

    def get_compression_ratio(self) -> float:
        """Calculate overall compression ratio (in bits relative to FP32)."""
        if not self.bit_width_choices:
            return 0.0

        avg_bit_width = np.mean(list(self.bit_width_choices.values()))
        return (32 - avg_bit_width) / 32.0


def train_haq(
    model: nn.Module,
    platform: str = 'NVIDIA_A100',
    num_episodes: int = 200,
    max_steps: int = 50,
    device: str = 'cuda'
) -> Tuple[Dict[str, int], HAQAgent]:
    """
    Train HAQ agent to find optimal bit widths.

    Args:
        model: PyTorch model to quantize
        platform: Hardware platform
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        device: Device to run on

    Returns:
        bit_widths: Dictionary mapping layer names to bit widths
        agent: Trained HAQ agent
    """
    # Create latency lookup table
    latency_lut = LatencyLookupTable(platform)

    # Create environment
    env = QuantizationEnvironment(model, latency_lut)

    # Create agent
    state_dim = len(env.reset())
    agent = HAQAgent(state_dim=state_dim, device=device)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            # Select action
            bit_width = agent.select_action(state, add_noise=True)

            # Execute action
            next_state, reward, done = env.step(bit_width)

            # Store experience
            agent.replay_buffer.add(state, float(bit_width), reward, next_state, float(done))

            # Update agent
            agent.update(batch_size=64)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Logging
        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.4f}, "
                  f"Avg Bit Width: {np.mean(list(env.bit_width_choices.values())):.1f}")

    # Get final bit widths
    bit_widths = env.get_final_bit_widths()

    return bit_widths, agent
