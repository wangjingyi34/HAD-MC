"""Rollout Buffer for HAD-MC 2.0"""

import torch
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    Rollout buffer for storing trajectories in PPO.

    Stores state, action, log_prob, reward, done, and value tuples
    for PPO training with Generalized Advantage Estimation (GAE).
    """

    def __init__(self, capacity: int = 2048):
        """
        Initialize Rollout Buffer.

        Args:
            capacity: Maximum capacity of the buffer
        """
        self.capacity = capacity

        # Storage for each component
        self.states = []          # State tensors
        self.actions = {
            'pruning': [],
            'quantization': [],
            'distillation': [],
            'fusion': [],
            'update': [],
        }
        self.log_probs = []       # Log probabilities
        self.rewards = []         # Rewards
        self.dones = []           # Episode termination flags
        self.values = []          # State values

        self.size = 0

        logger.info(f"RolloutBuffer initialized with capacity={capacity}")

    def add(
        self,
        state: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor
    ):
        """
        Add a single transition to the buffer.

        Args:
            state: State tensor
            actions: Actions dict for all agents
            log_prob: Total log probability
            reward: Reward value
            done: Whether episode is done
            value: State value
        """
        if self.size >= self.capacity:
            logger.warning(f"Buffer full ({self.size}/{self.capacity}), dropping oldest transition")
            self._pop_oldest()

        self.states.append(state)
        for key, action in actions.items():
            self.actions[key].append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

        self.size += 1

    def _pop_oldest(self):
        """Remove the oldest transition."""
        self.states.pop(0)
        for key in self.actions:
            self.actions[key].pop(0)
        self.log_probs.pop(0)
        self.rewards.pop(0)
        self.dones.pop(0)
        self.values.pop(0)
        self.size -= 1

    def get_all(self) -> tuple:
        """
        Get all stored experiences.

        Returns:
            tuple: (states, actions, log_probs, rewards, dones, values)
        """
        return (
            torch.stack(self.states),
            {k: torch.stack(v) for k, v in self.actions.items()},
            torch.stack(self.log_probs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32),
            torch.stack(self.values),
        )

    def get_batch(self, batch_size: int) -> Optional[tuple]:
        """
        Get a random batch of experiences.

        Args:
            batch_size: Size of the batch

        Returns:
            Optional[tuple]: Batch data or None if buffer is too small
        """
        if self.size < batch_size:
            return None

        # Random sampling
        indices = torch.randperm(self.size)[:batch_size]

        batch_states = torch.stack([self.states[i] for i in indices])
        batch_actions = {
            k: torch.stack([self.actions[k][i] for i in indices])
            for k in self.actions
        }
        batch_log_probs = torch.stack([self.log_probs[i] for i in indices])
        batch_rewards = torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32)
        batch_dones = torch.tensor([self.dones[i] for i in indices], dtype=torch.float32)
        batch_values = torch.stack([self.values[i] for i in indices])

        return (
            batch_states,
            batch_actions,
            batch_log_probs,
            batch_rewards,
            batch_dones,
            batch_values,
        )

    def clear(self):
        """Clear all stored experiences."""
        self.states = []
        self.actions = {
            'pruning': [],
            'quantization': [],
            'distillation': [],
            'fusion': [],
            'update': [],
        }
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.size = 0

        logger.info("RolloutBuffer cleared")

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def get_statistics(self) -> Dict:
        """
        Get statistics about the buffer.

        Returns:
            dict: Buffer statistics
        """
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'fill_ratio': 0.0,
            }

        return {
            'size': self.size,
            'capacity': self.capacity,
            'fill_ratio': self.size / self.capacity,
            'mean_reward': sum(self.rewards) / self.size,
            'std_reward': (sum((r - sum(self.rewards)/self.size)**2 for r in self.rewards) / self.size) ** 0.5,
            'mean_value': torch.stack(self.values).mean().item(),
            'num_dones': sum(self.dones),
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"RolloutBuffer(size={stats['size']}/{stats['capacity']}, "
                f"fill_ratio={stats['fill_ratio']:.2%}, "
                f"mean_reward={stats['mean_reward']:.4f})")
