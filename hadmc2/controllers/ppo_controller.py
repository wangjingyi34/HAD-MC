"""PPO Controller for HAD-MC 2.0 MARL framework"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    PPO Policy Network with shared feature extractor and agent-specific heads.

    The policy network outputs action distributions for all five agents:
    - Pruning Agent (discrete)
    - Quantization Agent (discrete)
    - Distillation Agent (continuous)
    - Fusion Agent (discrete)
    - Update Agent (discrete)
    """

    def __init__(self, state_dim: int, action_dims: Dict[str, int], hidden_dim: int = 256):
        super().__init__()

        self.state_dim = state_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim

        # ===== Shared Feature Extractor =====
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ===== Agent-Specific Policy Heads =====

        # Pruning head (discrete action)
        self.pruning_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['pruning']),
        )

        # Quantization head (discrete action)
        self.quantization_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['quantization']),
        )

        # Distillation head (continuous action: temperature and alpha)
        self.distillation_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # temperature, alpha
        )

        self.distillation_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softplus(),  # Ensure positive std
        )

        # Fusion head (discrete action)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['fusion']),
        )

        # Update head (discrete action)
        self.update_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dims['update']),
        )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            dict: Action distributions for all agents
        """
        # Extract shared features
        features = self.feature_extractor(state)

        # Compute action logits/probabilities for each agent
        pruning_logits = self.pruning_head(features)
        quantization_logits = self.quantization_head(features)
        distillation_mean = self.distillation_mean_head(features)
        distillation_std = self.distillation_std_head(features)
        fusion_logits = self.fusion_head(features)
        update_logits = self.update_head(features)

        return {
            'pruning': F.softmax(pruning_logits, dim=-1),
            'quantization': F.softmax(quantization_logits, dim=-1),
            'distillation_mean': distillation_mean,
            'distillation_std': distillation_std,
            'fusion': F.softmax(fusion_logits, dim=-1),
            'update': F.softmax(update_logits, dim=-1),
        }

    def sample_actions(self, state: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Sample actions from policy distributions.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            tuple: (actions dict, total log probability)
        """
        distributions = self.forward(state)

        actions = {}
        log_probs = {}

        # ===== Pruning Action (Discrete) =====
        pruning_dist = torch.distributions.Categorical(distributions['pruning'])
        actions['pruning'] = pruning_dist.sample()
        log_probs['pruning'] = pruning_dist.log_prob(actions['pruning'])

        # ===== Quantization Action (Discrete) =====
        quantization_dist = torch.distributions.Categorical(distributions['quantization'])
        actions['quantization'] = quantization_dist.sample()
        log_probs['quantization'] = quantization_dist.log_prob(actions['quantization'])

        # ===== Distillation Action (Continuous) =====
        distillation_dist = torch.distributions.Normal(
            distributions['distillation_mean'],
            distributions['distillation_std']
        )
        actions['distillation'] = distillation_dist.sample()
        # Log prob of continuous action
        log_probs['distillation'] = distillation_dist.log_prob(actions['distillation']).sum(dim=-1)

        # ===== Fusion Action (Discrete) =====
        fusion_dist = torch.distributions.Categorical(distributions['fusion'])
        actions['fusion'] = fusion_dist.sample()
        log_probs['fusion'] = fusion_dist.log_prob(actions['fusion'])

        # ===== Update Action (Discrete) =====
        update_dist = torch.distributions.Categorical(distributions['update'])
        actions['update'] = update_dist.sample()
        log_probs['update'] = update_dist.log_prob(actions['update'])

        # Total log probability
        total_log_prob = sum(log_probs.values())

        return actions, total_log_prob


class ValueNetwork(nn.Module):
    """
    PPO Value Network for state-value estimation.

    Estimates the expected return (value) from a given state.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            torch.Tensor: State value [batch_size, 1]
        """
        return self.network(state)


class ExperienceBuffer:
    """
    Experience buffer for storing trajectory data.

    Stores state, action, log_prob, reward, done, value tuples
    for PPO training.
    """

    def __init__(self):
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

    @property
    def size(self) -> int:
        """Return the number of stored experiences."""
        return len(self.states)

    def add(self, state, actions, log_prob, reward, done, value):
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
        self.states.append(state)
        for key, action in actions.items():
            self.actions[key].append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.item() if torch.is_tensor(value) else value)

    def get_all(self) -> Tuple:
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
            torch.tensor(self.values, dtype=torch.float32),
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

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.states)


class PPOController:
    """
    PPO (Proximal Policy Optimization) Controller.

    Implements the PPO algorithm with clipping for stable policy updates.
    Coordinates all five agents through a shared policy network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dims: Dict[str, int],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
    ):
        """
        Initialize PPO Controller.

        Args:
            state_dim: Dimension of state space
            action_dims: Action dimensions for each agent
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on
        """
        self.device = device
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Initialize networks
        self.policy_network = PolicyNetwork(state_dim, action_dims).to(device)
        self.value_network = ValueNetwork(state_dim).to(device)

        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=lr
        )

        # Initialize experience buffer
        self.buffer = ExperienceBuffer()

        logger.info(f"PPO Controller initialized on {device}")
        logger.info(f"State dim: {state_dim}, Action dims: {action_dims}")

    def select_actions(self, state: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Select actions using the current policy.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            tuple: (actions dict, log_prob, value)
        """
        with torch.no_grad():
            actions, log_prob = self.policy_network.sample_actions(state)
            value = self.value_network(state)

        return actions, log_prob, value

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward tensor [trajectory_length]
            values: Value tensor [trajectory_length]
            dones: Done flags [trajectory_length]
            next_value: Value of the state after the last trajectory step

        Returns:
            tuple: (advantages, returns)
        """
        advantages = []
        gae = 0

        # Compute advantages in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value if next_value is not None else 0
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        # Compute returns
        returns = advantages + values_tensor

        # Normalize advantages (important for training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, batch_size: int = 64, num_epochs: int = 10) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.

        Args:
            batch_size: Mini-batch size
            num_epochs: Number of PPO epochs per update

        Returns:
            dict: Dictionary with loss information
        """
        if len(self.buffer) < batch_size:
            logger.warning(f"Buffer size ({len(self.buffer)}) < batch_size, skipping update")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0,
            }

        # Get all experiences from buffer
        states, actions, old_log_probs, rewards, dones, values = self.buffer.get_all()

        # Compute GAE
        with torch.no_grad():
            next_value = self.value_network(states[-1:])
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # PPO update with multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for epoch in range(num_epochs):
            # Shuffle indices for stochastic gradient descent
            indices = torch.randperm(len(states))

            # Mini-batch updates
            num_updates = 0
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get new action probabilities and entropy
                batch_actions = {k: v[batch_indices] for k, v in actions.items()}
                distributions = self.policy_network(batch_states)

                # Compute entropy
                entropy = self._compute_entropy(distributions)

                # Compute log probs of ACTUAL actions taken under new policy
                # For discrete agents: compute log prob of taken action
                new_log_probs = 0.0
                for key in ['pruning', 'quantization', 'fusion', 'update']:
                    # Get log probs of taken actions
                    batch_log_probs_new = torch.log_softmax(distributions[key], dim=-1)[
                        torch.arange(len(batch_indices)), batch_actions[key]]
                    new_log_probs += batch_log_probs_new

                # For distillation (continuous), compute log prob of taken action
                distillation_dist = torch.distributions.Normal(
                    distributions['distillation_mean'],
                    distributions['distillation_std']
                )
                distillation_log_prob = distillation_dist.log_prob(batch_actions['distillation']).sum(dim=-1)
                new_log_probs += distillation_log_prob

                # Normalize by number of agents
                new_log_probs = new_log_probs / 5

                # Simplified PPO update using the computed log_probs
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Compute PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                new_values = self.value_network(batch_states).squeeze()
                value_loss = F.mse_loss(new_values, batch_returns)

                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Update policy network
                self.policy_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(),
                    self.max_grad_norm
                )
                self.policy_optimizer.step()

                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.value_network.parameters(),
                    self.max_grad_norm
                )
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Average over all updates
        if num_updates > 0:
            total_policy_loss /= num_updates
            total_value_loss /= num_updates
            total_entropy /= num_updates

        # Clear buffer
        self.buffer.clear()

        logger.info(f"PPO Update: policy_loss={total_policy_loss:.4f}, "
                    f"value_loss={total_value_loss:.4f}, "
                    f"entropy={total_entropy:.4f}")

        return {
            'policy_loss': total_policy_loss,
            'value_loss': total_value_loss,
            'entropy': total_entropy,
            'total_loss': total_policy_loss + self.value_coef * total_value_loss - self.entropy_coef * total_entropy,
        }

    def _compute_entropy(self, distributions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute policy entropy for exploration bonus.

        Args:
            distributions: Action distributions for all agents

        Returns:
            torch.Tensor: Total entropy
        """
        entropy = 0

        # Discrete distributions
        for key in ['pruning', 'quantization', 'fusion', 'update']:
            dist = torch.distributions.Categorical(distributions[key])
            entropy += dist.entropy().mean()

        # Continuous distribution (distillation)
        dist_dist = torch.distributions.Normal(
            distributions['distillation_mean'],
            distributions['distillation_std']
        )
        entropy += dist_dist.entropy().sum(dim=-1).mean()

        return entropy

    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, path)
        logger.info(f"Saved PPO controller to {path}")

    def load(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        logger.info(f"Loaded PPO controller from {path}")
