"""HAD-MC 2.0 Trainer"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import copy
import logging
import os

logger = logging.getLogger(__name__)


class HADMCTrainer:
    """
    HAD-MC 2.0 Training System.

    Coordinates all five compression agents and the PPO controller
    for multi-objective model compression optimization.

    The training loop:
    1. PPO selects actions for all agents
    2. Agents apply compression operations
    3. Compressed model is evaluated
    4. Reward is computed
    5. PPO is updated
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_model: Optional[nn.Module],
        train_loader,
        val_loader,
        hal,
        config: Dict,
    ):
        """
        Initialize HAD-MC 2.0 Trainer.

        Args:
            model: Original model to compress
            teacher_model: Teacher model for distillation (optional)
            train_loader: Training data loader
            val_loader: Validation data loader
            hal: Hardware Abstraction Layer
            config: Training configuration
        """
        self.model = model
        self.teacher_model = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hal = hal
        self.config = config

        # Training parameters
        self.num_episodes = config.get('num_episodes', 100)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 50)
        self.update_interval = config.get('update_interval', 10)
        self.device = config.get('device', 'cpu')

        # Initialize components
        self._init_components()

        # Training statistics
        self.history = {
            'rewards': [],
            'accuracies': [],
            'latencies': [],
            'energies': [],
            'sizes': [],
        }
        self.best_reward = float('-inf')
        self.best_config = None

        logger.info(f"HADMCTrainer initialized: {self.num_episodes} episodes, "
                   f"{self.max_steps_per_episode} steps/episode")

    def _init_components(self):
        """Initialize all training components."""
        from ..controllers.ppo_controller import PPOController
        from ..agents.pruning_agent import PruningAgent
        from ..agents.quantization_agent import QuantizationAgent
        from ..agents.distillation_agent import DistillationAgent
        from ..agents.fusion_agent import FusionAgent
        from ..agents.update_agent import UpdateAgent
        from ..rewards.reward_function import RewardFunction
        from ..utils.state import State, create_state_from_model_and_hardware
        from ..utils.action import ActionSpace

        # State dimension
        state = create_state_from_model_and_hardware(
            self.model, self.hal.get_hardware_config()
        )
        state_dim = state.to_tensor().shape[0]

        # Action space
        action_space = ActionSpace(
            num_layers=len([m for m in self.model.modules()
                          if isinstance(m, (nn.Conv2d, nn.Linear))])
        )
        action_dims = action_space.get_all_action_dims()

        # PPO Controller
        self.ppo = PPOController(
            state_dim=state_dim,
            action_dims=action_dims,
            lr=self.config.get('lr', 3e-4),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
            clip_epsilon=self.config.get('clip_epsilon', 0.2),
            value_coef=self.config.get('value_coef', 0.5),
            entropy_coef=self.config.get('entropy_coef', 0.01),
            max_grad_norm=self.config.get('max_grad_norm', 0.5),
            device=self.device,
        )

        # Five Agents
        self.pruning_agent = PruningAgent(self.model, self.train_loader, self.device)
        self.quantization_agent = QuantizationAgent(self.model, self.val_loader, self.device)

        if self.teacher_model is not None:
            self.distillation_agent = DistillationAgent(
                self.teacher_model, copy.deepcopy(self.model), self.device
            )
        else:
            self.distillation_agent = None

        self.fusion_agent = FusionAgent(self.model, self.device)
        self.update_agent = UpdateAgent(self.model, self.device)

        # Reward Function
        self.reward_fn = RewardFunction(
            accuracy_weight=self.config.get('accuracy_weight', 1.0),
            latency_weight=self.config.get('latency_weight', 0.5),
            energy_weight=self.config.get('energy_weight', 0.3),
            size_weight=self.config.get('size_weight', 0.2),
            accuracy_threshold=self.config.get('accuracy_threshold', 0.90),
            latency_threshold=self.config.get('latency_threshold', 10.0),
            energy_threshold=self.config.get('energy_threshold', 1.0),
            size_threshold=self.config.get('size_threshold', 5.0),
        )

        # Store for reference
        self.state_dim = state_dim
        self.action_dims = action_dims

    def train(self, save_dir: Optional[str] = None):
        """
        Run the full training loop.

        Args:
            save_dir: Directory to save checkpoints (optional)

        Returns:
            dict: Training results
        """
        logger.info(f"Starting HAD-MC 2.0 training for {self.num_episodes} episodes")

        # Evaluate baseline
        baseline_metrics = self._evaluate_model(self.model)
        logger.info(f"Baseline - Acc: {baseline_metrics['accuracy']:.4f}, "
                   f"Lat: {baseline_metrics['latency']:.2f}ms, "
                   f"Size: {baseline_metrics['size']:.2f}MB")

        self.baseline_metrics = baseline_metrics

        # Create save directory
        if save_dir is None:
            save_dir = './checkpoints'
        os.makedirs(save_dir, exist_ok=True)

        # Training loop
        for episode in range(self.num_episodes):
            episode_reward = self._train_episode(episode, save_dir)

            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = sum(self.history['rewards'][-10:]) / min(10, len(self.history['rewards']))
                avg_acc = sum(self.history['accuracies'][-10:]) / min(10, len(self.history['accuracies']))
                logger.info(f"Episode {episode + 1}/{self.num_episodes}: "
                           f"Avg Reward (last 10)={avg_reward:.4f}, "
                           f"Avg Acc (last 10)={avg_acc:.4f}")

        logger.info(f"Training complete. Best reward: {self.best_reward:.4f}")

        return {
            'best_reward': self.best_reward,
            'best_config': self.best_config,
            'history': self.history,
            'num_episodes': self.num_episodes,
        }

    def _train_episode(self, episode: int, save_dir: str) -> float:
        """
        Train for a single episode.

        Args:
            episode: Episode number
            save_dir: Directory to save checkpoints

        Returns:
            float: Total episode reward
        """
        from ..utils.state import create_state_from_model_and_hardware

        # Start with original model
        current_model = copy.deepcopy(self.model)
        current_model.to(self.device)

        # Get initial state
        state = self._get_state(current_model)
        episode_reward = 0

        for step in range(self.max_steps_per_episode):
            # Select actions using PPO
            actions, log_prob, value = self.ppo.select_actions(state)

            # Apply actions to model
            compressed_model = self._apply_actions(current_model, actions)

            # Evaluate compressed model
            metrics = self._evaluate_model(compressed_model)

            # Compute reward
            reward = self.reward_fn.compute(
                accuracy=metrics['accuracy'],
                latency=metrics['latency'],
                energy=metrics['energy'],
                size=metrics['size'],
                baseline_accuracy=self.baseline_metrics['accuracy'],
                baseline_latency=self.baseline_metrics['latency'],
                baseline_energy=self.baseline_metrics['energy'],
                baseline_size=self.baseline_metrics['size'],
            )

            # Get next state
            next_state = self._get_state(compressed_model)
            done = step == self.max_steps_per_episode - 1

            # Store experience in PPO buffer
            self.ppo.buffer.add(state, actions, log_prob, reward, done, value)

            # Update for next step
            current_model = compressed_model
            state = next_state
            episode_reward += reward

            # Update Pareto front
            self.reward_fn.update_pareto_front(
                accuracy=metrics['accuracy'],
                latency=metrics['latency'],
                energy=metrics['energy'],
                size=metrics['size'],
            )

            if done:
                break

        # Record history
        self.history['rewards'].append(episode_reward)
        self.history['accuracies'].append(metrics['accuracy'])
        self.history['latencies'].append(metrics['latency'])
        self.history['energies'].append(metrics['energy'])
        self.history['sizes'].append(metrics['size'])

        # Save best model
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_config = actions
            self._save_checkpoint(compressed_model, actions, episode, save_dir)

        # Update PPO periodically
        if (episode + 1) % self.update_interval == 0:
            update_info = self.ppo.update()
            logger.info(f"PPO Update: policy_loss={update_info['policy_loss']:.4f}, "
                       f"value_loss={update_info['value_loss']:.4f}")

        return episode_reward

    def _get_state(self, model: nn.Module) -> torch.Tensor:
        """
        Get state from model.

        Args:
            model: PyTorch model

        Returns:
            torch.Tensor: State tensor
        """
        from ..utils.state import create_state_from_model_and_hardware

        state = create_state_from_model_and_hardware(
            model, self.hal.get_hardware_config()
        )

        return state.to_tensor().to(self.device)

    def _apply_actions(self, model: nn.Module, actions: Dict) -> nn.Module:
        """
        Apply all agent actions to model.

        Args:
            model: PyTorch model
            actions: Actions dict from PPO

        Returns:
            nn.Module: Compressed model
        """
        from ..utils.action import ActionSpace

        action_space = ActionSpace(
            num_layers=len([m for m in model.modules()
                          if isinstance(m, (nn.Conv2d, nn.Linear))])
        )

        # Decode and apply actions
        pruning_config = action_space.decode_action('pruning', actions['pruning'])
        quantization_config = action_space.decode_action('quantization', actions['quantization'])
        distillation_config = action_space.decode_action('distillation', actions['distillation'])
        fusion_config = action_space.decode_action('fusion', actions['fusion'])
        # Update agent action is handled separately

        # Apply pruning
        if pruning_config.get('pruning_ratio', 0) > 0:
            model = self.pruning_agent.apply_action(model, pruning_config)

        # Apply quantization
        if quantization_config.get('bit_width', 32) < 32:
            model = self.quantization_agent.apply_action(model, quantization_config)

        # Apply fusion
        if fusion_config.get('pattern', 'none') != 'none':
            model = self.fusion_agent.apply_action(model, fusion_config)

        # Note: Distillation is applied during evaluation
        self.current_distillation_config = distillation_config

        return model

    def _evaluate_model(self, model: nn.Module) -> Dict:
        """
        Evaluate model metrics.

        Args:
            model: PyTorch model

        Returns:
            dict: Model metrics
        """
        from ..utils.metrics import MetricsCalculator

        calculator = MetricsCalculator(device=self.device)

        # Get input shape from dataloader
        sample_input, _ = next(iter(self.val_loader))
        input_shape = sample_input.shape

        # Calculate metrics
        accuracy = calculator.calculate_accuracy(model, self.val_loader)
        latency = calculator.calculate_latency(model, input_shape)
        size = calculator.calculate_model_size(model)

        # Simplified energy estimation
        energy = latency * 0.001  # J

        return {
            'accuracy': accuracy,
            'latency': latency,
            'energy': energy,
            'size': size,
        }

    def _save_checkpoint(
        self,
        model: nn.Module,
        actions: Dict,
        episode: int,
        save_dir: str
    ):
        """
        Save training checkpoint.

        Args:
            model: Model to save
            actions: Actions that led to this model
            episode: Episode number
            save_dir: Save directory
        """
        checkpoint = {
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'actions': actions,
            'reward': self.best_reward,
            'config': self.config,
        }

        save_path = os.path.join(save_dir, f'best_episode_{episode}.pt')
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> nn.Module:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            nn.Module: Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model from checkpoint
        loaded_model = copy.deepcopy(self.model)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Episode: {checkpoint['episode']}, Reward: {checkpoint['reward']:.4f}")

        return loaded_model

    def get_training_statistics(self) -> Dict:
        """
        Get training statistics.

        Returns:
            dict: Training statistics
        """
        return {
            'num_episodes': len(self.history['rewards']),
            'best_reward': self.best_reward,
            'mean_reward': sum(self.history['rewards']) / len(self.history['rewards']),
            'std_reward': (sum((r - sum(self.history['rewards']) / len(self.history['rewards'])) ** 2
                           for r in self.history['rewards']) / len(self.history['rewards'])) ** 0.5,
            'mean_accuracy': sum(self.history['accuracies']) / len(self.history['accuracies']),
            'mean_latency': sum(self.history['latencies']) / len(self.history['latencies']),
            'mean_size': sum(self.history['sizes']) / len(self.history['sizes']),
        }

    def __repr__(self) -> str:
        return (f"HADMCTrainer(episodes={self.num_episodes}, "
                f"steps_per_episode={self.max_steps_per_episode}, "
                f"device={self.device})")
