"""MARL Coordinator for HAD-MC 2.0"""

import torch
import torch.nn as nn
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MARLCoordinator:
    """
    Multi-Agent Reinforcement Learning Coordinator.

    Coordinates the five compression agents (pruning, quantization, distillation,
    fusion, update) and the PPO controller for training.
    """

    def __init__(
        self,
        ppo_controller,
        agents: Dict,
        hardware_abstraction_layer,
        reward_function,
    ):
        """
        Initialize MARL Coordinator.

        Args:
            ppo_controller: PPOController instance
            agents: Dictionary of agent instances
            hardware_abstraction_layer: HardwareAbstractionLayer instance
            reward_function: RewardFunction instance
        """
        self.ppo = ppo_controller
        self.agents = agents
        self.hal = hardware_abstraction_layer
        self.reward_fn = reward_function

        # Training statistics
        self.episode_rewards = []
        self.best_reward = float('-inf')

    def get_state(self, model: nn.Module) -> torch.Tensor:
        """
        Get current state from model and hardware.

        Args:
            model: PyTorch model

        Returns:
            torch.Tensor: State tensor
        """
        from ..utils.state import create_state_from_model_and_hardware

        hardware_config = self.hal.get_hardware_config()
        state = create_state_from_model_and_hardware(model, hardware_config)
        return state.to_tensor()

    def step(
        self,
        model: nn.Module,
        state: torch.Tensor,
        dataloader
    ) -> tuple:
        """
        Execute one step of training.

        Args:
            model: Current model
            state: Current state
            dataloader: Training/validation data loader

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Select actions using PPO
        actions, log_prob, value = self.ppo.select_actions(state)

        # Apply actions to model through agents
        # This is a simplified version - actual implementation would decode
        # actions and apply them properly
        compressed_model = self._apply_actions(model, actions)

        # Evaluate the compressed model
        from ..utils.metrics import MetricsCalculator
        calculator = MetricsCalculator(device=self.ppo.device)

        # Get input shape from dataloader
        sample_input, _ = next(iter(dataloader))
        input_shape = sample_input.shape

        # Calculate metrics
        accuracy = calculator.calculate_accuracy(compressed_model, dataloader)
        latency = calculator.calculate_latency(compressed_model, input_shape)
        model_size = calculator.calculate_model_size(compressed_model)

        # Calculate reward
        energy = latency * 0.1  # Simplified energy estimation
        reward = self.reward_fn.compute(
            accuracy=accuracy,
            latency=latency,
            energy=energy,
            size=model_size,
            baseline_accuracy=1.0,  # Would come from initial evaluation
            baseline_latency=latency * 2,  # Would come from initial evaluation
            baseline_energy=energy * 2,
            baseline_size=model_size * 2,
        )

        # Get next state
        next_state = self.get_state(compressed_model)

        # Done condition (simplified)
        done = False  # Would be based on episode length or convergence

        info = {
            'accuracy': accuracy,
            'latency': latency,
            'energy': energy,
            'size': model_size,
        }

        return next_state, reward, done, info

    def _apply_actions(self, model: nn.Module, actions: Dict) -> nn.Module:
        """
        Apply all agent actions to the model.

        Args:
            model: Original model
            actions: Actions dict for all agents

        Returns:
            nn.Module: Modified model
        """
        import copy
        model = copy.deepcopy(model)

        # Apply pruning
        if 'pruning_agent' in self.agents:
            model = self.agents['pruning_agent'].apply_action(
                model, actions['pruning']
            )

        # Apply quantization
        if 'quantization_agent' in self.agents:
            model = self.agents['quantization_agent'].apply_action(
                model, actions['quantization']
            )

        # Apply distillation
        if 'distillation_agent' in self.agents:
            model = self.agents['distillation_agent'].apply_action(
                model, actions['distillation']
            )

        # Apply fusion
        if 'fusion_agent' in self.agents:
            model = self.agents['fusion_agent'].apply_action(
                model, actions['fusion']
            )

        return model

    def update_agents(self, model: nn.Module, dataloader):
        """
        Update individual agents based on current model state.

        Args:
            model: Current model
            dataloader: Training data
        """
        # Update each agent's internal state
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'update_state'):
                agent.update_state(model, dataloader)

    def train_episode(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        max_steps: int = 50
    ) -> float:
        """
        Train for one episode.

        Args:
            model: Initial model
            train_loader: Training data
            val_loader: Validation data
            max_steps: Maximum steps per episode

        Returns:
            float: Total episode reward
        """
        state = self.get_state(model)
        episode_reward = 0

        for step in range(max_steps):
            # Execute step
            next_state, reward, done, info = self.step(model, state, val_loader)

            # Store experience in PPO buffer
            # Re-get actions for storage (simplified)
            actions, log_prob, value = self.ppo.select_actions(state)
            self.ppo.buffer.add(state, actions, log_prob, reward, done, value)

            # Update statistics
            episode_reward += reward
            state = next_state

            if done:
                break

        # Update PPO
        update_info = self.ppo.update()

        self.episode_rewards.append(episode_reward)

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward

        logger.info(f"Episode completed: reward={episode_reward:.4f}, "
                    f"accuracy={info['accuracy']:.4f}, "
                    f"latency={info['latency']:.2f}ms")

        return episode_reward

    def train(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        num_episodes: int = 100,
        max_steps_per_episode: int = 50
    ) -> Dict:
        """
        Train for multiple episodes.

        Args:
            model: Initial model
            train_loader: Training data
            val_loader: Validation data
            num_episodes: Number of training episodes
            max_steps_per_episode: Max steps per episode

        Returns:
            dict: Training statistics
        """
        logger.info(f"Starting MARL training for {num_episodes} episodes")

        for episode in range(num_episodes):
            self.train_episode(model, train_loader, val_loader, max_steps_per_episode)

            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                logger.info(f"Episode {episode + 1}/{num_episodes}, "
                            f"Avg reward (last 10): {avg_reward:.4f}")

        return {
            'best_reward': self.best_reward,
            'episode_rewards': self.episode_rewards,
            'num_episodes': num_episodes,
        }

    def get_training_statistics(self) -> Dict:
        """
        Get training statistics.

        Returns:
            dict: Training statistics
        """
        return {
            'episode_rewards': self.episode_rewards,
            'best_reward': self.best_reward,
            'avg_reward': sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0,
            'num_episodes': len(self.episode_rewards),
        }
