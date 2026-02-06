"""
TensorBoard Visualization for HAD-MC 2.0 Experiments

This module provides comprehensive TensorBoard logging for all experiments.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Any
import numpy as np
import os
from datetime import datetime


class TensorBoardLogger:
    """TensorBoard logger for HAD-MC 2.0 experiments."""

    def __init__(
        self,
        log_dir: str = 'experiments_r3/results/logs',
        experiment_name: Optional[str] = None
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"hadmc2_{timestamp}"

        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value.

        Args:
            tag: Tag for the scalar
            value: Scalar value
            step: Step number (uses global_step if None)
        """
        if step is None:
            step = self.global_step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log multiple scalars under a main tag.

        Args:
            main_tag: Main tag name
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Step number
        """
        if step is None:
            step = self.global_step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
        self,
        tag: str,
        values: np.ndarray,
        step: Optional[int] = None
    ):
        """
        Log a histogram.

        Args:
            tag: Tag for the histogram
            values: Array of values
            step: Step number
        """
        if step is None:
            step = self.global_step
        self.writer.add_histogram(tag, values, step)

    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number
        """
        if step is None:
            step = self.global_step

        # Main metrics
        for key, value in metrics.items():
            self.log_scalar(f'training/{key}', value, step)

    def log_episode(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log episode summary.

        Args:
            episode: Episode number
            episode_reward: Total reward for episode
            episode_length: Length of episode
            metrics: Additional metrics
        """
        self.log_scalar('episode/reward', episode_reward, episode)
        self.log_scalar('episode/length', episode_length, episode)

        if metrics:
            for key, value in metrics.items():
                self.log_scalar(f'episode/{key}', value, episode)

    def log_compression_metrics(
        self,
        accuracy: float,
        latency: float,
        energy: float,
        model_size: float,
        compression_ratio: float,
        step: Optional[int] = None
    ):
        """
        Log compression metrics.

        Args:
            accuracy: Model accuracy
            latency: Inference latency (ms)
            energy: Energy consumption (J)
            model_size: Model size (MB)
            compression_ratio: Compression ratio
            step: Step number
        """
        if step is None:
            step = self.global_step

        self.log_scalars('compression', {
            'accuracy': accuracy,
            'latency_ms': latency,
            'energy_j': energy,
            'model_size_mb': model_size,
            'compression_ratio': compression_ratio
        }, step)

    def log_agent_actions(
        self,
        episode: int,
        agent_actions: Dict[str, Any]
    ):
        """
        Log agent action distributions.

        Args:
            episode: Episode number
            agent_actions: Dictionary of agent actions
        """
        for agent_name, actions in agent_actions.items():
            if isinstance(actions, (list, np.ndarray)):
                self.log_histogram(f'actions/{agent_name}', np.array(actions), episode)

    def log_pareto_frontier(
        self,
        episode: int,
        frontier_size: int,
        hypervolume: float,
        new_points: int = 0
    ):
        """
        Log Pareto frontier metrics.

        Args:
            episode: Episode number
            frontier_size: Size of Pareto frontier
            hypervolume: Hypervolume value
            new_points: Number of new points added
        """
        self.log_scalar('pareto/frontier_size', frontier_size, episode)
        self.log_scalar('pareto/hypervolume', hypervolume, episode)
        self.log_scalar('pareto/new_points', new_points, episode)

    def log_model_parameters(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None
    ):
        """
        Log model parameter statistics.

        Args:
            model: PyTorch model
            step: Step number
        """
        if step is None:
            step = self.global_step

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f'parameters/{name}', param.data.cpu().numpy(), step)
                self.log_scalar(f'grad_norm/{name}', param.grad.norm().item(), step)

    def log_hyperparameters(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float]
    ):
        """
        Log hyperparameters and final metrics.

        Args:
            hparam_dict: Dictionary of hyperparameters
            metric_dict: Dictionary of final metrics
        """
        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_comparison(
        self,
        method_name: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log comparison metrics.

        Args:
            method_name: Name of method
            metrics: Dictionary of metrics
            step: Step number
        """
        if step is None:
            step = self.global_step

        for key, value in metrics.items():
            self.log_scalar(f'comparison/{method_name}/{key}', value, step)

    def log_ablation(
        self,
        variant_name: str,
        metrics: Dict[str, float]
    ):
        """
        Log ablation study results.

        Args:
            variant_name: Name of ablation variant
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            self.log_scalar(f'ablation/{variant_name}/{key}', value)

    def log_cross_dataset(
        self,
        dataset_name: str,
        metrics: Dict[str, float]
    ):
        """
        Log cross-dataset results.

        Args:
            dataset_name: Name of dataset
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            self.log_scalar(f'cross_dataset/{dataset_name}/{key}', value)

    def log_cross_platform(
        self,
        platform_name: str,
        metrics: Dict[str, float]
    ):
        """
        Log cross-platform results.

        Args:
            platform_name: Name of platform
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            self.log_scalar(f'cross_platform/{platform_name}/{key}', value)

    def step(self):
        """Increment global step."""
        self.global_step += 1

    def flush(self):
        """Flush writer."""
        self.writer.flush()

    def close(self):
        """Close writer."""
        self.writer.close()
