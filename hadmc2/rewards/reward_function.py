"""Reward Function for HAD-MC 2.0 MARL framework"""

import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RewardFunction:
    """
    Multi-objective Reward Function for HAD-MC 2.0.

    Balances multiple optimization objectives:
    - Model accuracy (higher is better)
    - Inference latency (lower is better)
    - Energy consumption (lower is better)
    - Model size (lower is better)

    Uses Pareto optimization to find optimal trade-offs.
    """

    def __init__(
        self,
        accuracy_weight: float = 1.0,
        latency_weight: float = 0.5,
        energy_weight: float = 0.3,
        size_weight: float = 0.2,
        accuracy_threshold: float = 0.90,
        latency_threshold: float = 10.0,  # ms
        energy_threshold: float = 1.0,    # J
        size_threshold: float = 5.0,       # MB
    ):
        """
        Initialize Reward Function.

        Args:
            accuracy_weight: Weight for accuracy objective
            latency_weight: Weight for latency objective
            energy_weight: Weight for energy objective
            size_weight: Weight for model size objective
            accuracy_threshold: Minimum acceptable accuracy
            latency_threshold: Maximum acceptable latency (ms)
            energy_threshold: Maximum acceptable energy (J)
            size_threshold: Maximum acceptable model size (MB)
        """
        self.accuracy_weight = accuracy_weight
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.size_weight = size_weight

        self.accuracy_threshold = accuracy_threshold
        self.latency_threshold = latency_threshold
        self.energy_threshold = energy_threshold
        self.size_threshold = size_threshold

        # Pareto frontier
        self.pareto_front = []

        logger.info(f"RewardFunction initialized with thresholds: "
                   f"acc>={accuracy_threshold}, lat<={latency_threshold}ms, "
                   f"eng<={energy_threshold}J, size<={size_threshold}MB")

    def compute(
        self,
        accuracy: float,
        latency: float,
        energy: float,
        size: float,
        baseline_accuracy: float,
        baseline_latency: float,
        baseline_energy: float,
        baseline_size: float,
    ) -> float:
        """
        Compute reward based on metrics and baselines.

        Args:
            accuracy: Current model accuracy [0, 1]
            latency: Current latency (ms)
            energy: Current energy (J)
            size: Current model size (MB)
            baseline_accuracy: Baseline accuracy
            baseline_latency: Baseline latency (ms)
            baseline_energy: Baseline energy (J)
            baseline_size: Baseline size (MB)

        Returns:
            float: Computed reward
        """
        # ===== Objective Rewards =====

        # Accuracy reward: positive if better than baseline
        accuracy_reward = (accuracy - baseline_accuracy) / baseline_accuracy

        # Latency reward: positive if latency decreased
        # Lower latency is better
        latency_reward = (baseline_latency - latency) / baseline_latency

        # Energy reward: positive if energy decreased
        energy_reward = (baseline_energy - energy) / baseline_energy

        # Size reward: positive if size decreased
        size_reward = (baseline_size - size) / baseline_size

        # ===== Constraint Penalties =====

        penalty = 0

        # Accuracy penalty: large penalty if below threshold
        if accuracy < self.accuracy_threshold:
            penalty += 10 * (self.accuracy_threshold - accuracy)

        # Latency penalty: penalty if above threshold
        if latency > self.latency_threshold:
            penalty += 5 * ((latency - self.latency_threshold) / self.latency_threshold)

        # Energy penalty: penalty if above threshold
        if energy > self.energy_threshold:
            penalty += 2 * ((energy - self.energy_threshold) / self.energy_threshold)

        # Size penalty: penalty if above threshold
        if size > self.size_threshold:
            penalty += 1 * ((size - self.size_threshold) / self.size_threshold)

        # ===== Total Reward =====

        total_reward = (
            self.accuracy_weight * accuracy_reward +
            self.latency_weight * latency_reward +
            self.energy_weight * energy_reward +
            self.size_weight * size_reward -
            penalty
        )

        return total_reward

    def compute_pareto_reward(
        self,
        accuracy: float,
        latency: float,
        energy: float,
        size: float,
        baseline_accuracy: float,
        baseline_latency: float,
        baseline_energy: float,
        baseline_size: float,
        use_pareto_bonus: bool = True
    ) -> float:
        """
        Compute reward with Pareto frontier consideration.

        Args:
            accuracy: Current model accuracy
            latency: Current latency (ms)
            energy: Current energy (J)
            size: Current model size (MB)
            baseline_accuracy: Baseline accuracy
            baseline_latency: Baseline latency (ms)
            baseline_energy: Baseline energy (J)
            baseline_size: Baseline size (MB)
            use_pareto_bonus: Whether to give Pareto bonus

        Returns:
            float: Computed reward with Pareto consideration
        """
        # Compute base reward
        base_reward = self.compute(
            accuracy, latency, energy, size,
            baseline_accuracy, baseline_latency, baseline_energy, baseline_size
        )

        if not use_pareto_bonus or not self.pareto_front:
            return base_reward

        # Check if current point is Pareto optimal
        point = np.array([accuracy, -latency, -energy, -size])  # Negative for minimization objectives

        is_pareto_optimal = self._is_pareto_optimal(point, self.pareto_front)

        if is_pareto_optimal:
            # Give bonus for Pareto optimality
            pareto_bonus = 1.0
        else:
            # Penalize distance from Pareto frontier
            distance = self._distance_to_pareto(point, self.pareto_front)
            pareto_bonus = -0.1 * distance

        total_reward = base_reward + pareto_bonus

        return total_reward

    def update_pareto_front(
        self,
        accuracy: float,
        latency: float,
        energy: float,
        size: float
    ):
        """
        Update Pareto frontier with a new point.

        Args:
            accuracy: Model accuracy
            latency: Model latency (ms)
            energy: Model energy (J)
            size: Model size (MB)
        """
        # Create point (negative for minimization objectives)
        new_point = {
            'accuracy': accuracy,
            'latency': latency,
            'energy': energy,
            'size': size,
        }

        # Remove points dominated by the new point
        self.pareto_front = [
            p for p in self.pareto_front
            if not self._dominates(new_point, p)
        ]

        # Check if new point is dominated by any existing point
        is_dominated = any(
            self._dominates(p, new_point)
            for p in self.pareto_front
        )

        if not is_dominated:
            self.pareto_front.append(new_point)

        logger.debug(f"Pareto front size: {len(self.pareto_front)}")

    def _dominates(self, point1: Dict, point2: Dict) -> bool:
        """
        Check if point1 dominates point2.

        Point1 dominates Point2 if:
        - Point1 is better or equal in all objectives
        - Point1 is strictly better in at least one objective

        Args:
            point1: First point
            point2: Second point

        Returns:
            bool: True if point1 dominates point2
        """
        # Higher accuracy is better
        acc_better = point1['accuracy'] >= point2['accuracy']
        # Lower latency is better
        lat_better = point1['latency'] <= point2['latency']
        # Lower energy is better
        eng_better = point1['energy'] <= point2['energy']
        # Lower size is better
        size_better = point1['size'] <= point2['size']

        all_better = acc_better and lat_better and eng_better and size_better
        any_strictly_better = (
            point1['accuracy'] > point2['accuracy'] or
            point1['latency'] < point2['latency'] or
            point1['energy'] < point2['energy'] or
            point1['size'] < point2['size']
        )

        return all_better and any_strictly_better

    def _is_pareto_optimal(self, point: np.ndarray, pareto_front: List[Dict]) -> bool:
        """
        Check if a point is Pareto optimal.

        Args:
            point: Numpy array representing the point
            pareto_front: List of Pareto-optimal points

        Returns:
            bool: True if point is Pareto optimal
        """
        for p in pareto_front:
            p_array = np.array([p['accuracy'], -p['latency'], -p['energy'], -p['size']])

            # Check if p dominates point
            if np.all(p_array >= point) and np.any(p_array > point):
                return False

        return True

    def _distance_to_pareto(self, point: np.ndarray, pareto_front: List[Dict]) -> float:
        """
        Compute Euclidean distance from a point to the Pareto frontier.

        Args:
            point: Numpy array representing the point
            pareto_front: List of Pareto-optimal points

        Returns:
            float: Minimum distance to Pareto frontier
        """
        if not pareto_front:
            return 0.0

        min_distance = float('inf')

        for p in pareto_front:
            p_array = np.array([p['accuracy'], -p['latency'], -p['energy'], -p['size']])
            distance = np.linalg.norm(point - p_array)
            min_distance = min(min_distance, distance)

        return min_distance

    def compute_shaped_reward(
        self,
        current_state: Dict,
        next_state: Dict,
        action: Dict
    ) -> float:
        """
        Compute shaped reward for better learning.

        Adds exploration bonus and progress bonus to base reward.

        Args:
            current_state: Current state dict
            next_state: Next state dict
            action: Action taken

        Returns:
            float: Shaped reward
        """
        # Base reward
        base_reward = self.compute(
            accuracy=next_state['accuracy'],
            latency=next_state['latency'],
            energy=next_state['energy'],
            size=next_state['size'],
            baseline_accuracy=current_state['accuracy'],
            baseline_latency=current_state['latency'],
            baseline_energy=current_state['energy'],
            baseline_size=current_state['size'],
        )

        # Exploration bonus (encourage trying new configurations)
        exploration_bonus = self._compute_novelty(action) * 0.1

        # Progress bonus (encourage moving toward objectives)
        progress_bonus = self._compute_progress(current_state, next_state) * 0.2

        shaped_reward = base_reward + exploration_bonus + progress_bonus

        return shaped_reward

    def _compute_novelty(self, action: Dict) -> float:
        """
        Compute novelty of an action.

        Args:
            action: Action dict

        Returns:
            float: Novelty score [0, 1]
        """
        # Simplified: use random exploration bonus
        # In practice, would track action history
        import random
        return random.random()

    def _compute_progress(self, current_state: Dict, next_state: Dict) -> float:
        """
        Compute progress toward objectives.

        Args:
            current_state: Current state
            next_state: Next state

        Returns:
            float: Progress score [0, 1]
        """
        # Distance to target (simplified)
        target_distance_current = max(
            0,
            self.accuracy_threshold - current_state['accuracy']
        )
        target_distance_next = max(
            0,
            self.accuracy_threshold - next_state['accuracy']
        )

        # Positive if moving closer to target
        progress = target_distance_current - target_distance_next

        return max(0, progress)

    def get_weights(self) -> Dict[str, float]:
        """
        Get current reward weights.

        Returns:
            dict: Dictionary of weights
        """
        return {
            'accuracy': self.accuracy_weight,
            'latency': self.latency_weight,
            'energy': self.energy_weight,
            'size': self.size_weight,
        }

    def get_thresholds(self) -> Dict[str, float]:
        """
        Get current constraint thresholds.

        Returns:
            dict: Dictionary of thresholds
        """
        return {
            'accuracy': self.accuracy_threshold,
            'latency': self.latency_threshold,
            'energy': self.energy_threshold,
            'size': self.size_threshold,
        }

    def set_weights(
        self,
        accuracy_weight: Optional[float] = None,
        latency_weight: Optional[float] = None,
        energy_weight: Optional[float] = None,
        size_weight: Optional[float] = None
    ):
        """
        Set reward weights.

        Args:
            accuracy_weight: Weight for accuracy
            latency_weight: Weight for latency
            energy_weight: Weight for energy
            size_weight: Weight for size
        """
        if accuracy_weight is not None:
            self.accuracy_weight = accuracy_weight
        if latency_weight is not None:
            self.latency_weight = latency_weight
        if energy_weight is not None:
            self.energy_weight = energy_weight
        if size_weight is not None:
            self.size_weight = size_weight

        logger.info(f"Updated weights: acc={self.accuracy_weight}, lat={self.latency_weight}, "
                   f"eng={self.energy_weight}, size={self.size_weight}")

    def set_thresholds(
        self,
        accuracy_threshold: Optional[float] = None,
        latency_threshold: Optional[float] = None,
        energy_threshold: Optional[float] = None,
        size_threshold: Optional[float] = None
    ):
        """
        Set constraint thresholds.

        Args:
            accuracy_threshold: Minimum accuracy threshold
            latency_threshold: Maximum latency threshold (ms)
            energy_threshold: Maximum energy threshold (J)
            size_threshold: Maximum size threshold (MB)
        """
        if accuracy_threshold is not None:
            self.accuracy_threshold = accuracy_threshold
        if latency_threshold is not None:
            self.latency_threshold = latency_threshold
        if energy_threshold is not None:
            self.energy_threshold = energy_threshold
        if size_threshold is not None:
            self.size_threshold = size_threshold

        logger.info(f"Updated thresholds: acc>={self.accuracy_threshold}, "
                   f"lat<={self.latency_threshold}ms, "
                   f"eng<={self.energy_threshold}J, "
                   f"size<={self.size_threshold}MB")

    def get_pareto_front_size(self) -> int:
        """Get size of current Pareto frontier."""
        return len(self.pareto_front)

    def clear_pareto_front(self):
        """Clear the Pareto frontier."""
        self.pareto_front = []
        logger.info("Cleared Pareto frontier")

    def __repr__(self) -> str:
        return (f"RewardFunction(weights=[acc={self.accuracy_weight}, "
                f"lat={self.latency_weight}, eng={self.energy_weight}, "
                f"size={self.size_weight}], "
                f"pareto_size={len(self.pareto_front)})")
