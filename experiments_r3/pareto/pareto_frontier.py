"""
Pareto Frontier Analysis for HAD-MC 2.0

This module implements Pareto frontier tracking and analysis for multi-objective
optimization in HAD-MC 2.0.

The Pareto frontier represents the set of non-dominated solutions where no
solution is better than another in all objectives.

Objectives:
- Accuracy (higher is better)
- Latency (lower is better)
- Energy (lower is better)
- Model size (lower is better)
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import torch


class ParetoPoint:
    """A single point in the multi-objective space."""

    def __init__(
        self,
        accuracy: float,
        latency: float,
        energy: float,
        model_size: float,
        config_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize Pareto point.

        Args:
            accuracy: Accuracy metric (higher is better)
            latency: Latency in ms (lower is better)
            energy: Energy in J (lower is better)
            model_size: Model size in MB (lower is better)
            config_id: Identifier for this configuration
            metadata: Additional metadata
        """
        self.accuracy = accuracy
        self.latency = latency
        self.energy = energy
        self.model_size = model_size
        self.config_id = config_id or f"config_{id(self)}"
        self.metadata = metadata or {}

    def dominates(self, other: 'ParetoPoint') -> bool:
        """
        Check if this point dominates another point.

        Point A dominates point B if:
        - A is not worse than B in any objective
        - A is strictly better than B in at least one objective

        Args:
            other: Other Pareto point

        Returns:
            True if this point dominates the other
        """
        at_least_one_better = False

        # For objectives where higher is better
        if self.accuracy > other.accuracy:
            at_least_one_better = True
        elif self.accuracy < other.accuracy:
            return False

        # For objectives where lower is better
        if self.latency < other.latency:
            at_least_one_better = True
        elif self.latency > other.latency:
            return False

        if self.energy < other.energy:
            at_least_one_better = True
        elif self.energy > other.energy:
            return False

        if self.model_size < other.model_size:
            at_least_one_better = True
        elif self.model_size > other.model_size:
            return False

        return at_least_one_better

    def distance_to(self, other: 'ParetoPoint') -> float:
        """
        Calculate Euclidean distance to another point.

        Args:
            other: Other Pareto point

        Returns:
            Euclidean distance
        """
        # Normalize objectives for distance calculation
        norm_acc = self.accuracy - other.accuracy
        norm_lat = (self.latency - other.latency) / 100.0  # Scale latency
        norm_eng = (self.energy - other.energy) / 10.0    # Scale energy
        norm_size = (self.model_size - other.model_size) / 100.0  # Scale size

        return np.sqrt(
            norm_acc**2 + norm_lat**2 + norm_eng**2 + norm_size**2
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'latency': self.latency,
            'energy': self.energy,
            'model_size': self.model_size,
            'config_id': self.config_id,
            'metadata': self.metadata
        }


class ParetoFrontier:
    """
    Pareto frontier for multi-objective optimization.

    Maintains a set of non-dominated points and provides utilities for
    Pareto analysis.
    """

    def __init__(self):
        """Initialize empty Pareto frontier."""
        self.points: List[ParetoPoint] = []
        self.history: List[List[ParetoPoint]] = []  # Track frontier evolution

    def add_point(self, point: ParetoPoint) -> bool:
        """
        Add a point to the Pareto frontier.

        Args:
            point: Pareto point to add

        Returns:
            True if point was added to frontier (is non-dominated)
        """
        # Check if point is dominated by any existing point
        for existing in self.points:
            if existing.dominates(point):
                return False

        # Remove any points dominated by the new point
        self.points = [p for p in self.points if not point.dominates(p)]

        # Add the new point
        self.points.append(point)
        self.history.append(self.points.copy())

        return True

    def get_points(self) -> List[ParetoPoint]:
        """Get all points in the frontier."""
        return self.points

    def get_frontier_size(self) -> int:
        """Get the number of points in the frontier."""
        return len(self.points)

    def is_dominated(self, point: ParetoPoint) -> bool:
        """
        Check if a point is dominated by any point in the frontier.

        Args:
            point: Pareto point to check

        Returns:
            True if point is dominated
        """
        for existing in self.points:
            if existing.dominates(point):
                return True
        return False

    def distance_to_frontier(self, point: ParetoPoint) -> float:
        """
        Calculate minimum distance from a point to the Pareto frontier.

        Args:
            point: Pareto point

        Returns:
            Minimum distance to any frontier point
        """
        if not self.points:
            return float('inf')

        distances = [point.distance_to(p) for p in self.points]
        return min(distances)

    def get_hypervolume(
        self,
        reference_point: Optional[ParetoPoint] = None
    ) -> float:
        """
        Calculate hypervolume of the Pareto frontier.

        Hypervolume measures the volume of objective space dominated by
        the Pareto frontier.

        Args:
            reference_point: Reference point for hypervolume calculation
                          (default: worst point seen so far)

        Returns:
            Hypervolume value
        """
        if not self.points:
            return 0.0

        # Use worst point as reference if not provided
        if reference_point is None:
            min_acc = min(p.accuracy for p in self.points)
            max_lat = max(p.latency for p in self.points)
            max_eng = max(p.energy for p in self.points)
            max_size = max(p.model_size for p in self.points)

            reference_point = ParetoPoint(
                accuracy=max(0, min_acc - 0.1),
                latency=max_lat * 1.1,
                energy=max_eng * 1.1,
                model_size=max_size * 1.1
            )

        # Simplified 2D hypervolume (accuracy vs latency)
        # For true multi-dimensional hypervolume, use a dedicated library
        area = 0.0
        sorted_points = sorted(self.points, key=lambda p: p.latency)

        prev_latency = reference_point.latency
        for point in sorted_points:
            width = prev_latency - point.latency
            height = point.accuracy - reference_point.accuracy
            area += width * height
            prev_latency = point.latency

        return area

    def get_tradeoff_curve(
        self,
        objective_x: str = 'latency',
        objective_y: str = 'accuracy'
    ) -> Tuple[List[float], List[float]]:
        """
        Get 2D tradeoff curve between two objectives.

        Args:
            objective_x: X-axis objective
            objective_y: Y-axis objective

        Returns:
            Tuple of (x_values, y_values)
        """
        if not self.points:
            return [], []

        x_values = [getattr(p, objective_x) for p in self.points]
        y_values = [getattr(p, objective_y) for p in self.points]

        # Sort by x-axis
        sorted_indices = np.argsort(x_values)
        x_values = [x_values[i] for i in sorted_indices]
        y_values = [y_values[i] for i in sorted_indices]

        return x_values, y_values

    def get_knee_point(self) -> Optional[ParetoPoint]:
        """
        Find the knee point in the Pareto frontier.

        The knee point represents the point where further improvement in
        one objective requires significant sacrifice in another.

        Returns:
            Knee point, or None if frontier is empty
        """
        if len(self.points) < 3:
            return None

        # Use distance to line method
        # Find point with maximum distance from line connecting extreme points

        # Sort by latency
        sorted_points = sorted(self.points, key=lambda p: p.latency)

        # Line from first to last point
        p1 = sorted_points[0]
        pn = sorted_points[-1]

        # Find point with maximum distance from line
        max_distance = -1
        knee_point = None

        for point in sorted_points[1:-1]:
            # Distance from point to line
            distance = abs(
                (pn.latency - p1.latency) * point.accuracy -
                (pn.accuracy - p1.accuracy) * point.latency +
                pn.accuracy * p1.latency -
                pn.latency * p1.accuracy
            ) / np.sqrt((pn.latency - p1.latency)**2 + (pn.accuracy - p1.accuracy)**2)

            if distance > max_distance:
                max_distance = distance
                knee_point = point

        return knee_point

    def compare_frontiers(
        self,
        other: 'ParetoFrontier'
    ) -> Dict[str, float]:
        """
        Compare this Pareto frontier with another.

        Args:
            other: Other Pareto frontier

        Returns:
            Dictionary of comparison metrics
        """
        # Calculate coverage
        coverage = 0.0
        for point in other.points:
            if self.is_dominated(point):
                coverage += 1
        coverage = coverage / len(other.points) if other.points else 0.0

        # Compare hypervolumes
        hv1 = self.get_hypervolume()
        hv2 = other.get_hypervolume()

        return {
            'coverage': coverage,
            'hypervolume_self': hv1,
            'hypervolume_other': hv2,
            'hypervolume_ratio': hv1 / hv2 if hv2 > 0 else 0,
            'size_self': len(self.points),
            'size_other': len(other.points)
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'points': [p.to_dict() for p in self.points],
            'size': len(self.points),
            'hypervolume': self.get_hypervolume()
        }


def track_pareto_during_training(
    points: List[Dict[str, float]],
    episode_rewards: Optional[List[float]] = None
) -> Tuple[ParetoFrontier, List[Dict]]:
    """
    Track Pareto frontier during MARL training.

    Args:
        points: List of points with objective values
        episode_rewards: Episode rewards

    Returns:
        Tuple of (final frontier, frontier evolution history)
    """
    frontier = ParetoFrontier()
    history = []

    for i, point_data in enumerate(points):
        point = ParetoPoint(
            accuracy=point_data.get('accuracy', 0),
            latency=point_data.get('latency', 100),
            energy=point_data.get('energy', 10),
            model_size=point_data.get('model_size', 100),
            config_id=point_data.get('config_id', f'episode_{i}'),
            metadata={'episode': i, 'reward': episode_rewards[i] if episode_rewards else None}
        )

        is_added = frontier.add_point(point)
        history.append({
            'episode': i,
            'point_added': is_added,
            'frontier_size': frontier.get_frontier_size(),
            'hypervolume': frontier.get_hypervolume()
        })

    return frontier, history


def visualize_pareto_frontier(
    frontier: ParetoFrontier,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize Pareto frontier (2D projection).

    Args:
        frontier: Pareto frontier to visualize
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt

        # Get tradeoff curve (accuracy vs latency)
        x_values, y_values = frontier.get_tradeoff_curve('latency', 'accuracy')

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, c='blue', s=100, label='Pareto Front', zorder=3)

        # Connect points
        if len(x_values) > 1:
            plt.plot(x_values, y_values, 'b--', alpha=0.5, linewidth=2)

        # Mark knee point if exists
        knee = frontier.get_knee_point()
        if knee:
            plt.scatter(knee.latency, knee.accuracy, c='red', s=200,
                       marker='*', label='Knee Point', zorder=4)

        # Add labels
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            plt.annotate(f'{i+1}', (x, y), textcoords="offset points",
                       xytext=(5, 5), ha='center', fontsize=8)

        plt.xlabel('Latency (ms)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Pareto Frontier: Accuracy vs. Latency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pareto frontier saved to {save_path}")

        plt.close()

    except ImportError:
        print("Matplotlib not available, skipping visualization")


def compare_multiple_frontiers(
    frontiers: Dict[str, ParetoFrontier],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize multiple Pareto frontiers for comparison.

    Args:
        frontiers: Dictionary mapping names to Pareto frontiers
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for (name, frontier), color in zip(frontiers.items(), colors[:len(frontiers)]):
            x_values, y_values = frontier.get_tradeoff_curve('latency', 'accuracy')

            plt.scatter(x_values, y_values, c=color, s=80, label=name, zorder=3)

            if len(x_values) > 1:
                plt.plot(x_values, y_values, color=color, linestyle='--',
                        alpha=0.5, linewidth=1.5)

        plt.xlabel('Latency (ms)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Pareto Frontier Comparison', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")

        plt.close()

    except ImportError:
        print("Matplotlib not available, skipping visualization")
