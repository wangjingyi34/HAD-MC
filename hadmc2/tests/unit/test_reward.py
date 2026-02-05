"""Unit tests for Reward Function"""

import unittest
import numpy as np
from hadmc2.rewards.reward_function import RewardFunction


class TestRewardFunction(unittest.TestCase):
    """Test RewardFunction class."""

    def test_initialization(self):
        """Test RewardFunction initialization."""
        reward_fn = RewardFunction()

        self.assertEqual(reward_fn.accuracy_weight, 1.0)
        self.assertEqual(reward_fn.latency_weight, 0.5)
        self.assertEqual(reward_fn.energy_weight, 0.3)
        self.assertEqual(reward_fn.size_weight, 0.2)

        self.assertEqual(reward_fn.accuracy_threshold, 0.90)
        self.assertEqual(reward_fn.latency_threshold, 10.0)
        self.assertEqual(reward_fn.energy_threshold, 1.0)
        self.assertEqual(reward_fn.size_threshold, 5.0)

    def test_compute_basic_reward(self):
        """Test basic reward computation."""
        reward_fn = RewardFunction()

        # Better than baseline
        reward = reward_fn.compute(
            accuracy=0.95,
            latency=5.0,
            energy=0.5,
            size=2.5,
            baseline_accuracy=0.90,
            baseline_latency=10.0,
            baseline_energy=1.0,
            baseline_size=5.0,
        )

        # All improvements should give positive reward
        self.assertGreater(reward, 0)

    def test_compute_with_accuracy_penalty(self):
        """Test reward with accuracy below threshold."""
        reward_fn = RewardFunction(accuracy_threshold=0.90)

        reward = reward_fn.compute(
            accuracy=0.85,  # Below threshold
            latency=5.0,
            energy=0.5,
            size=2.5,
            baseline_accuracy=0.90,
            baseline_latency=10.0,
            baseline_energy=1.0,
            baseline_size=5.0,
        )

        # Should have penalty
        self.assertLess(reward, 0)

    def test_compute_with_latency_penalty(self):
        """Test reward with latency above threshold."""
        reward_fn = RewardFunction(latency_threshold=10.0)

        reward = reward_fn.compute(
            accuracy=0.95,
            latency=15.0,  # Above threshold
            energy=0.5,
            size=2.5,
            baseline_accuracy=0.90,
            baseline_latency=10.0,
            baseline_energy=1.0,
            baseline_size=5.0,
        )

        # Should have penalty
        self.assertLess(reward, 0)

    def test_pareto_front_update(self):
        """Test Pareto frontier update."""
        reward_fn = RewardFunction()

        # Add initial point
        reward_fn.update_pareto_front(
            accuracy=0.90,
            latency=10.0,
            energy=1.0,
            size=5.0,
        )

        self.assertEqual(len(reward_fn.pareto_front), 1)

        # Add dominating point
        reward_fn.update_pareto_front(
            accuracy=0.95,  # Better accuracy
            latency=8.0,    # Better latency
            energy=0.8,    # Better energy
            size=4.0,       # Better size
        )

        # Should dominate and replace previous
        self.assertEqual(len(reward_fn.pareto_front), 1)

    def test_is_pareto_optimal(self):
        """Test Pareto optimality check."""
        reward_fn = RewardFunction()

        # Use non-dominated points (trade-offs between objectives)
        pareto_front = [
            {'accuracy': 0.95, 'latency': 10.0, 'energy': 1.0, 'size': 5.0},  # High accuracy, high latency
            {'accuracy': 0.90, 'latency': 8.0, 'energy': 0.8, 'size': 4.0},   # Lower accuracy, low latency
        ]

        # Non-dominated point (better than one in accuracy, worse than other in latency)
        point = np.array([0.92, -9.0, -0.9, -4.5])
        is_optimal = reward_fn._is_pareto_optimal(point, pareto_front)

        self.assertTrue(is_optimal)

        # Dominated point (worse than both)
        dominated_point = np.array([0.88, -12.0, -1.2, -6.0])
        is_optimal = reward_fn._is_pareto_optimal(dominated_point, pareto_front)

        self.assertFalse(is_optimal)

    def test_distance_to_pareto(self):
        """Test distance to Pareto frontier calculation."""
        reward_fn = RewardFunction()

        # Use non-dominated points (trade-offs between objectives)
        pareto_front = [
            {'accuracy': 0.95, 'latency': 10.0, 'energy': 1.0, 'size': 5.0},  # High accuracy, high latency
            {'accuracy': 0.90, 'latency': 8.0, 'energy': 0.8, 'size': 4.0},   # Lower accuracy, low latency
        ]

        point = np.array([0.92, -9.0, -0.9, -4.5])
        distance = reward_fn._distance_to_pareto(point, pareto_front)

        self.assertGreaterEqual(distance, 0)

    def test_compute_pareto_reward(self):
        """Test Pareto-aware reward computation."""
        reward_fn = RewardFunction()

        # Set up Pareto frontier
        reward_fn.update_pareto_front(
            accuracy=0.90,
            latency=10.0,
            energy=1.0,
            size=5.0,
        )

        # Test with Pareto bonus
        reward = reward_fn.compute_pareto_reward(
            accuracy=0.92,  # Better than Pareto point
            latency=9.0,    # Better than Pareto point
            energy=0.9,    # Better than Pareto point
            size=4.5,       # Better than Pareto point
            baseline_accuracy=0.90,
            baseline_latency=10.0,
            baseline_energy=1.0,
            baseline_size=5.0,
            use_pareto_bonus=True,
        )

        self.assertGreater(reward, 0)

    def test_get_weights(self):
        """Test getting reward weights."""
        reward_fn = RewardFunction(
            accuracy_weight=2.0,
            latency_weight=0.8,
            energy_weight=0.4,
            size_weight=0.6,
        )

        weights = reward_fn.get_weights()

        self.assertEqual(weights['accuracy'], 2.0)
        self.assertEqual(weights['latency'], 0.8)
        self.assertEqual(weights['energy'], 0.4)
        self.assertEqual(weights['size'], 0.6)

    def test_set_weights(self):
        """Test setting reward weights."""
        reward_fn = RewardFunction()

        reward_fn.set_weights(
            accuracy_weight=3.0,
            latency_weight=1.0,
            energy_weight=0.5,
            size_weight=0.7,
        )

        weights = reward_fn.get_weights()

        self.assertEqual(weights['accuracy'], 3.0)
        self.assertEqual(weights['latency'], 1.0)

    def test_get_thresholds(self):
        """Test getting constraint thresholds."""
        reward_fn = RewardFunction(
            accuracy_threshold=0.95,
            latency_threshold=8.0,
            energy_threshold=0.8,
            size_threshold=3.0,
        )

        thresholds = reward_fn.get_thresholds()

        self.assertEqual(thresholds['accuracy'], 0.95)
        self.assertEqual(thresholds['latency'], 8.0)
        self.assertEqual(thresholds['energy'], 0.8)
        self.assertEqual(thresholds['size'], 3.0)

    def test_set_thresholds(self):
        """Test setting constraint thresholds."""
        reward_fn = RewardFunction()

        reward_fn.set_thresholds(
            accuracy_threshold=0.92,
            latency_threshold=12.0,
            energy_threshold=1.5,
            size_threshold=6.0,
        )

        thresholds = reward_fn.get_thresholds()

        self.assertEqual(thresholds['accuracy'], 0.92)
        self.assertEqual(thresholds['latency'], 12.0)

    def test_clear_pareto_front(self):
        """Test clearing Pareto frontier."""
        reward_fn = RewardFunction()

        # Add non-dominated points (trade-offs between objectives)
        reward_fn.update_pareto_front(0.95, 10.0, 1.0, 5.0)  # High accuracy, high latency
        reward_fn.update_pareto_front(0.90, 8.0, 0.8, 4.0)   # Lower accuracy, low latency

        self.assertEqual(len(reward_fn.pareto_front), 2)

        reward_fn.clear_pareto_front()

        self.assertEqual(len(reward_fn.pareto_front), 0)

    def test_get_pareto_front_size(self):
        """Test getting Pareto front size."""
        reward_fn = RewardFunction()

        self.assertEqual(reward_fn.get_pareto_front_size(), 0)

        # Add non-dominated points (trade-offs between objectives)
        reward_fn.update_pareto_front(0.95, 10.0, 1.0, 5.0)  # High accuracy, high latency
        reward_fn.update_pareto_front(0.90, 8.0, 0.8, 4.0)   # Lower accuracy, low latency

        self.assertEqual(reward_fn.get_pareto_front_size(), 2)

    def test_compute_shaped_reward(self):
        """Test shaped reward computation."""
        reward_fn = RewardFunction()

        current_state = {
            'accuracy': 0.90,
            'latency': 10.0,
            'energy': 1.0,
            'size': 5.0,
        }

        next_state = {
            'accuracy': 0.92,  # Improved
            'latency': 9.0,    # Improved
            'energy': 0.9,    # Improved
            'size': 4.5,       # Improved
        }

        action = {'test': 'action'}

        reward = reward_fn.compute_shaped_reward(current_state, next_state, action)

        self.assertIsInstance(reward, float)


if __name__ == '__main__':
    unittest.main()
