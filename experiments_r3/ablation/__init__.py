"""
Ablation Study Experiments

This module contains ablation study experiments to understand the contribution
of each component in HAD-MC 2.0.

Ablation variants:
1. Baseline (no compression)
2. Pruning only
3. Quantization only
4. Distillation only
5. Fusion only
6. Update only
7. Pruning + Quantization
8. Pruning + Distillation
9. Quantization + Distillation
10. All five agents (full HAD-MC 2.0)

Additional ablations:
- Without PPO controller (random actions)
- Without Pareto-aware reward
- Different reward weight configurations
- Different PPO hyperparameters
"""

__all__ = [
    'run_ablation_study',
    'AblationConfig',
    'AblationResults'
]
