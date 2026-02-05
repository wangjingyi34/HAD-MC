"""HAD-MC 2.0: Hardware-Aware Deep Model Compression with MARL

HAD-MC 2.0 upgrades the original HAD-MC framework by introducing:
- Multi-Agent Reinforcement Learning (MARL) for coordinated optimization
- PPO controller for automated compression strategy learning
- Hardware Abstraction Layer (HAL) for cross-platform support
- Dedicated Inference Engine (DIE) for optimized deployment
- Five specialized agents: Pruning, Quantization, Distillation, Fusion, Update
"""

__version__ = "2.0.0"
__author__ = "HAD-MC Research Team"

from .agents import *
from .controllers import *
from .hardware import *
from .inference import *
from .rewards import *
from .training import *
from .utils import *

__all__ = [
    'PruningAgent',
    'QuantizationAgent',
    'DistillationAgent',
    'FusionAgent',
    'UpdateAgent',
    'PPOController',
    'PolicyNetwork',
    'ValueNetwork',
    'ExperienceBuffer',
    'MARLCoordinator',
    'HardwareAbstractionLayer',
    'HardwareConfig',
    'LatencyLookupTable',
    'HardwareProfiler',
    'DedicatedInferenceEngine',
    'TensorRTBackend',
    'RewardFunction',
    'HADMCTrainer',
    'RolloutBuffer',
    'State',
    'ActionSpace',
    'MetricsCalculator',
    'load_config',
]
