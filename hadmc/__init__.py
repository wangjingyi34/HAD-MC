"""HAD-MC: Hardware-Aware Dynamic Model Compression Framework"""

from .utils import create_simple_cnn, calculate_flops, calculate_model_size, evaluate_model
from .quantization import LayerwisePrecisionAllocator, AdaptiveQuantizer
from .pruning import GradientSensitivityPruner
from .distillation import FeatureAlignedDistiller
from .fusion import OperatorFuser, OperatorFusion

__version__ = "1.0.0"
__all__ = [
    'create_simple_cnn',
    'calculate_flops', 
    'calculate_model_size',
    'evaluate_model',
    'LayerwisePrecisionAllocator',
    'AdaptiveQuantizer',
    'GradientSensitivityPruner',
    'FeatureAlignedDistiller',
    'OperatorFuser',
    'OperatorFusion'
]
