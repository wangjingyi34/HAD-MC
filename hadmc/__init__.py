"""HAD-MC: Hardware-Aware Dynamic Model Compression Framework"""

from .utils import create_simple_cnn, calculate_flops, calculate_model_size, evaluate_model

__version__ = "1.0.0"
__all__ = [
    'create_simple_cnn',
    'calculate_flops', 
    'calculate_model_size',
    'evaluate_model'
]
