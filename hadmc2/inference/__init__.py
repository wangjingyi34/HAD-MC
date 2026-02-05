"""HAD-MC 2.0 Inference Module"""

from .die import DedicatedInferenceEngine
from .tensorrt_backend import TensorRTBackend

__all__ = [
    'DedicatedInferenceEngine',
    'TensorRTBackend',
]
