"""
Derp-VAE: Distribution Enforcement via Random Probe for Variational Auto-Encoder
Complete implementation for quantitative finance
"""

__version__ = "1.0.0"
__author__ = "Derp-VAE Project"

from .models.encoder import DerpEncoder
from .models.decoder import DerpDecoder
from .training.engine import DerpEngine
from .training.random_probe import RandomProbe
from .training.monitor import Monitor
from .config.hyperparameters import HyperParameters

__all__ = [
    'DerpEncoder',
    'DerpDecoder',
    'DerpEngine',
    'RandomProbe',
    'Monitor',
    'HyperParameters'
]
