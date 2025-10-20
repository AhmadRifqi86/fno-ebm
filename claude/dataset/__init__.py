"""
Synthetic Dataset Generation for FNO-EBM

This module provides synthetic data generation for various PDEs with:
- Multiple PDE types (Darcy, Burgers, Poisson, Wave)
- Configurable complexity levels
- Various noise models (Gaussian, heteroscedastic, spatially correlated)
- Reproducible with seed control
"""

from .darcy_flow import DarcyFlowGenerator
from .burgers import BurgersGenerator
from .poisson import PoissonGenerator
from .noise_models import (
    add_gaussian_noise,
    add_heteroscedastic_noise,
    add_spatially_correlated_noise,
    add_mixed_noise
)

__all__ = [
    'DarcyFlowGenerator',
    'BurgersGenerator',
    'PoissonGenerator',
    'add_gaussian_noise',
    'add_heteroscedastic_noise',
    'add_spatially_correlated_noise',
    'add_mixed_noise',
]
