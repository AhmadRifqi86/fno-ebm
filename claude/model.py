"""
Model registry for FNO-EBM project

This module imports all FNO and EBM models from their respective modules
and provides a centralized access point for backward compatibility.

For new code, import directly from fno.py, ebm.py, or trainer.py instead.

Note: FNO_EBM class has been moved to trainer.py to avoid circular imports.
"""

# Import all FNO models from fno.py
from fno import (
    SpectralConv2d,
    FNO2d,
    SpatialTransformerBlock,
    TransformerFNO2d,
    FourierTransformerBlock,
    FourierTransformerFNO2d,
    MambaBlock2D,
    MambaFNO2d,
    FourierMambaBlock,
    FourierMambaFNO2d,
)

# Import all EBM models from ebm.py
from ebm import (
    EBMPotential,
    KANLayer,
    KAN_EBM,
    FNO_KAN_EBM,
    GraphConvLayer,
    GNN_EBM,
)

# FNO_EBM is now in trainer.py (cannot import here due to circular dependency)
# Users should import directly: from trainer import FNO_EBM

# Define __all__ for explicit exports
__all__ = [
    # FNO models
    'SpectralConv2d',
    'FNO2d',
    'SpatialTransformerBlock',
    'TransformerFNO2d',
    'FourierTransformerBlock',
    'FourierTransformerFNO2d',
    'MambaBlock2D',
    'MambaFNO2d',
    'FourierMambaBlock',
    'FourierMambaFNO2d',
    # EBM models
    'EBMPotential',
    'KANLayer',
    'KAN_EBM',
    'FNO_KAN_EBM',
    'GraphConvLayer',
    'GNN_EBM',
]