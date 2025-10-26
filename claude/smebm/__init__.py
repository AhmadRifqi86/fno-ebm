"""
smebm - FNO-EBM implementation using torchebm library

This package contains the same FNO-EBM architecture as the scratch/ package,
but uses the torchebm library for EBM training instead of implementing
everything from scratch.

Key differences from scratch/:
- trainer.py uses torchebm.samplers.LangevinSampler instead of manual MCMC
- trainer.py uses torchebm.losses.ContrastiveDivergence instead of manual CD loss
- All other components (FNO, EBM models, data utilities) remain the same
"""

__version__ = "1.0.0"