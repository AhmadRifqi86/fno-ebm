"""
Noise Models for Realistic Data Simulation

Implements various noise models to simulate measurement uncertainty:
- Gaussian (homoscedastic): Constant noise level
- Heteroscedastic: Noise proportional to signal magnitude
- Spatially correlated: Smooth noise patterns (mimics environmental effects)
- Mixed: Combination of above
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Union, Tuple


def add_gaussian_noise(
    data: np.ndarray,
    noise_level: float,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add homoscedastic Gaussian noise.

    Args:
        data: Clean data, shape (*data_shape)
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility

    Returns:
        noisy_data: Data with noise added
        noise: The noise that was added (for analysis)
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(0, noise_level, size=data.shape)
    noisy_data = data + noise

    return noisy_data, noise


def add_heteroscedastic_noise(
    data: np.ndarray,
    base_noise: float = 0.001,
    scale_factor: float = 0.02,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add heteroscedastic noise (noise level varies with signal magnitude).

    This mimics real sensors where:
    - Larger values have larger absolute errors
    - Relative error stays more constant

    Args:
        data: Clean data
        base_noise: Minimum noise level (sensor floor)
        scale_factor: Noise proportional to signal (e.g., 0.02 = 2% noise)
        seed: Random seed

    Returns:
        noisy_data: Data with heteroscedastic noise
        noise: The noise added
    """
    if seed is not None:
        np.random.seed(seed)

    # Noise level varies with signal magnitude
    noise_std = base_noise + scale_factor * np.abs(data)

    # Generate noise with varying std
    noise = np.random.normal(0, 1, size=data.shape) * noise_std
    noisy_data = data + noise

    return noisy_data, noise


def add_spatially_correlated_noise(
    data: np.ndarray,
    noise_level: float,
    correlation_length: float = 3.0,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add spatially correlated noise using Gaussian smoothing.

    This mimics environmental effects like:
    - Temperature gradients
    - Vibrations
    - Electromagnetic interference

    Args:
        data: Clean data, shape (nx, ny, ...)
        noise_level: Overall noise magnitude
        correlation_length: Spatial correlation length in grid points
        seed: Random seed

    Returns:
        noisy_data: Data with spatially correlated noise
        noise: The noise added
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise
    white_noise = np.random.normal(0, noise_level, size=data.shape)

    # Apply Gaussian filter to create spatial correlation
    # Filter each spatial dimension
    if data.ndim == 2:
        noise = gaussian_filter(white_noise, sigma=correlation_length)
    elif data.ndim == 3:
        # For (nx, ny, channels), filter only spatial dims
        noise = np.zeros_like(white_noise)
        for c in range(data.shape[-1]):
            noise[..., c] = gaussian_filter(
                white_noise[..., c],
                sigma=correlation_length
            )
    else:
        noise = gaussian_filter(white_noise, sigma=correlation_length)

    # Renormalize to maintain noise_level
    noise = noise * (noise_level / (np.std(noise) + 1e-8))

    noisy_data = data + noise

    return noisy_data, noise


def add_mixed_noise(
    data: np.ndarray,
    gaussian_level: float = 0.005,
    hetero_scale: float = 0.01,
    spatial_level: float = 0.003,
    correlation_length: float = 2.0,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add mixed noise combining multiple sources.

    Realistic noise often has multiple components:
    - White Gaussian (sensor electronics)
    - Heteroscedastic (measurement principle)
    - Spatially correlated (environment)

    Args:
        data: Clean data
        gaussian_level: Homoscedastic component
        hetero_scale: Heteroscedastic scale factor
        spatial_level: Spatially correlated component
        correlation_length: Spatial correlation length
        seed: Random seed

    Returns:
        noisy_data: Data with mixed noise
        total_noise: Combined noise
    """
    if seed is not None:
        np.random.seed(seed)

    # Component 1: Gaussian white noise (sensor)
    noise_gaussian = np.random.normal(0, gaussian_level, size=data.shape)

    # Component 2: Heteroscedastic noise (physics)
    noise_std = hetero_scale * np.abs(data)
    noise_hetero = np.random.normal(0, 1, size=data.shape) * noise_std

    # Component 3: Spatially correlated noise (environment)
    white = np.random.normal(0, spatial_level, size=data.shape)
    if data.ndim == 2:
        noise_spatial = gaussian_filter(white, sigma=correlation_length)
    elif data.ndim == 3:
        noise_spatial = np.zeros_like(white)
        for c in range(data.shape[-1]):
            noise_spatial[..., c] = gaussian_filter(
                white[..., c],
                sigma=correlation_length
            )
    else:
        noise_spatial = gaussian_filter(white, sigma=correlation_length)

    # Combine all noise sources
    total_noise = noise_gaussian + noise_hetero + noise_spatial
    noisy_data = data + total_noise

    return noisy_data, total_noise


def get_noise_statistics(noise: np.ndarray) -> dict:
    """
    Compute statistics of noise for analysis.

    Args:
        noise: Noise array

    Returns:
        stats: Dictionary with mean, std, min, max, etc.
    """
    return {
        'mean': np.mean(noise),
        'std': np.std(noise),
        'min': np.min(noise),
        'max': np.max(noise),
        'median': np.median(noise),
        'snr_db': 10 * np.log10(np.mean(noise**2) + 1e-10)
    }