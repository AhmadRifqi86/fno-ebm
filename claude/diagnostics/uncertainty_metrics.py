"""
Uncertainty Quantification Metrics

This module implements standard metrics for evaluating probabilistic predictions:
- Calibration error
- Continuous Ranked Probability Score (CRPS)
- Negative Log-Likelihood (NLL)
- Coverage analysis
- Sharpness and interval scores

All metrics follow the convention: lower is better (except for coverage, which should be close to nominal).
"""

import numpy as np
import torch
from scipy.stats import gaussian_kde
from typing import Union, Tuple, Dict


def calibration_error(
    samples: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    num_bins: int = 10,
    return_details: bool = False
) -> Union[float, Dict]:
    """
    Compute calibration error: measures if predicted confidence matches actual coverage.

    For a well-calibrated model, if we predict 90% confidence intervals,
    the true value should fall within those intervals 90% of the time.

    Args:
        samples: Predictive samples, shape (n_samples, *data_shape)
        ground_truth: True values, shape (*data_shape)
        num_bins: Number of confidence levels to test
        return_details: If True, return detailed breakdown

    Returns:
        calibration_error: Mean absolute difference between predicted and actual coverage
        OR dictionary with detailed results if return_details=True

    Example:
        >>> samples = np.random.randn(100, 64, 64, 1)  # 100 samples
        >>> truth = np.random.randn(64, 64, 1)
        >>> cal_err = calibration_error(samples, truth)
        >>> print(f"Calibration error: {cal_err:.4f}")  # Should be < 0.05 for good calibration
    """
    # Convert to numpy if torch
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Confidence levels to test
    alphas = np.linspace(0.05, 0.95, num_bins)
    predicted_coverage = alphas
    actual_coverage = []
    errors = []

    for alpha in alphas:
        # Compute credible interval at confidence level alpha
        lower_quantile = (1 - alpha) / 2
        upper_quantile = (1 + alpha) / 2

        lower = np.quantile(samples, lower_quantile, axis=0)
        upper = np.quantile(samples, upper_quantile, axis=0)

        # Check if ground truth falls within interval
        within_interval = (ground_truth >= lower) & (ground_truth <= upper)
        actual_cov = np.mean(within_interval)

        actual_coverage.append(actual_cov)
        errors.append(abs(actual_cov - alpha))

    calibration_err = np.mean(errors)

    if return_details:
        return {
            'calibration_error': calibration_err,
            'predicted_coverage': predicted_coverage,
            'actual_coverage': np.array(actual_coverage),
            'errors': np.array(errors),
            'max_error': np.max(errors)
        }
    else:
        return calibration_err


def continuous_ranked_probability_score(
    samples: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).

    CRPS is a proper scoring rule that measures the quality of probabilistic predictions.
    It generalizes MAE to probabilistic forecasts.

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]

    where X, X' are independent samples from the predictive distribution, y is ground truth.

    Args:
        samples: Predictive samples, shape (n_samples, *data_shape)
        ground_truth: True values, shape (*data_shape)

    Returns:
        crps: Lower is better (0 = perfect)

    Reference:
        Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction, and Estimation"
    """
    # Convert to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    n_samples = samples.shape[0]

    # E[|X - y|]
    term1 = np.mean(np.abs(samples - ground_truth[np.newaxis, ...]), axis=0)

    # E[|X - X'|] - computed efficiently
    # Instead of double loop, use pairwise differences
    term2 = 0.0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            term2 += np.abs(samples[i] - samples[j])

    term2 = term2 / (n_samples * (n_samples - 1) / 2)

    crps = np.mean(term1 - 0.5 * term2)

    return crps


def negative_log_likelihood(
    samples: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    bandwidth: float = None
) -> float:
    """
    Compute Negative Log-Likelihood using kernel density estimation.

    Estimates the log probability of ground truth under the empirical distribution
    defined by samples.

    Args:
        samples: Predictive samples, shape (n_samples, *data_shape)
        ground_truth: True values, shape (*data_shape)
        bandwidth: KDE bandwidth (if None, uses Scott's rule)

    Returns:
        nll: Negative log-likelihood (lower is better)

    Note:
        This uses Gaussian KDE which may not be appropriate for all distributions.
        For high-dimensional data, this becomes unreliable.
    """
    # Convert to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Flatten spatial dimensions for KDE
    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1).T  # (n_features, n_samples)
    truth_flat = ground_truth.reshape(-1, 1)  # (n_features, 1)

    # Use Gaussian KDE
    try:
        if bandwidth is not None:
            kde = gaussian_kde(samples_flat, bw_method=bandwidth)
        else:
            kde = gaussian_kde(samples_flat)

        log_prob = kde.logpdf(truth_flat)
        nll = -np.mean(log_prob)
    except np.linalg.LinAlgError:
        # KDE failed (e.g., singular covariance)
        # Fall back to simple Gaussian assumption
        mean = np.mean(samples_flat, axis=1, keepdims=True)
        std = np.std(samples_flat, axis=1, keepdims=True) + 1e-6

        log_prob = -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((truth_flat - mean) / std)**2
        nll = -np.mean(log_prob)

    return nll


def coverage_analysis(
    samples: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    confidence_levels: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    Analyze coverage at different confidence levels.

    Args:
        samples: Predictive samples, shape (n_samples, *data_shape)
        ground_truth: True values, shape (*data_shape)
        confidence_levels: Array of confidence levels to test (default: [0.5, 0.68, 0.9, 0.95, 0.99])

    Returns:
        Dictionary with coverage statistics
    """
    if confidence_levels is None:
        confidence_levels = np.array([0.5, 0.68, 0.9, 0.95, 0.99])

    # Convert to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    results = {
        'confidence_levels': confidence_levels,
        'actual_coverage': [],
        'interval_widths': []
    }

    for alpha in confidence_levels:
        lower_q = (1 - alpha) / 2
        upper_q = (1 + alpha) / 2

        lower = np.quantile(samples, lower_q, axis=0)
        upper = np.quantile(samples, upper_q, axis=0)

        # Coverage
        within = (ground_truth >= lower) & (ground_truth <= upper)
        actual_cov = np.mean(within)
        results['actual_coverage'].append(actual_cov)

        # Mean interval width
        width = np.mean(upper - lower)
        results['interval_widths'].append(width)

    results['actual_coverage'] = np.array(results['actual_coverage'])
    results['interval_widths'] = np.array(results['interval_widths'])

    return results


def prediction_interval_coverage(
    samples: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.9
) -> float:
    """
    Compute empirical coverage of prediction intervals.

    Args:
        samples: Predictive samples, shape (n_samples, *data_shape)
        ground_truth: True values, shape (*data_shape)
        alpha: Nominal confidence level (e.g., 0.9 for 90% intervals)

    Returns:
        coverage: Fraction of points falling within intervals (should be ≈ alpha)
    """
    # Convert to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    lower_q = (1 - alpha) / 2
    upper_q = (1 + alpha) / 2

    lower = np.quantile(samples, lower_q, axis=0)
    upper = np.quantile(samples, upper_q, axis=0)

    within_interval = (ground_truth >= lower) & (ground_truth <= upper)
    coverage = np.mean(within_interval)

    return coverage


def sharpness_score(
    samples: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.9
) -> float:
    """
    Compute sharpness: average width of prediction intervals.

    Sharpness measures how concentrated the predictions are.
    Lower is better (but must be balanced with coverage).

    Args:
        samples: Predictive samples, shape (n_samples, *data_shape)
        alpha: Confidence level

    Returns:
        sharpness: Average interval width
    """
    # Convert to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()

    lower_q = (1 - alpha) / 2
    upper_q = (1 + alpha) / 2

    lower = np.quantile(samples, lower_q, axis=0)
    upper = np.quantile(samples, upper_q, axis=0)

    sharpness = np.mean(upper - lower)

    return sharpness


def interval_score(
    samples: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.9
) -> float:
    """
    Compute interval score (proper scoring rule for intervals).

    Combines sharpness and calibration into a single score.

    IS_α(l, u, y) = (u - l) + (2/α)·(l - y)·1{y < l} + (2/α)·(y - u)·1{y > u}

    Args:
        samples: Predictive samples, shape (n_samples, *data_shape)
        ground_truth: True values, shape (*data_shape)
        alpha: Confidence level

    Returns:
        interval_score: Lower is better

    Reference:
        Gneiting & Raftery (2007)
    """
    # Convert to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    lower_q = (1 - alpha) / 2
    upper_q = (1 + alpha) / 2

    lower = np.quantile(samples, lower_q, axis=0)
    upper = np.quantile(samples, upper_q, axis=0)

    # Width term (sharpness)
    width = upper - lower

    # Penalty for being below lower bound
    below = (lower - ground_truth) * (ground_truth < lower)

    # Penalty for being above upper bound
    above = (ground_truth - upper) * (ground_truth > upper)

    # Combined score
    score = width + (2 / alpha) * (below + above)

    return np.mean(score)