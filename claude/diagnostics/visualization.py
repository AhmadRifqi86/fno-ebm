"""
Visualization Tools for Uncertainty Quantification Diagnostics

This module provides publication-quality plotting functions for:
- Calibration curves
- Reliability diagrams
- Uncertainty heatmaps
- Prediction intervals
- Error vs uncertainty correlation
- Comparison radar charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from typing import Union, List, Dict, Optional, Tuple
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
sns.set_palette("husl")


def plot_calibration_curve(
    predicted_coverage: np.ndarray,
    actual_coverage: np.ndarray,
    method_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot calibration curve showing predicted vs actual coverage.

    A well-calibrated model should lie on the diagonal line.

    Args:
        predicted_coverage: Predicted confidence levels (e.g., [0.5, 0.68, 0.9, 0.95])
        actual_coverage: Actual empirical coverage at each level
        method_name: Name of the method for legend
        save_path: Path to save figure (if None, don't save)
        show: Whether to display the plot
        ax: Matplotlib axes (if None, creates new figure)

    Returns:
        fig: Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration', alpha=0.6)

    # Actual calibration
    ax.plot(predicted_coverage, actual_coverage, 'o-', linewidth=2,
            markersize=8, label=method_name, color='#2E86AB')

    # Shaded region for acceptable calibration (±5%)
    ax.fill_between([0, 1], [0, 1], [0.05, 1.05], alpha=0.1, color='green',
                     label='Good calibration (±5%)')
    ax.fill_between([0, 1], [-0.05, 0.95], [0, 1], alpha=0.1, color='green')

    # Styling
    ax.set_xlabel('Predicted Confidence Level', fontweight='bold')
    ax.set_ylabel('Actual Coverage', fontweight='bold')
    ax.set_title(f'Calibration Curve: {method_name}', fontweight='bold', pad=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_aspect('equal')

    # Add calibration error text
    cal_error = np.mean(np.abs(predicted_coverage - actual_coverage))
    ax.text(0.98, 0.02, f'Calibration Error: {cal_error:.4f}',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_reliability_diagram(
    predicted_coverage: np.ndarray,
    actual_coverage: np.ndarray,
    method_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot reliability diagram comparing multiple methods.

    Args:
        predicted_coverage: Array of shape (n_methods, n_bins)
        actual_coverage: Array of shape (n_methods, n_bins)
        method_names: List of method names
        save_path: Path to save figure
        show: Whether to display

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect', alpha=0.7)

    # Plot each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

    for i, (pred, actual, name) in enumerate(zip(predicted_coverage, actual_coverage, method_names)):
        ax.plot(pred, actual, 'o-', linewidth=2, markersize=7,
                label=name, color=colors[i], alpha=0.8)

    ax.set_xlabel('Predicted Confidence', fontweight='bold', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontweight='bold', fontsize=12)
    ax.set_title('Reliability Diagram: Multi-Method Comparison',
                 fontweight='bold', fontsize=14, pad=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_uncertainty_heatmap(
    ground_truth: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    Plot heatmap showing ground truth, prediction, and uncertainty.

    Args:
        ground_truth: True solution, shape (nx, ny) or (nx, ny, 1)
        predicted_mean: Mean prediction, same shape
        predicted_std: Predictive standard deviation, same shape
        save_path: Path to save
        show: Whether to display
        vmin, vmax: Color scale limits

    Returns:
        fig: Matplotlib figure
    """
    # Squeeze channel dimension if present
    if ground_truth.ndim == 3:
        ground_truth = ground_truth.squeeze(-1)
    if predicted_mean.ndim == 3:
        predicted_mean = predicted_mean.squeeze(-1)
    if predicted_std.ndim == 3:
        predicted_std = predicted_std.squeeze(-1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Determine color scale
    if vmin is None:
        vmin = min(ground_truth.min(), predicted_mean.min())
    if vmax is None:
        vmax = max(ground_truth.max(), predicted_mean.max())

    # Ground truth
    im0 = axes[0, 0].imshow(ground_truth, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Predicted mean
    im1 = axes[0, 1].imshow(predicted_mean, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Predicted Mean', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Absolute error
    error = np.abs(ground_truth - predicted_mean)
    im2 = axes[1, 0].imshow(error, cmap='Reds')
    axes[1, 0].set_title('Absolute Error', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Predicted uncertainty (std)
    im3 = axes[1, 1].imshow(predicted_std, cmap='plasma')
    axes[1, 1].set_title('Predicted Std Dev (Uncertainty)', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Overall title
    fig.suptitle('Uncertainty Quantification Heatmaps',
                 fontweight='bold', fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_prediction_intervals(
    x_coords: np.ndarray,
    ground_truth: np.ndarray,
    predicted_mean: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    confidence_level: float = 0.9,
    slice_index: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot prediction intervals along a 1D slice.

    Args:
        x_coords: Spatial coordinates for slice
        ground_truth: True values along slice
        predicted_mean: Mean predictions
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        confidence_level: Confidence level (e.g., 0.9 for 90%)
        slice_index: Which slice to plot (for 2D data)
        save_path: Save path
        show: Whether to display

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot confidence interval
    ax.fill_between(x_coords, lower_bound, upper_bound,
                     alpha=0.3, color='skyblue',
                     label=f'{int(confidence_level*100)}% Prediction Interval')

    # Plot mean prediction
    ax.plot(x_coords, predicted_mean, 'b-', linewidth=2, label='Predicted Mean')

    # Plot ground truth
    ax.plot(x_coords, ground_truth, 'r--', linewidth=2, label='Ground Truth', alpha=0.8)

    # Styling
    ax.set_xlabel('Spatial Coordinate', fontweight='bold')
    ax.set_ylabel('Solution Value', fontweight='bold')
    title = 'Prediction Intervals'
    if slice_index is not None:
        title += f' (Slice {slice_index})'
    ax.set_title(title, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    # Calculate coverage
    within = (ground_truth >= lower_bound) & (ground_truth <= upper_bound)
    coverage = np.mean(within)
    ax.text(0.02, 0.98, f'Empirical Coverage: {coverage:.2%}',
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_error_vs_uncertainty(
    absolute_errors: np.ndarray,
    predicted_std: np.ndarray,
    method_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot correlation between absolute error and predicted uncertainty.

    A well-calibrated model should have high correlation: when uncertainty
    is high, errors should also be high.

    Args:
        absolute_errors: Absolute prediction errors
        predicted_std: Predicted standard deviations
        method_name: Name of method
        save_path: Save path
        show: Whether to display

    Returns:
        fig: Matplotlib figure
    """
    # Flatten arrays
    errors_flat = absolute_errors.flatten()
    std_flat = predicted_std.flatten()

    # Remove any NaN or inf values
    mask = np.isfinite(errors_flat) & np.isfinite(std_flat)
    errors_flat = errors_flat[mask]
    std_flat = std_flat[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].hexbin(std_flat, errors_flat, gridsize=50, cmap='Blues', mincnt=1)
    axes[0].set_xlabel('Predicted Std Dev', fontweight='bold')
    axes[0].set_ylabel('Absolute Error', fontweight='bold')
    axes[0].set_title(f'Error vs Uncertainty: {method_name}', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Compute correlation
    correlation = np.corrcoef(std_flat, errors_flat)[0, 1]
    axes[0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                 transform=axes[0].transAxes, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
                 fontsize=11)

    # Binned statistics
    n_bins = 20
    bins = np.linspace(std_flat.min(), std_flat.max(), n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []

    for i in range(n_bins):
        mask = (std_flat >= bins[i]) & (std_flat < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(errors_flat[mask].mean())
            bin_stds.append(errors_flat[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    axes[1].errorbar(bin_centers, bin_means, yerr=bin_stds,
                     fmt='o-', capsize=5, capthick=2, linewidth=2,
                     color='darkblue', label='Mean Error ± Std')
    axes[1].plot(bin_centers, bin_centers, 'r--', linewidth=2,
                 label='Ideal (Error = Uncertainty)', alpha=0.7)
    axes[1].set_xlabel('Predicted Std Dev (Binned)', fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Error', fontweight='bold')
    axes[1].set_title('Binned Error vs Uncertainty', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_comparison_radar(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot radar chart comparing multiple methods across metrics.

    Args:
        metrics_dict: Dictionary mapping method names to metric dictionaries
                     e.g., {'FNO-EBM': {'MSE': 0.01, 'CalError': 0.02, ...}}
        save_path: Save path
        show: Whether to display

    Returns:
        fig: Matplotlib figure
    """
    # Extract metrics
    method_names = list(metrics_dict.keys())
    metric_names = list(metrics_dict[method_names[0]].keys())
    n_metrics = len(metric_names)

    # Normalize metrics to [0, 1] (lower is better → invert for visualization)
    values_array = np.array([[metrics_dict[method][metric]
                             for metric in metric_names]
                            for method in method_names])

    # Normalize each metric
    max_vals = values_array.max(axis=0)
    min_vals = values_array.min(axis=0)
    normalized = 1 - (values_array - min_vals) / (max_vals - min_vals + 1e-8)

    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Plot each method
    colors = plt.cm.Set2(np.linspace(0, 1, len(method_names)))

    for idx, method in enumerate(method_names):
        values = normalized[idx].tolist()
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2,
                label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title('Multi-Method Comparison (Radar Chart)\n(Higher is better after normalization)',
                 fontweight='bold', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig