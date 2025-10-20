"""
Diagnostics Suite for FNO-EBM Evaluation

This module provides comprehensive tools for evaluating uncertainty quantification
quality, physics compliance, and comparison with baseline methods.

Main modules:
- uncertainty_metrics: Calibration, CRPS, NLL, coverage analysis
- physics_metrics: PDE residuals, boundary condition violations
- visualization: Calibration plots, uncertainty heatmaps, comparison charts
- comparison: Framework for comparing multiple models
"""

from .uncertainty_metrics import (
    calibration_error,
    continuous_ranked_probability_score,
    negative_log_likelihood,
    coverage_analysis,
    prediction_interval_coverage,
    sharpness_score,
    interval_score
)

from .physics_metrics import (
    compute_pde_residual,
    boundary_condition_violation,
    conservation_law_check,
    physics_consistency_score
)

from .visualization import (
    plot_calibration_curve,
    plot_uncertainty_heatmap,
    plot_prediction_intervals,
    plot_error_vs_uncertainty,
    plot_reliability_diagram,
    plot_comparison_radar
)

from .comparison import (
    ModelComparator,
    BenchmarkSuite,
    generate_comparison_report
)

__all__ = [
    # Uncertainty metrics
    'calibration_error',
    'continuous_ranked_probability_score',
    'negative_log_likelihood',
    'coverage_analysis',
    'prediction_interval_coverage',
    'sharpness_score',
    'interval_score',

    # Physics metrics
    'compute_pde_residual',
    'boundary_condition_violation',
    'conservation_law_check',
    'physics_consistency_score',

    # Visualization
    'plot_calibration_curve',
    'plot_uncertainty_heatmap',
    'plot_prediction_intervals',
    'plot_error_vs_uncertainty',
    'plot_reliability_diagram',
    'plot_comparison_radar',

    # Comparison framework
    'ModelComparator',
    'BenchmarkSuite',
    'generate_comparison_report',
]