# Diagnostics Suite for FNO-EBM Evaluation

A comprehensive toolkit for evaluating probabilistic operator learning models with uncertainty quantification. This suite provides standardized metrics, visualizations, and comparison tools for research integrity and reproducibility.

## Features

### 📊 Uncertainty Quantification Metrics
- **Calibration Error**: Measures if predicted confidence matches actual coverage
- **CRPS (Continuous Ranked Probability Score)**: Proper scoring rule for probabilistic forecasts
- **Negative Log-Likelihood**: Evaluates probability assigned to ground truth
- **Coverage Analysis**: Checks empirical coverage at multiple confidence levels
- **Sharpness & Interval Scores**: Balances uncertainty width with accuracy

### ⚛️ Physics Compliance Metrics
- **PDE Residuals**: Measures how well predictions satisfy governing equations
  - Darcy flow: `-∇·(a∇u) = f`
  - Poisson: `-∇²u = f`
  - Burgers equation: `u·∂u/∂x = ν·∂²u/∂x²`
- **Boundary Condition Violations**: Checks BC satisfaction
- **Conservation Laws**: Validates mass/energy conservation

### 📈 Visualization Tools
- **Calibration Curves**: Predicted vs actual coverage
- **Reliability Diagrams**: Multi-model comparison
- **Uncertainty Heatmaps**: Spatial uncertainty visualization
- **Prediction Intervals**: Confidence bands with coverage
- **Error vs Uncertainty**: Correlation analysis
- **Radar Charts**: Multi-metric model comparison

### 🔄 Comparison Framework
- **ModelComparator**: Systematic comparison of multiple models
- **BenchmarkSuite**: Standard test problems (Darcy, Burgers, etc.)
- **Automated Report Generation**: HTML/JSON outputs

---

## Installation

```bash
# The diagnostics suite is already part of your project
cd /path/to/fno-ebm/claude
# All dependencies should already be installed with your main environment
```

---

## Quick Start

### Example 1: Evaluate Calibration

```python
from diagnostics import calibration_error, plot_calibration_curve
import numpy as np

# Your model predictions (100 samples per prediction)
samples = model.predict_samples(test_data, num_samples=100)  # (100, batch, nx, ny, 1)
ground_truth = test_labels  # (batch, nx, ny, 1)

# Compute calibration error
cal_err = calibration_error(samples, ground_truth, num_bins=10)
print(f"Calibration Error: {cal_err:.4f}")  # Should be < 0.05 for good calibration

# Visualize calibration curve
from diagnostics import calibration_error
result = calibration_error(samples, ground_truth, num_bins=10, return_details=True)

plot_calibration_curve(
    result['predicted_coverage'],
    result['actual_coverage'],
    method_name="FNO-EBM",
    save_path="calibration_curve.png"
)
```

### Example 2: Check Physics Compliance

```python
from diagnostics import compute_pde_residual, boundary_condition_violation

# For Darcy flow
residual = compute_pde_residual(
    u=predicted_solution,
    x=None,
    pde_type='darcy',
    a=permeability_field,
    f=source_term
)
print(f"PDE Residual: {residual:.6f}")  # Lower is better

# Check boundary conditions
bc_error = boundary_condition_violation(
    predicted_solution,
    bc_type='dirichlet_zero'
)
print(f"BC Violation: {bc_error:.6f}")
```

### Example 3: Uncertainty Visualization

```python
from diagnostics import plot_uncertainty_heatmap
import numpy as np

# Get predictions
samples = model.predict_samples(test_input, num_samples=100)
pred_mean = samples.mean(axis=0)
pred_std = samples.std(axis=0)

# Visualize
plot_uncertainty_heatmap(
    ground_truth=test_truth[0],  # First test sample
    predicted_mean=pred_mean[0],
    predicted_std=pred_std[0],
    save_path="uncertainty_heatmap.png"
)
```

### Example 4: Compare Multiple Models

```python
from diagnostics import ModelComparator

# Define models to compare
models = {
    'FNO-EBM': your_fno_ebm_model,
    'FNO-Dropout': baseline_dropout_model,
    'Vanilla-FNO': deterministic_fno
}

# Create comparator
comparator = ModelComparator(
    models=models,
    test_data=test_loader,
    num_samples=100
)

# Run comparison
results = comparator.run_comparison()

# Print summary table
comparator.print_summary()

# Generate report
comparator.generate_report('comparison_report')
```

---

## Complete Evaluation Pipeline

Here's a full example for your thesis/paper:

```python
import torch
from diagnostics import (
    calibration_error,
    continuous_ranked_probability_score,
    compute_pde_residual,
    plot_calibration_curve,
    plot_uncertainty_heatmap,
    plot_error_vs_uncertainty,
    ModelComparator
)

# 1. Load your trained model
from trainer import FNO_EBM
from fno import FNO2d
from ebm import EBMPotential

fno_model = FNO2d(modes1=12, modes2=12, width=64, num_layers=4)
ebm_model = EBMPotential(input_dim=4, hidden_dims=[128, 256, 256, 128])
model = FNO_EBM(fno_model, ebm_model)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# 2. Generate predictions on test set
from inference import inference_probabilistic

samples_list = []
truths_list = []

for x_test, y_test in test_loader:
    samples, stats = inference_probabilistic(
        model, x_test,
        num_samples=100,
        num_mcmc_steps=200,
        step_size=0.005
    )
    samples_list.append(samples)
    truths_list.append(y_test)

# Stack all samples
all_samples = torch.cat(samples_list, dim=1)  # (100, N_test, nx, ny, 1)
all_truths = torch.cat(truths_list, dim=0)    # (N_test, nx, ny, 1)

# 3. Compute Metrics
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

# Calibration
cal_err = calibration_error(all_samples, all_truths)
print(f"Calibration Error: {cal_err:.4f}")

# CRPS
crps = continuous_ranked_probability_score(all_samples, all_truths)
print(f"CRPS: {crps:.4f}")

# Physics residual
pred_mean = all_samples.mean(dim=0)
residual = compute_pde_residual(pred_mean[0], None, pde_type='darcy', a=a_test[0], f=f_test[0])
print(f"PDE Residual: {residual:.6f}")

# 4. Generate Visualizations
plot_calibration_curve(
    *calibration_error(all_samples, all_truths, return_details=True).values(),
    method_name="FNO-EBM",
    save_path="figures/calibration.png"
)

plot_uncertainty_heatmap(
    all_truths[0].numpy(),
    pred_mean[0].numpy(),
    all_samples.std(dim=0)[0].numpy(),
    save_path="figures/uncertainty_heatmap.png"
)

pred_std = all_samples.std(dim=0)
abs_errors = torch.abs(pred_mean - all_truths)
plot_error_vs_uncertainty(
    abs_errors.numpy(),
    pred_std.numpy(),
    method_name="FNO-EBM",
    save_path="figures/error_vs_uncertainty.png"
)

print("\nFigures saved to figures/")
```

---

## Metrics Reference

### Calibration Error
**Range**: [0, 1]
**Interpretation**: < 0.05 excellent, < 0.10 good, > 0.15 poor
**What it measures**: Whether predicted uncertainty matches actual error distribution

### CRPS (Continuous Ranked Probability Score)
**Range**: [0, ∞)
**Interpretation**: Lower is better (0 = perfect)
**What it measures**: Generalization of MAE to probabilistic forecasts
**Formula**: `E[|X - y|] - 0.5·E[|X - X'|]`

### Physics Residual
**Range**: [0, ∞)
**Interpretation**: Lower is better (0 = exact PDE satisfaction)
**What it measures**: Mean squared PDE residual over domain
**Formula**: For Darcy: `||-∇·(a∇u) - f||²`

### Coverage (90% / 95%)
**Range**: [0, 1]
**Interpretation**: Should match nominal level (0.90 or 0.95)
**What it measures**: Fraction of points within predicted intervals
**Good calibration**: Coverage ≈ Nominal ± 0.05

---

## Comparison with Papers #1 and #2

### Recommended Experiments

```python
# 1. Load baselines (you'll need to implement these)
from baselines import EnergyScoreFNO, ProbabilisticNO

# 2. Compare on Darcy flow
models = {
    'FNO-EBM (Yours)': your_model,
    'Energy Score FNO (Paper #1)': energy_score_model,
    'PNO (Paper #2)': pno_model
}

comparator = ModelComparator(models, test_data, num_samples=100)
results = comparator.run_comparison(metrics=[
    'mse', 'mae',
    'calibration_error', 'crps',
    'coverage_90', 'coverage_95',
    'sharpness', 'interval_score',
    'inference_time'
])

# 3. Generate comparison plots
from diagnostics import plot_comparison_radar, plot_reliability_diagram

plot_comparison_radar(
    {name: res['metrics'] for name, res in results.items()},
    save_path='figures/comparison_radar.png'
)
```

---

## Tips for Your Thesis

### 1. Always Report These Metrics
- MSE / MAE (accuracy)
- Calibration Error (primary UQ metric)
- CRPS (probabilistic score)
- Coverage at 90% and 95%
- Physics Residual (your advantage!)

### 2. Key Figures to Include
- Calibration curve (shows if UQ is trustworthy)
- Uncertainty heatmap (shows spatial variation)
- Error vs Uncertainty (shows correlation)
- Comparison radar (shows you're competitive)

### 3. Honest Reporting
- Include failure cases
- Report computational cost
- Show when physics constraints help
- Be clear about limitations

---

## FAQ

**Q: My calibration error is 0.15, is this bad?**
A: Yes, that's poorly calibrated. Target < 0.05. Check:
- Are you tuning the σ² parameter in `E(u,x) = ||u-μ||²/(2σ²) + V`?
- Are you running enough Langevin steps?
- Is your EBM trained sufficiently?

**Q: How many samples do I need for reliable metrics?**
A: For calibration: ≥ 50 samples. For CRPS: ≥ 100 samples. For NLL: ≥ 200 samples.

**Q: My model has lower MSE but worse calibration than baselines. What does this mean?**
A: Your model is accurate but overconfident. It's giving too narrow uncertainty estimates. This is common with EBMs if trained incorrectly.

**Q: Should I compare to Paper #1 or #2?**
A: **Both**. Use Paper #1 as a simpler baseline, Paper #2 as the SOTA. You need to be competitive with #1 and comparable to #2.

---

## Citation

If you use this diagnostics suite in your research, please cite:

```bibtex
@software{fno_ebm_diagnostics,
  title={Diagnostics Suite for FNO-EBM Evaluation},
  year={2025},
  author={Your Name},
  note={Part of FNO-EBM project}
}
```

---

## Contact & Support

For issues or questions about this diagnostics suite:
1. Check this README first
2. Look at example code above
3. Review metric definitions in source code
4. Ask your supervisor

**Remember**: The goal is honest, reproducible science. Use these tools to validate your work rigorously!