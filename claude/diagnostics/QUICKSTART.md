# Quick Start Guide - Diagnostics Suite

## 5-Minute Quick Start

### 1. Test the Installation

```python
# Test if diagnostics work
from diagnostics import calibration_error
import numpy as np

# Dummy data
samples = np.random.randn(100, 10, 10, 1)  # 100 samples
truth = np.random.randn(10, 10, 1)

# Compute calibration
cal_err = calibration_error(samples, truth)
print(f"Calibration error: {cal_err:.4f}")
```

If this works, you're ready!

---

### 2. Evaluate Your Model

```python
from diagnostics import calibration_error, plot_calibration_curve
from inference import inference_probabilistic

# Generate predictions
samples, stats = inference_probabilistic(
    your_model, test_input,
    num_samples=100,
    num_mcmc_steps=200
)

# Compute calibration
result = calibration_error(samples, test_truth, return_details=True)

# Plot
plot_calibration_curve(
    result['predicted_coverage'],
    result['actual_coverage'],
    method_name="FNO-EBM",
    save_path="calibration.png"
)
```

**Check if calibration error < 0.05** ✅

---

### 3. Run Full Evaluation

```bash
cd diagnostics
python example_evaluation.py
```

This generates all figures and metrics automatically.

---

## What Each File Does

### `uncertainty_metrics.py`
**Metrics for evaluating uncertainty quality**

Key functions:
- `calibration_error()` - Most important!
- `continuous_ranked_probability_score()` - CRPS
- `coverage_analysis()` - Check prediction intervals

### `physics_metrics.py`
**Metrics for physics compliance**

Key functions:
- `compute_pde_residual()` - PDE residual
- `boundary_condition_violation()` - BC errors

### `visualization.py`
**Publication-quality plots**

Key functions:
- `plot_calibration_curve()` - Calibration visualization
- `plot_uncertainty_heatmap()` - Spatial uncertainty
- `plot_error_vs_uncertainty()` - Correlation analysis

### `comparison.py`
**Compare multiple models**

Key class:
- `ModelComparator` - Systematic comparison

### `example_evaluation.py`
**Complete working example**

Run this to see everything in action!

---

## Common Issues

### Issue: "ImportError: No module named diagnostics"

**Solution:**
```python
import sys
sys.path.append('/path/to/fno-ebm/claude')
from diagnostics import ...
```

### Issue: "Calibration error is very high (> 0.2)"

**Possible causes:**
1. Not enough Langevin steps (increase to 500+)
2. Wrong σ² in energy function (try tuning)
3. EBM not trained enough

**Debug:**
```python
# Check if samples are diverse enough
std_per_point = samples.std(dim=0)
print(f"Mean std: {std_per_point.mean():.4f}")
# Should be > 0.01, if near 0, samples are too similar
```

### Issue: "CRPS is NaN"

**Cause:** All samples are identical (no uncertainty)

**Solution:** Check your sampling process is actually generating different samples.

---

## For Your Thesis - Minimum Viable Evaluation

### Required Metrics (Must Report)
1. ✅ Calibration Error
2. ✅ CRPS
3. ✅ MSE/MAE
4. ✅ Coverage at 90% and 95%

### Required Figures (Must Include)
1. ✅ Calibration curve
2. ✅ Uncertainty heatmap (1-2 examples)
3. ✅ Comparison table with baselines

### Nice to Have
- Error vs uncertainty plot
- Physics residual comparison
- Radar chart of multi-method comparison

---

## Interpreting Results

### Good Results ✅
```
Calibration Error: 0.035
CRPS: 0.012
Coverage 90%: 0.89
Coverage 95%: 0.94
```
→ **Well-calibrated, ready to publish**

### Borderline Results ⚠️
```
Calibration Error: 0.08
CRPS: 0.025
Coverage 90%: 0.82
Coverage 95%: 0.90
```
→ **Acceptable but needs discussion**

### Poor Results ❌
```
Calibration Error: 0.15
CRPS: 0.045
Coverage 90%: 0.70
Coverage 95%: 0.80
```
→ **Not ready - need to debug/improve**

---

## Next Steps

1. **Run example_evaluation.py**
2. **Check calibration error**
3. **If < 0.05: Great! Generate all figures**
4. **If > 0.10: Debug and tune**
5. **Compare with baseline (at least FNO-Dropout)**
6. **Write thesis section with honest results**

---

## Help

If stuck:
1. Check README.md for detailed docs
2. Look at example_evaluation.py
3. Read metric docstrings
4. Test on dummy data first

Remember: **Honest, rigorous evaluation > inflated claims!**