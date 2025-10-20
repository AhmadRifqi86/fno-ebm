"""
Example Evaluation Script

This script demonstrates how to use the diagnostics suite to evaluate
your FNO-EBM model and compare it with baselines.

Run this after training your model to generate comprehensive evaluation results.
"""

import torch
import numpy as np
from pathlib import Path

# Diagnostics imports
from diagnostics import (
    # Metrics
    calibration_error,
    continuous_ranked_probability_score,
    coverage_analysis,
    compute_pde_residual,
    boundary_condition_violation,

    # Visualization
    plot_calibration_curve,
    plot_uncertainty_heatmap,
    plot_prediction_intervals,
    plot_error_vs_uncertainty,

    # Comparison
    ModelComparator
)

# Your model imports
from trainer import FNO_EBM
from fno import FNO2d
from ebm import EBMPotential, KAN_EBM, FNO_KAN_EBM, GNN_EBM
from inference import inference_probabilistic


def evaluate_single_model(
    model,
    test_loader,
    device='cuda',
    num_samples=100,
    output_dir='evaluation_results'
):
    """
    Comprehensive evaluation of a single model.

    Args:
        model: Your FNO_EBM model
        test_loader: DataLoader with test data
        device: 'cuda' or 'cpu'
        num_samples: Number of samples for uncertainty quantification
        output_dir: Where to save results
    """

    print("=" * 70)
    print("SINGLE MODEL EVALUATION")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    # Collect predictions
    all_samples = []
    all_truths = []
    all_inputs = []

    print("\nGenerating predictions...")
    for i, (x, y) in enumerate(test_loader):
        print(f"  Batch {i+1}/{len(test_loader)}", end='\r')

        x = x.to(device)
        y = y.to(device)

        # Generate samples using Langevin dynamics
        samples, stats = inference_probabilistic(
            model, x,
            num_samples=num_samples,
            num_mcmc_steps=200,
            step_size=0.005,
            device=device
        )

        all_samples.append(samples.cpu())
        all_truths.append(y.cpu())
        all_inputs.append(x.cpu())

    print("\n\nAggregating results...")
    all_samples = torch.cat(all_samples, dim=1)  # (num_samples, N_test, nx, ny, 1)
    all_truths = torch.cat(all_truths, dim=0)     # (N_test, nx, ny, 1)
    all_inputs = torch.cat(all_inputs, dim=0)     # (N_test, nx, ny, 3)

    # Compute statistics
    pred_mean = all_samples.mean(dim=0)
    pred_std = all_samples.std(dim=0)

    # ==========================================
    # UNCERTAINTY METRICS
    # ==========================================
    print("\n" + "=" * 70)
    print("UNCERTAINTY QUANTIFICATION METRICS")
    print("=" * 70)

    # Calibration
    cal_result = calibration_error(
        all_samples.numpy(),
        all_truths.numpy(),
        num_bins=10,
        return_details=True
    )
    print(f"\nCalibration Error: {cal_result['calibration_error']:.4f}")
    print(f"Max Calibration Error: {cal_result['max_error']:.4f}")

    # CRPS
    crps = continuous_ranked_probability_score(
        all_samples.numpy(),
        all_truths.numpy()
    )
    print(f"CRPS: {crps:.6f}")

    # Coverage analysis
    cov_result = coverage_analysis(
        all_samples.numpy(),
        all_truths.numpy(),
        confidence_levels=np.array([0.5, 0.68, 0.9, 0.95, 0.99])
    )

    print("\nCoverage Analysis:")
    for level, actual in zip(cov_result['confidence_levels'], cov_result['actual_coverage']):
        print(f"  {level:.0%} confidence: {actual:.2%} actual coverage")

    # ==========================================
    # PHYSICS METRICS (Example for Darcy flow)
    # ==========================================
    print("\n" + "=" * 70)
    print("PHYSICS COMPLIANCE METRICS")
    print("=" * 70)

    # Assuming you have permeability and source term
    # Replace with your actual data structure
    # a = all_inputs[..., 2:3]  # Permeability from input
    # f = torch.zeros_like(pred_mean)  # Source term

    # For first test sample
    # residual = compute_pde_residual(
    #     pred_mean[0].numpy(),
    #     None, None,
    #     pde_type='darcy',
    #     a=a[0].numpy(),
    #     f=f[0].numpy(),
    #     return_details=True
    # )
    # print(f"\nPDE Residual (mean): {residual['mean_residual']:.6f}")
    # print(f"PDE Residual (max):  {residual['max_residual']:.6f}")

    # BC violation
    bc_error = boundary_condition_violation(
        pred_mean[0].numpy(),
        bc_type='dirichlet_zero'
    )
    print(f"Boundary Condition Violation: {bc_error:.6f}")

    # ==========================================
    # ACCURACY METRICS
    # ==========================================
    print("\n" + "=" * 70)
    print("ACCURACY METRICS")
    print("=" * 70)

    mse = torch.mean((pred_mean - all_truths) ** 2).item()
    mae = torch.mean(torch.abs(pred_mean - all_truths)).item()

    # Relative L2 error
    rel_l2 = (torch.norm(pred_mean - all_truths) / torch.norm(all_truths)).item()

    print(f"\nMSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Relative L2 Error: {rel_l2:.4f}")

    # ==========================================
    # GENERATE VISUALIZATIONS
    # ==========================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Calibration curve
    print("\n1. Calibration curve...")
    plot_calibration_curve(
        cal_result['predicted_coverage'],
        cal_result['actual_coverage'],
        method_name="FNO-EBM",
        save_path=output_dir / "calibration_curve.png",
        show=False
    )

    # 2. Uncertainty heatmap (first test sample)
    print("2. Uncertainty heatmap...")
    plot_uncertainty_heatmap(
        all_truths[0].squeeze().numpy(),
        pred_mean[0].squeeze().numpy(),
        pred_std[0].squeeze().numpy(),
        save_path=output_dir / "uncertainty_heatmap.png",
        show=False
    )

    # 3. Error vs Uncertainty
    print("3. Error vs uncertainty correlation...")
    abs_errors = torch.abs(pred_mean - all_truths)
    plot_error_vs_uncertainty(
        abs_errors.numpy(),
        pred_std.numpy(),
        method_name="FNO-EBM",
        save_path=output_dir / "error_vs_uncertainty.png",
        show=False
    )

    # 4. Prediction intervals (1D slice)
    print("4. Prediction intervals...")
    slice_idx = 32  # Middle slice
    x_coords = np.arange(all_truths.shape[2])

    plot_prediction_intervals(
        x_coords,
        all_truths[0, slice_idx, :, 0].numpy(),
        pred_mean[0, slice_idx, :, 0].numpy(),
        pred_mean[0, slice_idx, :, 0].numpy() - 1.96 * pred_std[0, slice_idx, :, 0].numpy(),
        pred_mean[0, slice_idx, :, 0].numpy() + 1.96 * pred_std[0, slice_idx, :, 0].numpy(),
        confidence_level=0.95,
        save_path=output_dir / "prediction_intervals.png",
        show=False
    )

    print(f"\nAll figures saved to: {output_dir}")

    # ==========================================
    # SAVE NUMERICAL RESULTS
    # ==========================================
    results = {
        'calibration_error': cal_result['calibration_error'],
        'crps': crps,
        'mse': mse,
        'mae': mae,
        'relative_l2': rel_l2,
        'bc_violation': bc_error,
        'coverage': {
            str(level): actual
            for level, actual in zip(cov_result['confidence_levels'], cov_result['actual_coverage'])
        }
    }

    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to: {output_dir / 'metrics.json'}")

    return results


def compare_models(
    models_dict,
    test_loader,
    device='cuda',
    num_samples=100,
    output_dir='comparison_results'
):
    """
    Compare multiple models systematically.

    Args:
        models_dict: Dictionary {'model_name': model_object}
        test_loader: Test data
        device: 'cuda' or 'cpu'
        num_samples: Samples per prediction
        output_dir: Where to save results
    """

    print("=" * 70)
    print("MULTI-MODEL COMPARISON")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparator
    comparator = ModelComparator(
        models=models_dict,
        test_data=list(test_loader),
        device=device,
        num_samples=num_samples
    )

    # Run comparison
    results = comparator.run_comparison(
        metrics=[
            'mse', 'mae',
            'calibration_error', 'crps',
            'coverage_90', 'coverage_95',
            'sharpness', 'interval_score',
            'inference_time'
        ],
        save_samples=False
    )

    # Print summary
    summary_df = comparator.print_summary()

    # Save to CSV
    summary_df.to_csv(output_dir / 'comparison_table.csv')

    # Generate report
    comparator.generate_report(output_dir / 'comparison_report')

    print(f"\nComparison results saved to: {output_dir}")

    return results


if __name__ == '__main__':
    """
    Example usage: Evaluate your trained model
    """

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load your trained model
    print("\nLoading model...")

    # Example: Load FNO-EBM with standard EBM
    fno_model = FNO2d(modes1=12, modes2=12, width=64, num_layers=4)
    ebm_model = EBMPotential(input_dim=4, hidden_dims=[128, 256, 256, 128])
    model = FNO_EBM(fno_model, ebm_model)

    # Load weights (adjust path to your checkpoint)
    checkpoint_path = '../checkpoints/best_model.pt'
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.u_fno.load_state_dict(checkpoint['fno_model'])
        model.V_ebm.load_state_dict(checkpoint['ebm_model'])
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model for demonstration")

    # Load test data (adjust to your data loader)
    from datautils import dummy_dataloaders
    from config import Config

    config = Config({
        'batch_size': 4,
        'device': device,
        'n_train': 100,
        'n_test': 20,
        'grid_size': 64
    })

    _, test_loader = dummy_dataloaders(config)

    # ==========================================
    # SINGLE MODEL EVALUATION
    # ==========================================
    print("\n" + "=" * 70)
    print("RUNNING SINGLE MODEL EVALUATION")
    print("=" * 70)

    results = evaluate_single_model(
        model=model,
        test_loader=test_loader,
        device=device,
        num_samples=50,  # Use fewer samples for faster demo
        output_dir='evaluation_results'
    )

    # ==========================================
    # OPTIONAL: MULTI-MODEL COMPARISON
    # ==========================================
    # Uncomment this section if you have baseline models

    # print("\n" + "=" * 70)
    # print("RUNNING MULTI-MODEL COMPARISON")
    # print("=" * 70)

    # # Create different EBM variants
    # kan_ebm = KAN_EBM(input_dim=4, hidden_dims=[64, 128, 64])
    # fno_kan_ebm = FNO_KAN_EBM(fno_width=64, kan_hidden_dims=[128, 64])
    # gnn_ebm = GNN_EBM(node_features=4, hidden_dims=[64, 128, 128, 64])

    # model_variants = {
    #     'FNO-EBM (MLP)': FNO_EBM(fno_model, ebm_model),
    #     'FNO-EBM (KAN)': FNO_EBM(fno_model, kan_ebm),
    #     'FNO-KAN-EBM': FNO_EBM(fno_model, fno_kan_ebm),
    #     'FNO-GNN-EBM': FNO_EBM(fno_model, gnn_ebm),
    # }

    # comparison_results = compare_models(
    #     models_dict=model_variants,
    #     test_loader=test_loader,
    #     device=device,
    #     num_samples=50,
    #     output_dir='comparison_results'
    # )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print("\nCheck the following directories:")
    print("  - evaluation_results/  : Single model evaluation")
    print("  - comparison_results/  : Multi-model comparison (if run)")