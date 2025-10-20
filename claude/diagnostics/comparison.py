"""
Model Comparison Framework

This module provides tools for systematic comparison of probabilistic models:
- ModelComparator: Compare multiple models on same data
- BenchmarkSuite: Run standardized benchmark problems
- Report generation with tables and plots

Usage:
    >>> from diagnostics import ModelComparator
    >>> comparator = ModelComparator(models_dict, test_data)
    >>> results = comparator.run_comparison()
    >>> comparator.generate_report('comparison_report.html')
"""

import numpy as np
import torch
from typing import Dict, List, Callable, Optional, Tuple
import pandas as pd
from pathlib import Path
import json
import time

from .uncertainty_metrics import (
    calibration_error,
    continuous_ranked_probability_score,
    negative_log_likelihood,
    coverage_analysis,
    sharpness_score,
    interval_score
)
from .physics_metrics import (
    compute_pde_residual,
    boundary_condition_violation,
    physics_consistency_score
)
from .visualization import (
    plot_calibration_curve,
    plot_reliability_diagram,
    plot_comparison_radar,
    plot_uncertainty_heatmap
)


class ModelComparator:
    """
    Compare multiple probabilistic models systematically.

    Example:
        >>> models = {
        ...     'FNO-EBM': fno_ebm_model,
        ...     'Baseline': baseline_model,
        ... }
        >>> comparator = ModelComparator(models, test_loader)
        >>> results = comparator.run_comparison()
        >>> comparator.print_summary()
    """

    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        test_data: List[Tuple[torch.Tensor, torch.Tensor]],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_samples: int = 100
    ):
        """
        Initialize comparator.

        Args:
            models: Dictionary mapping model names to model objects
            test_data: List of (input, ground_truth) tuples
            device: Device to run on
            num_samples: Number of samples to draw per prediction
        """
        self.models = models
        self.test_data = test_data
        self.device = device
        self.num_samples = num_samples
        self.results = {}

    def run_comparison(
        self,
        metrics: List[str] = None,
        save_samples: bool = False
    ) -> Dict:
        """
        Run full comparison across all models.

        Args:
            metrics: List of metric names to compute (if None, compute all)
            save_samples: Whether to save all samples (memory intensive)

        Returns:
            results: Dictionary with detailed results per model
        """
        if metrics is None:
            metrics = [
                'mse', 'mae', 'calibration_error', 'crps',
                'coverage_90', 'coverage_95', 'sharpness',
                'interval_score', 'inference_time'
            ]

        print("=" * 70)
        print("Running Model Comparison")
        print("=" * 70)

        for model_name, model in self.models.items():
            print(f"\nEvaluating: {model_name}")
            print("-" * 70)

            model.eval()
            model_results = {
                'predictions': [],
                'samples': [] if save_samples else None,
                'inference_times': []
            }

            # Collect predictions
            for x, y_true in self.test_data:
                x = x.to(self.device)
                y_true = y_true.to(self.device)

                # Time inference
                start_time = time.time()

                with torch.no_grad():
                    # Generate samples (model-specific logic)
                    samples = self._generate_samples(model, x, model_name)

                inference_time = time.time() - start_time
                model_results['inference_times'].append(inference_time)

                # Store results
                pred_mean = samples.mean(dim=0)
                pred_std = samples.std(dim=0)

                model_results['predictions'].append({
                    'input': x.cpu(),
                    'ground_truth': y_true.cpu(),
                    'mean': pred_mean.cpu(),
                    'std': pred_std.cpu(),
                    'samples': samples.cpu() if save_samples else None
                })

            # Compute metrics
            model_results['metrics'] = self._compute_metrics(model_results, metrics)

            self.results[model_name] = model_results

            # Print summary
            self._print_model_summary(model_name, model_results)

        print("\n" + "=" * 70)
        print("Comparison Complete")
        print("=" * 70)

        return self.results

    def _generate_samples(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        model_name: str
    ) -> torch.Tensor:
        """
        Generate samples from model (handles different model types).

        Args:
            model: The model
            x: Input
            model_name: Name of model (used to infer sampling strategy)

        Returns:
            samples: (num_samples, *shape)
        """
        samples = []

        # Try to use model's sampling method if available
        if hasattr(model, 'sample'):
            for _ in range(self.num_samples):
                sample = model.sample(x)
                samples.append(sample)

        # For FNO_EBM, use Langevin dynamics
        elif 'ebm' in model_name.lower() or 'energy' in model_name.lower():
            from inference import langevin_dynamics
            for _ in range(self.num_samples):
                sample = langevin_dynamics(
                    model, x,
                    num_steps=200,
                    step_size=0.005,
                    device=self.device
                )
                samples.append(sample)

        # For dropout-based models
        elif hasattr(model, 'train'):
            model.train()  # Enable dropout
            for _ in range(self.num_samples):
                sample = model(x)
                samples.append(sample)
            model.eval()

        # Default: deterministic
        else:
            pred = model(x)
            samples = [pred] * self.num_samples

        return torch.stack(samples, dim=0)

    def _compute_metrics(
        self,
        model_results: Dict,
        metric_names: List[str]
    ) -> Dict[str, float]:
        """
        Compute all requested metrics.

        Args:
            model_results: Results for one model
            metric_names: List of metrics to compute

        Returns:
            metrics: Dictionary of metric values
        """
        metrics = {}

        # Aggregate predictions
        all_truths = []
        all_means = []
        all_stds = []
        all_samples_list = []

        for pred in model_results['predictions']:
            all_truths.append(pred['ground_truth'])
            all_means.append(pred['mean'])
            all_stds.append(pred['std'])

            # Reconstruct samples from mean and std if not saved
            if pred['samples'] is not None:
                all_samples_list.append(pred['samples'])

        all_truths = torch.cat(all_truths, dim=0).numpy()
        all_means = torch.cat(all_means, dim=0).numpy()
        all_stds = torch.cat(all_stds, dim=0).numpy()

        # Compute metrics
        if 'mse' in metric_names:
            metrics['mse'] = np.mean((all_means - all_truths) ** 2)

        if 'mae' in metric_names:
            metrics['mae'] = np.mean(np.abs(all_means - all_truths))

        if 'calibration_error' in metric_names and len(all_samples_list) > 0:
            all_samples = torch.cat(all_samples_list, dim=1).numpy()
            metrics['calibration_error'] = calibration_error(all_samples, all_truths)

        if 'crps' in metric_names and len(all_samples_list) > 0:
            all_samples = torch.cat(all_samples_list, dim=1).numpy()
            metrics['crps'] = continuous_ranked_probability_score(all_samples, all_truths)

        if 'coverage_90' in metric_names:
            lower = all_means - 1.645 * all_stds
            upper = all_means + 1.645 * all_stds
            metrics['coverage_90'] = np.mean((all_truths >= lower) & (all_truths <= upper))

        if 'coverage_95' in metric_names:
            lower = all_means - 1.96 * all_stds
            upper = all_means + 1.96 * all_stds
            metrics['coverage_95'] = np.mean((all_truths >= lower) & (all_truths <= upper))

        if 'sharpness' in metric_names and len(all_samples_list) > 0:
            all_samples = torch.cat(all_samples_list, dim=1).numpy()
            metrics['sharpness'] = sharpness_score(all_samples, alpha=0.9)

        if 'interval_score' in metric_names and len(all_samples_list) > 0:
            all_samples = torch.cat(all_samples_list, dim=1).numpy()
            metrics['interval_score'] = interval_score(all_samples, all_truths, alpha=0.9)

        if 'inference_time' in metric_names:
            metrics['inference_time'] = np.mean(model_results['inference_times'])

        return metrics

    def _print_model_summary(self, model_name: str, results: Dict):
        """Print summary for one model."""
        metrics = results['metrics']

        print(f"\nMetrics for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:20s}: {value:.6f}")

    def print_summary(self):
        """Print comparison table."""
        if not self.results:
            print("No results yet. Run run_comparison() first.")
            return

        # Create DataFrame
        data = []
        for model_name, results in self.results.items():
            row = {'Model': model_name}
            row.update(results['metrics'])
            data.append(row)

        df = pd.DataFrame(data)
        df = df.set_index('Model')

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80)

        return df

    def generate_report(
        self,
        output_path: str = 'comparison_report',
        format: str = 'html'
    ):
        """
        Generate comprehensive comparison report.

        Args:
            output_path: Path for output file (without extension)
            format: 'html', 'pdf', or 'markdown'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create summary table
        df = self.print_summary()

        # Generate visualizations
        # TODO: Implement full HTML report generation
        # For now, save metrics to JSON

        report_data = {
            'models': list(self.models.keys()),
            'metrics': {
                model_name: results['metrics']
                for model_name, results in self.results.items()
            }
        }

        with open(f"{output_path}.json", 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nReport saved to: {output_path}.json")


class BenchmarkSuite:
    """
    Standard benchmark problems for operator learning with UQ.
    """

    @staticmethod
    def darcy_flow(
        n_train: int = 1000,
        n_test: int = 200,
        resolution: int = 64,
        noise_level: float = 0.01
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Generate Darcy flow benchmark.

        Returns:
            train_loader, test_loader
        """
        # TODO: Implement Darcy flow data generation
        # This is a placeholder
        pass

    @staticmethod
    def burgers_equation(
        n_train: int = 1000,
        n_test: int = 200,
        resolution: int = 256,
        viscosity: float = 0.01
    ):
        """Generate Burgers equation benchmark."""
        pass


def generate_comparison_report(
    results: Dict,
    output_dir: str = './reports'
):
    """
    Generate comprehensive comparison report with plots.

    Args:
        results: Results from ModelComparator
        output_dir: Directory to save reports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data for plots
    model_names = list(results.keys())

    # TODO: Generate full report with all plots
    # - Calibration curves
    # - Reliability diagrams
    # - Radar charts
    # - Summary tables

    print(f"Report generated in: {output_dir}")