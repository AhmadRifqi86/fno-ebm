"""
Training Tracking and Monitoring Tools for UQ Paper

This module provides decorators and utilities for:
- Gradient monitoring (norm, max, min, histogram)
- Weight statistics tracking
- TensorBoard logging integration
- Automatic anomaly detection (NaN, Inf, exploding gradients)

Usage:
    from track import track_gradients, track_weights, GradientTracker

    class MyTrainer:
        def __init__(self, ...):
            self.gradient_tracker = GradientTracker(model, log_dir='./runs')

        @track_gradients
        @track_weights
        def train_step(self, ...):
            ...
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from functools import wraps
import numpy as np
import time
import warnings
from typing import Optional, Dict, Any, Callable
from pathlib import Path


# ============================================================================
# Gradient and Weight Statistics
# ============================================================================

class GradientTracker:
    """
    Comprehensive gradient and weight tracking with TensorBoard integration.

    Tracks:
    - Gradient norms (L2, L∞)
    - Weight statistics (mean, std, min, max)
    - Gradient flow (layer-wise norms)
    - Anomaly detection (NaN, Inf, exploding gradients)
    """
    def __init__(
        self,
        model: nn.Module,
        log_dir: str = './runs',
        experiment_name: Optional[str] = None,
        track_interval: int = 10,
        histogram_interval: int = 100,
        gradient_clip_threshold: float = 50.0,
    ):
        """
        Args:
            model: PyTorch model to track
            log_dir: TensorBoard log directory
            experiment_name: Name for this experiment (auto-generated if None)
            track_interval: Log scalar metrics every N steps
            histogram_interval: Log histograms every N steps
            gradient_clip_threshold: Warn if gradient norm exceeds this
        """
        self.model = model
        self.track_interval = track_interval
        self.histogram_interval = histogram_interval
        self.gradient_clip_threshold = gradient_clip_threshold

        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"

        # Initialize TensorBoard writer
        log_path = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(log_dir=str(log_path))

        # Tracking state
        self.step = 0
        self.anomaly_counts = {
            'nan_gradients': 0,
            'inf_gradients': 0,
            'exploding_gradients': 0,
            'vanishing_gradients': 0,
        }

        # Hook storage
        self.hooks = []
        self.gradient_dict = {}  # Store gradients from hooks

        print(f"[GradientTracker] Initialized with log_dir: {log_path}")

    def register_hooks(self):
        """Register backward hooks to capture gradients."""
        def make_hook(name):
            def hook(grad):
                self.gradient_dict[name] = grad.detach().cpu()
                return None  # Don't modify gradient
            return hook

        # Register hooks for all parameters with gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(make_hook(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradient_dict = {}

    def compute_gradient_stats(self) -> Dict[str, float]:
        """
        Compute gradient statistics across all parameters.

        Returns:
            stats: Dictionary with gradient metrics
        """
        stats = {
            'grad_norm_l2': 0.0,
            'grad_norm_linf': 0.0,
            'grad_mean': 0.0,
            'grad_std': 0.0,
            'num_nan_grads': 0,
            'num_inf_grads': 0,
        }

        total_norm_squared = 0.0
        max_grad = 0.0
        all_grads = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()

                # Check for NaN/Inf
                if torch.isnan(grad).any():
                    stats['num_nan_grads'] += 1
                    warnings.warn(f"NaN gradient detected in {name}")
                if torch.isinf(grad).any():
                    stats['num_inf_grads'] += 1
                    warnings.warn(f"Inf gradient detected in {name}")

                # Compute norms
                param_norm = grad.norm(2).item()
                total_norm_squared += param_norm ** 2
                max_grad = max(max_grad, grad.abs().max().item())

                # Collect for statistics
                all_grads.append(grad.flatten())

        # L2 norm (total gradient norm)
        stats['grad_norm_l2'] = np.sqrt(total_norm_squared)

        # L∞ norm (max absolute gradient)
        stats['grad_norm_linf'] = max_grad

        # Mean and std across all gradients
        if all_grads:
            all_grads_tensor = torch.cat(all_grads)
            stats['grad_mean'] = all_grads_tensor.mean().item()
            stats['grad_std'] = all_grads_tensor.std().item()

        return stats

    def compute_weight_stats(self) -> Dict[str, float]:
        """
        Compute weight statistics across all parameters.

        Returns:
            stats: Dictionary with weight metrics
        """
        stats = {
            'weight_norm_l2': 0.0,
            'weight_mean': 0.0,
            'weight_std': 0.0,
            'weight_min': float('inf'),
            'weight_max': float('-inf'),
            'num_dead_neurons': 0,  # Weights with zero gradient
        }

        total_norm_squared = 0.0
        all_weights = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight = param.detach()

                # Compute norm
                param_norm = weight.norm(2).item()
                total_norm_squared += param_norm ** 2

                # Min/max
                stats['weight_min'] = min(stats['weight_min'], weight.min().item())
                stats['weight_max'] = max(stats['weight_max'], weight.max().item())

                # Dead neurons (zero gradient)
                if param.grad is not None:
                    dead_mask = (param.grad.abs() < 1e-8)
                    stats['num_dead_neurons'] += dead_mask.sum().item()

                # Collect for statistics
                all_weights.append(weight.flatten())

        # L2 norm
        stats['weight_norm_l2'] = np.sqrt(total_norm_squared)

        # Mean and std
        if all_weights:
            all_weights_tensor = torch.cat(all_weights)
            stats['weight_mean'] = all_weights_tensor.mean().item()
            stats['weight_std'] = all_weights_tensor.std().item()

        return stats

    def compute_gradient_flow(self) -> Dict[str, float]:
        """
        Compute layer-wise gradient norms to detect gradient flow issues.

        Returns:
            flow: Dictionary mapping layer names to gradient norms
        """
        flow = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                flow[name] = grad_norm

        return flow

    def detect_anomalies(self, grad_stats: Dict[str, float]):
        """
        Detect training anomalies and issue warnings.

        Args:
            grad_stats: Gradient statistics from compute_gradient_stats()
        """
        # NaN/Inf detection
        if grad_stats['num_nan_grads'] > 0:
            self.anomaly_counts['nan_gradients'] += 1
            warnings.warn(f"Step {self.step}: NaN gradients detected in {grad_stats['num_nan_grads']} parameters!")

        if grad_stats['num_inf_grads'] > 0:
            self.anomaly_counts['inf_gradients'] += 1
            warnings.warn(f"Step {self.step}: Inf gradients detected in {grad_stats['num_inf_grads']} parameters!")

        # Exploding gradients
        if grad_stats['grad_norm_l2'] > self.gradient_clip_threshold:
            self.anomaly_counts['exploding_gradients'] += 1
            warnings.warn(
                f"Step {self.step}: Exploding gradient detected! "
                f"Norm: {grad_stats['grad_norm_l2']:.2f} > {self.gradient_clip_threshold}"
            )

        # Vanishing gradients (very small norm)
        if grad_stats['grad_norm_l2'] < 1e-7 and self.step > 0:
            self.anomaly_counts['vanishing_gradients'] += 1
            warnings.warn(
                f"Step {self.step}: Vanishing gradient detected! "
                f"Norm: {grad_stats['grad_norm_l2']:.2e}"
            )

    def log_scalars(self, prefix: str, stats: Dict[str, float], step: int):
        """
        Log scalar metrics to TensorBoard.

        Args:
            prefix: Metric prefix (e.g., 'gradients', 'weights')
            stats: Dictionary of metrics
            step: Global step number
        """
        for key, value in stats.items():
            self.writer.add_scalar(f'{prefix}/{key}', value, step)

    def log_histograms(self, step: int):
        """
        Log weight and gradient histograms to TensorBoard.

        Args:
            step: Global step number
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Weight histogram
                self.writer.add_histogram(f'weights/{name}', param.detach().cpu(), step)

                # Gradient histogram
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad.detach().cpu(), step)

    def log_gradient_flow_chart(self, gradient_flow: Dict[str, float], step: int):
        """
        Log gradient flow as bar chart to TensorBoard.

        Args:
            gradient_flow: Layer-wise gradient norms
            step: Global step number
        """
        # Create figure for gradient flow
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        layers = list(gradient_flow.keys())
        norms = list(gradient_flow.values())

        ax.bar(range(len(layers)), norms)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'Gradient Flow - Step {step}')
        ax.set_yscale('log')
        plt.tight_layout()

        self.writer.add_figure('gradient_flow', fig, step)
        plt.close(fig)

    def track(self, loss: Optional[torch.Tensor] = None, custom_metrics: Optional[Dict[str, float]] = None):
        """
        Main tracking function - call after backward() but before optimizer.step().

        Args:
            loss: Training loss (optional)
            custom_metrics: Additional metrics to log (optional)
        """
        self.step += 1

        # Compute statistics
        grad_stats = self.compute_gradient_stats()
        weight_stats = self.compute_weight_stats()

        # Detect anomalies
        self.detect_anomalies(grad_stats)

        # Log scalars at specified interval
        if self.step % self.track_interval == 0:
            self.log_scalars('gradients', grad_stats, self.step)
            self.log_scalars('weights', weight_stats, self.step)

            if loss is not None:
                self.writer.add_scalar('loss/train', loss.item(), self.step)

            if custom_metrics:
                self.log_scalars('custom', custom_metrics, self.step)

        # Log histograms at specified interval
        if self.step % self.histogram_interval == 0:
            self.log_histograms(self.step)

            # Gradient flow chart
            gradient_flow = self.compute_gradient_flow()
            self.log_gradient_flow_chart(gradient_flow, self.step)

        # Print summary
        if self.step % self.track_interval == 0:
            print(
                f"[Step {self.step}] "
                f"Grad Norm: {grad_stats['grad_norm_l2']:.2e}, "
                f"Weight Norm: {weight_stats['weight_norm_l2']:.2e}, "
                f"NaN: {grad_stats['num_nan_grads']}, "
                f"Inf: {grad_stats['num_inf_grads']}"
            )

    def close(self):
        """Close TensorBoard writer and cleanup."""
        self.remove_hooks()
        self.writer.close()
        print(f"[GradientTracker] Closed. Anomalies: {self.anomaly_counts}")


# ============================================================================
# Decorator Functions
# ============================================================================

def track_gradients(func: Callable) -> Callable:
    """
    Decorator to automatically track gradients after training step.

    Usage:
        class Trainer:
            @track_gradients
            def train_step(self, x, y):
                loss = ...
                loss.backward()
                return loss

    Note: Requires self.gradient_tracker to be a GradientTracker instance
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Execute training step
        result = func(self, *args, **kwargs)

        # Track gradients if gradient_tracker exists
        if hasattr(self, 'gradient_tracker') and self.gradient_tracker is not None:
            # Extract loss from result
            loss = None
            if isinstance(result, tuple):
                # Assume first element is loss
                if isinstance(result[0], torch.Tensor):
                    loss = result[0]
            elif isinstance(result, torch.Tensor):
                loss = result

            # Track
            self.gradient_tracker.track(loss=loss)

        return result

    return wrapper


def track_weights(func: Callable) -> Callable:
    """
    Decorator to track weight statistics before training step.

    Usage:
        class Trainer:
            @track_weights
            def train_step(self, x, y):
                ...

    Note: Requires self.gradient_tracker to be a GradientTracker instance
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Track weights before step
        if hasattr(self, 'gradient_tracker') and self.gradient_tracker is not None:
            weight_stats = self.gradient_tracker.compute_weight_stats()

            # Log at interval
            if self.gradient_tracker.step % self.gradient_tracker.track_interval == 0:
                self.gradient_tracker.log_scalars(
                    'weights_pre_step',
                    weight_stats,
                    self.gradient_tracker.step
                )

        # Execute training step
        result = func(self, *args, **kwargs)

        return result

    return wrapper


def log_to_tensorboard(metric_prefix: str = 'custom'):
    """
    Decorator to log custom metrics to TensorBoard.

    Usage:
        @log_to_tensorboard(metric_prefix='fno_metrics')
        def train_step(self, x, y):
            ...
            return loss, {'metric1': value1, 'metric2': value2}

    Note: Function must return (loss, metrics_dict) or just loss
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            # Extract metrics
            if hasattr(self, 'gradient_tracker') and self.gradient_tracker is not None:
                metrics = None
                if isinstance(result, tuple) and len(result) == 2:
                    if isinstance(result[1], dict):
                        metrics = result[1]

                # Log metrics
                if metrics and self.gradient_tracker.step % self.gradient_tracker.track_interval == 0:
                    self.gradient_tracker.log_scalars(
                        metric_prefix,
                        metrics,
                        self.gradient_tracker.step
                    )

            return result

        return wrapper

    return decorator


# ============================================================================
# Standalone Tracking Functions
# ============================================================================

def log_gradient_norm(model: nn.Module, writer: SummaryWriter, step: int, prefix: str = 'gradients'):
    """
    Standalone function to log gradient norm.

    Args:
        model: PyTorch model
        writer: TensorBoard SummaryWriter
        step: Global step number
        prefix: Metric prefix
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(2).item()
            total_norm += param_norm ** 2

    total_norm = np.sqrt(total_norm)
    writer.add_scalar(f'{prefix}/total_norm', total_norm, step)
    return total_norm


def log_weight_norm(model: nn.Module, writer: SummaryWriter, step: int, prefix: str = 'weights'):
    """
    Standalone function to log weight norm.

    Args:
        model: PyTorch model
        writer: TensorBoard SummaryWriter
        step: Global step number
        prefix: Metric prefix
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            param_norm = param.norm(2).item()
            total_norm += param_norm ** 2

    total_norm = np.sqrt(total_norm)
    writer.add_scalar(f'{prefix}/total_norm', total_norm, step)
    return total_norm


def check_for_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check tensor for NaN or Inf values.

    Args:
        tensor: Tensor to check
        name: Name for warning message

    Returns:
        has_anomaly: True if NaN/Inf detected
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        warnings.warn(f"Anomaly detected in {name}: NaN={has_nan}, Inf={has_inf}")
        return True

    return False


# ============================================================================
# Utility Functions
# ============================================================================

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def log_learning_rate(optimizer: torch.optim.Optimizer, writer: SummaryWriter, step: int):
    """Log current learning rate to TensorBoard."""
    lr = get_lr(optimizer)
    writer.add_scalar('training/learning_rate', lr, step)


def log_model_summary(model: nn.Module, writer: SummaryWriter):
    """
    Log model architecture summary to TensorBoard.

    Args:
        model: PyTorch model
        writer: TensorBoard SummaryWriter
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create summary text
    summary = f"""
    Model Summary:
    - Total Parameters: {total_params:,}
    - Trainable Parameters: {trainable_params:,}
    - Non-trainable Parameters: {total_params - trainable_params:,}

    Architecture:
    {model}
    """

    writer.add_text('model/summary', summary, 0)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")