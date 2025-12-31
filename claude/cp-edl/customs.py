"""
Custom components for UQ Paper: KAN-EBM for Neural Operator Uncertainty Quantification

This module contains:
1. Custom loss functions for FNO and KAN-EBM training
2. Calibration metrics (ECE, NLL, CRPS)
3. Utility functions for uncertainty quantification
4. Custom schedulers and training utilities
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, Any, Optional, Tuple, List


# ============================================================================
# Custom Loss Functions for FNO Training
# ============================================================================

def gradient_penalty_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Gradient penalty loss to combat FNO over-smoothing.

    Penalizes predictions where spatial gradients don't match target gradients.
    This encourages the model to preserve sharp boundaries and fine details.

    Args:
        pred: Predicted field (batch, nx, ny) or (batch, nx, ny, 1)
        target: Ground truth field (batch, nx, ny) or (batch, nx, ny, 1)

    Returns:
        loss: Mean absolute difference between predicted and target gradients

    Usage:
        In FNO training loop:
        >>> fno_loss = F.mse_loss(pred, target) + 0.1 * gradient_penalty_loss(pred, target)

    Notes:
        - Weight (0.1) controls strength: higher = sharper but more noise
        - Use 0.05-0.15 range for reaction-diffusion
        - Use 0.2-0.3 for turbulence or sharp shocks
    """
    # Squeeze if needed
    if pred.dim() == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if target.dim() == 4 and target.shape[-1] == 1:
        target = target.squeeze(-1)

    # Compute spatial gradients using finite differences
    # X-direction: shape (batch, nx-1, ny)
    pred_grad_x = torch.diff(pred, dim=1)
    target_grad_x = torch.diff(target, dim=1)

    # Y-direction: shape (batch, nx, ny-1)
    pred_grad_y = torch.diff(pred, dim=2)
    target_grad_y = torch.diff(target, dim=2)

    # L1 loss on gradients (more robust than L2)
    loss_x = F.l1_loss(pred_grad_x, target_grad_x)
    loss_y = F.l1_loss(pred_grad_y, target_grad_y)

    return (loss_x + loss_y) / 2


def combined_fno_loss(pred: torch.Tensor,
                      target: torch.Tensor,
                      weight_mse: float = 1.0,
                      weight_grad: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined FNO loss with MSE and gradient penalty.

    Convenience function that combines standard MSE loss with gradient penalty
    to prevent over-smoothing while maintaining accuracy.

    Args:
        pred: FNO predictions (batch, nx, ny) or (batch, nx, ny, 1)
        target: Ground truth (batch, nx, ny) or (batch, nx, ny, 1)
        weight_mse: Weight for MSE loss (default: 1.0)
        weight_grad: Weight for gradient penalty (default: 0.1)

    Returns:
        loss: Combined loss
        loss_dict: Dictionary with individual loss components for logging

    Usage:
        In FNO training loop:
        >>> loss, loss_dict = combined_fno_loss(pred, target, weight_grad=0.15)
        >>> loss.backward()
        >>> # Log individual components
        >>> print(f"MSE: {loss_dict['mse']:.4f}, Grad: {loss_dict['grad']:.4f}")

    Recommended Weights:
        - Smooth PDEs (heat, diffusion): weight_grad=0.05-0.1
        - Medium complexity (reaction-diffusion): weight_grad=0.1-0.15
        - High frequency (turbulence, shocks): weight_grad=0.2-0.3
    """
    # MSE loss
    mse_loss = F.mse_loss(pred, target)

    # Gradient penalty
    grad_loss = gradient_penalty_loss(pred, target)

    # Combined
    total_loss = weight_mse * mse_loss + weight_grad * grad_loss

    loss_dict = {
        'mse': mse_loss.item(),
        'grad': grad_loss.item(),
        'total': total_loss.item()
    }

    return total_loss, loss_dict


# ============================================================================
# Custom Loss Functions for KAN-EBM Training
# ============================================================================

def error_aware_ebm_loss(ebm_std: torch.Tensor,
                         fno_pred: torch.Tensor,
                         ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Error-aware EBM loss for calibrated uncertainty quantification.

    Teaches the EBM to predict HIGH uncertainty where FNO makes large errors,
    and LOW uncertainty where FNO predictions are accurate. This creates
    spatially-structured uncertainty maps instead of uniform noise.

    Args:
        ebm_std: EBM predicted standard deviation (batch, nx, ny)
        fno_pred: FNO predictions (batch, nx, ny) or (batch, nx, ny, 1)
        ground_truth: True solution (batch, nx, ny) or (batch, nx, ny, 1)

    Returns:
        loss: Calibration loss encouraging correlation between error and uncertainty

    Usage:
        In EBM training loop (after score matching loss):
        >>> ebm_loss = score_matching_loss + 0.5 * error_aware_ebm_loss(ebm_std, fno_pred, gt)

    Theory:
        - Computes actual FNO error: |pred - truth|
        - Normalizes both error and std to [0, 1] range
        - Penalizes when: high error but low std (miscalibrated)
        - Reward when: error and std are correlated (well-calibrated)

    Expected Results:
        - Before: uncertainty is uniform noise (~0.02-0.03 everywhere)
        - After: uncertainty is structured (high at boundaries, low in smooth regions)
        - Calibration plot correlation: 0.05 → 0.6-0.8

    Notes:
        - Weight (0.5) is a good default, adjust to 0.3-0.7
        - Requires access to ground truth during EBM training
        - Use ONLY during training, not inference
    """
    # Squeeze if needed
    if fno_pred.dim() == 4 and fno_pred.shape[-1] == 1:
        fno_pred = fno_pred.squeeze(-1)
    if ground_truth.dim() == 4 and ground_truth.shape[-1] == 1:
        ground_truth = ground_truth.squeeze(-1)
    if ebm_std.dim() == 4 and ebm_std.shape[-1] == 1:
        ebm_std = ebm_std.squeeze(-1)

    # Compute actual FNO error (per pixel)
    actual_error = torch.abs(fno_pred - ground_truth)  # (batch, nx, ny)

    # Normalize to [0, 1] using percentile ranking
    # This makes the loss invariant to absolute error/std scales
    error_max = actual_error.view(actual_error.size(0), -1).max(dim=1, keepdim=True)[0]
    std_max = ebm_std.view(ebm_std.size(0), -1).max(dim=1, keepdim=True)[0]

    # Reshape back to spatial dimensions
    error_max = error_max.view(-1, 1, 1) + 1e-8
    std_max = std_max.view(-1, 1, 1) + 1e-8

    # Normalized values (0 = best, 1 = worst)
    error_norm = actual_error / error_max
    std_norm = ebm_std / std_max

    # MSE loss: EBM std should match FNO error distribution
    calibration_loss = F.mse_loss(std_norm, error_norm)

    return calibration_loss


def weighted_score_matching_loss(ebm_model: nn.Module,
                                  u_clean: torch.Tensor,
                                  x_coords: torch.Tensor,
                                  sigmas: List[float] = [0.01, 0.02, 0.05],
                                  weights: Optional[Dict[float, float]] = None,
                                  return_diagnostics: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Weighted score matching loss with balanced multi-scale learning.

    Standard score matching trains on multiple noise levels but small sigmas
    (high frequency) dominate the loss, preventing coarse-scale learning.
    This version uses inverse weighting: smaller sigma → smaller weight.

    Args:
        ebm_model: Energy-based model (outputs score function)
        u_clean: Clean field samples (batch, nx, ny, channels)
        x_coords: Input coordinates (batch, nx, ny, coord_channels)
        sigmas: List of noise levels to train on (default: [0.01, 0.02, 0.05])
        weights: Optional custom weights dict {sigma: weight}
                 If None, uses inverse weighting: {0.01: 0.2, 0.02: 0.3, 0.05: 0.5}
        return_diagnostics: If True, returns (loss, diagnostics_dict)

    Returns:
        loss: Weighted score matching loss (scalar)
        diagnostics: Dict with per-level losses and score norms (if return_diagnostics=True)

    Usage:
        Replace standard score matching in trainer.py:
        >>> loss = weighted_score_matching_loss(ebm, u_clean, x_coords, sigmas=[0.01, 0.02, 0.05])

        With diagnostics for debugging:
        >>> loss, diag = weighted_score_matching_loss(ebm, u_clean, x_coords, return_diagnostics=True)
        >>> print(f"σ=0.01 loss: {diag['losses'][0.01]:.2f}")
        >>> print(f"Score norm ratio: {diag['norm_ratios'][0.01]:.2f}")

    Theory:
        Standard loss: L = Σ_σ ||s_θ(u+ε, x) - (-ε/σ²)||²
        Problem: Small σ → large ||ε/σ²|| → dominates gradient
        Solution: L = Σ_σ w_σ · ||s_θ(u+ε, x) - (-ε/σ²)||²
                  where w_σ ∝ σ² (inverse of target score magnitude)

    Expected Improvements:
        - Score norm ratio: 0.04 → 0.7-0.9 (much better match)
        - Train loss: 14,000 → 500-1,000 (better convergence)
        - Learning: Coarse features first, then fine details
        - Convergence: 60 epochs → 200-300 epochs for full convergence

    Notes:
        - Default weights {0.01:0.2, 0.02:0.3, 0.05:0.5} work for most cases
        - For very high-frequency data, use {0.01:0.1, 0.02:0.3, 0.05:0.6}
        - Monitor norm_ratio: should be >0.5 for good convergence
    """
    # Default inverse weighting (smaller sigma = smaller weight)
    if weights is None:
        weights = {0.01: 0.2, 0.02: 0.3, 0.05: 0.5}
        # Normalize to sum to 1.0
        weight_sum = sum(weights.values())
        weights = {k: v/weight_sum for k, v in weights.items()}

    total_loss = 0.0
    diagnostics = {'losses': {}, 'score_norms': {}, 'target_norms': {}, 'norm_ratios': {}}

    for sigma in sigmas:
        # Add noise to the field
        noise = torch.randn_like(u_clean) * sigma
        u_noisy = u_clean + noise
        u_noisy.requires_grad_(True)

        # Compute energy of noisy field
        energy = ebm_model(u_noisy, x_coords)

        # Predict score: s_θ(u_noisy, x) = -∇_u E(u_noisy, x)
        predicted_score = -torch.autograd.grad(
            outputs=energy.sum(),
            inputs=u_noisy,
            create_graph=True  # Need gradients for backprop
        )[0]

        # Target score: -ε/σ²
        target_score = -noise / (sigma ** 2)

        # MSE loss for this noise level
        level_loss = F.mse_loss(predicted_score, target_score)

        # Apply weight
        weight = weights.get(sigma, 1.0 / len(sigmas))
        weighted_loss = weight * level_loss
        total_loss += weighted_loss

        # Collect diagnostics
        if return_diagnostics:
            diagnostics['losses'][sigma] = level_loss.item()
            diagnostics['score_norms'][sigma] = predicted_score.norm().item()
            diagnostics['target_norms'][sigma] = target_score.norm().item()
            ratio = predicted_score.norm().item() / (target_score.norm().item() + 1e-8)
            diagnostics['norm_ratios'][sigma] = ratio

    if return_diagnostics:
        return total_loss, diagnostics
    else:
        return total_loss, None


def combined_ebm_loss(ebm_model: nn.Module,
                      u_clean: torch.Tensor,
                      x_coords: torch.Tensor,
                      fno_pred: torch.Tensor,
                      ground_truth: torch.Tensor,
                      weight_score: float = 1.0,
                      weight_calibration: float = 0.5,
                      sigmas: List[float] = [0.01, 0.02, 0.05],
                      energy_reg_weight: float = 0.001) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined EBM loss with score matching and error-aware calibration.

    Trains EBM to both denoise (score matching) and predict meaningful
    uncertainty (calibration). This produces spatially-structured uncertainty
    maps that correlate with FNO prediction errors.

    Args:
        ebm_model: Energy-based model
        u_clean: Clean FNO predictions / field (batch, nx, ny, channels)
        x_coords: Input coordinates (batch, nx, ny, coord_channels)
        fno_pred: FNO predictions (batch, nx, ny) - for calibration
        ground_truth: True solution (batch, nx, ny) - for calibration
        weight_score: Weight for score matching loss (default: 1.0)
        weight_calibration: Weight for error-aware loss (default: 0.5)
        sigmas: Noise levels for score matching (default: [0.01, 0.02, 0.05])
        energy_reg_weight: Weight for energy regularization (default: 0.001)

    Returns:
        loss: Combined loss
        loss_dict: Dictionary with individual loss components for logging

    Usage:
        In EBM training loop:
        >>> loss, loss_dict = combined_ebm_loss(ebm, u_clean, x_coords, fno_pred, gt)
        >>> loss.backward()
        >>> # Log components
        >>> print(f"Score: {loss_dict['score']:.2f}, Calib: {loss_dict['calibration']:.4f}")

    Expected Behavior:
        - First 50 epochs: Score matching dominates, learns denoising
        - After 50 epochs: Calibration kicks in, learns error patterns
        - Final result: Uncertainty correlates with FNO errors (r > 0.6)

    Recommended Weights:
        - Early training: weight_calibration=0.3 (focus on denoising)
        - Late training: weight_calibration=0.7 (focus on calibration)
        - Can use curriculum: start 0.3, increase to 0.7 over epochs
    """
    # 1. Weighted score matching loss
    score_loss, score_diag = weighted_score_matching_loss(
        ebm_model, u_clean, x_coords, sigmas=sigmas, return_diagnostics=True
    )

    # 2. Energy regularization (prevent energy from growing unbounded)
    energy = ebm_model(u_clean, x_coords)
    energy_reg = torch.mean(energy ** 2)

    # 3. Error-aware calibration loss
    # Get EBM uncertainty prediction
    # Compute score for uncertainty estimation (need gradients for calibration)
    u_clean_copy = u_clean.detach().clone()
    u_clean_copy.requires_grad_(True)

    energy_for_score = ebm_model(u_clean_copy, x_coords)
    ebm_score = -torch.autograd.grad(
        outputs=energy_for_score.sum(),
        inputs=u_clean_copy,
        create_graph=True  # Need graph for backprop through calibration loss
    )[0]

    # Compute std from score norm (keep gradient flow)
    ebm_std = torch.norm(ebm_score, dim=-1)  # (batch, nx, ny)

    calibration_loss = error_aware_ebm_loss(ebm_std, fno_pred, ground_truth)

    # Combined loss
    total_loss = (
        weight_score * score_loss +
        energy_reg_weight * energy_reg +
        weight_calibration * calibration_loss
    )

    loss_dict = {
        'score': score_loss.item(),
        'energy_reg': energy_reg.item(),
        'calibration': calibration_loss.item(),
        'total': total_loss.item(),
        # Score matching diagnostics
        'score_norm_ratio_0.01': score_diag['norm_ratios'].get(0.01, 0.0) if score_diag else 0.0,
        'score_norm_ratio_0.05': score_diag['norm_ratios'].get(0.05, 0.0) if score_diag else 0.0,
    }

    return total_loss, loss_dict


# ============================================================================
# Calibration Metrics for UQ Evaluation
# ============================================================================

def expected_calibration_error(predicted_std: torch.Tensor,
                               actual_error: torch.Tensor,
                               n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) for regression.

    ECE measures the gap between predicted uncertainty and actual error.
    A well-calibrated model should have predicted_std ≈ actual_error.

    Args:
        predicted_std: Predicted standard deviation (N,)
        actual_error: Actual prediction error |pred - truth| (N,)
        n_bins: Number of bins for calibration curve

    Returns:
        ece: Expected calibration error (scalar, lower is better)

    Target: ECE < 0.05 (well-calibrated)
    """
    predicted_std = predicted_std.cpu().numpy().flatten()
    actual_error = actual_error.cpu().numpy().flatten()

    # Create bins based on predicted uncertainty
    bin_boundaries = np.linspace(predicted_std.min(), predicted_std.max(), n_bins + 1)

    ece = 0.0
    total_samples = len(predicted_std)

    for i in range(n_bins):
        # Find samples in this bin
        mask = (predicted_std >= bin_boundaries[i]) & (predicted_std < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Include upper boundary in last bin
            mask = (predicted_std >= bin_boundaries[i]) & (predicted_std <= bin_boundaries[i + 1])

        if mask.sum() == 0:
            continue

        # Average predicted uncertainty in this bin
        avg_predicted_std = predicted_std[mask].mean()

        # Average actual error in this bin (empirical uncertainty)
        avg_actual_error = actual_error[mask].mean()

        # Weighted contribution to ECE
        weight = mask.sum() / total_samples
        ece += weight * abs(avg_predicted_std - avg_actual_error)

    return ece


def negative_log_likelihood(samples: torch.Tensor,
                            ground_truth: torch.Tensor) -> float:
    """
    Compute Negative Log-Likelihood (NLL) for Gaussian assumption.

    Assumes samples are drawn from Gaussian: u ~ N(μ, σ²)
    NLL = -log p(u_true | μ, σ)

    Args:
        samples: Posterior samples (n_samples, batch, nx, ny)
        ground_truth: True solution (batch, nx, ny)

    Returns:
        nll: Negative log-likelihood (scalar, lower is better)
    """
    mean = samples.mean(dim=0)  # (batch, nx, ny)
    std = samples.std(dim=0) + 1e-8  # (batch, nx, ny)

    # Gaussian NLL: 0.5 * log(2π) + log(σ) + (x - μ)² / (2σ²)
    nll = (
        0.5 * np.log(2 * np.pi) +
        torch.log(std) +
        (ground_truth - mean) ** 2 / (2 * std ** 2)
    )

    return nll.mean().item()


def continuous_ranked_probability_score(samples: torch.Tensor,
                                        ground_truth: torch.Tensor) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).

    CRPS measures both sharpness and calibration of the predictive distribution.
    Lower CRPS = better uncertainty quantification.

    Args:
        samples: Posterior samples (n_samples, batch, nx, ny)
        ground_truth: True solution (batch, nx, ny)

    Returns:
        crps: Continuous ranked probability score (scalar, lower is better)

    Formula:
        CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
        where X, X' are independent samples, y is ground truth
    """
    n_samples = samples.shape[0]

    # E[|X - y|]: expected error
    error_term = torch.abs(samples - ground_truth.unsqueeze(0)).mean(dim=0)

    # E[|X - X'|]: expected distance between samples (sharpness)
    # Randomly pair samples
    idx1 = torch.randperm(n_samples)[:n_samples // 2]
    idx2 = torch.randperm(n_samples)[:n_samples // 2]
    diversity_term = torch.abs(samples[idx1] - samples[idx2]).mean(dim=0)

    crps = error_term - 0.5 * diversity_term

    return crps.mean().item()


def prediction_interval_coverage(samples: torch.Tensor,
                                 ground_truth: torch.Tensor,
                                 confidence_level: float = 0.95) -> float:
    """
    Compute prediction interval coverage.

    For a well-calibrated model, α% confidence intervals should contain
    the ground truth α% of the time.

    Args:
        samples: Posterior samples (n_samples, batch, nx, ny)
        ground_truth: True solution (batch, nx, ny)
        confidence_level: Confidence level (e.g., 0.95 for 95% intervals)

    Returns:
        coverage: Fraction of ground truth points inside prediction interval
                  Should be ≈ confidence_level for calibrated model

    Target: coverage ≈ confidence_level ± 0.05
    """
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    # Compute quantiles
    lower = torch.quantile(samples, lower_quantile, dim=0)
    upper = torch.quantile(samples, upper_quantile, dim=0)

    # Check if ground truth is inside interval
    inside = ((ground_truth >= lower) & (ground_truth <= upper)).float()

    coverage = inside.mean().item()

    return coverage


def prediction_interval_sharpness(samples: torch.Tensor,
                                  confidence_level: float = 0.95) -> float:
    """
    Compute prediction interval sharpness (average width).

    Among calibrated models, prefer sharper (narrower) intervals.
    Sharper = more confident predictions.

    Args:
        samples: Posterior samples (n_samples, batch, nx, ny)
        confidence_level: Confidence level (e.g., 0.95 for 95% intervals)

    Returns:
        sharpness: Average width of prediction intervals

    Note: Lower sharpness is better (more confident), but only meaningful
          if model is calibrated (coverage ≈ confidence_level).
    """
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    # Compute quantiles
    lower = torch.quantile(samples, lower_quantile, dim=0)
    upper = torch.quantile(samples, upper_quantile, dim=0)

    # Average interval width
    sharpness = (upper - lower).mean().item()

    return sharpness


def uncertainty_error_correlation(predicted_std: torch.Tensor,
                                   actual_error: torch.Tensor) -> float:
    """
    Compute Pearson correlation between predicted uncertainty and actual error.

    A good uncertainty model should have high correlation:
    high uncertainty → high error, low uncertainty → low error.

    Args:
        predicted_std: Predicted standard deviation (N,)
        actual_error: Actual prediction error |pred - truth| (N,)

    Returns:
        correlation: Pearson correlation coefficient

    Target: ρ > 0.7 (high correlation means uncertainty tracks actual error)
    """
    predicted_std = predicted_std.cpu().numpy().flatten()
    actual_error = actual_error.cpu().numpy().flatten()

    correlation = np.corrcoef(predicted_std, actual_error)[0, 1]

    return correlation


# ============================================================================
# Custom Schedulers
# ============================================================================

class CosineAnnealingWarmRestartsWithDecay(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warm Restarts and Learning Rate Decay.

    Extends CosineAnnealingWarmRestarts with:
    1. Decaying maximum learning rate after each restart
    2. Adjustable cycle frequency (make cycles shorter/longer)

    Args:
        optimizer: Wrapped optimizer
        T_0: Number of iterations for the first restart
        T_mult: Factor to increase cycle length after restart (default: 1.0)
        freq_mult: Multiplier for cycle length (e.g., 0.9 for shorter cycles)
        eta_min: Minimum learning rate (default: 0)
        decay: Decay factor for max LR after restart (default: 0.9)
        last_epoch: The index of last epoch (default: -1)

    Usage:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=50, decay=0.9)
        >>> for epoch in range(300):
        >>>     train(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, T_0, T_mult=1.0, freq_mult=1.0, eta_min=0, decay=0.9, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay = decay          # Decay factor for max LR
        self.freq_mult = freq_mult  # Multiplier for cycle length
        self.base_lrs = None
        self.current_max_lrs = None
        self.T_i = T_0
        self.cycle = 0
        self.epoch_since_restart = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.base_lrs is None or self.current_max_lrs is None:
            self.base_lrs = [group['initial_lr'] if 'initial_lr' in group else group['lr']
                             for group in self.optimizer.param_groups]
            self.current_max_lrs = self.base_lrs.copy()

        # Standard cosine annealing formula, but with decaying max LR
        return [
            self.eta_min + (max_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.epoch_since_restart / self.T_i)) / 2
            for max_lr in self.current_max_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.epoch_since_restart += 1

        if self.epoch_since_restart >= self.T_i:
            self.cycle += 1
            self.epoch_since_restart = 0
            self.T_i = int(self.T_i * self.freq_mult)
            self.current_max_lrs = [
                base_lr * (self.decay ** self.cycle)
                for base_lr in self.base_lrs
            ]
            for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, self.current_max_lrs)):
                name = group.get('name', f'group_{i}')
                print(f"[{name}] Decayed max LR: {lr:.6f}, T_i={self.T_i}")

        # Apply the new learning rates to param groups
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

        # Required for PyTorch's SequentialLR compatibility
        self._last_lr = lrs


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.

    Stops training if validation loss doesn't improve for 'patience' epochs.

    Args:
        patience: How long to wait after last improvement (default: 7)
        verbose: If True, prints messages (default: False)
        delta: Minimum change to qualify as improvement (default: 0)
        trace_func: Function for printing messages (default: print)

    Usage:
        >>> early_stopping = EarlyStopping(patience=20, verbose=True)
        >>> for epoch in range(epochs):
        >>>     train_loss = train(...)
        >>>     val_loss = validate(...)
        >>>     early_stopping(val_loss)
        >>>     if early_stopping.early_stop:
        >>>         print("Early stopping triggered")
        >>>         break
    """
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0


# ============================================================================
# Utility Functions
# ============================================================================

def compute_calibration_curve(predicted_std: np.ndarray,
                              actual_error: np.ndarray,
                              n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve for plotting.

    Returns bin centers, average predicted uncertainty, and average actual error
    for each bin. Use this to create calibration plots.

    Args:
        predicted_std: Predicted standard deviation (N,)
        actual_error: Actual prediction error (N,)
        n_bins: Number of bins

    Returns:
        bin_centers: Center of each bin (n_bins,)
        avg_predicted: Average predicted uncertainty per bin (n_bins,)
        avg_actual: Average actual error per bin (n_bins,)

    Usage:
        >>> bins, pred, actual = compute_calibration_curve(std, error)
        >>> plt.plot(bins, actual, label='Actual')
        >>> plt.plot(bins, pred, label='Predicted')
        >>> plt.plot([0, max(bins)], [0, max(bins)], 'k--', label='Perfect')
        >>> plt.xlabel('Predicted Uncertainty')
        >>> plt.ylabel('Actual Error')
        >>> plt.legend()
    """
    bin_boundaries = np.linspace(predicted_std.min(), predicted_std.max(), n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    avg_predicted = []
    avg_actual = []

    for i in range(n_bins):
        mask = (predicted_std >= bin_boundaries[i]) & (predicted_std < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (predicted_std >= bin_boundaries[i]) & (predicted_std <= bin_boundaries[i + 1])

        if mask.sum() == 0:
            avg_predicted.append(np.nan)
            avg_actual.append(np.nan)
        else:
            avg_predicted.append(predicted_std[mask].mean())
            avg_actual.append(actual_error[mask].mean())

    return bin_centers, np.array(avg_predicted), np.array(avg_actual)


# ============================================================================
# PART 2: UQ METHODS - LOSS FUNCTIONS AND EVALUATION
# ============================================================================
# This section contains implementations for:
# 1. Conformal Prediction (Idea 5)
# 2. Evidential Deep Learning (Idea 13)
# 3. Bayesian FNO
# 4. Unified evaluation framework
# ============================================================================


# ============================================================================
# 1. CONFORMAL PREDICTION (Idea 5)
# ============================================================================

def conformal_calibrate(model, cal_loader, alpha=0.05, score_fn='l2', device='cuda'):
    """
    Calibrate conformal prediction threshold on calibration set.
    
    Args:
        model: Trained FNO model
        cal_loader: Calibration data loader (held-out from training)
        alpha: Miscoverage rate (0.05 for 95% coverage)
        score_fn: 'l2', 'relative_l2', 'linf', or 'pointwise'
        device: Device to run on
    
    Returns:
        quantile: Calibrated threshold q̂ (scalar for l2/linf, tensor for pointwise)
    
    Example:
        >>> quantile = conformal_calibrate(fno, cal_loader, alpha=0.05, score_fn='l2')
        >>> print(f"95% coverage threshold: {quantile:.6f}")
    """
    import numpy as np
    
    model.eval()
    scores = []
    
    with torch.no_grad():
        for x, u_true in cal_loader:
            x, u_true = x.to(device), u_true.to(device)
            u_pred = model(x)
            
            # Compute nonconformity score
            if score_fn == 'l2':
                # Field-level L² norm
                score = torch.sqrt(((u_pred - u_true)**2).mean(dim=(1,2,3)))  # (batch,)
                
            elif score_fn == 'relative_l2':
                # Relative L² error
                numerator = torch.sqrt(((u_pred - u_true)**2).mean(dim=(1,2,3)))
                denominator = torch.sqrt((u_true**2).mean(dim=(1,2,3)))
                score = numerator / (denominator + 1e-8)
                
            elif score_fn == 'linf':
                # Maximum absolute error
                score = (u_pred - u_true).abs().flatten(1).max(dim=1)[0]  # (batch,)
                
            elif score_fn == 'pointwise':
                # Pointwise absolute residuals
                score = (u_pred - u_true).abs()  # (batch, nx, ny, 1)
            
            else:
                raise ValueError(f"Unknown score_fn: {score_fn}")
            
            scores.append(score)
    
    # Concatenate all scores
    if score_fn == 'pointwise':
        scores = torch.cat(scores, dim=0)  # (N_cal, nx, ny, 1)
    else:
        scores = torch.cat(scores, dim=0)  # (N_cal,)
    
    # Compute quantile with finite-sample correction
    n = scores.shape[0]
    level = np.ceil((n + 1) * (1 - alpha)) / n
    
    if score_fn == 'pointwise':
        quantile = torch.quantile(scores, level, dim=0)  # (nx, ny, 1)
    else:
        quantile = torch.quantile(scores, level)  # scalar
    
    return quantile.item() if quantile.numel() == 1 else quantile


def conformal_predict(model, x_test, quantile, score_fn='l2', device='cuda'):
    """
    Make prediction with conformal uncertainty interval.
    
    Args:
        model: Trained FNO
        x_test: Test input
        quantile: Calibrated threshold from conformal_calibrate()
        score_fn: Same as used in calibration
        device: Device
    
    Returns:
        Dictionary with prediction, lower, upper, interval_width
    """
    model.eval()
    
    with torch.no_grad():
        x_test = x_test.to(device)
        u_pred = model(x_test)
    
    # Construct prediction interval
    if isinstance(quantile, torch.Tensor):
        # Pointwise intervals
        quantile = quantile.to(device)
        lower = u_pred - quantile
        upper = u_pred + quantile
        interval_width = (2 * quantile).mean().item()
    else:
        # Uniform interval
        lower = u_pred - quantile
        upper = u_pred + quantile
        interval_width = 2 * quantile
    
    return {
        'prediction': u_pred,
        'lower': lower,
        'upper': upper,
        'interval_width': interval_width
    }


def conformal_coverage(model, test_loader, quantile, score_fn='l2', device='cuda'):
    """
    Evaluate conformal prediction coverage on test set.
    
    Returns:
        Dictionary with coverage and avg_interval_width
    """
    import numpy as np
    
    model.eval()
    covered = []
    interval_widths = []
    
    with torch.no_grad():
        for x, u_true in test_loader:
            x, u_true = x.to(device), u_true.to(device)
            
            result = conformal_predict(model, x, quantile, score_fn, device)
            u_pred = result['prediction']
            lower = result['lower']
            upper = result['upper']
            
            # Check coverage
            if score_fn == 'pointwise':
                is_covered = ((u_true >= lower) & (u_true <= upper)).float().mean()
            else:
                if score_fn == 'l2':
                    error = torch.sqrt(((u_pred - u_true)**2).mean(dim=(1,2,3)))
                    if isinstance(quantile, torch.Tensor):
                        q_val = quantile.mean().item()
                    else:
                        q_val = quantile
                    is_covered = (error <= q_val).float().mean()
                elif score_fn == 'linf':
                    error = (u_pred - u_true).abs().flatten(1).max(dim=1)[0]
                    is_covered = (error <= quantile).float().mean()
            
            covered.append(is_covered if isinstance(is_covered, float) else is_covered.item())
            interval_widths.append(result['interval_width'])
    
    coverage = np.mean(covered)
    avg_width = np.mean(interval_widths)
    
    return {
        'coverage': coverage,
        'avg_interval_width': avg_width
    }


# ============================================================================
# CQR (Conformalized Quantile Regression)
# ============================================================================

def quantile_loss(q_pred, y_true, tau):
    """
    Pinball loss for quantile regression.
    
    Args:
        q_pred: Predicted quantile
        y_true: True value
        tau: Quantile level (e.g., 0.025 for 2.5th percentile)
    
    Returns:
        loss: Pinball loss
    """
    error = y_true - q_pred
    loss = torch.max((tau - 1) * error, tau * error)
    return loss.mean()


def cqr_calibrate(quantile_model, cal_loader, alpha=0.05, device='cuda'):
    """
    Conformalized Quantile Regression calibration.
    
    Args:
        quantile_model: Trained QuantileFNO2d
        cal_loader: Calibration data loader
        alpha: Miscoverage rate
        device: Device
    
    Returns:
        correction: Conformal correction factor
    """
    import numpy as np
    
    quantile_model.eval()
    scores = []
    
    with torch.no_grad():
        for x, y in cal_loader:
            x, y = x.to(device), y.to(device)
            
            # Get predicted quantiles
            q_low, q_high = quantile_model(x)
            
            # Compute nonconformity score
            score = torch.max(
                q_low - y,  # Below lower bound
                y - q_high  # Above upper bound
            )
            
            scores.append(score)
    
    scores = torch.cat(scores, dim=0)
    
    # Compute correction
    n = scores.shape[0]
    level = np.ceil((n + 1) * (1 - alpha)) / n
    correction = torch.quantile(scores, level)
    
    return correction.item()


def cqr_predict(quantile_model, x, correction, device='cuda'):
    """
    Make CQR prediction with conformalized intervals.
    """
    quantile_model.eval()
    
    with torch.no_grad():
        x = x.to(device)
        q_low, q_high = quantile_model(x)
    
    # Apply conformal correction
    q_low_conf = q_low - correction
    q_high_conf = q_high + correction
    
    # Mean prediction
    mean = (q_low_conf + q_high_conf) / 2
    
    return {
        'mean': mean,
        'lower': q_low_conf,
        'upper': q_high_conf,
        'interval_width': (q_high_conf - q_low_conf).mean().item()
    }


# ============================================================================
# 2. EVIDENTIAL DEEP LEARNING (Idea 13)
# ============================================================================

def nig_nll(gamma, nu, alpha, beta, y):
    """
    Normal-Inverse-Gamma (NIG) Negative Log-Likelihood.
    
    This is the Student-t NLL derived from NIG prior.
    
    Args:
        gamma: Mean prediction (batch, *, 1)
        nu: Evidence for mean (batch, *, 1)
        alpha: Evidence strength (batch, *, 1)
        beta: Scale parameter (batch, *, 1)
        y: Ground truth (batch, *, 1)
    
    Returns:
        nll: Negative log-likelihood (scalar)
    """
    import numpy as np
    
    # Compute Δ² (scaled error)
    delta_sq = nu * (y - gamma) ** 2
    
    # Compute Ω
    omega = 2 * beta * (1 + nu)
    
    # NLL formula (stable version)
    nll = (
        0.5 * torch.log(torch.tensor(np.pi, device=gamma.device) / nu) -
        alpha * torch.log(omega) +
        (alpha + 0.5) * torch.log(omega + delta_sq) +
        torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    )
    
    return nll.mean()


def evidential_regularization(gamma, nu, alpha, beta, y):
    """
    Evidence regularization to prevent overconfidence on wrong predictions.
    
    Penalizes high evidence (ν, α) when prediction error |y - γ| is large.
    
    Args:
        gamma, nu, alpha, beta: Evidential parameters
        y: Ground truth
    
    Returns:
        reg: Regularization loss
    """
    error = torch.abs(y - gamma)
    evidence = 2 * nu + alpha
    reg = (error * evidence).mean()
    return reg


def evidential_loss(gamma, nu, alpha, beta, y, reg_weight=0.01,
                    beta_reg_weight=0.0, target_beta=0.5):
    """
    Combined evidential loss = NLL + regularization + optional beta regularization.

    Args:
        gamma, nu, alpha, beta: Evidential parameters from model
        y: Ground truth
        reg_weight: Weight for evidence regularization (default: 0.01)
        beta_reg_weight: Weight for beta regularization to prevent collapse (default: 0.0, disabled)
        target_beta: Target value for beta regularization (default: 0.5)

    Returns:
        loss: Total loss
        loss_dict: Dictionary with loss components
    """
    # NIG negative log-likelihood
    nll = nig_nll(gamma, nu, alpha, beta, y)

    # Evidence regularization
    reg = evidential_regularization(gamma, nu, alpha, beta, y)

    # Optional beta regularization (prevents beta collapse)
    beta_reg = torch.tensor(0.0, device=beta.device)
    if beta_reg_weight > 0.0:
        # L2 penalty to keep beta near target value
        beta_reg = torch.mean((beta - target_beta) ** 2)

    # Total loss
    total_loss = nll + reg_weight * reg + beta_reg_weight * beta_reg

    loss_dict = {
        'nll': nll.item(),
        'reg': reg.item(),
        'beta_reg': beta_reg.item() if beta_reg_weight > 0.0 else 0.0,
        'total': total_loss.item()
    }

    return total_loss, loss_dict


def evidential_uncertainty(gamma, nu, alpha, beta):
    """
    Compute epistemic and aleatoric uncertainty from evidential parameters.
    
    Args:
        gamma, nu, alpha, beta: Evidential parameters
    
    Returns:
        Dictionary with mean, aleatoric, epistemic, total, evidence
    """
    # Ensure α > 1 for valid variance
    alpha_safe = torch.clamp(alpha, min=1.01)
    
    # Aleatoric uncertainty (irreducible)
    aleatoric = beta / (alpha_safe - 1)
    
    # Epistemic uncertainty (reducible with more data)
    epistemic = beta / (nu * (alpha_safe - 1))
    
    # Total uncertainty
    total = beta * (1 + 1/nu) / (alpha_safe - 1)
    
    # Evidence score (higher = more confident)
    evidence = alpha_safe
    
    return {
        'mean': gamma,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
        'total': total,
        'evidence': evidence
    }


# ============================================================================
# IMPROVED EVIDENTIAL DEEP LEARNING (for comparison)
# ============================================================================

def improved_evidential_regularization(gamma, nu, alpha, beta, y):
    """
    Improved evidence regularization (Meinert & Lavin 2021).
    
    Uses logarithmic barrier instead of linear penalty for tighter bound.
    
    Original: R = |y - γ| * (2ν + α)
    Improved: R = log(1 + |y - γ|²) * (ν + α)
    """
    error_sq = (y - gamma) ** 2
    evidence = nu + alpha
    
    # Log barrier regularization
    reg = torch.log(1 + error_sq) * evidence
    
    return reg.mean()


def improved_evidential_loss(gamma, nu, alpha, beta, y, reg_weight=0.01):
    """
    Improved evidential loss with tighter bound.
    """
    # Same NLL as standard DER
    nll = nig_nll(gamma, nu, alpha, beta, y)

    # Improved regularization
    reg = improved_evidential_regularization(gamma, nu, alpha, beta, y)

    # Total loss
    total_loss = nll + reg_weight * reg

    loss_dict = {
        'nll': nll.item(),
        'reg': reg.item(),  # Changed from 'improved_reg' to 'reg' for consistency
        'total': total_loss.item()
    }

    return total_loss, loss_dict


# ============================================================================
# ADDITIONAL REGULARIZATION SCHEMES (for comparison)
# ============================================================================

def uncertainty_aware_regularization(gamma, nu, alpha, beta, y):
    """
    Uncertainty-Aware Regularization (Amini et al. 2020).

    Regularize based on prediction uncertainty rather than raw evidence.
    Penalizes high uncertainty when errors are large, and low uncertainty
    when errors are large (overconfidence).

    Formula: R = error² / (σ²_total + ε)

    Args:
        gamma, nu, alpha, beta: Evidential parameters
        y: Ground truth

    Returns:
        reg: Regularization loss
    """
    error = (y - gamma) ** 2

    # Compute total uncertainty
    alpha_safe = torch.clamp(alpha, min=1.01)
    aleatoric = beta / (alpha_safe - 1)
    epistemic = beta / (nu * (alpha_safe - 1))
    total_unc = aleatoric + epistemic

    # Penalize large errors with low uncertainty (overconfidence)
    # Add small epsilon to prevent division by zero
    reg = error / (total_unc + 1e-6)

    return reg.mean()


def annealed_regularization(gamma, nu, alpha, beta, y, epoch=0, max_epochs=100):
    """
    Annealed Regularization (Time-dependent).

    Gradually reduce regularization strength during training.
    Strong at start (prevent bad initialization), weak at end (allow refinement).

    Formula: R = λ(t) × |y - γ| × (2ν + α)
    where λ(t) = 0.1 + 0.9 × (1 - t/T)

    Args:
        gamma, nu, alpha, beta: Evidential parameters
        y: Ground truth
        epoch: Current epoch
        max_epochs: Total number of epochs

    Returns:
        reg: Regularization loss
    """
    error = torch.abs(y - gamma)
    evidence = 2 * nu + alpha

    # Annealing schedule: starts at 1.0, decays to 0.1
    anneal_factor = 0.1 + 0.9 * max(0.0, 1.0 - epoch / max(1, max_epochs))

    reg = anneal_factor * (error * evidence).mean()

    return reg


def l2_evidence_regularization(gamma, nu, alpha, beta, y):
    """
    L2 Evidence Regularization.

    L2 penalty on evidence parameters to prevent unbounded growth.
    Encourages conservative uncertainty estimates.

    Formula: R = error × (ν + α) + λ_L2 × (ν² + α²)

    Args:
        gamma, nu, alpha, beta: Evidential parameters
        y: Ground truth

    Returns:
        reg: Regularization loss
    """
    error = (y - gamma) ** 2

    # L2 penalty on ν and α (evidence parameters)
    evidence_penalty = (nu ** 2 + alpha ** 2).mean()

    # Combined: error-based + L2 penalty
    # Weight L2 term with 0.01 to balance magnitudes
    reg = (error * (nu + alpha)).mean() + 0.01 * evidence_penalty

    return reg


def adaptive_regularization(gamma, nu, alpha, beta, y):
    """
    Adaptive Regularization (Error-dependent weight).

    Adaptively weight regularization based on error magnitude.
    Stronger penalty for large errors, weaker for small errors.

    Formula: R = (exp(|y - γ|) - 1) × (2ν + α)

    Args:
        gamma, nu, alpha, beta: Evidential parameters
        y: Ground truth

    Returns:
        reg: Regularization loss
    """
    error = torch.abs(y - gamma)
    evidence = 2 * nu + alpha

    # Adaptive weighting: error acts as its own weight
    # Large errors → strong penalty, small errors → weak penalty
    # Use exp(e) - 1 ≈ e for small e, but grows faster for large e
    adaptive_weight = torch.exp(error) - 1

    reg = (adaptive_weight * evidence).mean()

    return reg


def kl_divergence_regularization(gamma, nu, alpha, beta, y, prior_nu=1.0, prior_alpha=1.0):
    """
    KL Divergence Regularization.

    KL divergence from prior NIG distribution.
    Encourages parameters to stay close to uninformative prior.

    Formula: R = KL(NIG(γ,ν,α,β) || NIG_prior) + error × (ν + α)

    Args:
        gamma, nu, alpha, beta: Evidential parameters
        y: Ground truth
        prior_nu: Prior value for ν (default: 1.0)
        prior_alpha: Prior value for α (default: 1.0)

    Returns:
        reg: Regularization loss
    """
    # KL(NIG || Prior) for Normal-Inverse-Gamma
    # For ν (precision parameter of gamma): KL ∝ (ν/ν₀ - log(ν/ν₀) - 1)
    kl_nu = 0.5 * (nu / prior_nu - torch.log(nu / prior_nu + 1e-8) - 1)

    # For α (shape parameter): KL ∝ (α - α₀ - ψ(α) + ψ(α₀))
    # Approximate with simpler form: (α - α₀)²
    kl_alpha = (alpha - prior_alpha) ** 2

    kl_total = (kl_nu + kl_alpha).mean()

    # Combine with error-based penalty
    error = (y - gamma) ** 2
    error_penalty = (error * (nu + alpha)).mean()

    reg = kl_total + error_penalty

    return reg


# ============================================================================
# GENERAL EVIDENTIAL LOSS (accepts regularization function)
# ============================================================================

def general_evidential_loss(gamma, nu, alpha, beta, y,
                           reg_fn=None, reg_weight=0.01, **reg_kwargs):
    """
    General evidential loss function that accepts any regularization scheme.

    Args:
        gamma, nu, alpha, beta: Evidential parameters from model
        y: Ground truth
        reg_fn: Regularization function (default: evidential_regularization)
                Should have signature: reg_fn(gamma, nu, alpha, beta, y, **kwargs)
        reg_weight: Weight for regularization term (default: 0.01)
        **reg_kwargs: Additional keyword arguments for regularization function

    Returns:
        loss: Total loss
        loss_dict: Dictionary with loss components

    Examples:
        # Standard DER
        loss, _ = general_evidential_loss(gamma, nu, alpha, beta, y)

        # Improved DER
        loss, _ = general_evidential_loss(gamma, nu, alpha, beta, y,
                                         reg_fn=improved_evidential_regularization)

        # Uncertainty-aware
        loss, _ = general_evidential_loss(gamma, nu, alpha, beta, y,
                                         reg_fn=uncertainty_aware_regularization)

        # Annealed
        loss, _ = general_evidential_loss(gamma, nu, alpha, beta, y,
                                         reg_fn=annealed_regularization,
                                         epoch=10, max_epochs=100)
    """
    # NIG negative log-likelihood
    nll = nig_nll(gamma, nu, alpha, beta, y)

    # Apply regularization
    if reg_fn is None:
        # Default to standard evidential regularization
        reg = evidential_regularization(gamma, nu, alpha, beta, y)
    else:
        # Use custom regularization function
        reg = reg_fn(gamma, nu, alpha, beta, y, **reg_kwargs)

    # Total loss
    total_loss = nll + reg_weight * reg

    loss_dict = {
        'nll': nll.item(),
        'reg': reg.item(),
        'total': total_loss.item()
    }

    return total_loss, loss_dict


# ============================================================================
# 3. BAYESIAN FNO
# ============================================================================

def bayesian_elbo_loss(model, x, y, n_samples=1, kl_weight=0.01):
    """
    Evidence Lower Bound (ELBO) for Bayesian neural networks.
    
    ELBO = E_q[log p(y|x,w)] - KL(q(w) || p(w))
         = -NLL - KL_weight * KL
    
    Args:
        model: BayesianFNO model
        x, y: Input and target
        n_samples: Number of samples for expectation
        kl_weight: Weight for KL term (default: 0.01)
    
    Returns:
        loss: ELBO loss (to be minimized)
        loss_dict: Dictionary with components
    """
    # Sample predictions
    total_nll = 0.0
    for _ in range(n_samples):
        pred = model.forward_single(x, sample=True)
        nll = F.mse_loss(pred, y)
        total_nll += nll
    
    avg_nll = total_nll / n_samples
    
    # KL divergence
    kl = model.kl_divergence()
    
    # ELBO
    loss = avg_nll + kl_weight * kl
    
    loss_dict = {
        'nll': avg_nll.item(),
        'kl': kl.item(),
        'total': loss.item()
    }
    
    return loss, loss_dict


# ============================================================================
# 4. PRIOR NETWORKS
# ============================================================================

def prior_network_loss(alphas, y, n_bins=50, output_range=(-1, 1)):
    """
    Reverse KL divergence for Prior Networks.
    
    KL(target || predicted) instead of KL(predicted || target).
    More robust to misspecification.
    """
    # Discretize targets into bins
    bin_edges = torch.linspace(output_range[0], output_range[1], n_bins + 1)
    bin_edges = bin_edges.to(y.device)
    
    # One-hot encoding of targets
    y_binned = torch.zeros_like(alphas)
    for b in range(y.size(0)):
        for i in range(y.size(1)):
            for j in range(y.size(2)):
                bin_idx = torch.searchsorted(bin_edges, y[b, i, j, 0])
                bin_idx = torch.clamp(bin_idx - 1, 0, n_bins - 1)
                y_binned[b, i, j, bin_idx] = 1.0
    
    # Reverse KL
    alpha_0 = alphas.sum(dim=-1, keepdim=True)
    log_p_expected = torch.digamma(alphas) - torch.digamma(alpha_0)
    reverse_kl = -(y_binned * log_p_expected).sum(dim=-1).mean()
    
    return reverse_kl


# ============================================================================
# 5. UNIFIED EVALUATION FRAMEWORK
# ============================================================================

def evaluate_uq_method(model, test_loader, method_name, device='cuda',
                      n_samples=100, confidence_level=0.95, quantile=None, score_fn='l2'):
    """
    Unified evaluation for all UQ methods.
    
    Args:
        model: UQ model
        test_loader: Test data loader
        method_name: 'mc_dropout', 'ensemble', 'bayesian', 'conformal', 'evidential'
        device: Device
        n_samples: Number of samples for sampling-based methods
        confidence_level: Confidence level for intervals
        quantile: For conformal method
        score_fn: For conformal method
    
    Returns:
        metrics: Dictionary with all evaluation metrics
    """
    import numpy as np
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_lower_bounds = []
    all_upper_bounds = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Get predictions and uncertainty based on method
            if method_name in ['mc_dropout', 'ensemble', 'bayesian']:
                # Sampling-based methods
                samples = model(x, return_samples=True)  # (T, batch, ...)
                
                mean = samples.mean(dim=0)
                std = samples.std(dim=0)
                
                # Construct intervals from samples
                alpha = 1 - confidence_level
                lower = torch.quantile(samples, alpha/2, dim=0)
                upper = torch.quantile(samples, 1 - alpha/2, dim=0)
                
            elif method_name == 'conformal':
                # Conformal prediction
                result = conformal_predict(model, x, quantile, score_fn, device)
                mean = result['prediction']
                lower = result['lower']
                upper = result['upper']
                std = (upper - lower) / (2 * 1.96)  # Approximate std
                
            elif method_name in ['evidential', 'der_nig', 'improved_der', 'natural_posterior',
                                'prior_networks', 'posterior_networks', 'dirichlet_evidential']:
                # Evidential deep learning methods
                if method_name in ['der_nig', 'improved_der', 'natural_posterior', 'evidential']:
                    # NIG-based methods
                    gamma, nu, alpha, beta = model(x)
                    uq_dict = evidential_uncertainty(gamma, nu, alpha, beta)

                    mean = uq_dict['mean']
                    std = torch.sqrt(uq_dict['total'])
                elif method_name in ['prior_networks', 'posterior_networks', 'dirichlet_evidential']:
                    # Other evidential methods - assume they return (mean, aleatoric, epistemic)
                    mean, aleatoric, epistemic = model(x)
                    std = torch.sqrt(aleatoric + epistemic)
                else:
                    # Fallback
                    gamma, nu, alpha, beta = model(x)
                    uq_dict = evidential_uncertainty(gamma, nu, alpha, beta)
                    mean = uq_dict['mean']
                    std = torch.sqrt(uq_dict['total'])

                # Construct intervals from Student-t
                t_quantile = 1.96  # Approximate for large df
                lower = mean - t_quantile * std
                upper = mean + t_quantile * std

            elif method_name in ['standard_fno']:
                # Standard FNO without uncertainty
                mean = model(x)
                std = torch.zeros_like(mean)  # No uncertainty
                lower = mean
                upper = mean

            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            all_predictions.append(mean.cpu())
            all_targets.append(y.cpu())
            all_uncertainties.append(std.cpu())
            all_lower_bounds.append(lower.cpu())
            all_upper_bounds.append(upper.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    uncertainties = torch.cat(all_uncertainties, dim=0)
    lower_bounds = torch.cat(all_lower_bounds, dim=0)
    upper_bounds = torch.cat(all_upper_bounds, dim=0)
    
    # Compute metrics
    
    # 1. Accuracy metrics
    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).item()
    rel_l2 = (torch.norm(predictions - targets) / torch.norm(targets)).item()
    
    # 2. Calibration metrics
    actual_errors = torch.abs(predictions - targets)
    
    ece = expected_calibration_error(
        uncertainties.flatten(),
        actual_errors.flatten(),
        n_bins=10
    )
    
    # 3. Coverage metrics
    coverage = ((targets >= lower_bounds) & (targets <= upper_bounds)).float().mean().item()
    
    # 4. Sharpness (interval width)
    interval_width = (upper_bounds - lower_bounds).mean().item()
    
    # 5. Uncertainty-error correlation
    correlation = uncertainty_error_correlation(
        uncertainties.flatten(),
        actual_errors.flatten()
    )
    
    # 6. NLL
    nll = 0.5 * np.log(2 * np.pi) + torch.log(uncertainties + 1e-8) + \
          (targets - predictions) ** 2 / (2 * (uncertainties ** 2 + 1e-8))
    nll = nll.mean().item()
    
    metrics = {
        'method': method_name,
        'mse': mse,
        'mae': mae,
        'rel_l2': rel_l2,
        'ece': ece,
        'coverage': coverage,
        'interval_width': interval_width,
        'correlation': correlation,
        'nll': nll,
    }

    return metrics


# ============================================================================
# Additional Conformal Prediction Methods (6 total)
# ============================================================================

def full_conformal_calibrate(model_class, model_kwargs, optimizer_class,
                             cal_data, alpha=0.05, score_fn='l2',
                             epochs=50, device='cuda'):
    """
    Full conformal prediction using leave-one-out.

    WARNING: Very expensive! Requires N retraining runs.
    Only use with small calibration sets (< 100 samples).

    Args:
        model_class: FNO class (e.g., FNO2d)
        model_kwargs: Arguments for model initialization
        optimizer_class: Optimizer class
        cal_data: Calibration dataset (not DataLoader!)
        alpha: Miscoverage rate
        score_fn: Score function
        epochs: Training epochs per model
        device: Device

    Returns:
        quantile: Calibrated threshold
    """
    from tqdm import tqdm
    from torch.utils.data import Subset, DataLoader

    n = len(cal_data)
    scores = []

    print(f"Full conformal: Training {n} models (leave-one-out)...")

    for i in tqdm(range(n)):
        # Create LOO training set (all except i)
        loo_indices = [j for j in range(n) if j != i]
        loo_dataset = Subset(cal_data, loo_indices)
        loo_loader = DataLoader(loo_dataset, batch_size=32, shuffle=True)

        # Train model on LOO set
        model_i = model_class(**model_kwargs).to(device)
        optimizer = optimizer_class(model_i.parameters(), lr=1e-3)

        for epoch in range(epochs):
            for x, y in loo_loader:
                x, y = x.to(device), y.to(device)
                pred = model_i(x)
                loss = F.mse_loss(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Test on held-out sample i
        x_i, y_i = cal_data[i]
        x_i = x_i.unsqueeze(0).to(device)
        y_i = y_i.unsqueeze(0).to(device)

        model_i.eval()
        with torch.no_grad():
            pred_i = model_i(x_i)

            # Compute score
            if score_fn == 'l2':
                score = torch.sqrt(((pred_i - y_i)**2).mean()).item()
            elif score_fn == 'linf':
                score = (pred_i - y_i).abs().max().item()

            scores.append(score)

    # Compute quantile
    quantile = np.quantile(scores, 1 - alpha)
    return quantile


def cross_conformal_calibrate(model_class, model_kwargs, optimizer_class,
                              cal_data, alpha=0.05, score_fn='l2',
                              k_folds=5, epochs=50, device='cuda'):
    """
    Cross-conformal prediction using K-fold cross-validation.

    Compromise between split and full conformal.

    Args:
        model_class, model_kwargs, optimizer_class: Model specification
        cal_data: Calibration dataset
        alpha: Miscoverage rate
        score_fn: Score function
        k_folds: Number of folds (default: 5)
        epochs: Training epochs per fold
        device: Device

    Returns:
        quantile: Calibrated threshold
    """
    from sklearn.model_selection import KFold
    from torch.utils.data import Subset, DataLoader

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores = []

    print(f"Cross-conformal: Training {k_folds} models (K-fold)...")

    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(cal_data)))):
        print(f"Fold {fold+1}/{k_folds}")

        # Create train/test splits
        train_subset = Subset(cal_data, train_idx.tolist())
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

        # Train model on K-1 folds
        model = model_class(**model_kwargs).to(device)
        optimizer = optimizer_class(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Test on held-out fold
        model.eval()
        with torch.no_grad():
            for i in test_idx:
                x_i, y_i = cal_data[i]
                x_i = x_i.unsqueeze(0).to(device)
                y_i = y_i.unsqueeze(0).to(device)

                pred_i = model(x_i)

                if score_fn == 'l2':
                    score = torch.sqrt(((pred_i - y_i)**2).mean()).item()
                elif score_fn == 'linf':
                    score = (pred_i - y_i).abs().max().item()

                scores.append(score)

    # Compute quantile
    quantile = np.quantile(scores, 1 - alpha)
    return quantile


def adaptive_conformal_calibrate(model, cal_loader, test_input,
                                alpha=0.05, score_fn='l2',
                                kernel='rbf', bandwidth=1.0,
                                device='cuda'):
    """
    Adaptive conformal prediction with weighted calibration.

    Weights calibration samples by similarity to test input.
    Better for distribution shift scenarios.

    Args:
        model: Trained FNO
        cal_loader: Calibration loader
        test_input: Test input for computing weights
        alpha: Miscoverage rate
        score_fn: Score function
        kernel: 'rbf' or 'cosine'
        bandwidth: Kernel bandwidth (for RBF)
        device: Device

    Returns:
        quantile: Weighted conformal threshold
    """
    from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity

    model.eval()
    scores = []
    cal_inputs = []

    # Collect scores and inputs
    with torch.no_grad():
        for x, y in cal_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            if score_fn == 'l2':
                score = torch.sqrt(((pred - y)**2).mean(dim=(1,2,3)))
            elif score_fn == 'linf':
                score = (pred - y).abs().flatten(1).max(dim=1)[0]

            scores.append(score.cpu().numpy())
            cal_inputs.append(x.flatten(1).cpu().numpy())

    scores = np.concatenate(scores)
    cal_inputs = np.concatenate(cal_inputs, axis=0)

    # Flatten test input for similarity
    test_input_flat = test_input.flatten(1).cpu().numpy()

    # Compute similarity weights
    if kernel == 'rbf':
        similarities = rbf_kernel(
            test_input_flat, cal_inputs, gamma=1.0/(2*bandwidth**2)
        )
    elif kernel == 'cosine':
        similarities = cosine_similarity(test_input_flat, cal_inputs)

    weights = similarities[0]  # (n_cal,)
    weights = weights / weights.sum()  # Normalize

    # Weighted quantile
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumsum = np.cumsum(sorted_weights)
    idx = np.searchsorted(cumsum, 1 - alpha)
    quantile = sorted_scores[min(idx, len(sorted_scores) - 1)]

    return quantile


def mondrian_conformal_calibrate(model, cal_loader, group_fn,
                                alpha=0.05, score_fn='l2',
                                device='cuda'):
    """
    Mondrian conformal prediction with stratified calibration.

    Maintains coverage guarantee PER GROUP, not just overall.

    Args:
        model: Trained FNO
        cal_loader: Calibration loader
        group_fn: Function mapping (x, y) -> group_id (e.g., 'low_Re', 'high_Re')
        alpha: Miscoverage rate
        score_fn: Score function
        device: Device

    Returns:
        quantiles: Dictionary {group_id: quantile}

    Example group_fn:
        def group_by_reynolds(x, y):
            # Extract Reynolds number from input
            Re = x[..., -1].mean().item()  # Assume last channel is Re
            if Re < 100:
                return 'low_Re'
            elif Re < 1000:
                return 'medium_Re'
            else:
                return 'high_Re'
    """
    from collections import defaultdict

    model.eval()

    # Group scores by category
    grouped_scores = defaultdict(list)

    with torch.no_grad():
        for x, y in cal_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Compute scores
            if score_fn == 'l2':
                scores = torch.sqrt(((pred - y)**2).mean(dim=(1,2,3)))
            elif score_fn == 'linf':
                scores = (pred - y).abs().flatten(1).max(dim=1)[0]

            # Assign to groups
            for i in range(x.size(0)):
                group_id = group_fn(x[i:i+1], y[i:i+1])
                grouped_scores[group_id].append(scores[i].item())

    # Compute quantile per group
    quantiles = {}
    for group_id, scores in grouped_scores.items():
        scores = np.array(scores)
        n = len(scores)
        level = np.ceil((n + 1) * (1 - alpha)) / n
        quantiles[group_id] = np.quantile(scores, level)

        print(f"Group '{group_id}': {n} samples, quantile = {quantiles[group_id]:.6f}")

    return quantiles


def mondrian_conformal_predict(model, x, quantiles, group_fn, device='cuda'):
    """
    Predict with Mondrian conformal intervals.

    Args:
        model: Trained FNO
        x: Input
        quantiles: Dictionary from mondrian_conformal_calibrate()
        group_fn: Same group function
        device: Device

    Returns:
        Dictionary with predictions and group-specific intervals
    """
    model.eval()

    with torch.no_grad():
        x = x.to(device)
        pred = model(x)

    # Determine group for each sample
    results = []
    for i in range(x.size(0)):
        group_id = group_fn(x[i:i+1], None)  # y not needed for grouping
        quantile = quantiles.get(group_id, quantiles[list(quantiles.keys())[0]])  # Default to first group

        lower = pred[i] - quantile
        upper = pred[i] + quantile

        results.append({
            'prediction': pred[i],
            'lower': lower,
            'upper': upper,
            'group': group_id,
            'quantile': quantile
        })

    return results


# ============================================================================
# Additional Evidential Deep Learning Methods (6 total)
# ============================================================================

def natural_nig_loss(gamma, nu, alpha, beta, y):
    """
    Natural parameterization of NIG for numerical stability.

    Uses natural parameters of exponential family:
    η₁ = γν, η₂ = -ν/2, η₃ = -α-0.5, η₄ = -β - γ²ν/2

    More stable for very large or small parameter values.

    Args:
        gamma, nu, alpha, beta: Evidential parameters
        y: Ground truth

    Returns:
        nll: Natural NIG negative log-likelihood
    """
    # Convert to natural parameters
    eta1 = gamma * nu
    eta2 = -nu / 2
    eta3 = -alpha - 0.5
    eta4 = -beta - (gamma ** 2 * nu) / 2

    # Sufficient statistics
    t1 = y
    t2 = y ** 2
    t3 = torch.ones_like(y)
    t4 = torch.ones_like(y)

    # Log partition function (normalizing constant)
    # A(η) for NIG distribution
    log_partition = (
        0.5 * torch.log(-2 * eta2 + 1e-8) -
        (eta3 + 0.5) * torch.log(-eta4 - eta1**2 / (4 * eta2 + 1e-8) + 1e-8) +
        torch.lgamma(-eta3 - 0.5 + 1e-8) -
        torch.lgamma(-eta3 + 1e-8)
    )

    # Natural NLL
    nll = -(eta1 * t1 + eta2 * t2 + eta3 * t3 + eta4 * t4) + log_partition

    return nll.mean()

