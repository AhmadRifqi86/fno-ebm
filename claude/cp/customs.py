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
