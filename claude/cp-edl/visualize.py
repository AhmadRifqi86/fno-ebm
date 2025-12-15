"""
Visualization utilities for FNO-EBM training results.

Creates comparison plots showing:
- Ground truth
- FNO prediction
- FNO error (absolute)
- EBM prediction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from scipy import stats


logger = logging.getLogger(__name__)


def visualize_predictions(
    x: torch.Tensor,
    u_gt: torch.Tensor,
    u_fno: torch.Tensor,
    u_ebm: torch.Tensor,
    save_path: Optional[str] = None,
    sample_idx: int = 0,
    pde_type: str = 'unknown',
    title_prefix: str = '',
) -> None:
    """
    Visualize ground truth, FNO prediction, FNO error, and EBM prediction.

    Args:
        x: Input coordinates (batch, n_x, [n_y], coord_channels)
        u_gt: Ground truth field (batch, n_x, [n_y], channels)
        u_fno: FNO prediction (batch, n_x, [n_y], channels)
        u_ebm: EBM prediction (batch, n_x, [n_y], channels)
        save_path: Path to save the figure (if None, displays instead)
        sample_idx: Which sample in the batch to visualize
        pde_type: Type of PDE for labeling
        title_prefix: Prefix for the figure title
    """
    # Move to CPU and convert to numpy
    u_gt_np = u_gt[sample_idx].cpu().numpy()
    u_fno_np = u_fno[sample_idx].cpu().numpy()
    u_ebm_np = u_ebm[sample_idx].cpu().numpy()

    # Compute FNO error
    fno_error_np = np.abs(u_gt_np - u_fno_np)

    # Check if 1D or 2D
    # Handle case where 1D data has shape (n_x, 1, channels) - squeeze out singleton dimension
    if len(u_gt_np.shape) == 3 and u_gt_np.shape[1] == 1:
        u_gt_np = u_gt_np.squeeze(1)  # (n_x, 1, channels) -> (n_x, channels)
        u_fno_np = u_fno_np.squeeze(1)
        u_ebm_np = u_ebm_np.squeeze(1)
        fno_error_np = fno_error_np.squeeze(1)

    is_1d = len(u_gt_np.shape) == 2  # (n_x, channels)
    is_2d = len(u_gt_np.shape) == 3  # (n_x, n_y, channels)

    if is_1d:
        _visualize_1d(u_gt_np, u_fno_np, fno_error_np, u_ebm_np,
                     save_path, pde_type, title_prefix)
    elif is_2d:
        _visualize_2d(u_gt_np, u_fno_np, fno_error_np, u_ebm_np,
                     save_path, pde_type, title_prefix)
    else:
        logger.warning(f"Unsupported data shape: {u_gt_np.shape}")


def _visualize_1d(
    u_gt: np.ndarray,
    u_fno: np.ndarray,
    fno_error: np.ndarray,
    u_ebm: np.ndarray,
    save_path: Optional[str],
    pde_type: str,
    title_prefix: str,
) -> None:
    """
    Visualize 1D fields (e.g., Burgers, Advection).

    Creates a 2x2 grid:
    - Top left: Ground truth
    - Top right: FNO prediction
    - Bottom left: FNO error (abs)
    - Bottom right: EBM prediction
    """
    n_x = u_gt.shape[0]
    x = np.linspace(0, 1, n_x)

    # Handle multi-channel by taking first channel
    if len(u_gt.shape) == 2:
        u_gt = u_gt[:, 0]
        u_fno = u_fno[:, 0]
        fno_error = fno_error[:, 0]
        u_ebm = u_ebm[:, 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Ground truth
    axes[0, 0].plot(x, u_gt, 'b-', linewidth=2)
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u')
    axes[0, 0].grid(True, alpha=0.3)

    # FNO prediction
    axes[0, 1].plot(x, u_fno, 'r-', linewidth=2, label='FNO')
    axes[0, 1].plot(x, u_gt, 'b--', linewidth=1, alpha=0.5, label='GT')
    axes[0, 1].set_title('FNO Prediction', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('u')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # FNO error
    axes[1, 0].plot(x, fno_error, 'purple', linewidth=2)
    axes[1, 0].fill_between(x, 0, fno_error, alpha=0.3, color='purple')
    axes[1, 0].set_title('FNO Error (|GT - FNO|)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].grid(True, alpha=0.3)

    # EBM prediction
    axes[1, 1].plot(x, u_ebm, 'g-', linewidth=2, label='EBM')
    axes[1, 1].plot(x, fno_error, 'purple', linewidth=1, alpha=0.5, label='FNO Error')
    axes[1, 1].set_title('EBM Prediction', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('u / Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Compute metrics
    fno_mse = np.mean((u_gt - u_fno) ** 2)
    fno_mae = np.mean(np.abs(u_gt - u_fno))

    # Overall title with metrics
    title = f'{title_prefix}{pde_type.upper()} - 1D Field Comparison\n'
    title += f'FNO MSE: {fno_mse:.6f}, MAE: {fno_mae:.6f}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved 1D visualization to {save_path}")
        plt.close()
    else:
        plt.show()


def _visualize_2d(
    u_gt: np.ndarray,
    u_fno: np.ndarray,
    fno_error: np.ndarray,
    u_ebm: np.ndarray,
    save_path: Optional[str],
    pde_type: str,
    title_prefix: str,
) -> None:
    """
    Visualize 2D fields (e.g., Navier-Stokes, Diffusion-Reaction).

    Creates a 2x2 grid with heatmaps:
    - Top left: Ground truth
    - Top right: FNO prediction
    - Bottom left: FNO error (abs)
    - Bottom right: EBM prediction
    """
    # Handle multi-channel by taking first channel
    if len(u_gt.shape) == 3:
        u_gt = u_gt[:, :, 0]
        u_fno = u_fno[:, :, 0]
        fno_error = fno_error[:, :, 0]
        u_ebm = u_ebm[:, :, 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Common colormap settings
    vmin = min(u_gt.min(), u_fno.min(), u_ebm.min())
    vmax = max(u_gt.max(), u_fno.max(), u_ebm.max())

    # Ground truth
    im0 = axes[0, 0].imshow(u_gt.T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])

    # FNO prediction
    im1 = axes[0, 1].imshow(u_fno.T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('FNO Prediction', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])

    # FNO error
    im2 = axes[1, 0].imshow(fno_error.T, origin='lower', cmap='hot')
    axes[1, 0].set_title('FNO Error (|GT - FNO|)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0])

    # EBM prediction
    im3 = axes[1, 1].imshow(u_ebm.T, origin='lower', cmap='plasma')
    axes[1, 1].set_title('EBM Prediction', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])

    # Compute metrics
    fno_mse = np.mean((u_gt - u_fno) ** 2)
    fno_mae = np.mean(np.abs(u_gt - u_fno))

    # Overall title with metrics
    title = f'{title_prefix}{pde_type.upper()} - 2D Field Comparison\n'
    title += f'FNO MSE: {fno_mse:.6f}, MAE: {fno_mae:.6f}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved 2D visualization to {save_path}")
        plt.close()
    else:
        plt.show()


def generate_ebm_visualization(
    ebm_trainer,
    fno_model: torch.nn.Module,
    test_loader,
    config,
    pde_type: str,
    output_dir: str,
    num_samples: int = 4,
) -> None:
    """
    Generate visualizations comparing GT, FNO, FNO error, and EBM predictions.

    This function should be called after EBM training is complete.

    Args:
        ebm_trainer: Trained EBMTrainer instance
        fno_model: Trained FNO model
        test_loader: Test data loader
        config: Configuration object
        pde_type: Type of PDE
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    logger.info(f"Generating EBM visualizations for {num_samples} samples...")

    device = ebm_trainer.device
    ebm_model = ebm_trainer.model
    ebm_model.eval()
    fno_model.eval()

    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Collect all EBM predictions and errors for calibration analysis
    all_ebm_uncertainties = []
    all_fno_errors = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if batch_idx >= num_samples:
                break

            # Unpack batch
            if len(batch_data) == 2:
                x, u_gt = batch_data
            else:
                x, u_gt, _ = batch_data

            x = x.to(device)
            u_gt = u_gt.to(device)

            # Get FNO prediction
            u_fno = fno_model(x)

            # Get EBM prediction
            # The EBM predicts energy, but we want to visualize the refined prediction
            # We can use Langevin sampling or just use the energy gradient as uncertainty

            # Option 1: Use energy gradient norm as EBM prediction (uncertainty map)
            # Need to enable gradients temporarily for score computation
            u_temp = u_gt.clone().detach()
            u_temp.requires_grad_(True)

            with torch.enable_grad():
                energy = ebm_model(u_temp, x, u_fno)

                ebm_score = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=u_temp,
                    create_graph=False
                )[0]

            # Use score norm as uncertainty (higher score = higher uncertainty)
            u_ebm = torch.norm(ebm_score, dim=-1, keepdim=True)

            # Scale EBM prediction to match FNO error magnitude for better visualization
            # Compute FNO error for reference
            fno_error_tensor = torch.abs(u_gt - u_fno)
            fno_error_mean = fno_error_tensor.mean()
            ebm_mean = u_ebm.mean()

            # Scale EBM to have similar magnitude as FNO error
            if ebm_mean > 1e-8:  # Avoid division by zero
                scale_factor = fno_error_mean / ebm_mean
                u_ebm = u_ebm * scale_factor

            # Expand to match channel dimension if needed
            if u_ebm.shape != u_gt.shape:
                # Replicate across channels
                u_ebm = u_ebm.expand_as(u_gt)

            # Collect data for calibration analysis
            all_ebm_uncertainties.append(u_ebm.cpu().numpy().flatten())
            all_fno_errors.append(fno_error_tensor.cpu().numpy().flatten())

            # Create visualization for each sample in batch
            batch_size = x.shape[0]
            for sample_idx in range(min(batch_size, 2)):  # Max 2 samples per batch
                save_path = viz_dir / f'sample_{batch_idx * batch_size + sample_idx:03d}.png'

                visualize_predictions(
                    x=x,
                    u_gt=u_gt,
                    u_fno=u_fno,
                    u_ebm=u_ebm,
                    save_path=str(save_path),
                    sample_idx=sample_idx,
                    pde_type=pde_type,
                    title_prefix=f'Sample {batch_idx * batch_size + sample_idx} - ',
                )

    logger.info(f"Visualizations saved to {viz_dir}")

    # Generate calibration analysis
    logger.info("Generating EBM calibration analysis...")
    all_ebm_uncertainties = np.concatenate(all_ebm_uncertainties)
    all_fno_errors = np.concatenate(all_fno_errors)

    calibration_path = Path(output_dir) / 'ebm_calibration.png'
    pearson_corr, spearman_corr = plot_ebm_calibration(
        ebm_uncertainties=all_ebm_uncertainties,
        fno_errors=all_fno_errors,
        save_path=str(calibration_path),
        pde_type=pde_type,
    )

    # Log correlation metrics
    logger.info(f"EBM Calibration Metrics:")
    logger.info(f"  Pearson correlation: {pearson_corr:.4f}")
    logger.info(f"  Spearman correlation: {spearman_corr:.4f}")

    # Interpretation
    if abs(pearson_corr) > 0.7 and abs(spearman_corr) > 0.7:
        logger.info("  ✓ Strong correlation - EBM is well-calibrated!")
    elif abs(pearson_corr) > 0.4 and abs(spearman_corr) > 0.4:
        logger.info("  ~ Moderate correlation - EBM shows some calibration")
    else:
        logger.info("  ✗ Weak correlation - EBM may not be learning error patterns")


def plot_ebm_calibration(
    ebm_uncertainties: np.ndarray,
    fno_errors: np.ndarray,
    save_path: str,
    pde_type: str,
) -> Tuple[float, float]:
    """
    Create calibration plots showing EBM uncertainty vs actual FNO error.

    Args:
        ebm_uncertainties: Flattened array of EBM uncertainty predictions
        fno_errors: Flattened array of actual FNO errors
        save_path: Path to save the figure
        pde_type: Type of PDE for labeling

    Returns:
        pearson_corr: Pearson correlation coefficient
        spearman_corr: Spearman rank correlation coefficient
    """
    # Compute correlations
    pearson_corr, pearson_p = stats.pearsonr(ebm_uncertainties, fno_errors)
    spearman_corr, spearman_p = stats.spearmanr(ebm_uncertainties, fno_errors)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot with regression line
    axes[0].scatter(ebm_uncertainties, fno_errors, alpha=0.3, s=10, c='blue', edgecolors='none')

    # Add regression line
    z = np.polyfit(ebm_uncertainties, fno_errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(ebm_uncertainties.min(), ebm_uncertainties.max(), 100)
    axes[0].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}')

    # Add diagonal (perfect calibration)
    max_val = max(ebm_uncertainties.max(), fno_errors.max())
    axes[0].plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')

    axes[0].set_xlabel('EBM Uncertainty', fontsize=12)
    axes[0].set_ylabel('Actual FNO Error', fontsize=12)
    axes[0].set_title(f'Calibration Scatter Plot\nPearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}',
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Binned calibration curve
    n_bins = 10
    bin_edges = np.percentile(ebm_uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_errors = []
    bin_stds = []

    for i in range(n_bins):
        mask = (ebm_uncertainties >= bin_edges[i]) & (ebm_uncertainties < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(ebm_uncertainties[mask].mean())
            bin_errors.append(fno_errors[mask].mean())
            bin_stds.append(fno_errors[mask].std())

    bin_centers = np.array(bin_centers)
    bin_errors = np.array(bin_errors)
    bin_stds = np.array(bin_stds)

    axes[1].errorbar(bin_centers, bin_errors, yerr=bin_stds, fmt='o-',
                    linewidth=2, markersize=8, capsize=5, label='Mean ± Std')
    axes[1].plot([bin_centers.min(), bin_centers.max()],
                [bin_centers.min(), bin_centers.max()],
                'k--', linewidth=1, alpha=0.5, label='Perfect calibration')

    axes[1].set_xlabel('EBM Uncertainty (binned)', fontsize=12)
    axes[1].set_ylabel('Mean FNO Error in Bin', fontsize=12)
    axes[1].set_title('Binned Calibration Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'{pde_type.upper()} - EBM Calibration Analysis\n' +
                f'Pearson R={pearson_corr:.4f} (p={pearson_p:.2e}), ' +
                f'Spearman ρ={spearman_corr:.4f} (p={spearman_p:.2e})',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved calibration plot to {save_path}")
    plt.close()

    return pearson_corr, spearman_corr


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None,
    title: str = 'Training Curves',
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the figure
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
        plt.close()
    else:
        plt.show()