"""
Diagnostic script to visualize MCMC sampling during inference
and confirm the anchor collapse hypothesis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from trainer import Trainer, FNO_EBM
from datautils import PDEDataset
import yaml

def diagnose_mcmc_sampling():
    """
    Visualize MCMC trajectory to see if samples are collapsing to FNO
    """
    print("="*80)
    print("DIAGNOSING INFERENCE MCMC SAMPLING")
    print("="*80)

    # Load config
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)

    from types import SimpleNamespace
    config = SimpleNamespace(**config_dict)

    # Load dataset
    from pathlib import Path
    data_dir = Path(config.data_dir)

    # Try different file naming patterns
    complexity = config.complexity if hasattr(config, 'complexity') else 'medium'
    noise_type = config.noise_type if hasattr(config, 'noise_type') else 'heteroscedastic'

    val_file = data_dir / f'{config.pde_type}_{complexity}_{noise_type}_res64_val.npz'

    if not val_file.exists():
        # Try alternative naming
        val_file = data_dir / f'{config.pde_type}_{complexity}_noisy_res64_val.npz'

    if not val_file.exists():
        print(f"ERROR: Validation file not found")
        print(f"Tried: {val_file}")
        return

    print(f"Loading: {val_file}")
    dataset = PDEDataset.from_file(str(val_file), normalize_output=True)

    # Load trained model directly
    from fno import FNO2d
    from ebm import ConvEBM

    fno_model = FNO2d(
        modes1=config.fno_modes,
        modes2=config.fno_modes,
        width=config.fno_width
    ).cuda()

    ebm_model = ConvEBM(
        in_channels=4,
        hidden_channels=[64, 128, 128, 64]
    ).cuda()

    # Create FNO_EBM wrapper
    model = FNO_EBM(fno_model, ebm_model).cuda()

    import os
    checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model_ebm.pt')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Load FNO and EBM separately
    fno_model.load_state_dict(checkpoint['fno_model'])
    ebm_model.load_state_dict(checkpoint['ebm_model'])
    model.eval()

    print(f"Loaded checkpoint: best_model_ebm.pt (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.3f})")
    print()

    # Get one test sample
    x, y_true = dataset[0]
    x = x.unsqueeze(0).cuda()
    y_true = y_true.unsqueeze(0).cuda()

    # Get FNO prediction
    with torch.no_grad():
        u_fno = model.u_fno(x)

    print(f"Ground truth range: [{y_true.min():.3f}, {y_true.max():.3f}]")
    print(f"FNO prediction range: [{u_fno.min():.3f}, {u_fno.max():.3f}]")
    print()

    # Run MCMC with 3 different sigma_squared values
    sigma_values = [1.0, 10.0, 100.0]

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    for row, sigma_sq in enumerate(sigma_values):
        print(f"\n{'='*60}")
        print(f"Testing sigma_squared_inference = {sigma_sq}")
        print(f"{'='*60}")

        # Initialize from FNO
        u = u_fno.clone()
        u.requires_grad_(True)

        num_steps = 200
        step_size = 0.005
        noise_scale = np.sqrt(2 * step_size)

        # Track trajectory
        trajectory = [u.detach().cpu().clone()]
        energies = []

        for k in range(num_steps):
            # Compute energy with specified sigma
            energy = model.energy(
                u, x,
                training=False,
                sigma_squared_inference=sigma_sq
            ).sum()

            energies.append(energy.item())

            # Compute gradient
            grad_u = torch.autograd.grad(energy, u)[0]

            with torch.no_grad():
                noise = torch.randn_like(u) * noise_scale
                u = u - step_size * grad_u + noise
                u.requires_grad_(True)

            # Save snapshots
            if k in [0, 49, 99, 199]:
                trajectory.append(u.detach().cpu().clone())

        u_final = u.detach()

        # Compute statistics
        distance_from_fno = (u_final - u_fno).abs().mean().item()
        print(f"  Mean distance from FNO: {distance_from_fno:.6f}")
        print(f"  Final energy: {energies[-1]:.3f}")
        print(f"  Energy change: {energies[-1] - energies[0]:.3f}")

        # Plot snapshots
        snapshot_steps = [0, 50, 100, 200]
        for col, step in enumerate(snapshot_steps):
            snap = trajectory[col][0, :, :, 0].numpy()

            axes[row, col].imshow(snap, cmap='RdBu_r', vmin=-3, vmax=3)
            axes[row, col].set_title(f'Step {step}')
            axes[row, col].axis('off')

        # Plot energy curve
        axes[row, 4].plot(energies)
        axes[row, 4].set_xlabel('MCMC Step')
        axes[row, 4].set_ylabel('Energy')
        axes[row, 4].set_title(f'σ²={sigma_sq}\nΔE={energies[-1]-energies[0]:.2f}')
        axes[row, 4].grid(True)

    axes[0, 0].text(-0.1, 0.5, 'σ²=1.0\n(Current)',
                     transform=axes[0, 0].transAxes,
                     fontsize=12, fontweight='bold',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    axes[1, 0].text(-0.1, 0.5, 'σ²=10.0\n(Weak anchor)',
                     transform=axes[1, 0].transAxes,
                     fontsize=12, fontweight='bold',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    axes[2, 0].text(-0.1, 0.5, 'σ²=100.0\n(Very weak)',
                     transform=axes[2, 0].transAxes,
                     fontsize=12, fontweight='bold',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    plt.suptitle('MCMC Sampling Trajectories with Different Anchor Strengths', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mcmc_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to mcmc_diagnosis.png")

    # Test multiple samples to see variance
    print(f"\n{'='*60}")
    print("Testing variance with 20 samples")
    print(f"{'='*60}")

    for sigma_sq in sigma_values:
        samples = []
        for i in range(20):
            u = u_fno.clone()
            u.requires_grad_(True)

            for k in range(200):
                energy = model.energy(u, x, training=False, sigma_squared_inference=sigma_sq).sum()
                grad_u = torch.autograd.grad(energy, u)[0]

                with torch.no_grad():
                    noise = torch.randn_like(u) * noise_scale
                    u = u - step_size * grad_u + noise
                    u.requires_grad_(True)

            samples.append(u.detach())

        samples = torch.stack(samples, dim=0)
        sample_std = samples.std(dim=0).mean().item()

        print(f"  σ²={sigma_sq:6.1f} → Sample std dev: {sample_std:.6f}")

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nExpected results:")
    print("  - σ²=1.0: Samples should be nearly identical (std ≈ 0)")
    print("  - σ²=10.0: Moderate variance")
    print("  - σ²=100.0: Higher variance (EBM can explore more)")
    print("\nIf σ²=1.0 gives std ≈ 0, that confirms the anchor is collapsing all samples!")

if __name__ == '__main__':
    diagnose_mcmc_sampling()