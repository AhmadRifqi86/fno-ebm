#!/usr/bin/env python3
"""
main_noisy.py - Single-Dataset Training Mode

Trains FNO-EBM on a single noisy dataset.
Both FNO and EBM train on the same noisy observations.

⭐ RECOMMENDED for fixing EBM training issues! ⭐

Why single-dataset mode?
- No distribution mismatch between FNO and EBM
- EBM can learn solution uncertainty (not just observation noise)
- FNO learns robust features from noisy data
- Better uncertainty calibration

Usage:
    python main_noisy.py

Requirements:
    - config.yaml with lambda_phys=0.0 (physics loss disabled)
    - Noisy dataset in data/ folder
"""

import yaml
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path

from config import Config
from fno import FNO2d
from ebm import SimpleFNO_EBM  # Using ConvEBM for spatial structure!
from trainer import Trainer, FNO_EBM
from customs import DarcyPhysicsLoss
from inference import inference_deterministic, inference_probabilistic
from datautils import PDEDataset, visualize_inference_results


def main():
    """
    Single-dataset training pipeline:
    - FNO trains on noisy data (no physics loss)
    - EBM trains on same noisy data
    """

    print("=" * 70)
    print("FNO-EBM TRAINING: SINGLE-DATASET MODE (Noisy Data Only)")
    print("=" * 70)

    # 1. Load Configuration
    print("\n--- Loading Configuration ---")
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)

    print(f"Device: {config.device}")
    print(f"PDE Type: {config.pde_type}")
    print(f"Complexity: {config.complexity}")
    print(f"Noise Type: {config.noise_type}")
    print(f"Physics Loss Weight: {config.lambda_phys} (should be 0.0 for noisy data)")

    if config.lambda_phys > 0:
        print("\n⚠️  WARNING: lambda_phys > 0 detected!")
        print("   For noisy data training, physics loss should be disabled (lambda_phys=0.0)")
        print("   Physics loss + noisy data = underfitting!")
        config.lambda_phys = 0.0

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 2. Initialize Models
    print("\n--- Initializing Models ---")

    # FNO model
    fno_model = FNO2d(
        modes1=config.fno_modes,
        modes2=config.fno_modes,
        width=config.fno_width,
        num_layers=4
    )

    # EBM model - Using ConvEBM for spatial structure
    ebm_model = SimpleFNO_EBM(in_channels=4, fno_width=32, fno_layers=3)

    # Combined model
    model = FNO_EBM(fno_model, ebm_model).to(config.device)

    print(f"FNO: modes={config.fno_modes}, width={config.fno_width}")
    print(f"EBM: ConvEBM with spatial convolutions")

    # 3. Load Noisy Dataset
    print("\n--- Loading Noisy Dataset ---")

    data_dir = Path(config.data_dir) if hasattr(config, 'data_dir') else Path('../data')
    resolution = config.grid_size if hasattr(config, 'grid_size') else 64
    complexity = config.complexity if hasattr(config, 'complexity') else 'medium'
    noise_type = config.noise_type if hasattr(config, 'noise_type') else 'heteroscedastic'

    # Construct file paths for noisy dataset
    noisy_train_file = data_dir / f"darcy_{complexity}_{noise_type}_res{resolution}_train.npz"
    noisy_val_file = data_dir / f"darcy_{complexity}_{noise_type}_res{resolution}_val.npz"

    # Check if files exist
    if not noisy_train_file.exists() or not noisy_val_file.exists():
        print(f"\n❌ ERROR: Noisy dataset files not found!")
        print(f"Expected files:")
        print(f"  - {noisy_train_file}")
        print(f"  - {noisy_val_file}")
        print("\nPlease generate dataset first (it should be auto-generated if missing)")
        print("Or check that config.yaml settings match your data files.")

    # Load noisy datasets
    print(f"Loading noisy dataset from {data_dir}...")
    train_dataset = PDEDataset.from_file(str(noisy_train_file), normalize_output=True)
    val_dataset = PDEDataset.from_file(str(noisy_val_file), normalize_output=True)

    print("✓ Dataset loaded (automatically normalized)")

    # 3b. Create DataLoaders
    print("\n--- Creating DataLoaders ---")

    batch_size = config.batch_size if hasattr(config, 'batch_size') else 16
    num_workers = config.num_workers if hasattr(config, 'num_workers') else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✓ DataLoaders created (batch_size={batch_size})")

    # 4. Initialize Trainer (Single-Dataset Mode)
    print("\n--- Initializing Trainer (Single-Dataset Mode) ---")

    # Initialize Darcy physics loss (won't be used since lambda_phys=0)
    phy_loss = DarcyPhysicsLoss(source_term=1.0)

    trainer = Trainer(
        model=model,
        phy_loss=phy_loss,
        train_loader=train_loader,   # Noisy data
        val_loader=val_loader,       # Noisy data
        config=config
        # No ebm_train_loader → single-dataset mode
    )

    print("Training mode: Single-dataset (noisy)")
    print("  - FNO trains on: noisy observations")
    print("  - EBM trains on: same noisy observations")

    # 5. Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    trainer.train_staged()
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    # 6. Inference
    print("\n--- Running Inference ---")
    best_fno_path = os.path.join(config.checkpoint_dir, 'best_model_fno.pt')
    best_ebm_path = os.path.join(config.checkpoint_dir, 'current_ebm.pt')

    if os.path.exists(best_fno_path) and os.path.exists(best_ebm_path):
        # Load best models
        fno_checkpoint = torch.load(best_fno_path, map_location=config.device)
        ebm_checkpoint = torch.load(best_ebm_path, map_location=config.device)

        model.u_fno.load_state_dict(fno_checkpoint['fno_model'])
        model.V_ebm.load_state_dict(ebm_checkpoint['ebm_model'])

        print("✓ Loaded best models")

        # Get validation batch
        x_samples, y_true_samples = next(iter(val_loader))
        x_samples = x_samples.to(config.device)

        # Deterministic inference (FNO)
        print("Running FNO deterministic inference...")
        y_fno_pred = inference_deterministic(model, x_samples, device=config.device)

        # Probabilistic inference (EBM)
        print("Running EBM probabilistic inference...")
        _, stats = inference_probabilistic(
            model,
            x_samples,
            num_samples=50,
            num_mcmc_steps=config.mcmc_steps,
            step_size=config.mcmc_step_size,
            device=config.device
        )

        # Denormalize predictions for visualization
        # (predictions are in normalized space, convert back to original scale)
        y_fno_pred_denorm = train_dataset.denormalize(y_fno_pred)
        y_true_denorm = train_dataset.denormalize(y_true_samples)

        # Denormalize stats
        stats_denorm = {
            'mean': train_dataset.denormalize(stats['mean']),
            'std': stats['std'] * train_dataset.u_std,  # Scale std by original std
        }

        # Visualize
        visualize_inference_results(y_true_denorm, y_fno_pred_denorm, stats_denorm, config)
        print("✓ Inference results saved")

    else:
        print("⚠️  Could not find best model checkpoints")

    print("\n" + "=" * 70)
    print("SUMMARY - SINGLE-DATASET MODE")
    print("=" * 70)
    print(f"Checkpoints saved in: {config.checkpoint_dir}/")
    print("  - best_model_fno.pt (trained on noisy data)")
    print("  - best_model_ebm.pt (trained on noisy data)")
    print("  - current_fno.pt")
    print("  - current_ebm.pt")
    print("\nDatasets used:")
    print(f"  - FNO training: {noisy_train_file}")
    print(f"  - EBM training: {noisy_train_file} (SAME AS FNO)")
    print("\nTo view results:")
    print("  - Check logs above for validation loss")
    print("  - View inference_results.png in checkpoint directory")
    print("\nKey differences from dual-dataset mode:")
    print("  ✓ No distribution mismatch between FNO and EBM")
    print("  ✓ EBM learns solution uncertainty (not just noise)")
    print("  ✓ FNO learns robust features from noisy observations")
    print("  ✓ Better calibrated uncertainty estimates")
    print("\nIf EBM uncertainty maps show spatial structure:")
    print("  → SUCCESS! Single-dataset mode fixed the issue")
    print("If EBM uncertainty is still random noise:")
    print("  → Check training logs for EBM loss progression")
    print("  → Try visualizing negative samples during training")
    print("  → Consider score matching as alternative")
    print("=" * 70)


if __name__ == '__main__':
    main()