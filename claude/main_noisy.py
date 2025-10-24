#!/usr/bin/env python3
"""
main_noisy.py - Single-Dataset Training Mode

Trains FNO-EBM on a single noisy dataset.
Both FNO and EBM train on the same noisy observations.

Usage:
    python main_noisy.py

Requirements:
    - config.yaml with lambda_phys=0.0 (physics loss disabled)
    - Noisy dataset generated via: python generate_data.py
"""

import yaml
import torch
from torch.utils.data import DataLoader
import os

from config import Config
from fno import FNO2d
from ebm import EBMPotential
from trainer import Trainer, FNO_EBM
from customs import DarcyPhysicsLoss
from inference import inference_deterministic, inference_probabilistic
from datautils import create_dataloaders, visualize_inference_results


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

    # EBM model
    ebm_model = EBMPotential(
        input_dim=4,  # u(1) + x(3)
        hidden_dims=[config.ebm_hidden_dim, config.ebm_hidden_dim * 2,
                     config.ebm_hidden_dim * 2, config.ebm_hidden_dim]
    )

    # Combined model
    model = FNO_EBM(fno_model, ebm_model).to(config.device)

    print(f"FNO: modes={config.fno_modes}, width={config.fno_width}")
    print(f"EBM: hidden_dim={config.ebm_hidden_dim}")

    # 3. Load Noisy Dataset
    print("\n--- Loading Noisy Dataset ---")
    train_loader, val_loader = create_dataloaders(config)

    print("✓ Dataset loaded (automatically normalized)")

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
    best_ebm_path = os.path.join(config.checkpoint_dir, 'best_model_ebm.pt')

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

        # Visualize
        visualize_inference_results(y_true_samples, y_fno_pred, stats, config)
        print("✓ Inference results saved")

    else:
        print("⚠️  Could not find best model checkpoints")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Checkpoints saved in: {config.checkpoint_dir}/")
    print("  - best_model_fno.pt")
    print("  - best_model_ebm.pt")
    print("  - current_fno.pt")
    print("  - current_ebm.pt")
    print("\nTo view results:")
    print("  - Check logs above for validation loss")
    print("  - View inference_results.png in checkpoint directory")
    print("=" * 70)


if __name__ == '__main__':
    main()