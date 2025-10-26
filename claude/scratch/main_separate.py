#!/usr/bin/env python3
"""
main_separate.py - Dual-Dataset Training Mode

Trains FNO on clean data (with physics loss) and EBM on noisy data (for uncertainty).

Usage:
    1. Generate dual dataset: python generate_dual_data.py
    2. Update config.yaml: set lambda_phys=0.05 (enable physics loss)
    3. Run training: python main_separate.py

Requirements:
    - Clean dataset: data/darcy_medium_clean_res64_train.npz
    - Noisy dataset: data/darcy_medium_noisy_res64_train.npz
    - config.yaml with lambda_phys>0 (physics loss enabled)
"""

import yaml
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path

from config import Config
from fno import FNO2d
from ebm import ConvEBM
from trainer import Trainer, FNO_EBM
from customs import DarcyPhysicsLoss
from inference import inference_deterministic, inference_probabilistic
from datautils import PDEDataset, visualize_inference_results


def main():
    """
    Dual-dataset training pipeline:
    - FNO trains on clean data WITH physics loss
    - EBM trains on noisy data for uncertainty quantification
    """

    print("=" * 70)
    print("FNO-EBM TRAINING: DUAL-DATASET MODE (Clean + Noisy)")
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
    print(f"Physics Loss Weight: {config.lambda_phys}")

    if config.lambda_phys == 0:
        print("\n⚠️  WARNING: lambda_phys = 0 detected!")
        print("   For dual-dataset mode with clean data, physics loss can be enabled")
        print("   Recommended: lambda_phys = 0.05")

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
    # ebm_model = EBMPotential(
    #     input_dim=4,  # u(1) + x(3)
    #     hidden_dims=[config.ebm_hidden_dim, config.ebm_hidden_dim * 2,
    #                  config.ebm_hidden_dim * 2, config.ebm_hidden_dim]
    # )
    ebm_model = ConvEBM(
      in_channels=4,  # u + (x, y, a)
      hidden_channels=[64, 128, 128, 64]  # Convolutional channels
   )


    # Combined model
    model = FNO_EBM(fno_model, ebm_model).to(config.device)

    print(f"FNO: modes={config.fno_modes}, width={config.fno_width}")
    print(f"EBM: hidden_dim={config.ebm_hidden_dim}")

    # 3. Load Dual Datasets
    print("\n--- Loading Dual Datasets ---")

    data_dir = Path(config.data_dir) if hasattr(config, 'data_dir') else Path('../data')
    resolution = config.grid_size if hasattr(config, 'grid_size') else 64
    complexity = config.complexity if hasattr(config, 'complexity') else 'medium'

    # Construct file paths
    clean_train_file = data_dir / f"darcy_{complexity}_clean_res{resolution}_train.npz"
    clean_val_file = data_dir / f"darcy_{complexity}_clean_res{resolution}_val.npz"
    noisy_train_file = data_dir / f"darcy_{complexity}_noisy_res{resolution}_train.npz"
    noisy_val_file = data_dir / f"darcy_{complexity}_noisy_res{resolution}_val.npz"

    # Check if files exist
    missing_files = []
    for f in [clean_train_file, clean_val_file, noisy_train_file, noisy_val_file]:
        if not f.exists():
            missing_files.append(str(f))

    if missing_files:
        print("\n❌ ERROR: Missing dataset files!")
        print("Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run: python generate_dual_data.py")
        return

    # Load clean datasets (for FNO)
    print(f"\nLoading CLEAN data for FNO training...")
    fno_train_dataset = PDEDataset.from_file(str(clean_train_file), normalize_output=True)
    fno_val_dataset = PDEDataset.from_file(str(clean_val_file), normalize_output=True)

    # Load noisy datasets (for EBM)
    print(f"Loading NOISY data for EBM training...")
    ebm_train_dataset = PDEDataset.from_file(str(noisy_train_file), normalize_output=True)
    ebm_val_dataset = PDEDataset.from_file(str(noisy_val_file), normalize_output=True)

    print("✓ All datasets loaded (automatically normalized)")

    # 4. Create DataLoaders
    print("\n--- Creating DataLoaders ---")

    batch_size = config.batch_size if hasattr(config, 'batch_size') else 16
    num_workers = config.num_workers if hasattr(config, 'num_workers') else 0

    fno_train_loader = DataLoader(
        fno_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    fno_val_loader = DataLoader(
        fno_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    ebm_train_loader = DataLoader(
        ebm_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    ebm_val_loader = DataLoader(
        ebm_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✓ DataLoaders created (batch_size={batch_size})")

    # 5. Initialize Trainer (Dual-Dataset Mode)
    print("\n--- Initializing Trainer (Dual-Dataset Mode) ---")

    # Initialize Darcy physics loss
    phy_loss = DarcyPhysicsLoss(source_term=1.0)

    trainer = Trainer(
        model=model,
        phy_loss=phy_loss,
        train_loader=fno_train_loader,   # Clean data for FNO
        val_loader=fno_val_loader,       # Clean data for FNO validation
        config=config,
        ebm_train_loader=ebm_train_loader,  # Noisy data for EBM
        ebm_val_loader=ebm_val_loader       # Noisy data for EBM validation
    )

    print("Training mode: Dual-dataset (clean + noisy)")
    print("  - FNO trains on: CLEAN data (physics loss enabled)")
    print("  - EBM trains on: NOISY data (uncertainty quantification)")

    # 6. Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    trainer.train_staged()
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    # 7. Inference
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

        # Use NOISY validation data for inference (realistic scenario)
        x_samples, y_true_samples = next(iter(ebm_val_loader))
        x_samples = x_samples.to(config.device)

        # Deterministic inference (FNO on clean-trained model)
        print("Running FNO deterministic inference...")
        y_fno_pred = inference_deterministic(model, x_samples, device=config.device)

        # Probabilistic inference (EBM captures noise uncertainty)
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
        # (predictions are in normalized space, convert back)
        y_fno_pred_denorm = fno_train_dataset.denormalize(y_fno_pred)
        y_true_denorm = ebm_train_dataset.denormalize(y_true_samples)

        # Denormalize stats
        stats_denorm = {
            'mean': fno_train_dataset.denormalize(stats['mean']),
            'std': stats['std'] * fno_train_dataset.u_std,  # Scale std
        }

        # Visualize
        visualize_inference_results(y_true_denorm, y_fno_pred_denorm, stats_denorm, config)
        print("✓ Inference results saved")

    else:
        print("⚠️  Could not find best model checkpoints")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Checkpoints saved in: {config.checkpoint_dir}/")
    print("  - best_model_fno.pt (trained on clean data)")
    print("  - best_model_ebm.pt (trained on noisy data)")
    print("  - current_fno.pt")
    print("  - current_ebm.pt")
    print("\nDatasets used:")
    print(f"  - FNO training: {clean_train_file}")
    print(f"  - EBM training: {noisy_train_file}")
    print("\nTo view results:")
    print("  - Check logs above for validation loss")
    print("  - View inference_results.png in checkpoint directory")
    print("\nPhysics-informed learning:")
    print(f"  - Physics loss weight: {config.lambda_phys}")
    print("  - FNO learned true PDE operator from clean data")
    print("  - EBM learned observation noise from noisy data")
    print("=" * 70)


if __name__ == '__main__':
    main()