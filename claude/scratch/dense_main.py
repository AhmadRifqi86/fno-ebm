#!/usr/bin/env python3
"""
dense_main.py - Dense Training with Autoregressive Rollout Validation

This script implements the standard FNO practice:
- Dense training: Train on consecutive t→t+1 pairs (time_step=1, consecutive=True)
- Sparse testing: Autoregressive rollout to evaluate at t=10, t=20, t=40, etc.

Usage:
    # Train with dense supervision and autoregressive validation
    python dense_main.py --data_path /path/to/file.h5 --pde_type diffusion_reaction --model_type FFNO

    # With custom horizons
    python dense_main.py --data_path /path/to/file.h5 --pde_type diffusion_reaction \
        --model_type UFNO --horizons 10 20 40 --num_rollout_steps 50
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from datetime import datetime

# Import existing modules
from datautils import PDEBenchH5Loader
from fno import FNO2d, FFNO2d, UFNO2d, UFFNO2d
from config import Config
from customs import DarcyPhysicsLoss, ReactionDiffusionPhysicsLoss, ShallowWaterPhysicsLoss, NavierStokesPhysicsLoss
from trainer import Trainer, FNO_EBM
from ebm import EBMPotential


def create_dense_config(pde_type: str, model_type: str, use_pinn: bool = False):
    """
    Create configuration for dense training with autoregressive validation.

    Args:
        pde_type: Type of PDE (diffusion_reaction, navier_stokes, etc.)
        model_type: Model architecture (FNO, FFNO, UFNO, UFFNO)
        use_pinn: Whether to use PINN loss

    Returns:
        dict: Configuration dictionary
    """
    config = {
        # Model config
        'model_type': model_type,
        'fno_modes': 12,
        'fno_width': 48,
        'fno_depth': 3 if model_type in ['UFNO', 'UFFNO'] else 4,
        'fno_dropout': 0.1,
        'fno_spectral_dropout': 0.05,  # Spectral dropout for regularization

        # Training config
        'batch_size': 16,
        'epochs': 100,
        'fno_epochs': 100,
        'patience': 20,
        'fno_learning_rate': 0.001,
        'ebm_learning_rate': 0.0001,

        # FNO Optimizer config
        'fno_optimizer_config': {
            'type': 'adamw',
            'lr': 0.001,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999]
        },

        # EBM Optimizer config
        'ebm_optimizer_config': {
            'type': 'adamw',
            'lr': 0.0001,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999]
        },

        # Physics loss
        'lambda_phys': 0.01 if use_pinn else 0.0,

        # EBM training config
        'train_ebm': False,  # Disable EBM training for baseline
        'langevin_steps': 20,
        'langevin_step_size': 0.01,
        'langevin_noise_scale': 0.005,

        # Data config - for dense training
        'pde_type': pde_type,
        'train_samples': 24000,  # More training pairs with dense sampling
        'val_samples': 1000,

        # Dense training specific
        'time_step': 1,           # Dense: single timestep transitions
        'pairs_per_sim': 20,      # First 20 consecutive pairs per simulation
        'consecutive': True,      # Consecutive sampling for dense training

        # Autoregressive validation config
        'num_rollout_steps': 50,  # Number of steps to rollout during validation
        'horizon_targets': [10, 20, 40],  # Evaluate at these horizons

        # Logging
        'checkpoint_dir': f'checkpoints_dense/{pde_type}/{model_type}_pinn{use_pinn}',
        'log_file': f'logs_dense/{pde_type}_{model_type}_pinn{use_pinn}.txt',

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    return config


def train_dense_autoregressive(data_path: str, pde_type: str, model_type: str,
                                use_pinn: bool = False, horizons: list = None,
                                num_rollout_steps: int = 50):
    """
    Train with dense supervision and validate with autoregressive rollout.

    Args:
        data_path: Path to PDEBench HDF5 file
        pde_type: Type of PDE
        model_type: Model architecture (FNO, FFNO, UFNO, UFFNO)
        use_pinn: Whether to use PINN loss
        horizons: List of timesteps to evaluate during autoregressive rollout
        num_rollout_steps: Total number of autoregressive steps
    """
    print("\n" + "="*80)
    print(f"Dense Training with Autoregressive Validation - {pde_type.upper()} with {model_type}")
    print("="*80)

    # Create config
    config_dict = create_dense_config(pde_type, model_type, use_pinn)

    # Override horizons if provided
    if horizons:
        config_dict['horizon_targets'] = horizons
    if num_rollout_steps:
        config_dict['num_rollout_steps'] = num_rollout_steps

    config = Config(config_dict)

    # Setup logging
    os.makedirs('logs_dense', exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Experiment Configuration")
    print(f"{'='*80}")
    print(f"PDE Type: {pde_type}")
    print(f"Model: {model_type}")
    print(f"PINN Loss: {use_pinn} (lambda={config.lambda_phys})")
    print(f"Modes: {config.fno_modes}, Width: {config.fno_width}, Depth: {config.fno_depth}")
    print(f"Batch size: {config.batch_size}, FNO LR: {config.fno_learning_rate}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}, Patience: {config.patience}")
    print(f"\nDense Training Config:")
    print(f"  time_step: {config.time_step} (single-step transitions)")
    print(f"  pairs_per_sim: {config.pairs_per_sim}")
    print(f"  consecutive: {config.consecutive}")
    print(f"\nAutoregressive Validation Config:")
    print(f"  Rollout steps: {config.num_rollout_steps}")
    print(f"  Evaluation horizons: {config.horizon_targets}")

    # Load data with lazy loading for memory efficiency
    print("\nLoading data with lazy loading...")
    with PDEBenchH5Loader(data_path) as loader:
        loader.print_info()

        # Use lazy loading for dense training (memory efficient)
        print(f"\n  Using lazy loading for dense training")
        print(f"  time_step={config.time_step}, pairs_per_sim={config.pairs_per_sim}, consecutive={config.consecutive}")

        full_dataset = loader.to_dataset_lazy(
            time_step=config.time_step,
            pairs_per_sim=config.pairs_per_sim,
            load_all_simulations=True,
            consecutive=config.consecutive
        )

        # Split into train/val
        n_total = len(full_dataset)
        n_train = min(config.train_samples, int(0.95 * n_total))
        n_val = min(config.val_samples, int(0.05 * n_total))

        indices = np.random.RandomState(seed=42).permutation(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]

        # Create subset datasets
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        print(f"\nDataset split:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Initialize FNO model
    print(f"\nInitializing {model_type} model...")
    if model_type == 'FNO':
        fno_model = FNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            num_layers=4,
            dropout=config.fno_dropout,
            spectral_dropout=getattr(config, 'fno_spectral_dropout', 0.0)
        )
    elif model_type == 'FFNO':
        fno_model = FFNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            num_layers=4,
            dropout=config.fno_dropout,
            spectral_dropout=getattr(config, 'fno_spectral_dropout', 0.0)
        )
    elif model_type == 'UFNO':
        fno_model = UFNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            depth=config.fno_depth,
            dropout=config.fno_dropout,
            spectral_dropout=getattr(config, 'fno_spectral_dropout', 0.0)
        )
    elif model_type == 'UFFNO':
        fno_model = UFFNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            depth=config.fno_depth,
            dropout=config.fno_dropout,
            spectral_dropout=getattr(config, 'fno_spectral_dropout', 0.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    fno_model = fno_model.to(config.device)
    n_fno_params = sum(p.numel() for p in fno_model.parameters() if p.requires_grad)
    print(f"FNO parameters: {n_fno_params:,}")

    # Initialize EBM model
    print("Initializing EBM model...")
    ebm_model = EBMPotential(input_dim=4, hidden_dims=[64, 128, 64]).to(config.device)
    n_ebm_params = sum(p.numel() for p in ebm_model.parameters() if p.requires_grad)
    print(f"EBM parameters: {n_ebm_params:,}")
    print(f"Total parameters: {n_fno_params + n_ebm_params:,}")

    # Wrap in FNO_EBM
    model = FNO_EBM(fno_model, ebm_model)

    # Create physics loss function based on PDE type
    print(f"\nCreating physics loss for {pde_type}...")
    if 'darcy' in pde_type.lower():
        phy_loss = DarcyPhysicsLoss()
    elif 'diffusion' in pde_type.lower() or 'reaction' in pde_type.lower():
        phy_loss = ReactionDiffusionPhysicsLoss()
    elif 'shallow' in pde_type.lower() or 'water' in pde_type.lower():
        phy_loss = ShallowWaterPhysicsLoss()
    elif 'navier' in pde_type.lower() or 'stokes' in pde_type.lower():
        phy_loss = NavierStokesPhysicsLoss()
    else:
        print(f"WARNING: Unknown PDE type '{pde_type}', using Darcy physics loss as default")
        phy_loss = DarcyPhysicsLoss()

    # Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        phy_loss=phy_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Train the model
    print(f"\n{'='*80}")
    print("Starting Dense Training with Autoregressive Validation")
    print("="*80)
    trainer.train_staged()

    # Run final autoregressive validation
    print(f"\n{'='*80}")
    print("Final Autoregressive Validation")
    print("="*80)
    horizon_errors = trainer.validate_autoregressive(
        num_steps=config.num_rollout_steps,
        horizon_targets=config.horizon_targets
    )

    print("\nAutoregressive Rollout Results:")
    for horizon, error in horizon_errors.items():
        print(f"  t={horizon:3d}: RelL2 = {error:.4%}")

    print(f"\n{'='*80}")
    print("✓ Training Complete!")
    print("="*80)
    print(f"Best model saved to: {config.checkpoint_dir}")
    print(f"Logs saved to: logs_dense/")
    print(f"\nComparison to Sparse Training:")
    print(f"  Dense training trains on t→t+1 pairs (more supervision)")
    print(f"  Sparse training trains on t=0→t=10 directly (larger gaps)")
    print(f"  Both are evaluated at the same horizons for fair comparison")
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Dense Training with Autoregressive Rollout")

    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to PDEBench HDF5 file')

    parser.add_argument('--pde_type', type=str, required=True,
                       help='PDE type (e.g., diffusion_reaction, darcy, navier_stokes)')

    parser.add_argument('--model_type', type=str, default='FFNO',
                       choices=['FNO', 'FFNO', 'UFNO', 'UFFNO'],
                       help='Model architecture to use')

    parser.add_argument('--use_pinn', action='store_true',
                       help='Enable PINN loss')

    parser.add_argument('--horizons', nargs='+', type=int,
                       default=[10, 20, 40],
                       help='Timesteps to evaluate during autoregressive rollout')

    parser.add_argument('--num_rollout_steps', type=int, default=50,
                       help='Total number of autoregressive steps to rollout')

    args = parser.parse_args()

    # Execute training
    train_dense_autoregressive(
        data_path=args.data_path,
        pde_type=args.pde_type,
        model_type=args.model_type,
        use_pinn=args.use_pinn,
        horizons=args.horizons,
        num_rollout_steps=args.num_rollout_steps
    )


if __name__ == '__main__':
    main()