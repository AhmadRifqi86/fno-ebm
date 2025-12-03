#!/usr/bin/env python3
"""
main.py - UQ Paper Training Script

Trains FNO-EBM models on 4 PDEs with multi-nu support.
"""

import argparse
import json
import os
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

from config import Config, Factory
from datautils import load_pde_data, create_dataloaders
from fno import FNOTrainer
from kanebm import EBMTrainer


def train_fno_ebm(config_dict: dict, data_path: str, pde_type: str,
                  nu_values: list = None, output_dir: str = 'experiments'):
    """
    Main training function for FNO-EBM on a single PDE.

    Args:
        config_dict: Configuration dictionary
        data_path: Path to data file
        pde_type: 'burgers', 'advection', 'diffusion_reaction', 'navier_stokes'
        nu_values: List of nu values (for 1D PDEs)
        output_dir: Output directory for checkpoints/logs
    """
    # Setup output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{pde_type}_nu{len(nu_values) if nu_values else 'all'}_{timestamp}"
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_dict['log_dir'] = str(exp_dir / 'logs')
    config_dict['checkpoint_dir'] = str(exp_dir / 'checkpoints')
    os.makedirs(config_dict['log_dir'], exist_ok=True)
    os.makedirs(config_dict['checkpoint_dir'], exist_ok=True)

    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    config = Config(config_dict)

    print(f"\n{'='*80}")
    print(f"Training: {pde_type}")
    print(f"Nu values: {nu_values if nu_values else 'all'}")
    print(f"Device: {config.device}")
    print(f"Output: {exp_dir}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    dataset = load_pde_data(
        filepath=data_path,
        pde_type=pde_type,
        nu_values=nu_values,
        max_samples=config.max_samples if hasattr(config, 'max_samples') else None
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Input shape: {dataset.X[0].shape}")
    print(f"Output shape: {dataset.U[0].shape}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        train_ratio=config.train_ratio if hasattr(config, 'train_ratio') else 0.8,
        val_ratio=config.val_ratio if hasattr(config, 'val_ratio') else 0.1,
        batch_size=config.batch_size,
        seed=config.seed if hasattr(config, 'seed') else 42
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Initialize FNO model
    print("\nInitializing FNO model...")
    fno_model = Factory.create_fno(config)
    print(f"FNO parameters: {sum(p.numel() for p in fno_model.parameters()):,}")

    # Train FNO
    print("\n" + "="*80)
    print("Stage 1: Training FNO")
    print("="*80)
    fno_trainer = FNOTrainer(model=fno_model, config=config)
    fno_trainer.train(train_loader, val_loader)

    # Initialize EBM
    print("\n" + "="*80)
    print("Stage 2: Training EBM")
    print("="*80)
    # ebm_model = KANEBM(#change this by calling a Factory function, eg: ebm_model = create_ebm(config)
    #     input_dim=4,  # [x, y, y_fno, y_true]
    #     hidden_dim=config.ebm_hidden_dim if hasattr(config, 'ebm_hidden_dim') else 64,
    #     num_layers=config.ebm_layers if hasattr(config, 'ebm_layers') else 3
    # ).to(config.device)
    ebm_model = Factory.create_ebm(config)
    print(f"EBM parameters: {sum(p.numel() for p in ebm_model.parameters()):,}")

    ebm_trainer = EBMTrainer(
        fno_model=fno_model,
        ebm_model=ebm_model,
        config=config
    )
    ebm_trainer.train(train_loader, val_loader)

    # Evaluate on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    test_metrics = ebm_trainer.evaluate(test_loader)
    print(f"Test Metrics:")
    for key, val in test_metrics.items():
        print(f"  {key}: {val:.6f}")

    # Save final results
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}\n")


def create_default_config(pde_type: str) -> dict:
    """Create default configuration for a PDE type."""
    config = {
        # Model
        'fno_modes': 12,
        'fno_width': 64,
        'fno_depth': 4,
        'ebm_hidden_dim': 64,
        'ebm_layers': 3,

        # Training
        'batch_size': 32,
        'fno_epochs': 100,
        'ebm_epochs': 50,
        'fno_lr': 1e-3,
        'ebm_lr': 1e-4,
        'patience': 20,

        # Data
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'max_samples': None,
        'seed': 42,

        # Tracking
        'enable_tracking': True,
        'tracking_backend': 'custom',  # 'custom' or 'tensorboard'

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # PDE-specific adjustments
    if pde_type in ['burgers', 'advection']:
        # 1D PDEs need fewer parameters
        config['fno_modes'] = 16
        config['batch_size'] = 64
    elif pde_type == 'navier_stokes':
        # NS needs more capacity
        config['fno_modes'] = 20
        config['fno_width'] = 96
        config['batch_size'] = 16

    return config


def main():
    parser = argparse.ArgumentParser(description="UQ Paper Training Script")

    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file (.hdf5, .h5, or .pt)')
    parser.add_argument('--pde_type', type=str, required=True,
                        choices=['burgers', 'advection', 'diffusion_reaction', 'navier_stokes'],
                        help='PDE type')

    # Optional arguments
    parser.add_argument('--nu_values', nargs='+', type=float, default=None,
                        help='Nu values to train on (for 1D PDEs only)')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config JSON')

    # Model hyperparameters
    parser.add_argument('--fno_modes', type=int, default=None)
    parser.add_argument('--fno_width', type=int, default=None)
    parser.add_argument('--fno_depth', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--fno_epochs', type=int, default=None)
    parser.add_argument('--ebm_epochs', type=int, default=None)

    # Tracking backend
    parser.add_argument('--tracking_backend', type=str, default=None,
                        choices=['custom', 'tensorboard'],
                        help='Tracking backend (custom or tensorboard)')

    args = parser.parse_args()

    # Create config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = create_default_config(args.pde_type)

    # Override with command line arguments
    if args.fno_modes is not None:
        config_dict['fno_modes'] = args.fno_modes
    if args.fno_width is not None:
        config_dict['fno_width'] = args.fno_width
    if args.fno_depth is not None:
        config_dict['fno_depth'] = args.fno_depth
    if args.batch_size is not None:
        config_dict['batch_size'] = args.batch_size
    if args.fno_epochs is not None:
        config_dict['fno_epochs'] = args.fno_epochs
    if args.ebm_epochs is not None:
        config_dict['ebm_epochs'] = args.ebm_epochs
    if args.tracking_backend is not None:
        config_dict['tracking_backend'] = args.tracking_backend

    # Run training
    train_fno_ebm(
        config_dict=config_dict,
        data_path=args.data_path,
        pde_type=args.pde_type,
        nu_values=args.nu_values,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
