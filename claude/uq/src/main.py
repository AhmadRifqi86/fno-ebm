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

    # Load data with memory-safe defaults
    print("Loading data...")
    max_samples = getattr(config, 'max_samples', 100)  # Default to 100 samples per nu value
    time_step_spacing = getattr(config, 'time_step_spacing', 10)
    max_pairs_per_sample = getattr(config, 'max_pairs_per_sample', 20)

    print(f"Data loading settings:")
    print(f"  max_samples: {max_samples}")
    print(f"  time_step_spacing: {time_step_spacing}")
    print(f"  max_pairs_per_sample: {max_pairs_per_sample}")

    dataset = load_pde_data(
        filepath=data_path,
        pde_type=pde_type,
        nu_values=nu_values,
        max_samples=max_samples,
        time_step_spacing=time_step_spacing,
        max_pairs_per_sample=max_pairs_per_sample
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
    fno_model = Factory.create_fno(config, pde_type=pde_type)
    print(f"FNO Model type: {fno_model.__class__.__name__}")
    print(f"FNO parameters: {sum(p.numel() for p in fno_model.parameters()):,}")

    # Train FNO
    print("\n" + "="*80)
    print("Stage 1: Training FNO")
    print("="*80)
    fno_trainer = FNOTrainer(model=fno_model, config=config)
    fno_epochs = getattr(config, 'fno_epochs', 100)
    fno_trainer.train(train_loader, val_loader, fno_epochs)

    # Initialize EBM
    print("\n" + "="*80)
    print("Stage 2: Training EBM")
    print("="*80)

    # Calculate EBM input_dim based on data shape
    sample_x, sample_u = dataset[0]
    n_spatial = sample_x.shape[0] * sample_x.shape[1]  # n_x * n_y
    n_input_channels = sample_x.shape[-1]  # coordinate channels
    n_output_channels = sample_u.shape[-1]  # output channels

    # Total flattened size: u + x + u_fno
    ebm_input_dim = n_spatial * (n_output_channels + n_input_channels + n_output_channels)

    print(f"EBM input calculation: n_spatial={n_spatial}, input_ch={n_input_channels}, output_ch={n_output_channels}")
    print(f"EBM input_dim: {ebm_input_dim}")

    # Update config_dict BEFORE creating Config object
    config_dict['ebm_input_dim'] = ebm_input_dim

    # Recreate config with updated input_dim
    config = Config(config_dict)

    ebm_model = Factory.create_ebm(config)
    print(f"EBM parameters: {sum(p.numel() for p in ebm_model.parameters()):,}")

    ebm_trainer = EBMTrainer(
        model=ebm_model,
        config=config,
        fno_model=fno_model
    )
    ebm_epochs = getattr(config, 'ebm_epochs', 50)
    ebm_trainer.train(train_loader, val_loader, ebm_epochs)

    # Evaluate on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    test_loss = ebm_trainer.validate(test_loader)
    test_metrics = {'test_loss': test_loss}
    print(f"Test Metrics:")
    print(f"  test_loss: {test_loss:.6f}")

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
        'fno_epochs': 20,
        'ebm_epochs': 20,
        'fno_lr': 1e-3,
        'ebm_lr': 1e-4,
        'patience': 20,

        # Data - Memory-safe defaults
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'max_samples': 250,  # Default to 100 samples per nu value to prevent OOM
        'time_step_spacing': 10,  # Time step spacing for 1D PDEs
        'max_pairs_per_sample': 20,  # Max pairs per sample for 1D PDEs
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
