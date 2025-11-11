#!/usr/bin/env python3
"""
main.py - PDEBench Benchmarking Entry Point

Week 1: Infrastructure & Setup
- Inspect PDEBench HDF5 files
- Test data loading for different PDE types
- Set up experiment tracking
- Run baseline training experiments
- Run experiment grid searches

Usage:
    # Inspect single HDF5 file
    python main.py --mode inspect --data_path /path/to/pdebench/file.h5

    # Inspect all files in directory
    python main.py --mode inspect_dir --data_dir /path/to/pdebench/

    # Test data loading
    python main.py --mode test_loading --data_dir /path/to/pdebench/

    # Train single configuration
    python main.py --mode train --data_path file.h5 --pde_type diffusion_reaction --model_type FNO

    # Run experiment grid (default hyperparameters)
    python main.py --mode experiment_grid --data_path file.h5 --pde_type diffusion_reaction

    # Run experiment grid with custom hyperparameters
    python main.py --mode experiment_grid --data_path file.h5 --pde_type diffusion_reaction \
        --models FNO UFNO --modes 8 12 16 --widths 48 64 --depths 2 3 \
        --pinn_constants 0.0 0.001 0.01 --output_dir my_experiments

    # Run with temporal augmentation (2x data)
    python main.py --mode train --data_path file.h5 --pde_type diffusion_reaction \
        --temporal_augmentation --num_temporal_splits 2

    # Run with temporal augmentation (4x data)
    python main.py --mode train --data_path file.h5 --pde_type diffusion_reaction \
        --temporal_augmentation --num_temporal_splits 4
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import pandas as pd
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt

# Import existing modules
from datautils import PDEBenchH5Loader
from fno import FNO2d, FFNO2d, UFNO2d, UFFNO2d
from config import Config
from customs import DarcyPhysicsLoss, ReactionDiffusionPhysicsLoss, ShallowWaterPhysicsLoss, NavierStokesPhysicsLoss
from trainer import Trainer, FNO_EBM
from ebm import EBMPotential

def inspect_h5_file(filepath: str):
    """
    Inspect a single PDEBench HDF5 file.

    Args:
        filepath: Path to .h5 file
    """
    print("\n" + "="*80)
    print("WEEK 1 TASK: Inspecting PDEBench HDF5 File")
    print("="*80)

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return

    with PDEBenchH5Loader(filepath) as loader:
        loader.print_info()

        # Try to peek at first sample
        try:
            sample = loader.peek_sample(sample_idx=0, timestep=0)
            print(f"\nFirst sample (t=0) statistics:")
            print(f"  Min: {sample.min():.6f}")
            print(f"  Max: {sample.max():.6f}")
            print(f"  Mean: {sample.mean():.6f}")
            print(f"  Std: {sample.std():.6f}")
        except Exception as e:
            print(f"\nCould not peek sample: {e}")


def inspect_directory(data_dir: str):
    """
    Inspect all HDF5 files in a directory.

    Args:
        data_dir: Directory containing .h5 files
    """
    print("\n" + "="*80)
    print("WEEK 1 TASK: Inspecting All PDEBench Files in Directory")
    print("="*80)

    h5_files = list(Path(data_dir).glob("*.h5")) + list(Path(data_dir).glob("*.hdf5"))

    if not h5_files:
        print(f"No .h5 or .hdf5 files found in {data_dir}")
        return

    print(f"\nFound {len(h5_files)} HDF5 files:")
    for i, filepath in enumerate(h5_files, 1):
        print(f"\n{'='*80}")
        print(f"File {i}/{len(h5_files)}: {filepath.name}")
        print('='*80)
        inspect_h5_file(str(filepath))


def test_data_loading(data_dir: str, pde_type: str = None):
    """
    Test loading PDEBench data for different PDE types.

    Args:
        data_dir: Directory containing PDEBench .h5 files
        pde_type: Specific PDE to test (if None, tests all found)
    """
    print("\n" + "="*80)
    print("WEEK 1 TASK: Testing PDEBench Data Loading")
    print("="*80)

    h5_files = list(Path(data_dir).glob("*.h5")) + list(Path(data_dir).glob("*.hdf5"))

    if not h5_files:
        print(f"No HDF5 files found in {data_dir}")
        return

    # Filter by PDE type if specified
    if pde_type:
        h5_files = [f for f in h5_files if pde_type.lower() in f.name.lower()]
        if not h5_files:
            print(f"No files matching PDE type '{pde_type}' found")
            return

    for filepath in h5_files:
        print(f"\n{'='*70}")
        print(f"Testing: {filepath.name}")
        print('='*70)

        try:
            with PDEBenchH5Loader(str(filepath)) as loader:
                # Inspect first
                shape = loader.get_shape()
                print(f"Data shape: {shape}")

                # Try loading as temporal dataset
                if len(shape) == 4:
                    print("\nAttempting to load as temporal dataset (t=0 → t=-1)...")
                    dataset = loader.to_dataset(
                        input_t=0,
                        output_t=-1,
                        num_samples=100  # Load only 100 for testing
                    )
                    print(f"✓ Successfully created dataset:")
                    print(f"  Samples: {len(dataset)}")
                    print(f"  Input shape: {dataset.X[0].shape}")
                    print(f"  Output shape: {dataset.U[0].shape}")

                    # Test DataLoader
                    loader_test = DataLoader(dataset, batch_size=4, shuffle=True)
                    x_batch, u_batch = next(iter(loader_test))
                    print(f"\n  Test batch shapes:")
                    print(f"    X: {x_batch.shape}")
                    print(f"    U: {u_batch.shape}")

                elif len(shape) == 3:
                    print("\nSteady-state data detected.")
                    print("Need to identify input_key and output_key for this PDE.")
                    print("Common pairs: ('nu', 'tensor') for Darcy Flow")

        except Exception as e:
            print(f"✗ Error loading {filepath.name}: {e}")
            import traceback
            traceback.print_exc()


def create_baseline_config(pde_type: str, model_type: str, use_pinn: bool = False):
    """
    Create baseline training configuration for a PDE + model combination.
    
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
        
        # Training config
        'batch_size': 16,
        'epochs': 100,
        'fno_epochs': 100,
        'patience': 20,
        'fno_learning_rate': 0.001,   # FNO learning rate
        'ebm_learning_rate': 0.0001,  # EBM learning rate

        # FNO Optimizer config (nested)
        'fno_optimizer_config': {
            'type': 'adamw',
            'lr': 0.001,              # Moderate LR for stable convergence
            'weight_decay': 0.01,     # Light regularization
            'betas': [0.9, 0.999]
        },
        
        # EBM Optimizer config (nested) - for future use
        'ebm_optimizer_config': {
            'type': 'adamw',
            'lr': 0.0001,             # EBM typically trains slower
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
        
        # Data config
        'pde_type': pde_type,
        'train_samples': 200,  # Further reduced for high-res Darcy data
        'val_samples': 50,     # Further reduced
        'test_samples': 50,    # Further reduced
        
        # Logging
        'checkpoint_dir': f'checkpoints/{pde_type}/{model_type}_pinn{use_pinn}',
        'log_file': f'logs/{pde_type}_{model_type}_pinn{use_pinn}.txt',
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    return config


def train_baseline(data_path: str, pde_type: str, model_type: str, use_pinn: bool = False,
                   temporal_augmentation: bool = False, num_temporal_splits: int = 2):
    """
    Run baseline training experiment with Trainer integration.

    Args:
        data_path: Path to PDEBench HDF5 file
        pde_type: Type of PDE
        model_type: Model architecture (FNO, FFNO, UFNO, UFFNO)
        use_pinn: Whether to use PINN loss
        temporal_augmentation: Whether to use temporal data augmentation
        num_temporal_splits: Number of temporal splits for augmentation
    """
    print("\n" + "="*80)
    print(f"Baseline Training - {pde_type.upper()} with {model_type}")
    print("="*80)

    # Create config
    config_dict = create_baseline_config(pde_type, model_type, use_pinn)
    config = Config(config_dict)

    # Setup logging
    os.makedirs('logs', exist_ok=True)
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
    if temporal_augmentation:
        print(f"Temporal Augmentation: ENABLED ({num_temporal_splits} splits = {num_temporal_splits}x data)")

    # Load data
    print("\nLoading data...")
    with PDEBenchH5Loader(data_path) as loader:
        loader.print_info()

        # Calculate how many samples to load (to avoid OOM)
        # Load enough for train + val + some buffer
        max_samples_needed = config.train_samples + config.val_samples + 200
        print(f"\n  Loading only {max_samples_needed} samples to save memory")

        # Load only needed samples with optional temporal augmentation
        full_dataset = loader.to_dataset(
            input_t=0,
            output_t=-1,
            num_samples=max_samples_needed,  # Only load what we need!
            temporal_augmentation=temporal_augmentation,
            num_temporal_splits=num_temporal_splits
        )

        # Split into train/val/test
        n_total = len(full_dataset)
        n_train = min(config.train_samples, int(0.8 * n_total))
        n_val = min(config.val_samples, int(0.1 * n_total))

        indices = np.random.permutation(n_total)
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
        pin_memory=False  # Disabled to reduce memory usage
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False  # Disabled to reduce memory usage
    )

    # Initialize FNO model
    print(f"\nInitializing {model_type} model...")
    if model_type == 'FNO':
        fno_model = FNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            num_layers=4,
            dropout=config.fno_dropout
        )
    elif model_type == 'FFNO':
        fno_model = FFNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            num_layers=6,
            dropout=config.fno_dropout
        )
    elif model_type == 'UFNO':
        fno_model = UFNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            depth=config.fno_depth,
            dropout=config.fno_dropout
        )
    elif model_type == 'UFFNO':
        fno_model = UFFNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            depth=config.fno_depth,
            dropout=config.fno_dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    fno_model = fno_model.to(config.device)
    n_fno_params = sum(p.numel() for p in fno_model.parameters() if p.requires_grad)
    print(f"FNO parameters: {n_fno_params:,}")

    # Initialize EBM model (simple placeholder for baseline)
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
    print("Starting Training")
    print("="*80)
    trainer.train_staged()

    print(f"\n{'='*80}")
    print("✓ Training Complete!")
    print("="*80)
    print(f"Best model saved to: {config.checkpoint_dir}")
    print(f"Logs saved to: logs/")
    print(f"\n{'='*80}\n")


def generate_experiment_configs(pde_type, model_types, modes_list, width_list, depth_list, pinn_constants):
    """
    Generate all experiment configurations from hyperparameter grid.

    Args:
        pde_type: Type of PDE (e.g., 'darcy', 'diffusion_reaction')
        model_types: List of model architectures (e.g., ['FNO', 'FFNO', 'UFNO', 'UFFNO'])
        modes_list: List of Fourier modes (e.g., [8, 12, 16, 20])
        width_list: List of hidden dimensions (e.g., [48, 64, 96])
        depth_list: List of depths for U-shaped models (e.g., [2, 3, 4])
        pinn_constants: List of PINN loss weights (e.g., [0.0, 0.001, 0.01])

    Returns:
        List of configuration dictionaries
    """
    configs = []
    exp_id = 0

    for model_type in model_types:
        for modes in modes_list:
            for width in width_list:
                # Depth only relevant for U-shaped architectures
                if model_type in ['UFNO', 'UFFNO']:
                    depth_options = depth_list
                else:
                    depth_options = [4]  # Default depth for non-U architectures

                for depth in depth_options:
                    for lambda_phys in pinn_constants:
                        config = {
                            'exp_id': exp_id,
                            'pde_type': pde_type,
                            'model_type': model_type,
                            'fno_modes': modes,
                            'fno_width': width,
                            'fno_depth': depth,
                            'lambda_phys': lambda_phys,
                            'fno_dropout': 0.1,
                            'batch_size': 32,
                            'learning_rate': 0.001,
                            'fno_learning_rate': 0.005,
                            'ebm_learning_rate': 0.0001,
                            'fno_epochs': 5, #100
                            'patience': 20,
                            'train_samples': 4000,
                            'val_samples': 500,
                            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                        }
                        configs.append(config)
                        exp_id += 1

    return configs

def run_single_experiment(config, data_path, experiment_dir):
    """
    Run a single training experiment with given configuration using Trainer.

    Args:
        config: Configuration dictionary
        data_path: Path to HDF5 data file
        experiment_dir: Directory to save results

    Returns:
        results_dict: Dictionary with experiment results
    """
    exp_id = config['exp_id']
    pde_type = config['pde_type']
    model_type = config['model_type']

    # Create experiment-specific directories
    exp_name = f"exp{exp_id:03d}_{model_type}_m{config['fno_modes']}_w{config['fno_width']}_d{config['fno_depth']}_p{config['lambda_phys']}"
    exp_log_dir = os.path.join(experiment_dir, 'logs')
    exp_checkpoint_dir = os.path.join(experiment_dir, 'checkpoints', exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)
    os.makedirs(exp_checkpoint_dir, exist_ok=True)

    log_file = os.path.join(exp_log_dir, f'{exp_name}.txt')

    def log(message):
        """Log to both console and file."""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    log(f"\n{'='*80}")
    log(f"Experiment {exp_id}/{config.get('total_experiments', '?')}")
    log(f"{'='*80}")
    log(f"Model: {model_type}, Modes: {config['fno_modes']}, Width: {config['fno_width']}, Depth: {config['fno_depth']}")
    log(f"PINN λ: {config['lambda_phys']}")

    try:
        # Convert config dict to Config object
        config_dict = config.copy()
        config_dict['checkpoint_dir'] = exp_checkpoint_dir
        config_obj = Config(config_dict)

        # Load data
        log("Loading data...")
        with PDEBenchH5Loader(data_path) as loader:
            full_dataset = loader.to_dataset(time_step=1, pairs_per_sim=10, load_all_simulations=True)

            # Split data
            n_total = len(full_dataset)
            n_train = min(config['train_samples'], int(0.8 * n_total))
            n_val = min(config['val_samples'], int(0.1 * n_total))

            indices = np.random.RandomState(seed=42).permutation(n_total)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]

            train_dataset = Subset(full_dataset, train_idx)
            val_dataset = Subset(full_dataset, val_idx)

            log(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        # Initialize FNO model
        log(f"Initializing {model_type} model...")
        if model_type == 'FNO':
            fno_model = FNO2d(
                modes1=config['fno_modes'],
                modes2=config['fno_modes'],
                width=config['fno_width'],
                num_layers=4,
                dropout=config['fno_dropout']
            )
        elif model_type == 'FFNO':
            fno_model = FFNO2d(
                modes1=config['fno_modes'],
                modes2=config['fno_modes'],
                width=config['fno_width'],
                num_layers=4,
                dropout=config['fno_dropout']
            )
        elif model_type == 'UFNO':
            fno_model = UFNO2d(
                modes1=config['fno_modes'],
                modes2=config['fno_modes'],
                width=config['fno_width'],
                depth=config['fno_depth'],
                dropout=config['fno_dropout']
            )
        elif model_type == 'UFFNO':
            fno_model = UFFNO2d(
                modes1=config['fno_modes'],
                modes2=config['fno_modes'],
                width=config['fno_width'],
                depth=config['fno_depth'],
                dropout=config['fno_dropout']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        fno_model = fno_model.to(config['device'])
        n_fno_params = sum(p.numel() for p in fno_model.parameters() if p.requires_grad)
        log(f"FNO parameters: {n_fno_params:,}")

        # Initialize EBM model
        log("Initializing EBM model...")
        ebm_model = EBMPotential(input_dim=4, hidden_dims=[64, 128, 64]).to(config['device'])
        n_ebm_params = sum(p.numel() for p in ebm_model.parameters() if p.requires_grad)
        log(f"EBM parameters: {n_ebm_params:,}")
        log(f"Total parameters: {n_fno_params + n_ebm_params:,}")

        # Wrap in FNO_EBM
        model = FNO_EBM(fno_model, ebm_model)

        # Create physics loss function based on PDE type
        log(f"Creating physics loss for {pde_type}...")
        if 'darcy' in pde_type.lower():
            phy_loss = DarcyPhysicsLoss()
        elif 'diffusion' in pde_type.lower() or 'reaction' in pde_type.lower():
            phy_loss = ReactionDiffusionPhysicsLoss()
        elif 'shallow' in pde_type.lower() or 'water' in pde_type.lower():
            phy_loss = ShallowWaterPhysicsLoss()
        elif 'navier' in pde_type.lower() or 'stokes' in pde_type.lower():
            phy_loss = NavierStokesPhysicsLoss()
        else:
            log(f"WARNING: Unknown PDE type '{pde_type}', using Darcy physics loss as default")
            phy_loss = DarcyPhysicsLoss()

        # Initialize Trainer
        log("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            phy_loss=phy_loss,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_obj
        )

        # Train the model
        log("Starting training...")
        trainer.train_staged()

        # Training completed - trainer doesn't maintain history
        # Best model is saved via checkpoint mechanism
        log(f"Training completed - Best model saved to checkpoint")

        # Collect results
        results = {
            'exp_id': exp_id,
            'model_type': model_type,
            'modes': config['fno_modes'],
            'width': config['fno_width'],
            'depth': config['fno_depth'],
            'lambda_phys': config['lambda_phys'],
            'n_params': n_fno_params + n_ebm_params,
            'best_val_loss': trainer.best_val_loss,  # Use trainer's tracked best_val_loss
            'status': 'success'
        }

        log("✓ Experiment completed successfully")

    except Exception as e:
        log(f"✗ Experiment failed: {e}")
        import traceback
        log(traceback.format_exc())

        results = {
            'exp_id': exp_id,
            'model_type': model_type,
            'modes': config['fno_modes'],
            'width': config['fno_width'],
            'depth': config['fno_depth'],
            'lambda_phys': config['lambda_phys'],
            'status': 'failed',
            'error': str(e)
        }

    return results


def run_single_experiment_old(config, data_path, experiment_dir):
    """
    Run a single training experiment with given configuration.

    Args:
        config: Configuration dictionary
        data_path: Path to HDF5 data file
        experiment_dir: Directory to save results

    Returns:
        results_dict: Dictionary with experiment results
    """
    exp_id = config['exp_id']
    pde_type = config['pde_type']
    model_type = config['model_type']

    # Create experiment-specific directories
    exp_name = f"exp{exp_id:03d}_{model_type}_m{config['fno_modes']}_w{config['fno_width']}_d{config['fno_depth']}_p{config['lambda_phys']}"
    exp_log_dir = os.path.join(experiment_dir, 'logs')
    exp_checkpoint_dir = os.path.join(experiment_dir, 'checkpoints', exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)
    os.makedirs(exp_checkpoint_dir, exist_ok=True)

    log_file = os.path.join(exp_log_dir, f'{exp_name}.txt')

    def log(message):
        """Log to both console and file."""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    log(f"\n{'='*80}")
    log(f"Experiment {exp_id}/{config.get('total_experiments', '?')}")
    log(f"{'='*80}")
    log(f"Model: {model_type}, Modes: {config['fno_modes']}, Width: {config['fno_width']}, Depth: {config['fno_depth']}")
    log(f"PINN λ: {config['lambda_phys']}")

    try:
        # Load data (use caching if possible)
        with PDEBenchH5Loader(data_path) as loader:
            full_dataset = loader.to_dataset(input_t=0, output_t=-1, num_samples=None)

        # Split data
        n_total = len(full_dataset)
        n_train = min(config['train_samples'], int(0.8 * n_total))
        n_val = min(config['val_samples'], int(0.1 * n_total))

        indices = np.random.RandomState(seed=42).permutation(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Initialize model
        if model_type == 'FNO':
            model = FNO2d(
                modes1=config['fno_modes'],
                modes2=config['fno_modes'],
                width=config['fno_width'],
                num_layers=4,
                dropout=config['fno_dropout']
            )
        elif model_type == 'FFNO':
            model = FFNO2d(
                modes1=config['fno_modes'],
                modes2=config['fno_modes'],
                width=config['fno_width'],
                num_layers=6,
                dropout=config['fno_dropout']
            )
        elif model_type == 'UFNO':
            model = UFNO2d(
                modes1=config['fno_modes'],
                modes2=config['fno_modes'],
                width=config['fno_width'],
                depth=config['fno_depth'],
                dropout=config['fno_dropout']
            )
        elif model_type == 'UFFNO':
            model = UFFNO2d(
                modes1=config['fno_modes'],
                modes2=config['fno_modes'],
                width=config['fno_width'],
                depth=config['fno_depth'],
                dropout=config['fno_dropout']
            )

        model = model.to(config['device'])
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"Parameters: {n_params:,}")
        #train_loader nya ga kepake
        # Quick evaluation (placeholder for full training)
        model.eval()
        with torch.no_grad():
            x_batch, u_batch = next(iter(val_loader))
            x_batch = x_batch.to(config['device'])
            u_batch = u_batch.to(config['device'])

            u_pred = model(x_batch)
            mse = torch.nn.functional.mse_loss(u_pred, u_batch).item()
            rel_l2 = (torch.norm(u_pred - u_batch) / torch.norm(u_batch)).item()

        log(f"Val MSE: {mse:.6f}, Rel L2: {rel_l2:.6f}")

        # Collect results
        results = {
            'exp_id': exp_id,
            'model_type': model_type,
            'modes': config['fno_modes'],
            'width': config['fno_width'],
            'depth': config['fno_depth'],
            'lambda_phys': config['lambda_phys'],
            'n_params': n_params,
            'val_mse': mse,
            'val_rel_l2': rel_l2,
            'status': 'success'
        }

        log("✓ Experiment completed successfully")

    except Exception as e:
        log(f"✗ Experiment failed: {e}")
        import traceback
        log(traceback.format_exc())

        results = {
            'exp_id': exp_id,
            'model_type': model_type,
            'modes': config['fno_modes'],
            'width': config['fno_width'],
            'depth': config['fno_depth'],
            'lambda_phys': config['lambda_phys'],
            'status': 'failed',
            'error': str(e)
        }

    return results


def run_experiment_grid(data_path, pde_type, model_types, modes_list, width_list, depth_list, pinn_constants, output_dir='experiments'):
    """
    Run full grid of experiments with different hyperparameters.

    Args:
        data_path: Path to PDEBench HDF5 file
        pde_type: Type of PDE
        model_types: List of model architectures to test
        modes_list: List of Fourier mode counts
        width_list: List of hidden dimensions
        depth_list: List of depths for U-shaped models
        pinn_constants: List of PINN loss weights
        output_dir: Directory to save all experiment results

    Returns:
        results_df: Pandas DataFrame with all results
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(output_dir, f'{pde_type}_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)

    # Generate all configurations
    configs = generate_experiment_configs(pde_type, model_types, modes_list, width_list, depth_list, pinn_constants)

    # Add total count to each config for progress tracking
    for cfg in configs:
        cfg['total_experiments'] = len(configs)

    print("\n" + "="*80)
    print("EXPERIMENT GRID SEARCH")
    print("="*80)
    print(f"PDE Type: {pde_type}")
    print(f"Total experiments: {len(configs)}")
    print(f"Models: {model_types}")
    print(f"Modes: {modes_list}")
    print(f"Width: {width_list}")
    print(f"Depth: {depth_list}")
    print(f"PINN constants: {pinn_constants}")
    print(f"Output directory: {experiment_dir}")
    print("="*80 + "\n")

    # Save experiment configuration
    config_file = os.path.join(experiment_dir, 'experiment_config.json')
    with open(config_file, 'w') as f:
        json.dump({
            'pde_type': pde_type,
            'model_types': model_types,
            'modes_list': modes_list,
            'width_list': width_list,
            'depth_list': depth_list,
            'pinn_constants': pinn_constants,
            'total_experiments': len(configs),
            'timestamp': timestamp
        }, f, indent=2)

    # Run all experiments
    all_results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running experiment {config['exp_id']}...")
        results = run_single_experiment(config, data_path, experiment_dir)
        all_results.append(results)

        # Save intermediate results
        results_df = pd.DataFrame(all_results)
        results_file = os.path.join(experiment_dir, 'results.csv')
        results_df.to_csv(results_file, index=False)

    # Final results
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(experiment_dir, 'results_final.csv')
    results_df.to_csv(results_file, index=False)

    print("\n" + "="*80)
    print("EXPERIMENT GRID COMPLETE")
    print("="*80)
    print(f"Results saved to: {results_file}")
    print(f"Total experiments: {len(results_df)}")
    print(f"Successful: {(results_df['status'] == 'success').sum()}")
    print(f"Failed: {(results_df['status'] == 'failed').sum()}")

    # Generate plots
    plot_experiment_results(results_df, experiment_dir, pde_type)

    return results_df


def plot_experiment_results(results_df, experiment_dir, pde_type):
    """
    Create visualization plots comparing experiment results.

    Args:
        results_df: Pandas DataFrame with experiment results
        experiment_dir: Directory to save plots
        pde_type: Type of PDE
    """
    # Filter only successful experiments
    df = results_df[results_df['status'] == 'success'].copy()

    if len(df) == 0:
        print("⚠ No successful experiments to plot")
        return

    plot_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Model comparison (across all configs)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    model_stats = df.groupby('model_type').agg({'best_val_loss': 'mean'}).reset_index()

    ax.bar(model_stats['model_type'], model_stats['best_val_loss'])
    ax.set_ylabel('Best Validation Loss')
    ax.set_title(f'{pde_type}: Model Comparison')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '01_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Mode ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_type in df['model_type'].unique():
        model_df = df[df['model_type'] == model_type]
        mode_stats = model_df.groupby('modes')['best_val_loss'].mean()
        ax.plot(mode_stats.index, mode_stats.values, marker='o', label=model_type)

    ax.set_xlabel('Fourier Modes')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title(f'{pde_type}: Effect of Fourier Modes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '02_mode_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Width ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_type in df['model_type'].unique():
        model_df = df[df['model_type'] == model_type]
        width_stats = model_df.groupby('width')['best_val_loss'].mean()
        ax.plot(width_stats.index, width_stats.values, marker='s', label=model_type)

    ax.set_xlabel('Hidden Dimension Width')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title(f'{pde_type}: Effect of Width')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '03_width_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. PINN loss effect
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_type in df['model_type'].unique():
        model_df = df[df['model_type'] == model_type]
        pinn_stats = model_df.groupby('lambda_phys')['best_val_loss'].mean()
        ax.plot(pinn_stats.index, pinn_stats.values, marker='^', label=model_type)

    ax.set_xlabel('PINN Loss Weight (λ_phys)')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title(f'{pde_type}: Effect of PINN Loss')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, '04_pinn_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Parameters vs Performance
    # fig, ax = plt.subplots(figsize=(10, 6))
    # for model_type in df['model_type'].unique():
    #     model_df = df[df['model_type'] == model_type]
    #     ax.scatter(model_df['n_params'], model_df['best_val_loss'], label=model_type, alpha=0.6, s=100)

    # ax.set_xlabel('Number of Parameters')
    # ax.set_ylabel('Best Validation Loss')
    # ax.set_title(f'{pde_type}: Parameters vs Performance')
    # ax.set_xscale('log')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    # plt.savefig(os.path.join(plot_dir, '05_params_vs_performance.png'), dpi=150, bbox_inches='tight')
    # plt.close()

    print(f"\n✓ Plots saved to: {plot_dir}")
    print(f"  - Model comparison")
    print(f"  - Mode ablation")
    print(f"  - Width ablation")
    print(f"  - PINN loss effect")
    print(f"  - Parameters vs performance")


def main():
    parser = argparse.ArgumentParser(description="PDEBench FNO Benchmarking - Week 1 Infrastructure")

    parser.add_argument('--mode', type=str, required=True,
                       choices=['inspect', 'inspect_dir', 'test_loading', 'train', 'experiment_grid'],
                       help='Operation mode')

    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to single HDF5 file (for inspect/train modes)')

    parser.add_argument('--data_dir', type=str, default='../pdebench_data',
                       help='Directory containing PDEBench files (for inspect_dir/test_loading)')

    parser.add_argument('--pde_type', type=str, default=None,
                       help='PDE type to test/train (e.g., diffusion_reaction, darcy, navier_stokes)')

    parser.add_argument('--model_type', type=str, default='FNO',
                       choices=['FNO', 'FFNO', 'UFNO', 'UFFNO'],
                       help='Model architecture to use')

    parser.add_argument('--use_pinn', action='store_true',
                       help='Enable PINN loss')

    # Experiment grid arguments
    parser.add_argument('--models', nargs='+',
                       choices=['FNO', 'FFNO', 'UFNO', 'UFFNO'],
                       default=['FNO', 'FFNO', 'UFNO', 'UFFNO'],
                       help='List of models to test in experiment grid')

    parser.add_argument('--modes', nargs='+', type=int,
                       default=[8, 12, 16, 20],
                       help='List of Fourier mode values to test')

    parser.add_argument('--widths', nargs='+', type=int,
                       default=[48, 64, 96],
                       help='List of width values to test')

    parser.add_argument('--depths', nargs='+', type=int,
                       default=[2, 3, 4],
                       help='List of depth values to test (for U-shaped models)')

    parser.add_argument('--pinn_constants', nargs='+', type=float,
                       default=[0.0, 0.0001, 0.0005, 0.001],
                       help='List of PINN loss weights to test')

    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for experiment results')

    # Temporal augmentation arguments
    parser.add_argument('--temporal_augmentation', action='store_true',
                       help='Enable temporal data augmentation (split trajectories)')

    parser.add_argument('--num_temporal_splits', type=int, default=2,
                       help='Number of temporal segments per trajectory (default: 2 = 2x data)')

    args = parser.parse_args()

    # Execute based on mode
    if args.mode == 'inspect':
        if not args.data_path:
            print("ERROR: --data_path required for inspect mode")
            sys.exit(1)
        inspect_h5_file(args.data_path)

    elif args.mode == 'inspect_dir':
        inspect_directory(args.data_dir)

    elif args.mode == 'test_loading':
        test_data_loading(args.data_dir, args.pde_type)

    elif args.mode == 'train':
        if not args.data_path:
            print("ERROR: --data_path required for train mode")
            sys.exit(1)
        if not args.pde_type:
            print("ERROR: --pde_type required for train mode")
            sys.exit(1)
        train_baseline(
            data_path=args.data_path,
            pde_type=args.pde_type,
            model_type=args.model_type,
            use_pinn=args.use_pinn,
            temporal_augmentation=args.temporal_augmentation,
            num_temporal_splits=args.num_temporal_splits
        )

    elif args.mode == 'experiment_grid':
        if not args.data_path:
            print("ERROR: --data_path required for experiment_grid mode")
            sys.exit(1)
        if not args.pde_type:
            print("ERROR: --pde_type required for experiment_grid mode")
            sys.exit(1)
        run_experiment_grid(
            data_path=args.data_path,
            pde_type=args.pde_type,
            model_types=args.models,
            modes_list=args.modes,
            width_list=args.widths,
            depth_list=args.depths,
            pinn_constants=args.pinn_constants,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()