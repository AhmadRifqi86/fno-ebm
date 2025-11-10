#!/usr/bin/env python3
"""
main.py - PDEBench Benchmarking Entry Point

Week 1: Infrastructure & Setup
- Inspect PDEBench HDF5 files
- Test data loading for different PDE types
- Set up experiment tracking
- Run baseline training experiments

Usage:
    python main.py --mode inspect --data_path /path/to/pdebench/file.h5
    python main.py --mode test_loading --data_dir /path/to/pdebench/
    python main.py --mode train --pde_type diffusion_reaction --model_type FNO
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Import existing modules
from datautils import PDEBenchH5Loader
from fno import FNO2d, FFNO2d, UFNO2d, UFFNO2d
from config import Config
from customs import DarcyPhysicsLoss, ReactionDiffusionPhysicsLoss


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
        'learning_rate': 0.001,
        'epochs': 100,
        'patience': 20,

        # Physics loss
        'lambda_phys': 0.01 if use_pinn else 0.0,

        # Data config
        'pde_type': pde_type,
        'train_samples': 4000,
        'val_samples': 500,
        'test_samples': 500,

        # Logging
        'checkpoint_dir': f'checkpoints/{pde_type}/{model_type}_pinn{use_pinn}',
        'log_file': f'logs/{pde_type}_{model_type}_pinn{use_pinn}.txt',

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    return config


def train_baseline(data_path: str, pde_type: str, model_type: str, use_pinn: bool = False):
    """
    Run baseline training experiment.

    Args:
        data_path: Path to PDEBench HDF5 file
        pde_type: Type of PDE
        model_type: Model architecture (FNO, FFNO, UFNO, UFFNO)
        use_pinn: Whether to use PINN loss
    """
    print("\n" + "="*80)
    print(f"WEEK 1 TASK: Baseline Training - {pde_type.upper()} with {model_type}")
    print("="*80)

    # Create config
    config_dict = create_baseline_config(pde_type, model_type, use_pinn)
    config = Config(config_dict)

    # Setup logging
    os.makedirs('logs', exist_ok=True)
    log_file = config_dict['log_file']

    def log(message):
        """Log to both console and file."""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    log(f"\n{'='*80}")
    log(f"Experiment Configuration")
    log(f"{'='*80}")
    log(f"PDE Type: {pde_type}")
    log(f"Model: {model_type}")
    log(f"PINN Loss: {use_pinn} (lambda={config_dict['lambda_phys']})")
    log(f"Modes: {config_dict['fno_modes']}, Width: {config_dict['fno_width']}, Depth: {config_dict['fno_depth']}")
    log(f"Batch size: {config_dict['batch_size']}, LR: {config_dict['learning_rate']}")
    log(f"Device: {config_dict['device']}")
    log(f"Checkpoint dir: {config_dict['checkpoint_dir']}")
    log(f"Log file: {log_file}")

    # Load data
    print("\nLoading data...")
    with PDEBenchH5Loader(data_path) as loader:
        loader.print_info()

        # Load full dataset
        full_dataset = loader.to_dataset(
            input_t=0,
            output_t=-1,
            num_samples=None  # Load all
        )

        # Split into train/val/test
        n_total = len(full_dataset)
        n_train = min(config_dict['train_samples'], int(0.8 * n_total))
        n_val = min(config_dict['val_samples'], int(0.1 * n_total))

        indices = np.random.permutation(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]

        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

        print(f"\nDataset split:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Initialize model
    print(f"\nInitializing {model_type} model...")
    if model_type == 'FNO':
        model = FNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            num_layers=4,
            dropout=config.fno_dropout
        )
    elif model_type == 'FFNO':
        model = FFNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            num_layers=6,
            dropout=config.fno_dropout
        )
    elif model_type == 'UFNO':
        model = UFNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            depth=config.fno_depth,
            dropout=config.fno_dropout
        )
    elif model_type == 'UFFNO':
        model = UFFNO2d(
            modes1=config.fno_modes,
            modes2=config.fno_modes,
            width=config.fno_width,
            depth=config.fno_depth,
            dropout=config.fno_dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(config.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\nModel parameters: {n_params:,}")

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # For now, we'll do simple validation test without full training
    # Full Trainer integration coming in Week 2
    log("\n⚠ Note: Full Trainer integration coming in Week 2")
    log("  Running quick validation test instead...")

    # Quick validation test
    model.eval()
    with torch.no_grad():
        x_batch, u_batch = next(iter(val_loader))
        x_batch = x_batch.to(config.device)
        u_batch = u_batch.to(config.device)

        u_pred = model(x_batch)

        mse = torch.nn.functional.mse_loss(u_pred, u_batch)
        rel_l2 = torch.norm(u_pred - u_batch) / torch.norm(u_batch)

        log(f"\nInitial validation metrics (untrained model):")
        log(f"  MSE: {mse.item():.6f}")
        log(f"  Relative L2: {rel_l2.item():.6f}")

    log(f"\n{'='*80}")
    log(f"✓ Baseline setup complete!")
    log(f"{'='*80}")
    log(f"Model: {model_type}")
    log(f"PDE: {pde_type}")
    log(f"PINN: {use_pinn}")
    log(f"Checkpoint dir: {config.checkpoint_dir}")
    log(f"Log file: {log_file}")
    log(f"\nNext steps:")
    log(f"  1. Verify data loading works correctly")
    log(f"  2. Week 2: Implement full training loop")
    log(f"  3. Run ablation studies")
    log(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="PDEBench FNO Benchmarking - Week 1 Infrastructure")

    parser.add_argument('--mode', type=str, required=True,
                       choices=['inspect', 'inspect_dir', 'test_loading', 'train'],
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
        train_baseline(args.data_path, args.pde_type, args.model_type, args.use_pinn)


if __name__ == '__main__':
    main()