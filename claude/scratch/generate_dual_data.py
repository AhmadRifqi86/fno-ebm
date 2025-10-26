#!/usr/bin/env python3
"""
Generate dual dataset (clean + noisy) for physics-informed FNO-EBM training.

Usage:
    python generate_dual_data.py
"""

import numpy as np
from dataset import DarcyFlowGenerator
from pathlib import Path

def main():
    """Generate clean and noisy datasets for FNO and EBM training."""

    # Configuration
    config = {
        'resolution': 64,
        'complexity': 'medium',
        'n_train': 1000,
        'n_val': 200,
        'seed': 42,
        'noise_type': 'heteroscedastic',
        'noise_params': {
            'base_noise': 0.001,
            'scale_factor': 0.02
        }
    }

    # Create data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("DUAL DATASET GENERATION (Clean + Noisy)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Resolution: {config['resolution']}x{config['resolution']}")
    print(f"  Complexity: {config['complexity']}")
    print(f"  Training samples: {config['n_train']}")
    print(f"  Validation samples: {config['n_val']}")
    print(f"  Noise type: {config['noise_type']}")
    print(f"  Noise params: {config['noise_params']}")
    print()

    # Generate training data
    print("Generating TRAINING set...")
    gen_train = DarcyFlowGenerator(
        resolution=config['resolution'],
        complexity=config['complexity'],
        seed=config['seed']
    )

    X_train, U_clean_train, U_noisy_train = gen_train.generate_dual_dataset(
        n_samples=config['n_train'],
        noise_type=config['noise_type'],
        noise_params=config['noise_params']
    )

    # Save training data
    train_file_clean = data_dir / f"darcy_{config['complexity']}_clean_res{config['resolution']}_train.npz"
    train_file_noisy = data_dir / f"darcy_{config['complexity']}_noisy_res{config['resolution']}_train.npz"

    np.savez_compressed(train_file_clean, X=X_train, U=U_clean_train)
    np.savez_compressed(train_file_noisy, X=X_train, U=U_noisy_train)

    print(f"\n✓ Training data saved:")
    print(f"  Clean: {train_file_clean} ({train_file_clean.stat().st_size / 1024**2:.1f} MB)")
    print(f"  Noisy: {train_file_noisy} ({train_file_noisy.stat().st_size / 1024**2:.1f} MB)")

    # Generate validation data
    print(f"\nGenerating VALIDATION set...")
    gen_val = DarcyFlowGenerator(
        resolution=config['resolution'],
        complexity=config['complexity'],
        seed=config['seed'] + 10000  # Different seed for validation
    )

    X_val, U_clean_val, U_noisy_val = gen_val.generate_dual_dataset(
        n_samples=config['n_val'],
        noise_type=config['noise_type'],
        noise_params=config['noise_params']
    )

    # Save validation data
    val_file_clean = data_dir / f"darcy_{config['complexity']}_clean_res{config['resolution']}_val.npz"
    val_file_noisy = data_dir / f"darcy_{config['complexity']}_noisy_res{config['resolution']}_val.npz"

    np.savez_compressed(val_file_clean, X=X_val, U=U_clean_val)
    np.savez_compressed(val_file_noisy, X=X_val, U=U_noisy_val)

    print(f"\n✓ Validation data saved:")
    print(f"  Clean: {val_file_clean} ({val_file_clean.stat().st_size / 1024**2:.1f} MB)")
    print(f"  Noisy: {val_file_noisy} ({val_file_noisy.stat().st_size / 1024**2:.1f} MB)")

    # Statistics
    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)

    print("\nTraining Set:")
    print(f"  Clean data:")
    print(f"    Mean: {U_clean_train.mean():.6f}")
    print(f"    Std:  {U_clean_train.std():.6f}")
    print(f"    Range: [{U_clean_train.min():.6f}, {U_clean_train.max():.6f}]")

    print(f"  Noisy data:")
    print(f"    Mean: {U_noisy_train.mean():.6f}")
    print(f"    Std:  {U_noisy_train.std():.6f}")
    print(f"    Range: [{U_noisy_train.min():.6f}, {U_noisy_train.max():.6f}]")

    noise = U_noisy_train - U_clean_train
    print(f"  Noise:")
    print(f"    Mean: {noise.mean():.6f} (should be ~0)")
    print(f"    Std:  {noise.std():.6f}")
    print(f"    SNR: {20 * np.log10(U_clean_train.std() / noise.std()):.2f} dB")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Use CLEAN data for FNO training:")
    print(f"   train_loader = load_data('{train_file_clean}')")
    print("   Loss = MSE + lambda_phys * ||PDE_residual||²")
    print()
    print("2. Use NOISY data for EBM training:")
    print(f"   train_loader = load_data('{train_file_noisy}')")
    print("   EBM learns uncertainty from mismatch")
    print()
    print("3. Update config.yaml:")
    print("   lambda_phys: 0.05  # Can use physics loss with clean data!")
    print("=" * 70)

if __name__ == '__main__':
    main()