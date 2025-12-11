#!/usr/bin/env python3
"""
Quick diagnostic script to check Burgers data quality.
"""
import h5py
import numpy as np

def diagnose_burgers(filepath):
    """Check what's in the Burgers data file."""
    print(f"Analyzing: {filepath}\n")

    with h5py.File(filepath, 'r') as f:
        print(f"Available keys: {list(f.keys())}\n")

        if 'tensor' in f.keys():
            data = f['tensor']
            print(f"Data shape: {data.shape}")  # (n_total, n_t, n_x)
            n_total, n_t, n_x = data.shape

            # Assume 4 nu values stacked
            n_nu = 4
            samples_per_nu = n_total // n_nu
            print(f"Samples per nu: {samples_per_nu}")
            print(f"Time steps: {n_t}")
            print(f"Spatial points: {n_x}\n")

            # Load first sample from each nu group
            print("="*80)
            print("Checking first sample from each nu group:")
            print("="*80)
            for nu_idx in range(n_nu):
                start_idx = nu_idx * samples_per_nu
                sample = data[start_idx, :, :]  # (n_t, n_x)

                print(f"\nNu group {nu_idx}:")
                print(f"  Initial (t=0):")
                print(f"    min={sample[0].min():.6f}, max={sample[0].max():.6f}, mean={sample[0].mean():.6f}, std={sample[0].std():.6f}")
                print(f"  Middle (t={n_t//2}):")
                print(f"    min={sample[n_t//2].min():.6f}, max={sample[n_t//2].max():.6f}, mean={sample[n_t//2].mean():.6f}, std={sample[n_t//2].std():.6f}")
                print(f"  Final (t={n_t-1}):")
                print(f"    min={sample[-1].min():.6f}, max={sample[-1].max():.6f}, mean={sample[-1].mean():.6f}, std={sample[-1].std():.6f}")

                # Check temporal decay
                initial_std = sample[0].std()
                final_std = sample[-1].std()
                print(f"  Temporal decay: {initial_std:.6f} → {final_std:.6f} (ratio: {final_std/initial_std:.3f})")

            # Check what happens with time_step_spacing=10
            print("\n" + "="*80)
            print("Checking temporal pairs with spacing=10:")
            print("="*80)
            sample = data[0, :, :]  # First sample
            for t_in in [0, 10, 20, 40, 80]:
                t_out = min(t_in + 10, n_t - 1)
                input_field = sample[t_in]
                output_field = sample[t_out]

                print(f"\nt_in={t_in}, t_out={t_out}:")
                print(f"  Input:  min={input_field.min():.6f}, max={input_field.max():.6f}, std={input_field.std():.6f}")
                print(f"  Output: min={output_field.min():.6f}, max={output_field.max():.6f}, std={output_field.std():.6f}")
                print(f"  Difference: std={np.abs(output_field - input_field).std():.6f}")

            # Overall statistics
            print("\n" + "="*80)
            print("Overall statistics (all data):")
            print("="*80)
            # Sample 100 random points to avoid loading everything
            random_samples = np.random.choice(n_total, size=min(100, n_total), replace=False)
            sample_data = data[random_samples, :, :]

            print(f"Global min: {sample_data.min():.6f}")
            print(f"Global max: {sample_data.max():.6f}")
            print(f"Global mean: {sample_data.mean():.6f}")
            print(f"Global std: {sample_data.std():.6f}")

        else:
            print("No 'tensor' key found, checking individual nu keys...")
            nu_keys = [k for k in f.keys() if k.startswith('nu_')]
            for nu_key in nu_keys:
                data = f[nu_key]
                print(f"\n{nu_key}:")
                print(f"  Shape: {data.shape}")
                sample = data[0, :, :]
                print(f"  Initial: min={sample[0].min():.6f}, max={sample[0].max():.6f}, std={sample[0].std():.6f}")
                print(f"  Final: min={sample[-1].min():.6f}, max={sample[-1].max():.6f}, std={sample[-1].std():.6f}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diagnose_burgers.py <path_to_burgers.hdf5>")
        sys.exit(1)

    diagnose_burgers(sys.argv[1])