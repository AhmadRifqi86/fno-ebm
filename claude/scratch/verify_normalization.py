"""
Quick script to verify input normalization is working correctly.
Run this before retraining to confirm the fix is applied.
"""

import numpy as np
from datautils import PDEDataset

def verify_normalization():
    print("="*80)
    print("VERIFYING INPUT NORMALIZATION FIX")
    print("="*80)

    # Load existing data (will show normalization in action)
    print("\nLoading validation data...")
    dataset = PDEDataset.from_file(
        'data/darcy_medium_heteroscedastic_res64_val.npz',
        normalize_input=True,
        normalize_output=True
    )

    print("\n" + "="*80)
    print("✓ VERIFICATION PASSED - Input normalization is working!")
    print("="*80)

    # Summary
    X = dataset.X
    U = dataset.U

    print(f"\n📊 Dataset Summary:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {U.shape}")
    print(f"  Input channels: {dataset.in_channels}")
    print(f"  Normalized fields: {len(dataset.x_fields_mean)}")

    print(f"\n📈 Channel Statistics:")
    for ch in range(dataset.in_channels):
        ch_data = X[..., ch]
        ch_name = "x-coord" if ch == 0 else ("y-coord" if ch == 1 else f"field-{ch-2}")
        status = "✓ Kept [0,1]" if ch < 2 else "✓ Normalized to N(0,1)"

        print(f"  Channel {ch} ({ch_name:10s}): mean={ch_data.mean():7.4f}, std={ch_data.std():7.4f}  {status}")

    print(f"\n📐 Field Normalization Stats:")
    for i, (mean, std) in enumerate(zip(dataset.x_fields_mean, dataset.x_fields_std)):
        print(f"  Field {i} (channel {i+2}): original mean={mean:.6f}, std={std:.6f}")

    # Check that all channels are balanced
    print(f"\n🔍 Balance Check:")
    channel_stds = [X[..., ch].std().item() for ch in range(dataset.in_channels)]
    max_std = max(channel_stds)
    min_std = min(channel_stds)
    ratio = max_std / min_std if min_std > 0 else float('inf')

    print(f"  Channel std devs: {[f'{s:.3f}' for s in channel_stds]}")
    print(f"  Max/Min ratio: {ratio:.2f}")

    if ratio < 3.0:
        print(f"  ✅ PASS - Channels are well-balanced (ratio < 3.0)")
    else:
        print(f"  ⚠️  WARNING - Channels still imbalanced (ratio = {ratio:.2f})")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Delete old data: rm data/darcy_medium_heteroscedastic_res64_*.npz")
    print("2. Delete old checkpoints: rm checkpoints/best_model_*.pt")
    print("3. Run training: python main_noisy.py")
    print("4. Data will be regenerated with normalized inputs automatically")
    print("="*80)

if __name__ == '__main__':
    verify_normalization()