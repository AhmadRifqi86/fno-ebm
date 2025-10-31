"""
Diagnose why uncertainty map (4th column) shows noise instead of spatial structure.

This script will:
1. Load the best EBM model
2. Run inference and collect samples
3. Analyze the samples to understand why std is noisy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from pathlib import Path

from config import Config
from fno import FNO2d
from ebm import SimpleFNO_EBM
from trainer import FNO_EBM
from datautils import PDEDataset
from inference import inference_probabilistic

print("=" * 80)
print("DIAGNOSING INFERENCE UNCERTAINTY MAP")
print("=" * 80)

# Load config
print("\n--- Loading Configuration ---")
with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)
config = Config(config_dict)

# Load dataset
print("\nLoading validation dataset...")
data_dir = Path(config.data_dir) if hasattr(config, 'data_dir') else Path('../data')
resolution = config.grid_size if hasattr(config, 'grid_size') else 64
complexity = config.complexity if hasattr(config, 'complexity') else 'medium'
noise_type = config.noise_type if hasattr(config, 'noise_type') else 'heteroscedastic'
noisy_val_file = data_dir / f"darcy_{complexity}_{noise_type}_res{resolution}_val.npz"
val_dataset = PDEDataset.from_file(str(noisy_val_file), normalize_output=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

# Build model
print("\nBuilding models...")
# fno_model = FNO2d(
#     modes1=config.fno_modes,
#     modes2=config.fno_modes,
#     width=config.fno_width,
#     initial_step=1
# )
fno_model = FNO2d(
        modes1=config.fno_modes,
        modes2=config.fno_modes,
        width=config.fno_width,
        num_layers=4
    )
ebm_model = SimpleFNO_EBM(in_channels=4, fno_width=32, fno_layers=3)
model = FNO_EBM(fno_model, ebm_model).to(config.device)

# Load checkpoint
print("\nLoading best EBM checkpoint...")
ebm_ckpt_path = os.path.join(config.checkpoint_dir, "best_model_ebm.pt")
if not os.path.exists(ebm_ckpt_path):
    print(f"❌ Checkpoint not found: {ebm_ckpt_path}")
    exit(1)

best_fno_path = os.path.join(config.checkpoint_dir, 'best_model_fno.pt')
best_ebm_path = os.path.join(config.checkpoint_dir, 'best_model_ebm.pt')

if os.path.exists(best_fno_path) and os.path.exists(best_ebm_path):
        # Load best models
    fno_checkpoint = torch.load(best_fno_path, map_location=config.device)
    ebm_checkpoint = torch.load(best_ebm_path, map_location=config.device)

    model.u_fno.load_state_dict(fno_checkpoint['fno_model'])
    model.V_ebm.load_state_dict(ebm_checkpoint['ebm_model'])

    print("✓ Loaded best models")

# Get one test batch
x_test, y_test = next(iter(val_loader))
x_test = x_test.to(config.device)
y_test = y_test.to(config.device)

print(f"\nTest batch shape: x={x_test.shape}, y={y_test.shape}")

# Run inference with detailed analysis
print("\n" + "=" * 80)
print("RUNNING INFERENCE WITH SAMPLING")
print("=" * 80)

num_samples = 50  # Generate 50 samples
print(f"Generating {num_samples} samples via MCMC...")

samples_list = []
model.eval()

# Import langevin_dynamics from inference
from inference import langevin_dynamics

for i in range(num_samples):
    # Note: langevin_dynamics needs gradients, so we can't use torch.no_grad()
    u_sample = langevin_dynamics(
        model, x_test,
        num_steps=200,
        step_size=0.005,
        device=config.device
    )
    # Detach after sampling to save memory
    samples_list.append(u_sample.detach().cpu())

    if (i+1) % 10 == 0:
        print(f"  Generated {i+1}/{num_samples} samples...")

samples = torch.stack(samples_list, dim=0)  # (num_samples, batch, nx, ny, 1)
print(f"\nSamples shape: {samples.shape}")

# Compute statistics
mean = samples.mean(dim=0)  # (batch, nx, ny, 1)
std = samples.std(dim=0)    # (batch, nx, ny, 1)

print("\n" + "=" * 80)
print("ANALYZING UNCERTAINTY MAP")
print("=" * 80)

# Take first sample from batch
sample_idx = 0
mean_map = mean[sample_idx, ..., 0].numpy()
std_map = std[sample_idx, ..., 0].numpy()
y_true_map = y_test[sample_idx, ..., 0].cpu().numpy()

# Check individual samples
sample0 = samples[0, sample_idx, ..., 0].numpy()
sample10 = samples[10, sample_idx, ..., 0].numpy()
sample20 = samples[20, sample_idx, ..., 0].numpy()

print(f"\nSample statistics:")
print(f"  Sample 0:  mean={sample0.mean():.4f}, std={sample0.std():.4f}")
print(f"  Sample 10: mean={sample10.mean():.4f}, std={sample10.std():.4f}")
print(f"  Sample 20: mean={sample20.mean():.4f}, std={sample20.std():.4f}")
print(f"  Mean:      mean={mean_map.mean():.4f}, std={mean_map.std():.4f}")

# Check if samples are identical (collapsed)
sample_diff = np.abs(sample10 - sample0).mean()
print(f"\n  Difference between sample 0 and 10: {sample_diff:.6f}")

if sample_diff < 0.001:
    print("  ❌ WARNING: Samples are nearly identical (mode collapse!)")
else:
    print("  ✓ Samples show diversity")

# Analyze std map
print(f"\nUncertainty map statistics:")
print(f"  Mean: {std_map.mean():.6f}")
print(f"  Std:  {std_map.std():.6f}")
print(f"  Min:  {std_map.min():.6f}")
print(f"  Max:  {std_map.max():.6f}")
print(f"  Range ratio: {std_map.max() / (std_map.min() + 1e-8):.2f}x")

# Check for spatial structure
# Compute radial profile
center_y, center_x = 32, 32
y_grid, x_grid = np.ogrid[:64, :64]
r = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)

# Bin by radius
radial_bins = np.arange(0, 45, 5)
radial_profile = []
for i in range(len(radial_bins) - 1):
    mask = (r >= radial_bins[i]) & (r < radial_bins[i+1])
    radial_profile.append(std_map[mask].mean())

print(f"\nRadial uncertainty profile (from center to edge):")
for i, val in enumerate(radial_profile):
    print(f"  Radius {radial_bins[i]:2.0f}-{radial_bins[i+1]:2.0f}: {val:.6f}")

# Check if there's radial structure
radial_std = np.std(radial_profile)
if radial_std / np.mean(radial_profile) < 0.1:
    print(f"\n  ❌ NO spatial structure detected (radial variation < 10%)")
    print(f"     Uncertainty is spatially uniform (like noise)")
else:
    print(f"\n  ✓ Spatial structure detected (radial variation = {radial_std/np.mean(radial_profile)*100:.1f}%)")

# Visualization
print("\n" + "=" * 80)
print("VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Row 1: Individual samples
axes[0, 0].imshow(sample0, cmap='viridis')
axes[0, 0].set_title('Sample 0')
axes[0, 0].axis('off')

axes[0, 1].imshow(sample10, cmap='viridis')
axes[0, 1].set_title('Sample 10')
axes[0, 1].axis('off')

axes[0, 2].imshow(sample20, cmap='viridis')
axes[0, 2].set_title('Sample 20')
axes[0, 2].axis('off')

axes[0, 3].imshow(np.abs(sample10 - sample0), cmap='hot')
axes[0, 3].set_title(f'|Sample 10 - Sample 0|\nmean={sample_diff:.6f}')
axes[0, 3].axis('off')

# Row 2: Mean and Ground Truth
axes[1, 0].imshow(y_true_map, cmap='viridis')
axes[1, 0].set_title('Ground Truth')
axes[1, 0].axis('off')

axes[1, 1].imshow(mean_map, cmap='viridis')
axes[1, 1].set_title('EBM Mean')
axes[1, 1].axis('off')

axes[1, 2].imshow(np.abs(mean_map - y_true_map), cmap='hot')
axes[1, 2].set_title('|Mean - Truth|')
axes[1, 2].axis('off')

axes[1, 3].axis('off')

# Row 3: Uncertainty analysis
im = axes[2, 0].imshow(std_map, cmap='hot')
axes[2, 0].set_title(f'Uncertainty (Std)\nrange: [{std_map.min():.4f}, {std_map.max():.4f}]')
axes[2, 0].axis('off')
plt.colorbar(im, ax=axes[2, 0])

# Radial profile plot
axes[2, 1].plot(radial_bins[:-1], radial_profile, 'o-')
axes[2, 1].set_xlabel('Radius from center')
axes[2, 1].set_ylabel('Mean uncertainty')
axes[2, 1].set_title('Radial Uncertainty Profile')
axes[2, 1].grid(True, alpha=0.3)

# Histogram of uncertainty
axes[2, 2].hist(std_map.flatten(), bins=50, alpha=0.7)
axes[2, 2].set_xlabel('Uncertainty (std)')
axes[2, 2].set_ylabel('Frequency')
axes[2, 2].set_title('Uncertainty Distribution')
axes[2, 2].grid(True, alpha=0.3)

axes[2, 3].axis('off')

plt.tight_layout()
plt.savefig('inference_uncertainty_diagnosis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to inference_uncertainty_diagnosis.png")

# ============================================================================
# DIAGNOSIS SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

if sample_diff < 0.001:
    print("""
❌ ROOT CAUSE: MODE COLLAPSE

All MCMC samples are nearly identical!
- Sample diversity: {:.6f} (should be > 0.01)
- Uncertainty map is FAKE - it's just numerical noise, not true uncertainty
- The EBM's energy function has collapsed to a narrow mode

WHY THIS HAPPENS:
1. Energy function is too peaked around the mean
2. MCMC step size (0.005) is too small to explore
3. Training only learned to match the mean, not the distribution

SOLUTIONS:
1. Increase MCMC step size during inference (try 0.05 or 0.1)
2. Increase number of MCMC steps (try 500-1000)
3. Add noise injection during MCMC sampling
4. Use different training objective (score matching, denoising score matching)
""".format(sample_diff))

elif radial_std / np.mean(radial_profile) < 0.1:
    print("""
⚠️  ISSUE: NO SPATIAL STRUCTURE IN UNCERTAINTY

Samples show diversity, but uncertainty has no spatial pattern.
- Radial variation: {:.1f}% (should be > 30%)
- The EBM didn't learn WHERE uncertainty should be high/low

POSSIBLE CAUSES:
1. SimpleFNO_EBM isn't leveraging its global receptive field
2. Training data variance is truly spatially uniform
3. Energy function is too smooth (not discriminative enough)

SOLUTIONS:
1. Check if data actually has spatially-structured uncertainty
2. Visualize energy landscapes to see if EBM learned structure
3. Try increasing fno_width or fno_layers for more capacity
""".format(radial_std/np.mean(radial_profile)*100))

else:
    print("""
✓ INFERENCE LOOKS HEALTHY

- Samples show diversity: {:.6f}
- Uncertainty has spatial structure: {:.1f}% radial variation
- SimpleFNO_EBM is working as expected

If visualization still looks noisy, check:
1. Color scale normalization
2. Number of samples (try num_samples=100 for smoother maps)
3. Denormalization of std values
""".format(sample_diff, radial_std/np.mean(radial_profile)*100))

print("=" * 80)