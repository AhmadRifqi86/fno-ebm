"""
Inspect the actual data to determine:
1. Does the data have spatially-heterogeneous uncertainty?
2. Can ConvEBM learn spatial uncertainty patterns?
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

print("=" * 80)
print("INSPECTING DATA FOR SPATIAL UNCERTAINTY PATTERNS")
print("=" * 80)

# Load the data
data_train = np.load('data/darcy_medium_heteroscedastic_res64_train.npz')
data_val = np.load('data/darcy_medium_heteroscedastic_res64_val.npz')

X_train = data_train['X']  # (1000, 64, 64, 3)
U_train = data_train['U']  # (1000, 64, 64, 1)

X_val = data_val['X']
U_val = data_val['U']

print(f"\nDataset shapes:")
print(f"  Train: X={X_train.shape}, U={U_train.shape}")
print(f"  Val: X={X_val.shape}, U={U_val.shape}")

# ============================================================================
# QUESTION 1: Does the data have spatially-heterogeneous uncertainty?
# ============================================================================

print("\n" + "=" * 80)
print("QUESTION 1: Spatial heterogeneity of uncertainty")
print("=" * 80)

# For heteroscedastic noise, the variance should correlate with something spatial
# Let's check: does variance correlate with permeability?

# Get permeability field (channel 2)
permeability = X_train[..., 2]  # (1000, 64, 64)

# Compute local statistics across samples for each spatial location
# This tells us: at each (x,y) location, what's the variance across different samples?
spatial_mean = U_train.mean(axis=0).squeeze()  # (64, 64)
spatial_std = U_train.std(axis=0).squeeze()    # (64, 64)

print(f"\nSpatial statistics across samples:")
print(f"  Mean of spatial_std: {spatial_std.mean():.6f}")
print(f"  Std of spatial_std: {spatial_std.std():.6f}")
print(f"  Min spatial_std: {spatial_std.min():.6f}")
print(f"  Max spatial_std: {spatial_std.max():.6f}")
print(f"  Ratio (max/min): {spatial_std.max() / spatial_std.min():.2f}x")

if spatial_std.std() / spatial_std.mean() < 0.1:
    print("\n⚠️  FINDING: Spatial uncertainty is nearly HOMOGENEOUS!")
    print("   All locations have similar uncertainty - no spatial pattern to learn")
else:
    print("\n✓ FINDING: Spatial uncertainty is HETEROGENEOUS")
    print("   Different locations have different uncertainty levels")

# Check correlation with permeability
permeability_mean = permeability.mean(axis=0)  # (64, 64)
correlation = np.corrcoef(permeability_mean.flatten(), spatial_std.flatten())[0, 1]
print(f"\nCorrelation between permeability and uncertainty: {correlation:.4f}")

if abs(correlation) > 0.3:
    print("✓ Strong correlation found - uncertainty depends on permeability")
else:
    print("⚠️  Weak correlation - uncertainty doesn't depend on permeability")

# ============================================================================
# Let's also check per-sample variation
# ============================================================================

print("\n" + "-" * 80)
print("Analyzing individual samples")
print("-" * 80)

# Take 5 samples and compute their local variance
fig1, axes = plt.subplots(5, 4, figsize=(16, 20))

for i in range(5):
    sample_x = X_train[i]
    sample_u = U_train[i, ..., 0]
    sample_perm = sample_x[..., 2]

    # To check heteroscedasticity, we need the TRUE uncertainty at each point
    # For synthetic data, this might be encoded in the generation process
    # Let's visualize what we have

    # Plot permeability
    im0 = axes[i, 0].imshow(sample_perm, cmap='viridis')
    axes[i, 0].set_title(f'Sample {i+1}: Permeability')
    plt.colorbar(im0, ax=axes[i, 0])

    # Plot solution
    im1 = axes[i, 1].imshow(sample_u, cmap='RdBu_r')
    axes[i, 1].set_title(f'Sample {i+1}: Solution')
    plt.colorbar(im1, ax=axes[i, 1])

    # Plot solution gradient magnitude (proxy for difficulty)
    grad_y, grad_x = np.gradient(sample_u)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    im2 = axes[i, 2].imshow(grad_mag, cmap='hot')
    axes[i, 2].set_title(f'Sample {i+1}: Gradient Magnitude')
    plt.colorbar(im2, ax=axes[i, 2])

    # Expected uncertainty location (high gradients = high uncertainty typically)
    im3 = axes[i, 3].imshow(spatial_std, cmap='hot')
    axes[i, 3].set_title(f'Cross-sample Std Dev')
    plt.colorbar(im3, ax=axes[i, 3])

    for ax in axes[i]:
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('data_uncertainty_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to data_uncertainty_analysis.png")

# ============================================================================
# QUESTION 2: Is ConvEBM architecture capable of learning spatial patterns?
# ============================================================================

print("\n" + "=" * 80)
print("QUESTION 2: ConvEBM architecture analysis")
print("=" * 80)

from ebm import ConvEBM

# Check receptive field
model = ConvEBM(in_channels=4, hidden_channels=[64, 128, 128, 64])

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Analyze receptive field
print("\nArchitecture:")
print("  Input: 4 channels (u + x,y,permeability)")
print("  Conv blocks: [64, 128, 128, 64]")
print("  Each block: 3x3 conv -> norm -> 3x3 conv -> skip connection")
print("  Final: 1x1 conv -> global avg pool -> scalar energy")

# Calculate receptive field
receptive_field = 1
for i in range(4):  # 4 blocks
    receptive_field += 2 * 2  # Two 3x3 convs per block
print(f"\nReceptive field: {receptive_field}x{receptive_field} pixels")
print(f"Coverage: {receptive_field/64*100:.1f}% of 64x64 grid")

if receptive_field < 32:
    print("⚠️  ISSUE: Receptive field is too small!")
    print("   Model cannot see global context - only local patterns")
else:
    print("✓ Receptive field covers good portion of the grid")

# Test forward pass to check gradient flow
print("\nTesting gradient flow...")
x_test = torch.randn(2, 64, 64, 3)
u_test = torch.randn(2, 64, 64, 1)
u_test.requires_grad = True

energy = model(u_test, x_test)
print(f"  Energy output shape: {energy.shape}")

grad = torch.autograd.grad(energy.sum(), u_test)[0]
print(f"  Gradient shape: {grad.shape}")
print(f"  Gradient mean: {grad.mean().item():.6f}")
print(f"  Gradient std: {grad.std().item():.6f}")

if grad.std().item() < 1e-4:
    print("⚠️  ISSUE: Gradients are too small - vanishing gradient problem")
else:
    print("✓ Gradients are healthy")

# ============================================================================
# FINAL DIAGNOSIS
# ============================================================================

print("\n" + "=" * 80)
print("DIAGNOSIS & RECOMMENDATION")
print("=" * 80)

print("\n1. Spatial uncertainty in data:")
if spatial_std.std() / spatial_std.mean() < 0.1:
    print("   ❌ Data has HOMOGENEOUS uncertainty (no spatial pattern)")
    print("   → The noise is the same everywhere")
    print("   → There's nothing for the model to learn spatially!")
    print()
    print("   ROOT CAUSE: Your data generation might be using i.i.d. Gaussian noise")
    print("   instead of truly heteroscedastic (spatially-varying) noise.")
else:
    print("   ✓ Data has HETEROGENEOUS uncertainty")

print("\n2. ConvEBM architecture capability:")
if receptive_field >= 32:
    print("   ✓ Architecture has sufficient receptive field")
else:
    print("   ⚠️  Architecture receptive field is limited")

print("\n" + "=" * 80)
print("ACTIONABLE RECOMMENDATION")
print("=" * 80)

if spatial_std.std() / spatial_std.mean() < 0.1:
    print("""
The problem is NOT the training method or architecture!

Your data has HOMOGENEOUS uncertainty - the noise is the same everywhere.
This means:
  • The EBM has nothing to learn spatially
  • The 4th column (uncertainty map) SHOULD be uniform noise
  • No amount of training or architecture changes will fix this

SOLUTION: You need to regenerate your data with TRUE heteroscedastic noise.

For Darcy flow, heteroscedastic noise should:
  • Be higher in high-permeability regions (flow is faster → more unstable)
  • Be higher near boundaries
  • Be higher in high-gradient regions

DON'T switch to score matching - it won't help!

FIX your data generation first, then re-evaluate.
""")
else:
    print("""
The data has spatial uncertainty patterns!

If your uncertainty maps are still noisy, the issue is likely:
  1. Training: MCMC isn't working (you already know this)
  2. Architecture: Needs more capacity or global context

Recommendations (in order):
  1. Try score matching EBM (eliminates MCMC during training)
  2. Add attention/transformer layers for global context
  3. Use FNO encoder instead of pure convolutions
""")

print("=" * 80)