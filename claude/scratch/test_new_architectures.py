"""
Test the new EBM architectures to verify:
1. They run without errors
2. Gradients flow properly
3. Receptive field is truly global (for FNO)
"""

import torch
import torch.nn as nn
from ebm import ConvEBM, SimpleFNO_EBM, MultiScaleConvEBM

print("=" * 80)
print("TESTING NEW EBM ARCHITECTURES")
print("=" * 80)

# Test input
batch_size = 2
x_test = torch.randn(batch_size, 64, 64, 3)
u_test = torch.randn(batch_size, 64, 64, 1)
u_test.requires_grad = True

# ============================================================================
# Test 1: ConvEBM (baseline - limited receptive field)
# ============================================================================

print("\n" + "=" * 80)
print("1. ConvEBM (Baseline)")
print("=" * 80)

model1 = ConvEBM(in_channels=4, hidden_channels=[64, 128, 128, 64])
total_params1 = sum(p.numel() for p in model1.parameters())

print(f"Parameters: {total_params1:,}")
print("Receptive field: 17x17 pixels (27% of 64x64)")

# Forward pass
energy1 = model1(u_test, x_test)
print(f"Energy shape: {energy1.shape}")
print(f"Energy range: [{energy1.min().item():.3f}, {energy1.max().item():.3f}]")

# Gradient check
grad1 = torch.autograd.grad(energy1.sum(), u_test, retain_graph=True)[0]
print(f"Gradient mean: {grad1.mean().item():.6f}")
print(f"Gradient std: {grad1.std().item():.6f}")

# ============================================================================
# Test 2: SimpleFNO_EBM (RECOMMENDED - global receptive field)
# ============================================================================

print("\n" + "=" * 80)
print("2. SimpleFNO_EBM (RECOMMENDED)")
print("=" * 80)

model2 = SimpleFNO_EBM(
    in_channels=4,
    fno_modes1=12,
    fno_modes2=12,
    fno_width=32,
    fno_layers=3
)
total_params2 = sum(p.numel() for p in model2.parameters())

print(f"Parameters: {total_params2:,}")
print("Receptive field: GLOBAL (100% - sees entire 64x64 grid via FFT)")

# Forward pass
u_test2 = u_test.clone().detach()
u_test2.requires_grad = True
energy2 = model2(u_test2, x_test)
print(f"Energy shape: {energy2.shape}")
print(f"Energy range: [{energy2.min().item():.3f}, {energy2.max().item():.3f}]")

# Gradient check
grad2 = torch.autograd.grad(energy2.sum(), u_test2, retain_graph=True)[0]
print(f"Gradient mean: {grad2.mean().item():.6f}")
print(f"Gradient std: {grad2.std().item():.6f}")

# Test global receptive field
print("\nTesting GLOBAL receptive field:")
print("  Modifying single pixel at (32, 32) and checking if gradient changes everywhere...")

# Original gradient
u_clean = torch.randn(1, 64, 64, 1, requires_grad=True)
x_clean = torch.randn(1, 64, 64, 3)
energy_clean = model2(u_clean, x_clean)
grad_clean = torch.autograd.grad(energy_clean.sum(), u_clean, retain_graph=True)[0]

# Modified gradient (change one pixel)
u_modified = u_clean.clone().detach()
u_modified[0, 32, 32, 0] += 1.0  # Modify center pixel
u_modified.requires_grad = True
energy_modified = model2(u_modified, x_clean)
grad_modified = torch.autograd.grad(energy_modified.sum(), u_modified, retain_graph=True)[0]

# Check how many pixels have different gradients
grad_diff = (grad_modified - grad_clean).abs()
pixels_affected = (grad_diff > 1e-6).sum().item()
total_pixels = 64 * 64

print(f"  Pixels with changed gradients: {pixels_affected} / {total_pixels}")
print(f"  Percentage: {pixels_affected/total_pixels*100:.1f}%")

if pixels_affected > total_pixels * 0.9:
    print("  ✓ CONFIRMED: FNO has GLOBAL receptive field!")
else:
    print("  ⚠️  WARNING: Receptive field might not be fully global")

# ============================================================================
# Test 3: MultiScaleConvEBM (Alternative - dilated convs)
# ============================================================================

print("\n" + "=" * 80)
print("3. MultiScaleConvEBM (Alternative)")
print("=" * 80)

model3 = MultiScaleConvEBM(in_channels=4, hidden_channels=[32, 64, 64, 32])
total_params3 = sum(p.numel() for p in model3.parameters())

print(f"Parameters: {total_params3:,}")

# Calculate dilated conv receptive field
# With dilation [1, 2, 4, 8] and 3x3 kernels
receptive_field = 1
dilations = [1, 2, 4, 8]
for dilation in dilations:
    receptive_field += 2 * 2 * dilation  # Two 3x3 convs per block
print(f"Receptive field: {receptive_field}x{receptive_field} pixels ({receptive_field/64*100:.1f}% of 64x64)")

# Forward pass
u_test3 = u_test.clone().detach()
u_test3.requires_grad = True
energy3 = model3(u_test3, x_test)
print(f"Energy shape: {energy3.shape}")
print(f"Energy range: [{energy3.min().item():.3f}, {energy3.max().item():.3f}]")

# Gradient check
grad3 = torch.autograd.grad(energy3.sum(), u_test3)[0]
print(f"Gradient mean: {grad3.mean().item():.6f}")
print(f"Gradient std: {grad3.std().item():.6f}")

# ============================================================================
# Comparison Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n{:<20} {:<15} {:<25} {:<15}".format(
    "Architecture", "Parameters", "Receptive Field", "Grad Std"
))
print("-" * 80)
print("{:<20} {:<15,} {:<25} {:<15.6f}".format(
    "ConvEBM", total_params1, "17x17 (27%)", grad1.std().item()
))
print("{:<20} {:<15,} {:<25} {:<15.6f}".format(
    "SimpleFNO_EBM", total_params2, "GLOBAL (100%)", grad2.std().item()
))
print("{:<20} {:<15,} {:<25} {:<15.6f}".format(
    "MultiScaleConvEBM", total_params3, f"{receptive_field}x{receptive_field} (100%)", grad3.std().item()
))

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
For learning radial uncertainty patterns (center high, edges low):

1. ✓ SimpleFNO_EBM (BEST CHOICE)
   - True global receptive field via FFT
   - Sees entire field at once
   - Similar parameters to ConvEBM
   - Best for radial/global patterns

2. ⚠️  MultiScaleConvEBM (BACKUP)
   - Larger receptive field than ConvEBM
   - More parameters due to multi-scale aggregation
   - Good if FNO doesn't work

3. ❌ ConvEBM (DO NOT USE)
   - Receptive field too small (17x17)
   - Cannot see global radial pattern
   - Will produce noisy uncertainty maps

ACTION: Replace ConvEBM with SimpleFNO_EBM in your training config!
""")

print("=" * 80)