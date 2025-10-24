#!/usr/bin/env python3
"""
Diagnostic script to check if EBM training is working correctly.

This script checks:
1. Does EBM energy landscape have structure?
2. Is energy gradient meaningful?
3. Does MCMC sampling produce diverse samples?
4. Is uncertainty spatially structured or just noise?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
import yaml
import os
from pathlib import Path

print("="*70)
print("EBM TRAINING QUALITY DIAGNOSTIC")
print("="*70)

# Load config
with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)
    config = Config(config_dict)

print(f"\n📋 Configuration:")
print(f"   MCMC Steps:     {config.mcmc_steps}")
print(f"   MCMC Step Size: {config.mcmc_step_size}")
print(f"   Device:         {config.device}")

# Load models
print(f"\n🔄 Loading models...")

from fno import FNO2d
from ebm import EBMPotential

# Load FNO
fno_checkpoint = os.path.join(config.checkpoint_dir, 'best_model_fno.pt')
if not os.path.exists(fno_checkpoint):
    print(f"❌ FNO checkpoint not found: {fno_checkpoint}")
    exit(1)

fno_model = FNO2d(
    modes1=config.fno_modes,
    modes2=config.fno_modes,
    width=config.fno_width,
    num_layers=4
).to(config.device)

fno_ckpt = torch.load(fno_checkpoint, map_location=config.device, weights_only=False)
if 'fno_model' in fno_ckpt:
    fno_model.load_state_dict(fno_ckpt['fno_model'])
else:
    fno_model.load_state_dict(fno_ckpt)
fno_model.eval()
print("   ✓ FNO loaded")

# Load EBM
ebm_checkpoint = os.path.join(config.checkpoint_dir, 'best_model_ebm.pt')
if not os.path.exists(ebm_checkpoint):
    print(f"❌ EBM checkpoint not found: {ebm_checkpoint}")
    exit(1)

ebm_model = EBMPotential(
    input_dim=4,  # u + x coords + input field
    hidden_dims=[config.ebm_hidden_dim, config.ebm_hidden_dim*2, config.ebm_hidden_dim*2, config.ebm_hidden_dim]
).to(config.device)

ebm_ckpt = torch.load(ebm_checkpoint, map_location=config.device, weights_only=False)
if 'ebm_model' in ebm_ckpt:
    ebm_model.load_state_dict(ebm_ckpt['ebm_model'])
else:
    ebm_model.load_state_dict(ebm_ckpt)
ebm_model.eval()
print("   ✓ EBM loaded")

# Load validation data
print(f"\n📊 Loading validation data...")
from datautils import create_dataloaders

_, val_loader = create_dataloaders(config)
x_batch, y_batch = next(iter(val_loader))
x_batch = x_batch.to(config.device)
y_batch = y_batch.to(config.device)

# Take first sample for detailed analysis
x = x_batch[0:1]  # (1, 64, 64, 3)
y_true = y_batch[0:1]  # (1, 64, 64, 1)

print(f"   Data shape: {x.shape}")
print(f"   Truth range: [{y_true.min():.4f}, {y_true.max():.4f}]")

# Get FNO prediction
with torch.no_grad():
    y_fno = fno_model(x)

print(f"   FNO range: [{y_fno.min():.4f}, {y_fno.max():.4f}]")

print("\n" + "="*70)
print("TEST 1: Energy Landscape Structure")
print("="*70)

# Define energy function
def compute_energy(u, x, fno_pred):
    """Compute total energy E(u,x) = quadratic + V_ebm(u,x)"""
    # Quadratic term
    quadratic = 0.5 * torch.mean((u - fno_pred)**2, dim=[1, 2, 3])

    # EBM potential
    v_ebm = ebm_model(x, u)

    # Total energy
    energy = quadratic + v_ebm
    return energy, quadratic, v_ebm

# Test energy at different points
print("\n🔬 Testing energy at different solution values:")

test_points = {
    "FNO prediction": y_fno,
    "Ground truth": y_true,
    "Small perturbation (+10%)": y_fno + 0.1 * torch.randn_like(y_fno),
    "Large perturbation (+50%)": y_fno + 0.5 * torch.randn_like(y_fno),
    "Constant field (zeros)": torch.zeros_like(y_fno),
    "Random field": torch.randn_like(y_fno) * y_true.std()
}

energy_results = {}
with torch.no_grad():
    for name, u in test_points.items():
        E_total, E_quad, E_ebm = compute_energy(u, x, y_fno)
        energy_results[name] = {
            'total': E_total.item(),
            'quadratic': E_quad.item(),
            'ebm': E_ebm.item()
        }
        print(f"\n   {name}:")
        print(f"      Total Energy: {E_total.item():8.4f}")
        print(f"      Quadratic:    {E_quad.item():8.4f}")
        print(f"      V_EBM:        {E_ebm.item():8.4f}")

# Check if energy landscape has structure
print("\n📊 Energy Analysis:")

E_fno = energy_results["FNO prediction"]["total"]
E_perturbed_small = energy_results["Small perturbation (+10%)"]["total"]
E_perturbed_large = energy_results["Large perturbation (+50%)"]["total"]

energy_sensitivity_small = abs(E_perturbed_small - E_fno)
energy_sensitivity_large = abs(E_perturbed_large - E_fno)

print(f"   Energy at FNO:           {E_fno:.6f}")
print(f"   Energy change (+10%):    {energy_sensitivity_small:.6f}")
print(f"   Energy change (+50%):    {energy_sensitivity_large:.6f}")

# Critical check 1: Is energy landscape flat?
if energy_sensitivity_large < 0.01:
    print("\n   ❌ CRITICAL: Energy landscape is TOO FLAT!")
    print("      Energy barely changes when solution is perturbed.")
    print("      EBM did not learn meaningful structure.")
    ebm_learned_structure = False
elif energy_sensitivity_large < 0.1:
    print("\n   ⚠️  WARNING: Energy landscape is quite flat.")
    print("      EBM learned weak structure.")
    ebm_learned_structure = False
else:
    print("\n   ✅ GOOD: Energy landscape has structure!")
    print("      EBM learned to distinguish good vs bad solutions.")
    ebm_learned_structure = True

print("\n" + "="*70)
print("TEST 2: Energy Gradient Magnitude")
print("="*70)

# Check energy gradients
print("\n🔬 Computing energy gradients...")

u_test = y_fno.clone().detach().requires_grad_(True)
E_total, E_quad, E_ebm = compute_energy(u_test, x, y_fno)
grad_u = torch.autograd.grad(E_total.sum(), u_test)[0]

grad_magnitude = grad_u.abs().mean().item()
grad_std = grad_u.std().item()
grad_max = grad_u.abs().max().item()

print(f"\n   Gradient statistics at FNO prediction:")
print(f"      Mean |∇E|: {grad_magnitude:.6f}")
print(f"      Std  ∇E:   {grad_std:.6f}")
print(f"      Max |∇E|:  {grad_max:.6f}")

# Critical check 2: Are gradients too small?
if grad_magnitude < 1e-4:
    print("\n   ❌ CRITICAL: Gradients are TOO SMALL!")
    print("      MCMC will barely move from initialization.")
    meaningful_gradients = False
elif grad_magnitude < 1e-3:
    print("\n   ⚠️  WARNING: Gradients are quite small.")
    print("      MCMC will move slowly.")
    meaningful_gradients = False
else:
    print("\n   ✅ GOOD: Gradients are meaningful!")
    print("      MCMC can explore the landscape.")
    meaningful_gradients = True

print("\n" + "="*70)
print("TEST 3: MCMC Sampling Quality")
print("="*70)

print(f"\n🔬 Running Langevin MCMC with {config.mcmc_steps} steps...")

# Run MCMC sampling
def langevin_sample(u_init, x, fno_pred, num_steps, step_size):
    """Run Langevin dynamics"""
    u = u_init.clone()
    trajectory = [u.clone().detach()]
    energies = []

    for step in range(num_steps):
        u.requires_grad_(True)
        E_total, _, _ = compute_energy(u, x, fno_pred)
        grad_u = torch.autograd.grad(E_total.sum(), u)[0]

        with torch.no_grad():
            noise = torch.randn_like(u) * np.sqrt(2 * step_size)
            u = u - step_size * grad_u + noise

            # Record trajectory
            if step % 10 == 0:
                trajectory.append(u.clone())
                energies.append(E_total.item())

    return u, trajectory, energies

# Generate multiple samples
num_samples = 20
samples = []
final_energies = []

for i in range(num_samples):
    u_sample, traj, energies = langevin_sample(
        y_fno, x, y_fno,
        num_steps=config.mcmc_steps,
        step_size=config.mcmc_step_size
    )
    samples.append(u_sample.detach().cpu())
    final_energies.append(energies[-1] if energies else E_fno)

samples_tensor = torch.stack(samples, dim=0)  # (num_samples, 1, 64, 64, 1)

# Compute sample statistics
sample_mean = samples_tensor.mean(dim=0)
sample_std = samples_tensor.std(dim=0)

print(f"\n   Generated {num_samples} samples")
print(f"\n   Sample statistics:")
print(f"      Mean:  {sample_mean.mean():.6f} ± {sample_mean.std():.6f}")
print(f"      Range: [{samples_tensor.min():.4f}, {samples_tensor.max():.4f}]")

# Check sample diversity
sample_range = samples_tensor.max() - samples_tensor.min()
fno_range = y_fno.max() - y_fno.min()
diversity_ratio = sample_range.item() / (fno_range.item() + 1e-8)

print(f"\n   Sample diversity:")
print(f"      Sample range:     {sample_range.item():.6f}")
print(f"      FNO range:        {fno_range.item():.6f}")
print(f"      Diversity ratio:  {diversity_ratio:.4f}")

# Critical check 3: Are samples diverse?
if diversity_ratio < 0.1:
    print("\n   ❌ CRITICAL: Samples have NO diversity!")
    print("      All samples are nearly identical.")
    print("      MCMC is not exploring the energy landscape.")
    diverse_samples = False
elif diversity_ratio < 0.5:
    print("\n   ⚠️  WARNING: Samples have low diversity.")
    print("      MCMC needs more steps or larger step size.")
    diverse_samples = False
else:
    print("\n   ✅ GOOD: Samples are diverse!")
    print("      MCMC is exploring different solutions.")
    diverse_samples = True

print("\n" + "="*70)
print("TEST 4: Uncertainty Spatial Structure")
print("="*70)

print("\n🔬 Analyzing uncertainty maps...")

# Compute pointwise statistics
uncertainty_map = sample_std.squeeze().numpy()  # (64, 64)

# Check if uncertainty is spatially structured
# Compute spatial autocorrelation
from scipy.ndimage import convolve

# Simple autocorrelation: uncertainty similar to neighbors?
kernel = np.array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]]) / 4.0

uncertainty_smoothed = convolve(uncertainty_map, kernel, mode='constant')
autocorr = np.corrcoef(uncertainty_map.flatten(), uncertainty_smoothed.flatten())[0, 1]

print(f"\n   Uncertainty map statistics:")
print(f"      Mean uncertainty:  {uncertainty_map.mean():.6f}")
print(f"      Std uncertainty:   {uncertainty_map.std():.6f}")
print(f"      Min uncertainty:   {uncertainty_map.min():.6f}")
print(f"      Max uncertainty:   {uncertainty_map.max():.6f}")
print(f"      Spatial autocorr:  {autocorr:.4f}")

# Critical check 4: Is uncertainty spatially structured?
if autocorr < 0.1:
    print("\n   ❌ CRITICAL: Uncertainty is RANDOM NOISE!")
    print("      No spatial correlation - just pixel-level noise.")
    print("      EBM uncertainty is meaningless.")
    structured_uncertainty = False
elif autocorr < 0.5:
    print("\n   ⚠️  WARNING: Uncertainty has weak spatial structure.")
    print("      Some correlation but mostly noise.")
    structured_uncertainty = False
else:
    print("\n   ✅ GOOD: Uncertainty is spatially structured!")
    print("      Meaningful spatial patterns in uncertainty.")
    structured_uncertainty = True

# Check if uncertainty is uniform (bad) or varying (good)
uncertainty_variation = uncertainty_map.std() / (uncertainty_map.mean() + 1e-8)
print(f"\n   Uncertainty variation:")
print(f"      Coefficient of variation: {uncertainty_variation:.4f}")

if uncertainty_variation < 0.1:
    print("      ❌ Nearly uniform - no information!")
elif uncertainty_variation < 0.3:
    print("      ⚠️  Somewhat varying")
else:
    print("      ✅ Good variation across space")

print("\n" + "="*70)
print("OVERALL ASSESSMENT")
print("="*70)

# Count passed checks
checks = {
    "Energy landscape structure": ebm_learned_structure,
    "Meaningful gradients": meaningful_gradients,
    "Sample diversity": diverse_samples,
    "Structured uncertainty": structured_uncertainty
}

passed = sum(checks.values())
total = len(checks)

print(f"\n📊 Test Results: {passed}/{total} passed\n")

for check_name, result in checks.items():
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"   {status}: {check_name}")

print("\n💡 Overall Verdict:\n")

if passed == 4:
    print("   🎉 EXCELLENT! EBM is working properly!")
    print("      Your uncertainty quantification is meaningful.")
    print("      The spatial patterns in uncertainty are real.")
elif passed == 3:
    print("   👍 GOOD! EBM is mostly working.")
    print("      Minor issues but uncertainty is reasonable.")
elif passed == 2:
    print("   ⚠️  MEDIOCRE. EBM has significant issues.")
    print("      Uncertainty may not be reliable.")
else:
    print("   ❌ POOR! EBM is NOT working.")
    print("      Uncertainty is likely meaningless noise.")

print("\n📋 Recommendations:\n")

if not ebm_learned_structure:
    print("   🔧 Energy landscape is flat:")
    print("      → EBM did not learn during training")
    print("      → Check EBM training loss - did it decrease?")
    print("      → May need to retrain EBM with different hyperparameters")

if not meaningful_gradients:
    print("   🔧 Gradients too small:")
    print("      → Increase mcmc_step_size in config.yaml")
    print("      → Try: mcmc_step_size: 0.05 or 0.1")

if not diverse_samples:
    print("   🔧 Samples not diverse:")
    print("      → Increase mcmc_steps in config.yaml")
    print("      → Try: mcmc_steps: 200 or 400")
    print("      → Increase mcmc_step_size: 0.05")

if not structured_uncertainty:
    print("   🔧 Uncertainty is noise:")
    print("      → This is the combination of above issues")
    print("      → Fix: Increase both mcmc_steps AND mcmc_step_size")
    print("      → If still fails, retrain EBM")

if passed == 4:
    print("   ✨ Your current settings are working well!")
    print("      No changes needed.")

print("\n" + "="*70)

# Create visualization
print("\n📊 Creating diagnostic visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Solutions
im0 = axes[0, 0].imshow(y_true.squeeze().cpu().numpy(), cmap='viridis')
axes[0, 0].set_title('Ground Truth')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(y_fno.squeeze().cpu().numpy(), cmap='viridis')
axes[0, 1].set_title('FNO Prediction')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(sample_mean.squeeze().numpy(), cmap='viridis')
axes[0, 2].set_title('EBM Mean')
plt.colorbar(im2, ax=axes[0, 2])

# Row 2: Uncertainty analysis
im3 = axes[1, 0].imshow(uncertainty_map, cmap='hot')
axes[1, 0].set_title(f'EBM Std Dev (Autocorr={autocorr:.2f})')
plt.colorbar(im3, ax=axes[1, 0])

# Histogram of uncertainty
axes[1, 1].hist(uncertainty_map.flatten(), bins=50, color='orange', alpha=0.7)
axes[1, 1].set_xlabel('Uncertainty')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Uncertainty Distribution')
axes[1, 1].axvline(uncertainty_map.mean(), color='red', linestyle='--', label='Mean')
axes[1, 1].legend()

# Sample diversity: show std of each pixel
sample_diversity = samples_tensor.std(dim=0).squeeze().numpy()
im5 = axes[1, 2].imshow(sample_diversity, cmap='hot')
axes[1, 2].set_title(f'Sample Diversity (CoV={uncertainty_variation:.2f})')
plt.colorbar(im5, ax=axes[1, 2])

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
output_path = os.path.join(config.checkpoint_dir, 'ebm_diagnostic.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   Saved diagnostic plot to: {output_path}")

print("\n✅ Diagnostic complete!")
print("\n💡 Next steps:")
print("   1. Review the diagnostic plot: ebm_diagnostic.png")
print("   2. Follow the recommendations above")
print("   3. If needed, update config.yaml and rerun inference")
print("\n" + "="*70)