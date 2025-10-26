"""Quick test to debug EBM training stall"""
import torch
import torch.nn as nn
from ebm import ConvEBM, ConditionalEnergyWrapper
from torchebm.samplers import LangevinDynamics as LangevinSampler
from torchebm.losses import ContrastiveDivergence
import numpy as np

# Create small test data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 2
n_x, n_y = 16, 16  # Small grid for fast testing

# x: conditioning (coordinates + permeability)
x = torch.randn(batch_size, n_x, n_y, 3).to(device)
# y: solution to be sampled
y = torch.randn(batch_size, n_x, n_y, 1).to(device)

# Create EBM
ebm_model = ConvEBM(in_channels=4, hidden_channels=[32, 64, 32]).to(device)

print("Testing direct EBM call...")
try:
    energy = ebm_model(y, x)
    print(f"✓ Direct call works: energy shape = {energy.shape}")
except Exception as e:
    print(f"✗ Direct call failed: {e}")

print("\nTesting ConditionalEnergyWrapper...")
try:
    conditional_energy = ConditionalEnergyWrapper(
        energy_fn=ebm_model,
        condition=x
    )
    energy = conditional_energy(y)
    print(f"✓ Conditional wrapper works: energy shape = {energy.shape}")
except Exception as e:
    print(f"✗ Conditional wrapper failed: {e}")

print("\nTesting Langevin sampler with ConditionalEnergyWrapper...")
try:
    conditional_sampler = LangevinSampler(
        energy_function=conditional_energy,
        step_size=0.01,
        noise_scale=np.sqrt(2 * 0.01)
    )
    print("✓ Sampler created")

    # Try sampling with very few steps
    print("  Attempting 5 MCMC steps...")
    y_sampled = conditional_sampler.sample(
        x=y,
        n_steps=5
    )
    print(f"✓ Sampling works: sampled shape = {y_sampled.shape}")
except Exception as e:
    print(f"✗ Sampling failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting ContrastiveDivergence loss...")
try:
    cd_loss = ContrastiveDivergence(
        energy_function=conditional_energy,
        sampler=conditional_sampler,
        k_steps=5
    )
    print("✓ CD loss created")

    print("  Computing CD loss...")
    loss, neg_samples = cd_loss(y)
    print(f"✓ CD loss works: loss = {loss.item():.4f}")
except Exception as e:
    print(f"✗ CD loss failed: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")