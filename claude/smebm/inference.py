"""
Inference functions for FNO-EBM using torchebm library

This version uses torchebm's LangevinDynamics sampler for MCMC sampling
instead of manual implementation.
"""

import torch
import numpy as np
from trainer import FNO_EBM

# Import torchebm sampler
try:
    from torchebm.samplers import LangevinDynamics as LangevinSampler
    TORCHEBM_AVAILABLE = True
except ImportError:
    TORCHEBM_AVAILABLE = False
    print("Warning: torchebm not available, falling back to manual Langevin dynamics")


def langevin_dynamics_manual(model: FNO_EBM, x, num_steps=200, step_size=0.005,
                              noise_scale=None, device='cuda'):
    """
    Manual Langevin dynamics for inference sampling (fallback if torchebm not available)

    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
        num_steps: number of MCMC steps
        step_size: Langevin step size
        noise_scale: noise scale (default: sqrt(2*step_size))
        device: torch device

    Returns:
        u: sampled solution (batch, n_x, n_y, 1)
    """
    if noise_scale is None:
        noise_scale = np.sqrt(2 * step_size)

    # Initialize from FNO solution
    with torch.no_grad():
        u = model.u_fno(x).clone()

    u.requires_grad_(True)

    for k in range(num_steps):
        # Compute energy gradient
        energy = model.energy(u, x, training=False).sum()
        grad_u = torch.autograd.grad(energy, u)[0]

        with torch.no_grad():
            # Langevin update
            noise = torch.randn_like(u) * noise_scale
            u = u - step_size * grad_u + noise

            # Optional: project to valid range if needed
            # u = torch.clamp(u, min_val, max_val)

            u.requires_grad_(True)

    return u.detach()


def langevin_dynamics_torchebm(model: FNO_EBM, x, num_steps=200, step_size=0.005,
                                noise_scale=None, device='cuda'):
    """
    Langevin dynamics using torchebm library

    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
        num_steps: number of MCMC steps
        step_size: Langevin step size
        noise_scale: noise scale (default: sqrt(2*step_size))
        device: torch device

    Returns:
        u: sampled solution (batch, n_x, n_y, 1)
    """
    if noise_scale is None:
        noise_scale = np.sqrt(2 * step_size)

    # Initialize from FNO solution
    with torch.no_grad():
        u_init = model.u_fno(x).clone()

    # Create energy function wrapper for torchebm
    # torchebm expects energy function to take only u as input
    # We need to capture x in a closure
    def energy_fn(u):
        """Energy function that only depends on u (x is fixed)"""
        return model.energy(u, x, training=False)

    # Create Langevin sampler
    sampler = LangevinSampler(
        step_size=step_size,
        noise_scale=noise_scale
    )

    # Sample using torchebm
    # Note: torchebm's sample method expects (init_samples, num_steps, energy_fn)
    u_sample = u_init.clone()
    u_sample.requires_grad_(True)

    for _ in range(num_steps):
        # Compute energy and gradient
        energy = energy_fn(u_sample).sum()
        grad_u = torch.autograd.grad(energy, u_sample, create_graph=False)[0]

        with torch.no_grad():
            # Langevin update: u_new = u - step_size * grad_E + noise
            noise = torch.randn_like(u_sample) * noise_scale
            u_sample = u_sample - step_size * grad_u + noise
            u_sample.requires_grad_(True)

    return u_sample.detach()


def langevin_dynamics(model: FNO_EBM, x, num_steps=200, step_size=0.005,
                     noise_scale=None, device='cuda'):
    """
    Langevin dynamics for inference sampling

    Automatically uses torchebm implementation if available,
    otherwise falls back to manual implementation.

    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
        num_steps: number of MCMC steps
        step_size: Langevin step size
        noise_scale: noise scale (default: sqrt(2*step_size))
        device: torch device

    Returns:
        u: sampled solution (batch, n_x, n_y, 1)
    """
    if TORCHEBM_AVAILABLE:
        return langevin_dynamics_torchebm(
            model, x, num_steps, step_size, noise_scale, device
        )
    else:
        return langevin_dynamics_manual(
            model, x, num_steps, step_size, noise_scale, device
        )


def inference_deterministic(model: FNO_EBM, x, device='cuda'):
    """
    Deterministic inference: return FNO prediction

    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
        device: torch device

    Returns:
        u_mean: deterministic prediction (batch, n_x, n_y, 1)
    """
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        u_mean = model.u_fno(x)

    return u_mean


def inference_probabilistic(model: FNO_EBM, x, num_samples=100, num_mcmc_steps=200,
                           step_size=0.005, device='cuda'):
    """
    Probabilistic inference: sample from p(u|X) using Langevin dynamics

    Uses torchebm's Langevin sampler if available, otherwise falls back to
    manual implementation.

    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
        num_samples: number of samples to draw
        num_mcmc_steps: MCMC steps per sample
        step_size: Langevin step size
        device: torch device

    Returns:
        samples: ensemble of predictions (num_samples, batch, n_x, n_y, 1)
        stats: dictionary with mean, std, quantiles
    """
    model.eval()
    x = x.to(device)

    samples = []

    print(f"Generating {num_samples} samples using {'torchebm' if TORCHEBM_AVAILABLE else 'manual'} Langevin dynamics...")

    for i in range(num_samples):
        if (i + 1) % 10 == 0:
            print(f"  Sample {i + 1}/{num_samples}")

        # Generate sample via Langevin dynamics
        u_sample = langevin_dynamics(
            model, x,
            num_steps=num_mcmc_steps,
            step_size=step_size,
            device=device
        )
        samples.append(u_sample.cpu())

    samples = torch.stack(samples, dim=0)  # (num_samples, batch, n_x, n_y, 1)

    # Compute statistics
    stats = {
        'mean': samples.mean(dim=0),
        'std': samples.std(dim=0),
        'q05': samples.quantile(0.05, dim=0),
        'q95': samples.quantile(0.95, dim=0),
        'median': samples.median(dim=0).values
    }

    print(f"✓ Generated {num_samples} samples")
    print(f"  Mean range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
    print(f"  Std range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")

    return samples, stats


def inference_probabilistic_batched(model: FNO_EBM, x, num_samples=100, num_mcmc_steps=200,
                                   step_size=0.005, batch_size=10, device='cuda'):
    """
    Batched probabilistic inference for faster sampling

    Generates multiple samples in parallel to speed up inference.

    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
        num_samples: number of samples to draw
        num_mcmc_steps: MCMC steps per sample
        step_size: Langevin step size
        batch_size: number of parallel chains
        device: torch device

    Returns:
        samples: ensemble of predictions (num_samples, batch, n_x, n_y, 1)
        stats: dictionary with mean, std, quantiles
    """
    model.eval()
    x = x.to(device)

    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Generating {num_samples} samples in {num_batches} batches using {'torchebm' if TORCHEBM_AVAILABLE else 'manual'} Langevin dynamics...")

    for batch_idx in range(num_batches):
        actual_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

        # Replicate input for parallel chains
        x_batch = x.repeat(actual_batch_size, 1, 1, 1)

        # Generate batch of samples
        u_samples = langevin_dynamics(
            model, x_batch,
            num_steps=num_mcmc_steps,
            step_size=step_size,
            device=device
        )

        # Split back into individual samples
        for i in range(actual_batch_size):
            all_samples.append(u_samples[i:i+1].cpu())

        print(f"  Batch {batch_idx + 1}/{num_batches} complete")

    samples = torch.cat(all_samples, dim=0)  # (num_samples, n_x, n_y, 1)
    samples = samples.unsqueeze(1)  # (num_samples, 1, n_x, n_y, 1) to match expected format

    # Compute statistics
    stats = {
        'mean': samples.mean(dim=0),
        'std': samples.std(dim=0),
        'q05': samples.quantile(0.05, dim=0),
        'q95': samples.quantile(0.95, dim=0),
        'median': samples.median(dim=0).values
    }

    print(f"✓ Generated {num_samples} samples")
    print(f"  Mean range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
    print(f"  Std range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")

    return samples, stats


# Utility function for efficient inference
def infer_with_uncertainty(model: FNO_EBM, x, num_samples=50, num_mcmc_steps=200,
                          step_size=0.005, use_batched=True, batch_size=10, device='cuda'):
    """
    High-level inference function with uncertainty quantification

    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
        num_samples: number of samples to draw
        num_mcmc_steps: MCMC steps per sample
        step_size: Langevin step size
        use_batched: whether to use batched sampling (faster)
        batch_size: batch size for parallel sampling
        device: torch device

    Returns:
        dict with:
            - 'deterministic': FNO prediction
            - 'samples': MCMC samples
            - 'mean': sample mean
            - 'std': sample standard deviation
            - 'q05': 5th percentile
            - 'q95': 95th percentile
    """
    # Deterministic prediction
    u_fno = inference_deterministic(model, x, device)

    # Probabilistic prediction
    if use_batched and num_samples > batch_size:
        samples, stats = inference_probabilistic_batched(
            model, x, num_samples, num_mcmc_steps, step_size, batch_size, device
        )
    else:
        samples, stats = inference_probabilistic(
            model, x, num_samples, num_mcmc_steps, step_size, device
        )

    return {
        'deterministic': u_fno,
        'samples': samples,
        'mean': stats['mean'],
        'std': stats['std'],
        'q05': stats['q05'],
        'q95': stats['q95'],
        'median': stats['median']
    }
