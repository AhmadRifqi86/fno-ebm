
import torch
import numpy as np
from trainer import FNO_EBM

def langevin_dynamics(model: FNO_EBM, x, num_steps=200, step_size=0.005, 
                     noise_scale=None, device='cuda'):
    """
    Langevin dynamics for inference sampling
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

def inference_deterministic(model: FNO_EBM, x, device='cuda'):
    """
    Deterministic inference: return FNO prediction
    
    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
    
    Returns:
        u_mean: deterministic prediction (batch, n_x, n_y, 1)
    """
    model.eval()
    x = x.to(device)
    
    with torch.no_grad():
        u_mean = model.u_fno(x)
    
    return u_mean


def inference_probabilistic(model: FNO_EBM, x, num_samples=100, num_mcmc_steps=200,
                           step_size=0.0001, device='cuda'):
    """
    Probabilistic inference: sample from p(u|X) using Langevin dynamics
    
    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
        num_samples: number of samples to draw
        num_mcmc_steps: MCMC steps per sample
        step_size: Langevin step size
    
    Returns:
        samples: ensemble of predictions (num_samples, batch, n_x, n_y, 1)
        stats: dictionary with mean, std, quantiles
    """
    model.eval()
    x = x.to(device)
    
    samples = []
    
    for i in range(num_samples):
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
    
    return samples, stats