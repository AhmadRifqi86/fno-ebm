"""
Physics Compliance Metrics

This module implements metrics for evaluating how well predictions satisfy
physical constraints:
- PDE residuals
- Boundary condition violations
- Conservation laws
- Energy conservation (for applicable systems)

These metrics are critical for physics-informed machine learning models.
"""

import numpy as np
import torch
from typing import Union, Dict, Callable, Optional
import torch.nn.functional as F


def compute_pde_residual(
    u: Union[np.ndarray, torch.Tensor],
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor] = None,
    pde_type: str = 'darcy',
    a: Union[np.ndarray, torch.Tensor] = None,
    f: Union[np.ndarray, torch.Tensor] = None,
    nu: float = 0.01,
    return_details: bool = False
) -> Union[float, Dict]:
    """
    Compute PDE residual for common equations.

    Args:
        u: Solution field, shape (batch, nx, ny, 1) or (nx, ny, 1)
        x: x-coordinates
        y: y-coordinates (if None, assumes uniform grid)
        pde_type: Type of PDE ('darcy', 'poisson', 'burgers', 'wave')
        a: Coefficient for Darcy flow (permeability)
        f: Forcing term
        nu: Viscosity for Burgers/Navier-Stokes
        return_details: If True, return detailed breakdown

    Returns:
        residual: Mean squared PDE residual
        OR dictionary with detailed results if return_details=True
    """
    # Convert to torch if numpy
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u).float()
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).float()
    if isinstance(f, np.ndarray):
        f = torch.from_numpy(f).float()

    u = u.requires_grad_(True)

    if pde_type == 'darcy':
        residual = _darcy_residual(u, a, f)
    elif pde_type == 'poisson':
        residual = _poisson_residual(u, f)
    elif pde_type == 'burgers':
        residual = _burgers_residual(u, nu)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")

    if return_details:
        return {
            'mean_residual': torch.mean(residual**2).item(),
            'max_residual': torch.max(torch.abs(residual)).item(),
            'std_residual': torch.std(residual).item(),
            'residual_field': residual.detach().cpu().numpy()
        }
    else:
        return torch.mean(residual**2).item()


def _darcy_residual(
    u: torch.Tensor,
    a: torch.Tensor,
    f: torch.Tensor
) -> torch.Tensor:
    """
    Compute residual for Darcy flow: -∇·(a∇u) = f

    Args:
        u: Solution (batch, nx, ny, 1)
        a: Permeability field (batch, nx, ny, 1)
        f: Source term (batch, nx, ny, 1)

    Returns:
        residual: -∇·(a∇u) - f
    """
    # Ensure correct shape
    if u.dim() == 3:
        u = u.unsqueeze(0)
    if a.dim() == 3:
        a = a.unsqueeze(0)
    if f.dim() == 3:
        f = f.unsqueeze(0)

    # Reshape for gradient computation: (batch, channels, H, W)
    u = u.permute(0, 3, 1, 2)  # (batch, 1, nx, ny)
    a = a.permute(0, 3, 1, 2)
    f = f.permute(0, 3, 1, 2)

    # Compute gradients using finite differences
    # ∂u/∂x
    du_dx = (u[:, :, 2:, 1:-1] - u[:, :, :-2, 1:-1]) / 2.0
    # ∂u/∂y
    du_dy = (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2]) / 2.0

    # Coefficient at interior points
    a_interior = a[:, :, 1:-1, 1:-1]

    # ∂(a·∂u/∂x)/∂x
    a_du_dx = a_interior * du_dx
    d_adu_dx = (a_du_dx[:, :, 1:, :] - a_du_dx[:, :, :-1, :])

    # ∂(a·∂u/∂y)/∂y
    a_du_dy = a_interior * du_dy
    d_adu_dy = (a_du_dy[:, :, :, 1:] - a_du_dy[:, :, :, :-1])

    # Laplacian: ∇·(a∇u)
    # Need to align shapes
    nx_res = min(d_adu_dx.shape[2], d_adu_dy.shape[2])
    ny_res = min(d_adu_dx.shape[3], d_adu_dy.shape[3])

    laplacian = -(d_adu_dx[:, :, :nx_res, :ny_res] + d_adu_dy[:, :, :nx_res, :ny_res])

    # Forcing term at same points
    f_interior = f[:, :, 2:2+nx_res, 2:2+ny_res]

    # Residual: -∇·(a∇u) - f
    residual = laplacian - f_interior

    return residual.squeeze()


def _poisson_residual(
    u: torch.Tensor,
    f: torch.Tensor
) -> torch.Tensor:
    """
    Compute residual for Poisson equation: -∇²u = f

    Args:
        u: Solution
        f: Source term

    Returns:
        residual: -∇²u - f
    """
    # Ensure correct shape
    if u.dim() == 3:
        u = u.unsqueeze(0)
    if f.dim() == 3:
        f = f.unsqueeze(0)

    u = u.permute(0, 3, 1, 2)
    f = f.permute(0, 3, 1, 2)

    # Compute Laplacian using finite differences
    # ∂²u/∂x²
    d2u_dx2 = u[:, :, 2:, 1:-1] - 2*u[:, :, 1:-1, 1:-1] + u[:, :, :-2, 1:-1]

    # ∂²u/∂y²
    d2u_dy2 = u[:, :, 1:-1, 2:] - 2*u[:, :, 1:-1, 1:-1] + u[:, :, 1:-1, :-2]

    # Laplacian
    laplacian = -(d2u_dx2[:, :, :, :-1] + d2u_dy2[:, :, :-1, :])

    # Forcing at interior points
    f_interior = f[:, :, 2:-2, 2:-2]

    # Residual
    residual = laplacian - f_interior

    return residual.squeeze()


def _burgers_residual(
    u: torch.Tensor,
    nu: float = 0.01
) -> torch.Tensor:
    """
    Compute residual for steady Burgers equation: u·∂u/∂x = ν·∂²u/∂x²

    Args:
        u: Solution
        nu: Viscosity

    Returns:
        residual
    """
    if u.dim() == 3:
        u = u.unsqueeze(0)

    u = u.permute(0, 3, 1, 2)

    # First derivative
    du_dx = (u[:, :, 2:, :] - u[:, :, :-2, :]) / 2.0

    # Second derivative
    d2u_dx2 = u[:, :, 2:, :] - 2*u[:, :, 1:-1, :] + u[:, :, :-2, :]

    # Align shapes
    u_interior = u[:, :, 1:-1, :]
    du_dx = du_dx[:, :, :-1, :]
    d2u_dx2 = d2u_dx2[:, :, :-1, :]

    # Residual: u·∂u/∂x - ν·∂²u/∂x²
    residual = u_interior * du_dx - nu * d2u_dx2

    return residual.squeeze()


def boundary_condition_violation(
    u: Union[np.ndarray, torch.Tensor],
    bc_type: str = 'dirichlet_zero',
    bc_value: float = 0.0,
    bc_mask: Union[np.ndarray, torch.Tensor] = None
) -> float:
    """
    Compute boundary condition violation.

    Args:
        u: Solution field, shape (*data_shape)
        bc_type: Type of BC ('dirichlet_zero', 'dirichlet', 'neumann', 'periodic')
        bc_value: Value for Dirichlet BC
        bc_mask: Boolean mask indicating boundary points

    Returns:
        bc_error: Mean squared boundary condition error
    """
    # Convert to numpy
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()

    if bc_mask is not None:
        if isinstance(bc_mask, torch.Tensor):
            bc_mask = bc_mask.detach().cpu().numpy()
    else:
        # Default: assume boundary is outer edge
        bc_mask = np.zeros_like(u, dtype=bool)
        bc_mask[0, :] = True   # Top
        bc_mask[-1, :] = True  # Bottom
        bc_mask[:, 0] = True   # Left
        bc_mask[:, -1] = True  # Right

    if bc_type == 'dirichlet_zero':
        bc_error = np.mean(u[bc_mask]**2)
    elif bc_type == 'dirichlet':
        bc_error = np.mean((u[bc_mask] - bc_value)**2)
    elif bc_type == 'neumann':
        # Check if derivative is close to bc_value (approximate with finite diff)
        # This is simplified - proper implementation needs gradient computation
        bc_error = 0.0  # Placeholder
    elif bc_type == 'periodic':
        # Check if opposite boundaries match
        left_edge = u[:, 0]
        right_edge = u[:, -1]
        top_edge = u[0, :]
        bottom_edge = u[-1, :]

        bc_error = (np.mean((left_edge - right_edge)**2) +
                   np.mean((top_edge - bottom_edge)**2)) / 2
    else:
        raise ValueError(f"Unknown BC type: {bc_type}")

    return bc_error


def conservation_law_check(
    u: Union[np.ndarray, torch.Tensor],
    conserved_quantity: str = 'mass',
    initial_value: float = None
) -> float:
    """
    Check conservation law violations.

    Args:
        u: Solution field
        conserved_quantity: Type of conservation ('mass', 'energy', 'momentum')
        initial_value: Initial value of conserved quantity (if known)

    Returns:
        violation: Deviation from conservation
    """
    # Convert to numpy
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()

    if conserved_quantity == 'mass':
        # Check total mass
        total = np.sum(u)
        if initial_value is not None:
            violation = abs(total - initial_value)
        else:
            violation = 0.0  # Can't check without reference

    elif conserved_quantity == 'energy':
        # Check total energy (assuming u represents energy density)
        total_energy = np.sum(u**2)
        if initial_value is not None:
            violation = abs(total_energy - initial_value)
        else:
            violation = 0.0

    else:
        raise ValueError(f"Unknown conserved quantity: {conserved_quantity}")

    return violation


def physics_consistency_score(
    samples: Union[np.ndarray, torch.Tensor],
    pde_type: str = 'darcy',
    **kwargs
) -> Dict[str, float]:
    """
    Compute comprehensive physics consistency scores across samples.

    Args:
        samples: Multiple solution samples, shape (n_samples, *data_shape)
        pde_type: Type of PDE
        **kwargs: Additional arguments for PDE residual computation

    Returns:
        Dictionary with various physics metrics
    """
    # Convert to torch
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples).float()

    n_samples = samples.shape[0]

    residuals = []
    bc_errors = []

    for i in range(n_samples):
        u = samples[i]

        # PDE residual
        res = compute_pde_residual(u, None, None, pde_type=pde_type, **kwargs)
        residuals.append(res)

        # BC violation
        bc_err = boundary_condition_violation(u)
        bc_errors.append(bc_err)

    return {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'max_residual': np.max(residuals),
        'mean_bc_error': np.mean(bc_errors),
        'max_bc_error': np.max(bc_errors),
        'physics_score': np.mean(residuals) + np.mean(bc_errors)  # Combined score
    }