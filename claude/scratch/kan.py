"""
Kolmogorov-Arnold Network (KAN) Layer with Multiple Approximators

Supports three types of basis functions:
- Gaussian RBF (Radial Basis Functions)
- Chebyshev Polynomials
- Fourier Basis

Usage:
    from kan import KANLayer, GaussianRBF, ChebyshevPoly, FourierBasis

    # Using Gaussian RBF
    kan = KANLayer(in_features=8, out_features=1, approx=GaussianRBF(grid_size=10))

    # Using Chebyshev polynomials
    kan = KANLayer(in_features=8, out_features=1, approx=ChebyshevPoly(degree=4))

    # Using Fourier basis
    kan = KANLayer(in_features=8, out_features=1, approx=FourierBasis(grid_size=5))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Approximator Base Class
# ============================================================================

class Approximator(nn.Module):
    """Base class for KAN approximators"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Compute basis functions for input x
        Args:
            x: input tensor (..., features)
        Returns:
            basis: basis function evaluations (..., features, basis_dim)
        """
        raise NotImplementedError

    def basis_dim(self):
        """Return the number of basis functions"""
        raise NotImplementedError


# ============================================================================
# Gaussian RBF Approximator
# ============================================================================

class GaussianRBF(Approximator):
    """
    Gaussian Radial Basis Function approximator
    Uses Gaussian kernels: φ(x) = exp(-((x - μ) / σ)²)
    """
    def __init__(self, grid_size=10, sigma=1.0):
        super().__init__()
        self.grid_size = grid_size

        # Create grid of centers uniformly in [-1, 1]
        centers = torch.linspace(-1, 1, grid_size)
        self.register_buffer('centers', centers)

        # Learnable width parameter
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))

    def forward(self, x):
        """
        Args:
            x: (..., features)
        Returns:
            (..., features, grid_size)
        """
        # Clamp input to [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)

        # Expand dimensions for broadcasting
        x_exp = x.unsqueeze(-1)  # (..., features, 1)
        centers = self.centers.unsqueeze(0)  # (1, grid_size)

        # Compute Gaussian basis
        sigma = torch.exp(self.log_sigma)
        diff = (x_exp - centers) / sigma
        basis = torch.exp(-0.5 * diff ** 2)

        return basis

    def basis_dim(self):
        return self.grid_size


# ============================================================================
# Chebyshev Polynomial Approximator
# ============================================================================

class ChebyshevPoly(Approximator):
    """
    Chebyshev polynomial approximator
    Uses Chebyshev polynomials: T_n(x) = cos(n * arccos(x))
    Recursion: T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
    """
    def __init__(self, degree=4):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        """
        Args:
            x: (..., features)
        Returns:
            (..., features, degree+1)
        """
        # Normalize input to [-1, 1] using tanh
        x = torch.tanh(x)

        # Initialize first two Chebyshev polynomials
        shape = x.shape
        chebs = []

        # T_0(x) = 1
        T0 = torch.ones_like(x)
        chebs.append(T0)

        if self.degree >= 1:
            # T_1(x) = x
            T1 = x
            chebs.append(T1)

        # Compute higher order polynomials using recurrence
        for n in range(2, self.degree + 1):
            # T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
            Tn = 2 * x * chebs[-1] - chebs[-2]
            chebs.append(Tn)

        # Stack along last dimension
        basis = torch.stack(chebs, dim=-1)  # (..., features, degree+1)

        return basis

    def basis_dim(self):
        return self.degree + 1


# ============================================================================
# Fourier Basis Approximator
# ============================================================================

class FourierBasis(Approximator):
    """
    Fourier basis approximator
    Uses sine and cosine functions: sin(kx), cos(kx) for k=1..grid_size
    """
    def __init__(self, grid_size=5):
        super().__init__()
        self.grid_size = grid_size

        # Frequency indices
        freqs = torch.arange(1, grid_size + 1, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        """
        Args:
            x: (..., features)
        Returns:
            (..., features, 2*grid_size)  [cos terms, sin terms]
        """
        # Expand for broadcasting
        x_exp = x.unsqueeze(-1)  # (..., features, 1)
        freqs = self.freqs.unsqueeze(0)  # (1, grid_size)

        # Compute angular frequencies
        angle = x_exp * freqs  # (..., features, grid_size)

        # Compute cos and sin
        cos_basis = torch.cos(angle)
        sin_basis = torch.sin(angle)

        # Concatenate [cos, sin]
        basis = torch.cat([cos_basis, sin_basis], dim=-1)  # (..., features, 2*grid_size)

        return basis

    def basis_dim(self):
        return 2 * self.grid_size


# ============================================================================
# KAN Layer
# ============================================================================

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer with flexible approximators

    Args:
        in_features: number of input features
        out_features: number of output features
        approx: approximator instance (GaussianRBF, ChebyshevPoly, or FourierBasis)
        use_base_weight: whether to include linear base transformation

    Example:
        layer = KANLayer(8, 1, approx=GaussianRBF(grid_size=10))
        y = layer(x)  # x: (batch, 8) -> y: (batch, 1)
    """
    def __init__(self, in_features, out_features, approx, use_base_weight=True):
        super().__init__()

        if not isinstance(approx, Approximator):
            raise TypeError("approx must be an instance of Approximator")

        self.in_features = in_features
        self.out_features = out_features
        self.approx = approx
        self.use_base_weight = use_base_weight

        # Get basis dimension from approximator
        basis_dim = approx.basis_dim()

        # Learnable coefficients for basis functions
        # Shape: (out_features, in_features, basis_dim)
        self.basis_weight = nn.Parameter(
            torch.randn(out_features, in_features, basis_dim) * 0.1
        )

        # Optional base linear transformation (residual)
        if use_base_weight:
            self.base_weight = nn.Parameter(
                torch.randn(out_features, in_features) * 0.1
            )
        else:
            self.register_parameter('base_weight', None)

    def forward(self, x):
        """
        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)
        """
        # Compute basis functions using approximator
        basis = self.approx(x)  # (..., in_features, basis_dim)

        # Apply basis weights
        # basis_weight: (out_features, in_features, basis_dim)
        # basis: (..., in_features, basis_dim)
        # Output: (..., out_features)
        output = torch.einsum('...ik,oik->...o', basis, self.basis_weight)

        # Add base linear transformation if enabled
        if self.use_base_weight:
            base_output = F.linear(x, self.base_weight)
            output = output + base_output

        return output


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test with different approximators
    batch_size = 32
    in_features = 8
    out_features = 1

    x = torch.randn(batch_size, in_features)

    # Test Gaussian RBF
    print("Testing Gaussian RBF...")
    rbf_layer = KANLayer(in_features, out_features, approx=GaussianRBF(grid_size=10))
    y_rbf = rbf_layer(x)
    print(f"  Input shape: {x.shape}, Output shape: {y_rbf.shape}")
    print(f"  Parameters: {sum(p.numel() for p in rbf_layer.parameters())}")

    # Test Chebyshev
    print("\nTesting Chebyshev Polynomials...")
    cheb_layer = KANLayer(in_features, out_features, approx=ChebyshevPoly(degree=4))
    y_cheb = cheb_layer(x)
    print(f"  Input shape: {x.shape}, Output shape: {y_cheb.shape}")
    print(f"  Parameters: {sum(p.numel() for p in cheb_layer.parameters())}")

    # Test Fourier
    print("\nTesting Fourier Basis...")
    fourier_layer = KANLayer(in_features, out_features, approx=FourierBasis(grid_size=5))
    y_fourier = fourier_layer(x)
    print(f"  Input shape: {x.shape}, Output shape: {y_fourier.shape}")
    print(f"  Parameters: {sum(p.numel() for p in fourier_layer.parameters())}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    loss = y_rbf.sum()
    loss.backward()
    print(f"  Gradient computed successfully!")