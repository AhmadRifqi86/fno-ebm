"""
Poisson Equation Data Generator

Generates synthetic data for Poisson equation:
    -∇²u(x) = f(x)  in Ω=[0,1]²
    u(x) = 0         on ∂Ω

where:
- f(x): source term (input)
- u(x): potential/solution (output)

This is a simpler benchmark than Darcy flow (constant coefficient).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional
from .noise_models import add_gaussian_noise, add_heteroscedastic_noise, add_mixed_noise


class PoissonGenerator:
    """
    Generate Poisson equation training data.

    Example:
        >>> gen = PoissonGenerator(resolution=64, complexity='medium')
        >>> X, U = gen.generate_dataset(n_samples=1000, noise_type='gaussian')
        >>> print(X.shape)  # (1000, 64, 64, 3)
        >>> print(U.shape)  # (1000, 64, 64, 1)
    """

    def __init__(
        self,
        resolution: int = 64,
        domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        complexity: str = 'medium',
        seed: int = None
    ):
        """
        Initialize Poisson equation generator.

        Args:
            resolution: Grid resolution (nx = ny = resolution)
            domain: (xmin, xmax, ymin, ymax)
            complexity: 'simple', 'medium', 'hard' - controls source complexity
            seed: Random seed for reproducibility
        """
        self.resolution = resolution
        self.domain = domain
        self.complexity = complexity
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Create spatial grid
        self.x = np.linspace(domain[0], domain[1], resolution)
        self.y = np.linspace(domain[2], domain[3], resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Complexity parameters
        self.length_scale, self.variance = self._get_complexity_params()

    def _get_complexity_params(self) -> Tuple[float, float]:
        """Get GP parameters for source term based on complexity level."""
        if self.complexity == 'simple':
            return 0.3, 1.0  # Large length scale, smooth sources
        elif self.complexity == 'medium':
            return 0.15, 2.0  # Medium length scale
        elif self.complexity == 'hard':
            return 0.08, 3.0  # Small length scale (rough sources)
        else:
            raise ValueError(f"Unknown complexity: {self.complexity}")

    def generate_source_term(self, seed_offset: int = 0) -> np.ndarray:
        """
        Generate random source term f(x,y) using Gaussian Process.

        Args:
            seed_offset: Offset for seed

        Returns:
            f: Source term, shape (resolution, resolution)
        """
        if self.seed is not None:
            np.random.seed(self.seed + seed_offset)

        # Generate Gaussian random field using FFT method
        nx, ny = self.resolution, self.resolution

        # Frequency grid
        kx = np.fft.fftfreq(nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2

        # Gaussian covariance in frequency domain
        power_spectrum = self.variance * np.exp(-self.length_scale**2 * K2 / 2)

        # Generate white noise in frequency domain
        noise_real = np.random.randn(nx, ny)
        noise_imag = np.random.randn(nx, ny)
        noise_fft = noise_real + 1j * noise_imag

        # Apply power spectrum
        field_fft = np.sqrt(power_spectrum) * noise_fft

        # Transform to spatial domain
        field = np.fft.ifft2(field_fft).real

        # Center around zero with controlled magnitude
        f = field - np.mean(field)
        f = f / (np.std(f) + 1e-8) * self.variance

        return f

    def solve_poisson(self, f: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation using finite differences.

        -∇²u = f with Dirichlet BC: u=0 on boundary

        Args:
            f: Source term, shape (nx, ny)

        Returns:
            u: Solution field, shape (nx, ny)
        """
        nx, ny = self.resolution, self.resolution

        # Interior points (excluding boundary)
        n_interior = (nx - 2) * (ny - 2)

        # Build sparse linear system Au = b
        # Using 5-point stencil for -∇²u

        # Map 2D interior indices to 1D
        def idx(i, j):
            """Map (i,j) interior point to linear index."""
            return (i - 1) * (ny - 2) + (j - 1)

        # Build matrix A and RHS b
        A = sparse.lil_matrix((n_interior, n_interior))
        b = np.zeros(n_interior)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                k = idx(i, j)

                # 5-point stencil for -∇²u
                # -∇²u ≈ -(u[i+1,j] + u[i-1,j] - 2u[i,j])/dx²
                #        -(u[i,j+1] + u[i,j-1] - 2u[i,j])/dy²

                # Diagonal
                A[k, k] = 2.0 / self.dx**2 + 2.0 / self.dy**2

                # Neighbors
                if i < nx - 2:  # i+1
                    A[k, idx(i+1, j)] = -1.0 / self.dx**2
                if i > 1:  # i-1
                    A[k, idx(i-1, j)] = -1.0 / self.dx**2
                if j < ny - 2:  # j+1
                    A[k, idx(i, j+1)] = -1.0 / self.dy**2
                if j > 1:  # j-1
                    A[k, idx(i, j-1)] = -1.0 / self.dy**2

                # RHS
                b[k] = f[i, j]

        # Solve sparse system
        A = A.tocsr()
        u_interior = spsolve(A, b)

        # Embed in full grid with BC
        u = np.zeros((nx, ny))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = u_interior[idx(i, j)]

        # Boundary conditions (already zero)

        return u

    def generate_sample(
        self,
        seed_offset: int = 0,
        noise_type: str = None,
        noise_params: dict = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate a single (input, output) pair.

        Args:
            seed_offset: Seed offset for reproducibility
            noise_type: 'gaussian', 'heteroscedastic', 'mixed', or None
            noise_params: Parameters for noise model

        Returns:
            x: Input tensor (nx, ny, 3) = [x_coords, y_coords, source_term]
            u: Output tensor (nx, ny, 1) = [solution]
            metadata: Dictionary with additional info
        """
        # Generate source term
        f = self.generate_source_term(seed_offset)

        # Solve PDE
        u_clean = self.solve_poisson(f)

        # Add noise to solution
        if noise_type is None:
            u = u_clean
            noise_added = np.zeros_like(u_clean)
        else:
            if noise_params is None:
                noise_params = {}

            if noise_type == 'gaussian':
                u, noise_added = add_gaussian_noise(
                    u_clean,
                    noise_level=noise_params.get('noise_level', 0.01),
                    seed=self.seed + seed_offset if self.seed else None
                )
            elif noise_type == 'heteroscedastic':
                u, noise_added = add_heteroscedastic_noise(
                    u_clean,
                    base_noise=noise_params.get('base_noise', 0.001),
                    scale_factor=noise_params.get('scale_factor', 0.02),
                    seed=self.seed + seed_offset if self.seed else None
                )
            elif noise_type == 'mixed':
                u, noise_added = add_mixed_noise(
                    u_clean,
                    seed=self.seed + seed_offset if self.seed else None,
                    **noise_params
                )
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

        # Package as FNO input format
        x = np.stack([self.X, self.Y, f], axis=-1)  # (nx, ny, 3)
        u = u.reshape(self.resolution, self.resolution, 1)  # (nx, ny, 1)

        # Metadata
        metadata = {
            'source_mean': np.mean(f),
            'source_std': np.std(f),
            'solution_mean': np.mean(u),
            'solution_std': np.std(u),
            'noise_std': np.std(noise_added) if noise_type else 0.0
        }

        return x, u, metadata

    def generate_dataset(
        self,
        n_samples: int,
        noise_type: str = 'heteroscedastic',
        noise_params: dict = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate full dataset.

        Args:
            n_samples: Number of samples to generate
            noise_type: Type of noise to add
            noise_params: Noise parameters
            verbose: Print progress

        Returns:
            X: Input tensor (n_samples, nx, ny, 3)
            U: Output tensor (n_samples, nx, ny, 1)
        """
        X = np.zeros((n_samples, self.resolution, self.resolution, 3))
        U = np.zeros((n_samples, self.resolution, self.resolution, 1))

        if verbose:
            print(f"Generating {n_samples} Poisson equation samples...")
            print(f"  Resolution: {self.resolution}x{self.resolution}")
            print(f"  Complexity: {self.complexity}")
            print(f"  Noise type: {noise_type}")

        for i in range(n_samples):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{n_samples} samples", end='\r')

            x, u, metadata = self.generate_sample(
                seed_offset=i,
                noise_type=noise_type,
                noise_params=noise_params
            )

            X[i] = x
            U[i] = u

        if verbose:
            print(f"  Generated {n_samples}/{n_samples} samples ✓")
            print(f"  X shape: {X.shape}")
            print(f"  U shape: {U.shape}")

        return X, U

    def save_dataset(
        self,
        X: np.ndarray,
        U: np.ndarray,
        filepath: str
    ):
        """Save dataset to file."""
        np.savez_compressed(
            filepath,
            X=X,
            U=U,
            resolution=self.resolution,
            domain=self.domain,
            complexity=self.complexity
        )
        print(f"Dataset saved to: {filepath}")

    @staticmethod
    def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Load dataset from file."""
        data = np.load(filepath)
        X = data['X']
        U = data['U']
        metadata = {
            'resolution': int(data['resolution']),
            'domain': tuple(data['domain']),
            'complexity': str(data['complexity'])
        }
        return X, U, metadata