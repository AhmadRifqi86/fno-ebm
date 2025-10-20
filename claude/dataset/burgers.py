"""
Burgers Equation Data Generator

Generates synthetic data for viscous Burgers equation:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²  in [0, L] × [0, T]
    u(x, 0) = u₀(x)                (initial condition)
    u(0, t) = u(L, t) = 0          (periodic or zero BC)

where:
- u₀(x): initial condition (input)
- u(x,t): solution trajectory (output)
- ν: viscosity coefficient

This is a standard benchmark for time-evolution operator learning.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from typing import Tuple, Optional
from .noise_models import add_gaussian_noise, add_heteroscedastic_noise, add_mixed_noise


class BurgersGenerator:
    """
    Generate Burgers equation training data.

    Example:
        >>> gen = BurgersGenerator(nx=256, nt=100, T=1.0, viscosity=0.01)
        >>> X, U = gen.generate_dataset(n_samples=1000, noise_type='heteroscedastic')
        >>> print(X.shape)  # (1000, 256, 2) - [x_coords, initial_condition]
        >>> print(U.shape)  # (1000, 100, 256, 1) - [time_trajectory]
    """

    def __init__(
        self,
        nx: int = 256,
        nt: int = 100,
        L: float = 2.0 * np.pi,
        T: float = 1.0,
        viscosity: float = 0.01,
        complexity: str = 'medium',
        seed: int = None
    ):
        """
        Initialize Burgers equation generator.

        Args:
            nx: Spatial resolution
            nt: Temporal resolution (output timesteps)
            L: Domain length [0, L]
            T: Final time
            viscosity: Viscosity coefficient ν
            complexity: 'simple', 'medium', 'hard' - controls initial condition complexity
            seed: Random seed for reproducibility
        """
        self.nx = nx
        self.nt = nt
        self.L = L
        self.T = T
        self.viscosity = viscosity
        self.complexity = complexity
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Spatial grid
        self.x = np.linspace(0, L, nx, endpoint=False)
        self.dx = self.x[1] - self.x[0]

        # Time grid
        self.t = np.linspace(0, T, nt)
        self.dt = self.t[1] - self.t[0]

        # Wavenumbers for spectral method
        self.k = fftfreq(nx, d=self.dx) * 2 * np.pi

        # Complexity parameters
        self.n_modes, self.amplitude = self._get_complexity_params()

    def _get_complexity_params(self) -> Tuple[int, float]:
        """Get initial condition complexity parameters."""
        if self.complexity == 'simple':
            return 3, 1.0  # Few modes, moderate amplitude
        elif self.complexity == 'medium':
            return 5, 2.0  # Medium modes
        elif self.complexity == 'hard':
            return 8, 3.0  # Many modes, large amplitude
        else:
            raise ValueError(f"Unknown complexity: {self.complexity}")

    def generate_initial_condition(self, seed_offset: int = 0) -> np.ndarray:
        """
        Generate random initial condition u₀(x).

        Uses Fourier series with random coefficients:
        u₀(x) = Σ [aₙ·sin(nx) + bₙ·cos(nx)]

        Args:
            seed_offset: Offset for seed

        Returns:
            u0: Initial condition, shape (nx,)
        """
        if self.seed is not None:
            np.random.seed(self.seed + seed_offset)

        u0 = np.zeros(self.nx)

        # Random Fourier modes
        for n in range(1, self.n_modes + 1):
            a_n = np.random.randn() * self.amplitude / n
            b_n = np.random.randn() * self.amplitude / n

            u0 += a_n * np.sin(2 * np.pi * n * self.x / self.L)
            u0 += b_n * np.cos(2 * np.pi * n * self.x / self.L)

        return u0

    def solve_burgers(self, u0: np.ndarray) -> np.ndarray:
        """
        Solve Burgers equation using pseudo-spectral method.

        Time integration: 4th-order Runge-Kutta
        Spatial derivatives: Spectral differentiation (FFT)

        Args:
            u0: Initial condition, shape (nx,)

        Returns:
            u_trajectory: Solution at all timesteps, shape (nt, nx)
        """
        u_trajectory = np.zeros((self.nt, self.nx))
        u_trajectory[0] = u0

        u = u0.copy()

        for i in range(1, self.nt):
            # RK4 time stepping
            k1 = self._burgers_rhs(u)
            k2 = self._burgers_rhs(u + 0.5 * self.dt * k1)
            k3 = self._burgers_rhs(u + 0.5 * self.dt * k2)
            k4 = self._burgers_rhs(u + self.dt * k3)

            u = u + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            u_trajectory[i] = u

        return u_trajectory

    def _burgers_rhs(self, u: np.ndarray) -> np.ndarray:
        """
        Compute RHS of Burgers equation: -u·∂u/∂x + ν·∂²u/∂x²

        Uses spectral derivatives in Fourier space.
        """
        # FFT
        u_hat = fft(u)

        # Spectral derivatives
        # ∂u/∂x = ifft(ik·û)
        # ∂²u/∂x² = ifft(-k²·û)
        ux_hat = 1j * self.k * u_hat
        uxx_hat = -(self.k**2) * u_hat

        # Transform back to physical space
        ux = np.real(ifft(ux_hat))
        uxx = np.real(ifft(uxx_hat))

        # RHS: -u·∂u/∂x + ν·∂²u/∂x²
        rhs = -u * ux + self.viscosity * uxx

        return rhs

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
            x: Input tensor (nx, 2) = [x_coords, initial_condition]
            u: Output tensor (nt, nx, 1) = [trajectory]
            metadata: Dictionary with additional info
        """
        # Generate initial condition
        u0 = self.generate_initial_condition(seed_offset)

        # Solve PDE
        u_clean = self.solve_burgers(u0)

        # Add noise to trajectory
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
        # Input: (nx, 2) = [x_coords, u0]
        x = np.stack([self.x, u0], axis=-1)

        # Output: (nt, nx, 1) = trajectory
        u = u.reshape(self.nt, self.nx, 1)

        # Metadata
        metadata = {
            'u0_mean': np.mean(u0),
            'u0_std': np.std(u0),
            'solution_mean': np.mean(u),
            'solution_std': np.std(u),
            'noise_std': np.std(noise_added) if noise_type else 0.0,
            'viscosity': self.viscosity
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
            X: Input tensor (n_samples, nx, 2)
            U: Output tensor (n_samples, nt, nx, 1)
        """
        X = np.zeros((n_samples, self.nx, 2))
        U = np.zeros((n_samples, self.nt, self.nx, 1))

        if verbose:
            print(f"Generating {n_samples} Burgers equation samples...")
            print(f"  Spatial resolution: {self.nx}")
            print(f"  Temporal resolution: {self.nt}")
            print(f"  Viscosity: {self.viscosity}")
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
            nx=self.nx,
            nt=self.nt,
            L=self.L,
            T=self.T,
            viscosity=self.viscosity,
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
            'nx': int(data['nx']),
            'nt': int(data['nt']),
            'L': float(data['L']),
            'T': float(data['T']),
            'viscosity': float(data['viscosity']),
            'complexity': str(data['complexity'])
        }
        return X, U, metadata