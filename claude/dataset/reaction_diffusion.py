"""
Reaction-Diffusion Data Generator

Generates synthetic data for Gray-Scott reaction-diffusion system:
    ∂u/∂t = D_u ∇²u - uv² + F(1-u)
    ∂v/∂t = D_v ∇²v + uv² - (F+k)v

where:
- u, v: concentrations of two chemical species
- D_u, D_v: diffusion coefficients
- F: feed rate
- k: kill rate

Different (F, k) values produce various patterns:
- Spots, stripes, spiral waves, chaos, etc.

This provides much more complex spatial patterns than Darcy flow.
"""

import numpy as np
from typing import Tuple, Optional
from .noise_models import add_gaussian_noise, add_heteroscedastic_noise, add_mixed_noise


class ReactionDiffusionGenerator:
    """
    Generate Reaction-Diffusion training data using Gray-Scott model.

    Example:
        >>> gen = ReactionDiffusionGenerator(resolution=64, complexity='medium')
        >>> X, U = gen.generate_dataset(n_samples=1000, noise_type='heteroscedastic')
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
        Initialize Reaction-Diffusion generator.

        Args:
            resolution: Grid resolution (nx = ny = resolution)
            domain: (xmin, xmax, ymin, ymax)
            complexity: 'simple', 'medium', 'hard' - controls pattern complexity
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

        # Fixed diffusion coefficients (standard Gray-Scott values)
        self.D_u = 2e-5
        self.D_v = 1e-5

        # Complexity determines (F, k) parameters and pattern type
        self.F, self.k, self.pattern_type = self._get_complexity_params()

    def _get_complexity_params(self) -> Tuple[float, float, str]:
        """
        Get Gray-Scott parameters based on complexity level.

        Returns:
            F: Feed rate
            k: Kill rate
            pattern_type: Description of expected pattern
        """
        if self.complexity == 'simple':
            # Stable spots/bubbles - simple circular patterns
            return 0.055, 0.062, 'spots'
        elif self.complexity == 'medium':
            # Stripes and labyrinth patterns - more complex
            return 0.035, 0.060, 'stripes'
        elif self.complexity == 'hard':
            # Complex spirals and chaotic patterns
            return 0.018, 0.051, 'spirals'
        else:
            raise ValueError(f"Unknown complexity: {self.complexity}")

    def generate_initial_condition(
        self,
        seed_offset: int = 0,
        perturbation_type: str = 'random'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial conditions for Gray-Scott system.

        Args:
            seed_offset: Offset for seed (to generate different ICs)
            perturbation_type: 'random', 'spots', 'squares'

        Returns:
            u0: Initial concentration of species u (nx, ny)
            v0: Initial concentration of species v (nx, ny)
        """
        if self.seed is not None:
            np.random.seed(self.seed + seed_offset)

        nx, ny = self.resolution, self.resolution

        # Start with uniform state: u=1, v=0
        u0 = np.ones((nx, ny))
        v0 = np.zeros((nx, ny))

        if perturbation_type == 'random':
            # Random perturbations in center region
            center_x, center_y = nx // 2, ny // 2
            size = max(nx // 8, 4)

            # Add multiple random spots
            n_spots = np.random.randint(3, 8)
            for _ in range(n_spots):
                x_offset = np.random.randint(-size, size)
                y_offset = np.random.randint(-size, size)
                spot_size = np.random.randint(2, size // 2)

                x_center = center_x + x_offset
                y_center = center_y + y_offset

                for i in range(max(0, x_center - spot_size), min(nx, x_center + spot_size)):
                    for j in range(max(0, y_center - spot_size), min(ny, y_center + spot_size)):
                        r = np.sqrt((i - x_center)**2 + (j - y_center)**2)
                        if r < spot_size:
                            u0[i, j] = 0.5
                            v0[i, j] = 0.25

        elif perturbation_type == 'spots':
            # Single or few circular spots
            center_x, center_y = nx // 2, ny // 2
            spot_radius = max(nx // 10, 3)

            for i in range(nx):
                for j in range(ny):
                    r = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if r < spot_radius:
                        u0[i, j] = 0.5
                        v0[i, j] = 0.25

        elif perturbation_type == 'squares':
            # Square perturbation (creates different dynamics)
            center_x, center_y = nx // 2, ny // 2
            size = max(nx // 8, 4)

            u0[center_x - size:center_x + size, center_y - size:center_y + size] = 0.5
            v0[center_x - size:center_x + size, center_y - size:center_y + size] = 0.25

        # Add small random noise to break symmetry
        u0 += np.random.randn(nx, ny) * 0.01
        v0 += np.random.randn(nx, ny) * 0.01

        # Ensure non-negative
        u0 = np.clip(u0, 0, 1)
        v0 = np.clip(v0, 0, 1)

        return u0, v0

    def laplacian_periodic(self, u: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian with periodic boundary conditions.

        Args:
            u: Field to compute Laplacian of (nx, ny)

        Returns:
            lap_u: Laplacian (nx, ny)
        """
        # 5-point stencil: ∇²u ≈ (u_left + u_right + u_up + u_down - 4*u_center) / dx²
        lap = (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
        ) / self.dx**2

        return lap

    def step_gray_scott(
        self,
        u: np.ndarray,
        v: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single timestep of Gray-Scott model using forward Euler.

        Args:
            u: Current u field (nx, ny)
            v: Current v field (nx, ny)
            dt: Time step size

        Returns:
            u_new: Updated u field
            v_new: Updated v field
        """
        # Compute Laplacians
        lap_u = self.laplacian_periodic(u)
        lap_v = self.laplacian_periodic(v)

        # Reaction terms
        uvv = u * v * v

        # Gray-Scott equations
        du_dt = self.D_u * lap_u - uvv + self.F * (1 - u)
        dv_dt = self.D_v * lap_v + uvv - (self.F + self.k) * v

        # Forward Euler update
        u_new = u + dt * du_dt
        v_new = v + dt * dv_dt

        # Ensure non-negative (clip to physically meaningful range)
        u_new = np.clip(u_new, 0, 1)
        v_new = np.clip(v_new, 0, 1)

        return u_new, v_new

    def solve_gray_scott(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        t_final: float = 10000.0,
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Gray-Scott system from initial condition to steady state.

        Args:
            u0: Initial u field (nx, ny)
            v0: Initial v field (nx, ny)
            t_final: Final time (longer = more evolved patterns)
            dt: Time step

        Returns:
            u_final: Final u field (nx, ny)
            v_final: Final v field (nx, ny)
        """
        u = u0.copy()
        v = v0.copy()

        n_steps = int(t_final / dt)

        for _ in range(n_steps):
            u, v = self.step_gray_scott(u, v, dt)

        return u, v

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
            x: Input tensor (nx, ny, 3) = [x_coords, y_coords, initial_v]
            u: Output tensor (nx, ny, 1) = [final_u pattern]
            metadata: Dictionary with additional info
        """
        # Generate initial conditions
        u0, v0 = self.generate_initial_condition(seed_offset, perturbation_type='random')

        # Solve to quasi-steady state
        # Adjust simulation time based on complexity
        if self.complexity == 'simple':
            t_final = 5000.0  # Simple patterns form quickly
        elif self.complexity == 'medium':
            t_final = 8000.0  # Medium patterns need more time
        else:  # hard
            t_final = 12000.0  # Complex spirals need long evolution

        u_clean, v_clean = self.solve_gray_scott(u0, v0, t_final=t_final, dt=1.0)

        # Use u field as the output (has richer patterns than v)
        output = u_clean

        # Add noise to output
        if noise_type is None:
            u_final = output
            noise_added = np.zeros_like(output)
        else:
            if noise_params is None:
                noise_params = {}

            if noise_type == 'gaussian':
                u_final, noise_added = add_gaussian_noise(
                    output,
                    noise_level=noise_params.get('noise_level', 0.01),
                    seed=self.seed + seed_offset if self.seed else None
                )
            elif noise_type == 'heteroscedastic':
                u_final, noise_added = add_heteroscedastic_noise(
                    output,
                    base_noise=noise_params.get('base_noise', 0.001),
                    scale_factor=noise_params.get('scale_factor', 0.02),
                    seed=self.seed + seed_offset if self.seed else None
                )
            elif noise_type == 'mixed':
                u_final, noise_added = add_mixed_noise(
                    output,
                    seed=self.seed + seed_offset if self.seed else None,
                    **noise_params
                )
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

        # Package as FNO input format
        # Input: coordinates + initial v field (which determines the pattern)
        x = np.stack([self.X, self.Y, v0], axis=-1)  # (nx, ny, 3)
        u = u_final.reshape(self.resolution, self.resolution, 1)  # (nx, ny, 1)

        # Metadata
        metadata = {
            'F': self.F,
            'k': self.k,
            'pattern_type': self.pattern_type,
            'output_mean': np.mean(u_final),
            'output_std': np.std(u_final),
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
            print(f"Generating {n_samples} Reaction-Diffusion samples...")
            print(f"  Resolution: {self.resolution}x{self.resolution}")
            print(f"  Complexity: {self.complexity} ({self.pattern_type} patterns)")
            print(f"  Parameters: F={self.F:.3f}, k={self.k:.3f}")
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
            complexity=self.complexity,
            F=self.F,
            k=self.k,
            pattern_type=self.pattern_type
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
            'complexity': str(data['complexity']),
            'F': float(data['F']),
            'k': float(data['k']),
            'pattern_type': str(data['pattern_type'])
        }
        return X, U, metadata