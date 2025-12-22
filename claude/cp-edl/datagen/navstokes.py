#!/usr/bin/env python3
"""
Navier-Stokes Data Generation for Experiment 4: OOD Detection

Generates 2D incompressible Navier-Stokes data at multiple Reynolds numbers
using PhiFlow for physics simulation.

Usage:
    # Generate training data (ID): Re=1000, 2000, 3000
    python navstokes.py --mode train --reynolds 1000 2000 3000 --samples 500

    # Generate OOD test data: Re=5000, 10000
    python navstokes.py --mode ood --reynolds 5000 10000 --samples 200

    # Generate all data for Experiment 4
    python navstokes.py --mode exp4 --samples 500

Dependencies:
    pip install phiflow torch
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
import json

try:
    from phi.torch.flow import *
    PHIFLOW_AVAILABLE = True
except ImportError:
    PHIFLOW_AVAILABLE = False
    print("WARNING: PhiFlow not installed. Install with: pip install phiflow")


class NavierStokesGenerator:
    """
    Generator for 2D incompressible Navier-Stokes data.

    Creates flow fields at different Reynolds numbers for OOD detection experiments.
    """

    def __init__(
        self,
        resolution: int = 64,
        domain_size: float = 1.0,
        dt: float = 0.01,
        num_steps: int = 100,
        seed: int = 42
    ):
        """
        Args:
            resolution: Spatial resolution (64x64, 128x128, etc.)
            domain_size: Physical domain size
            dt: Time step size
            num_steps: Number of simulation steps
            seed: Random seed for reproducibility
        """
        self.resolution = resolution
        self.domain_size = domain_size
        self.dt = dt
        self.num_steps = num_steps
        self.seed = seed

        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_initial_condition(
        self,
        ic_type: str = 'vortex',
        magnitude: float = 1.0
    ):
        """
        Create initial velocity field.

        Args:
            ic_type: Type of initial condition ('vortex', 'shear', 'random')
            magnitude: Velocity magnitude scale

        Returns:
            CenteredGrid with initial velocity field
        """
        if ic_type == 'vortex':
            # Taylor-Green vortex using lambda function
            def vortex_field(pos):
                px, py = math.unstack(pos, 'vector')
                vx = magnitude * math.sin(2 * np.pi * px) * math.cos(2 * np.pi * py)
                vy = -magnitude * math.cos(2 * np.pi * px) * math.sin(2 * np.pi * py)
                return math.stack([vx, vy], channel('vector'))

            velocity = CenteredGrid(
                vortex_field,
                extrapolation=extrapolation.PERIODIC,
                x=self.resolution,
                y=self.resolution,
                bounds=Box(x=self.domain_size, y=self.domain_size)
            )

        elif ic_type == 'shear':
            # Kelvin-Helmholtz shear layer
            def shear_field(pos):
                px, py = math.unstack(pos, 'vector')
                vx = magnitude * math.tanh(30 * (py - 0.25)) - magnitude * math.tanh(30 * (py - 0.75))
                vy = magnitude * 0.05 * math.sin(2 * np.pi * px)
                return math.stack([vx, vy], channel('vector'))

            velocity = CenteredGrid(
                shear_field,
                extrapolation=extrapolation.PERIODIC,
                x=self.resolution,
                y=self.resolution,
                bounds=Box(x=self.domain_size, y=self.domain_size)
            )

        elif ic_type == 'random':
            # Random Fourier modes
            def random_field(pos):
                px, py = math.unstack(pos, 'vector')
                num_modes = 4
                vx = px * 0
                vy = py * 0

                for kx in range(1, num_modes):
                    for ky in range(1, num_modes):
                        amp_x = np.random.randn() * magnitude / (kx + ky)
                        amp_y = np.random.randn() * magnitude / (kx + ky)
                        phase_x = np.random.rand() * 2 * np.pi
                        phase_y = np.random.rand() * 2 * np.pi

                        vx = vx + amp_x * math.sin(2 * np.pi * kx * px + phase_x) * math.cos(2 * np.pi * ky * py)
                        vy = vy + amp_y * math.cos(2 * np.pi * kx * px) * math.sin(2 * np.pi * ky * py + phase_y)

                return math.stack([vx, vy], channel('vector'))

            velocity = CenteredGrid(
                random_field,
                extrapolation=extrapolation.PERIODIC,
                x=self.resolution,
                y=self.resolution,
                bounds=Box(x=self.domain_size, y=self.domain_size)
            )

        else:
            raise ValueError(f"Unknown IC type: {ic_type}")

        # Convert to staggered grid for simulation
        velocity = field.stagger(velocity, math.minimum, extrapolation.PERIODIC)

        return velocity

    def simulate(
        self,
        reynolds_number: float,
        ic_type: str = 'vortex',
        magnitude: float = 1.0,
        save_every: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Run Navier-Stokes simulation.

        Args:
            reynolds_number: Reynolds number (Re = UL/ν)
            ic_type: Initial condition type
            magnitude: Velocity magnitude
            save_every: Save snapshot every N steps

        Returns:
            Dictionary with velocity fields and metadata
        """
        # Kinematic viscosity: ν = UL/Re
        viscosity = (magnitude * self.domain_size) / reynolds_number

        print(f"  Simulating Re={reynolds_number:.0f}, ν={viscosity:.6f}")

        # Initialize
        velocity = self.create_initial_condition(ic_type, magnitude)

        # Storage for snapshots
        snapshots = []
        times = []

        # Time integration
        for step in range(self.num_steps):
            # Advection (convective term)
            velocity = advect.semi_lagrangian(velocity, velocity, dt=self.dt)

            # Diffusion (viscous term)
            velocity = diffuse.explicit(velocity, viscosity, dt=self.dt)

            # Pressure projection (incompressibility) with relaxed tolerance
            solve_config = math.Solve('CG', rel_tol=1e-4, abs_tol=1e-4, max_iterations=2000)
            velocity, pressure = fluid.make_incompressible(velocity, solve=solve_config)

            # Save snapshot
            if step % save_every == 0:
                # Extract velocity components using reshaped_native
                vel_tensor = velocity.staggered_tensor()
                vel_data = math.reshaped_native(vel_tensor, ['vector', 'x', 'y'])  # Shape: (2, resolution, resolution)
                # Move to CPU if on CUDA
                if torch.is_tensor(vel_data) and vel_data.is_cuda:
                    vel_data = vel_data.cpu()
                snapshots.append(vel_data)
                times.append(step * self.dt)

        return {
            'velocity': np.array(snapshots),  # (num_snapshots, 2, H, W)
            'times': np.array(times),
            'reynolds': reynolds_number,
            'viscosity': viscosity,
            'resolution': self.resolution,
            'dt': self.dt
        }

    def generate_dataset(
        self,
        reynolds_numbers: List[float],
        num_samples: int,
        ic_types: List[str] = ['vortex', 'shear', 'random'],
        output_path: str = 'nsforcing_data.pt'
    ):
        """
        Generate full dataset for multiple Reynolds numbers.

        Args:
            reynolds_numbers: List of Reynolds numbers
            num_samples: Number of samples per Reynolds number
            ic_types: List of initial condition types to sample from
            output_path: Path to save dataset
        """
        print(f"\n{'='*80}")
        print(f"Generating Navier-Stokes Dataset")
        print(f"{'='*80}")
        print(f"Reynolds numbers: {reynolds_numbers}")
        print(f"Samples per Re: {num_samples}")
        print(f"Resolution: {self.resolution}x{self.resolution}")
        print(f"IC types: {ic_types}")
        print(f"{'='*80}\n")

        all_data = []

        for re in reynolds_numbers:
            print(f"\nGenerating {num_samples} samples for Re={re:.0f}...")

            for i in range(num_samples):
                # Randomly select IC type
                ic_type = np.random.choice(ic_types)

                # Random magnitude variation
                magnitude = np.random.uniform(0.8, 1.2)

                # Simulate
                data = self.simulate(
                    reynolds_number=re,
                    ic_type=ic_type,
                    magnitude=magnitude,
                    save_every=10
                )

                all_data.append(data)

                if (i + 1) % 50 == 0:
                    print(f"  Generated {i+1}/{num_samples} samples")

        # Save to PyTorch format
        print(f"\nSaving to {output_path}...")
        torch.save({
            'data': all_data,
            'metadata': {
                'reynolds_numbers': reynolds_numbers,
                'num_samples_per_re': num_samples,
                'resolution': self.resolution,
                'dt': self.dt,
                'num_steps': self.num_steps,
                'ic_types': ic_types,
                'generated_at': datetime.now().isoformat()
            }
        }, output_path)

        print(f"\n✓ Dataset saved! Total samples: {len(all_data)}")
        print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

        return all_data


def generate_experiment4_data(
    output_dir: str = '../data/exp4',
    train_samples: int = 500,
    ood_samples: int = 200,
    resolution: int = 64
):
    """
    Generate complete dataset for Experiment 4: OOD Detection.

    Creates:
    - Training data (ID): Re=1000, 2000, 3000
    - OOD test data: Re=5000, 10000

    Args:
        output_dir: Directory to save datasets
        train_samples: Samples per Reynolds number for training
        ood_samples: Samples per Reynolds number for OOD testing
        resolution: Spatial resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = NavierStokesGenerator(
        resolution=resolution,
        domain_size=1.0,
        dt=0.005,  # Reduced for stability
        num_steps=200,  # Increased to maintain same total time
        seed=42
    )

    # Generate training data (In-Distribution)
    print("\n" + "="*80)
    print("PART 1: Training Data (In-Distribution)")
    print("="*80)

    train_re = [1000.0, 2000.0, 3000.0]
    generator.generate_dataset(
        reynolds_numbers=train_re,
        num_samples=train_samples,
        output_path=str(output_dir / 'ns_train_id.pt')
    )

    # Generate OOD test data
    print("\n" + "="*80)
    print("PART 2: OOD Test Data (Out-of-Distribution)")
    print("="*80)

    ood_re = [5000.0, 10000.0]
    generator.generate_dataset(
        reynolds_numbers=ood_re,
        num_samples=ood_samples,
        output_path=str(output_dir / 'ns_test_ood.pt')
    )

    # Save experiment metadata
    metadata = {
        'experiment': 'Experiment 4: OOD Detection',
        'id_reynolds': train_re,
        'ood_reynolds': ood_re,
        'train_samples_per_re': train_samples,
        'ood_samples_per_re': ood_samples,
        'resolution': resolution,
        'generated_at': datetime.now().isoformat()
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("EXPERIMENT 4 DATASET COMPLETE!")
    print("="*80)
    print(f"Training data: {output_dir / 'ns_train_id.pt'}")
    print(f"OOD data: {output_dir / 'ns_test_ood.pt'}")
    print(f"Metadata: {output_dir / 'metadata.json'}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Navier-Stokes data for OOD detection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate complete Experiment 4 dataset
  python navstokes.py --mode exp4 --train-samples 500 --ood-samples 200

  # Generate custom training data
  python navstokes.py --mode train --reynolds 1000 2000 3000 --samples 300

  # Generate custom OOD data
  python navstokes.py --mode ood --reynolds 5000 10000 --samples 150

  # High resolution dataset
  python navstokes.py --mode exp4 --resolution 128 --train-samples 200
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'ood', 'exp4'],
                        help='Generation mode: train (ID data), ood (OOD data), or exp4 (both)')

    parser.add_argument('--reynolds', nargs='+', type=float,
                        help='Reynolds numbers to generate (for train/ood modes)')

    parser.add_argument('--samples', type=int, default=500,
                        help='Number of samples per Reynolds number')

    parser.add_argument('--train-samples', type=int, default=500,
                        help='Training samples per Re (exp4 mode)')

    parser.add_argument('--ood-samples', type=int, default=200,
                        help='OOD samples per Re (exp4 mode)')

    parser.add_argument('--resolution', type=int, default=64,
                        help='Spatial resolution (64, 128, 256)')

    parser.add_argument('--output-dir', type=str, default='../data/exp4',
                        help='Output directory')

    parser.add_argument('--output-file', type=str, default=None,
                        help='Output filename (for train/ood modes)')

    args = parser.parse_args()

    # Check PhiFlow availability
    if not PHIFLOW_AVAILABLE:
        print("ERROR: PhiFlow is required but not installed.")
        print("Install with: pip install phiflow")
        return

    if args.mode == 'exp4':
        # Generate complete Experiment 4 dataset
        generate_experiment4_data(
            output_dir=args.output_dir,
            train_samples=args.train_samples,
            ood_samples=args.ood_samples,
            resolution=args.resolution
        )

    elif args.mode in ['train', 'ood']:
        if not args.reynolds:
            print(f"ERROR: --reynolds required for {args.mode} mode")
            return

        # Determine output filename
        if args.output_file is None:
            suffix = 'id' if args.mode == 'train' else 'ood'
            args.output_file = f'ns_{args.mode}_{suffix}.pt'

        output_path = Path(args.output_dir) / args.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate dataset
        generator = NavierStokesGenerator(
            resolution=args.resolution,
            domain_size=1.0,
            dt=0.005,  # Reduced for stability
            num_steps=200,  # Increased to maintain same total time
            seed=42
        )

        generator.generate_dataset(
            reynolds_numbers=args.reynolds,
            num_samples=args.samples,
            output_path=str(output_path)
        )


if __name__ == '__main__':
    main()