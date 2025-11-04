"""
Data Generation Script

This script demonstrates how to generate synthetic PDE datasets
for training FNO-EBM models.

Usage:
    python generate_data.py --pde darcy --complexity medium --noise heteroscedastic
    python generate_data.py --pde burgers --complexity hard --noise mixed
    python generate_data.py --pde poisson --complexity simple --noise gaussian

Or simply run with default parameters from config.yaml:
    python generate_data.py
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path to import from ../dataset/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import DarcyFlowGenerator, BurgersGenerator, PoissonGenerator, ReactionDiffusionGenerator


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def generate_darcy_data(config):
    """Generate Darcy flow dataset."""
    print("\n=== Generating Darcy Flow Dataset ===")

    resolution = config.get('grid_size', 64)
    complexity = config.get('complexity', 'medium')
    seed = config.get('data_seed', 42)
    n_train = config.get('n_train', 1000)
    n_test = config.get('n_test', 200)
    noise_type = config.get('noise_type', 'heteroscedastic')

    # Get noise parameters
    noise_params = {}
    if noise_type == 'gaussian':
        noise_params['noise_level'] = config.get('noise_level', 0.01)
    elif noise_type == 'heteroscedastic':
        noise_params['base_noise'] = config.get('base_noise', 0.001)
        noise_params['scale_factor'] = config.get('scale_factor', 0.02)
    elif noise_type == 'mixed':
        noise_params['gaussian_level'] = config.get('gaussian_level', 0.005)
        noise_params['hetero_scale'] = config.get('hetero_scale', 0.01)
        noise_params['spatial_level'] = config.get('spatial_level', 0.003)
        noise_params['correlation_length'] = config.get('mixed_correlation_length', 2.0)

    # Create generators
    gen_train = DarcyFlowGenerator(resolution=resolution, complexity=complexity, seed=seed)
    gen_test = DarcyFlowGenerator(resolution=resolution, complexity=complexity, seed=seed + 10000)

    # Generate datasets
    X_train, U_train = gen_train.generate_dataset(
        n_samples=n_train,
        noise_type=noise_type,
        noise_params=noise_params
    )

    X_test, U_test = gen_test.generate_dataset(
        n_samples=n_test,
        noise_type=noise_type,
        noise_params=noise_params
    )

    # Save datasets
    data_dir = Path(config.get('data_dir', './data'))
    data_dir.mkdir(exist_ok=True)

    train_file = data_dir / f"darcy_{complexity}_{noise_type}_res{resolution}_train.npz"
    test_file = data_dir / f"darcy_{complexity}_{noise_type}_res{resolution}_val.npz"

    gen_train.save_dataset(X_train, U_train, str(train_file))
    gen_test.save_dataset(X_test, U_test, str(test_file))

    print(f"\nDatasets saved:")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")


def generate_burgers_data(config):
    """Generate Burgers equation dataset."""
    print("\n=== Generating Burgers Equation Dataset ===")

    nx = config.get('burgers_nx', 256)
    nt = config.get('burgers_nt', 100)
    viscosity = config.get('burgers_viscosity', 0.01)
    complexity = config.get('complexity', 'medium')
    seed = config.get('data_seed', 42)
    n_train = config.get('n_train', 1000)
    n_test = config.get('n_test', 200)
    noise_type = config.get('noise_type', 'heteroscedastic')

    # Get noise parameters
    noise_params = {}
    if noise_type == 'gaussian':
        noise_params['noise_level'] = config.get('noise_level', 0.01)
    elif noise_type == 'heteroscedastic':
        noise_params['base_noise'] = config.get('base_noise', 0.001)
        noise_params['scale_factor'] = config.get('scale_factor', 0.02)
    elif noise_type == 'mixed':
        noise_params['gaussian_level'] = config.get('gaussian_level', 0.005)
        noise_params['hetero_scale'] = config.get('hetero_scale', 0.01)
        noise_params['spatial_level'] = config.get('spatial_level', 0.003)
        noise_params['correlation_length'] = config.get('mixed_correlation_length', 2.0)

    # Create generators
    gen_train = BurgersGenerator(nx=nx, nt=nt, viscosity=viscosity, complexity=complexity, seed=seed)
    gen_test = BurgersGenerator(nx=nx, nt=nt, viscosity=viscosity, complexity=complexity, seed=seed + 10000)

    # Generate datasets
    X_train, U_train = gen_train.generate_dataset(
        n_samples=n_train,
        noise_type=noise_type,
        noise_params=noise_params
    )

    X_test, U_test = gen_test.generate_dataset(
        n_samples=n_test,
        noise_type=noise_type,
        noise_params=noise_params
    )

    # Save datasets
    data_dir = Path(config.get('data_dir', './data'))
    data_dir.mkdir(exist_ok=True)

    train_file = data_dir / f"burgers_{complexity}_{noise_type}_nx{nx}_nt{nt}_train.npz"
    test_file = data_dir / f"burgers_{complexity}_{noise_type}_nx{nx}_nt{nt}_val.npz"

    gen_train.save_dataset(X_train, U_train, str(train_file))
    gen_test.save_dataset(X_test, U_test, str(test_file))

    print(f"\nDatasets saved:")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")


def generate_poisson_data(config):
    """Generate Poisson equation dataset."""
    print("\n=== Generating Poisson Equation Dataset ===")

    resolution = config.get('grid_size', 64)
    complexity = config.get('complexity', 'medium')
    seed = config.get('data_seed', 42)
    n_train = config.get('n_train', 1000)
    n_test = config.get('n_test', 200)
    noise_type = config.get('noise_type', 'heteroscedastic')

    # Get noise parameters
    noise_params = {}
    if noise_type == 'gaussian':
        noise_params['noise_level'] = config.get('noise_level', 0.01)
    elif noise_type == 'heteroscedastic':
        noise_params['base_noise'] = config.get('base_noise', 0.001)
        noise_params['scale_factor'] = config.get('scale_factor', 0.02)
    elif noise_type == 'mixed':
        noise_params['gaussian_level'] = config.get('gaussian_level', 0.005)
        noise_params['hetero_scale'] = config.get('hetero_scale', 0.01)
        noise_params['spatial_level'] = config.get('spatial_level', 0.003)
        noise_params['correlation_length'] = config.get('mixed_correlation_length', 2.0)

    # Create generators
    gen_train = PoissonGenerator(resolution=resolution, complexity=complexity, seed=seed)
    gen_test = PoissonGenerator(resolution=resolution, complexity=complexity, seed=seed + 10000)

    # Generate datasets
    X_train, U_train = gen_train.generate_dataset(
        n_samples=n_train,
        noise_type=noise_type,
        noise_params=noise_params
    )

    X_test, U_test = gen_test.generate_dataset(
        n_samples=n_test,
        noise_type=noise_type,
        noise_params=noise_params
    )

    # Save datasets
    data_dir = Path(config.get('data_dir', './data'))
    data_dir.mkdir(exist_ok=True)

    train_file = data_dir / f"poisson_{complexity}_{noise_type}_res{resolution}_train.npz"
    test_file = data_dir / f"poisson_{complexity}_{noise_type}_res{resolution}_val.npz"

    gen_train.save_dataset(X_train, U_train, str(train_file))
    gen_test.save_dataset(X_test, U_test, str(test_file))

    print(f"\nDatasets saved:")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")


def generate_reaction_diffusion_data(config):
    """Generate Reaction-Diffusion dataset."""
    print("\n=== Generating Reaction-Diffusion Dataset ===")

    resolution = config.get('grid_size', 64)
    complexity = config.get('complexity', 'medium')
    seed = config.get('data_seed', 42)
    n_train = config.get('n_train', 1000)
    n_test = config.get('n_test', 200)
    noise_type = config.get('noise_type', 'heteroscedastic')

    # Get noise parameters
    noise_params = {}
    if noise_type == 'gaussian':
        noise_params['noise_level'] = config.get('noise_level', 0.01)
    elif noise_type == 'heteroscedastic':
        noise_params['base_noise'] = config.get('base_noise', 0.001)
        noise_params['scale_factor'] = config.get('scale_factor', 0.02)
    elif noise_type == 'mixed':
        noise_params['gaussian_level'] = config.get('gaussian_level', 0.005)
        noise_params['hetero_scale'] = config.get('hetero_scale', 0.01)
        noise_params['spatial_level'] = config.get('spatial_level', 0.003)
        noise_params['correlation_length'] = config.get('mixed_correlation_length', 2.0)

    # Create generators
    gen_train = ReactionDiffusionGenerator(resolution=resolution, complexity=complexity, seed=seed)
    gen_test = ReactionDiffusionGenerator(resolution=resolution, complexity=complexity, seed=seed + 10000)

    # Generate datasets
    X_train, U_train = gen_train.generate_dataset(
        n_samples=n_train,
        noise_type=noise_type,
        noise_params=noise_params
    )

    X_test, U_test = gen_test.generate_dataset(
        n_samples=n_test,
        noise_type=noise_type,
        noise_params=noise_params
    )

    # Save datasets
    data_dir = Path(config.get('data_dir', './data'))
    data_dir.mkdir(exist_ok=True)

    train_file = data_dir / f"reaction_diffusion_{complexity}_{noise_type}_res{resolution}_train.npz"
    test_file = data_dir / f"reaction_diffusion_{complexity}_{noise_type}_res{resolution}_val.npz"

    gen_train.save_dataset(X_train, U_train, str(train_file))
    gen_test.save_dataset(X_test, U_test, str(test_file))

    print(f"\nDatasets saved:")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic PDE datasets')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--pde', type=str, choices=['darcy', 'burgers', 'poisson', 'reaction_diffusion', 'all'],
                        help='PDE type to generate (overrides config)')
    parser.add_argument('--complexity', type=str, choices=['simple', 'medium', 'hard'],
                        help='Complexity level (overrides config)')
    parser.add_argument('--noise', type=str,
                        choices=['gaussian', 'heteroscedastic', 'spatially_correlated', 'mixed', 'none'],
                        help='Noise type (overrides config)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.pde:
        config['pde_type'] = args.pde
    if args.complexity:
        config['complexity'] = args.complexity
    if args.noise:
        config['noise_type'] = None if args.noise == 'none' else args.noise

    # Generate data based on PDE type
    pde_type = config.get('pde_type', 'darcy')

    if pde_type == 'all':
        # Generate all datasets
        config['pde_type'] = 'darcy'
        generate_darcy_data(config)

        config['pde_type'] = 'burgers'
        generate_burgers_data(config)

        config['pde_type'] = 'poisson'
        generate_poisson_data(config)

        config['pde_type'] = 'reaction_diffusion'
        generate_reaction_diffusion_data(config)

    elif pde_type == 'darcy':
        generate_darcy_data(config)
    elif pde_type == 'burgers':
        generate_burgers_data(config)
    elif pde_type == 'poisson':
        generate_poisson_data(config)
    elif pde_type == 'reaction_diffusion':
        generate_reaction_diffusion_data(config)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")

    print("\n=== Data Generation Complete ===")


if __name__ == '__main__':
    main()