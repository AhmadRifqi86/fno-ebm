# Synthetic Dataset Generation for FNO-EBM

This module provides comprehensive synthetic data generation for various PDEs with realistic noise models, suitable for training and evaluating FNO-EBM models.

## Features

- **Multiple PDE Types**: Darcy flow, Burgers equation, Poisson equation
- **Configurable Complexity**: Simple, medium, hard (controls field roughness)
- **Realistic Noise Models**: Gaussian, heteroscedastic, spatially correlated, mixed
- **Reproducible**: Seed-controlled generation
- **Production-Ready**: Efficient implementations using FFT and sparse solvers

## Quick Start

### 1. Generate Data Using Config File

```bash
# Generate datasets based on config.yaml
python generate_data.py

# Generate all PDE types
python generate_data.py --pde all

# Generate with specific parameters
python generate_data.py --pde darcy --complexity hard --noise mixed
```

### 2. Programmatic Usage

```python
from dataset import DarcyFlowGenerator

# Create generator
gen = DarcyFlowGenerator(
    resolution=64,
    complexity='medium',
    seed=42
)

# Generate dataset
X, U = gen.generate_dataset(
    n_samples=1000,
    noise_type='heteroscedastic',
    noise_params={'base_noise': 0.001, 'scale_factor': 0.02}
)

# Save to disk
gen.save_dataset(X, U, 'data/darcy_train.npz')
```

### 3. Use with PyTorch DataLoader

```python
from datautils import create_dataloaders
from config import Config
import yaml

# Load config
with open('config.yaml') as f:
    config_dict = yaml.safe_load(f)
config = Config(config_dict)

# Create dataloaders (auto-generates if needed)
train_loader, val_loader = create_dataloaders(config)

# Training loop
for x, u in train_loader:
    # x: (batch, channels, nx, ny) - input
    # u: (batch, channels, nx, ny) - output
    pass
```

## PDE Types

### 1. Darcy Flow

**Equation**: `-∇·(a(x)∇u(x)) = f(x)` in Ω=[0,1]²

**Input**: Permeability field `a(x,y)`
**Output**: Pressure/hydraulic head `u(x,y)`
**Data Format**:
- Input: `(n_samples, nx, ny, 3)` = [x_coords, y_coords, permeability]
- Output: `(n_samples, nx, ny, 1)` = [solution]

**Complexity Control**: GP length scale for permeability
- Simple: 0.2 (smooth fields)
- Medium: 0.1
- Hard: 0.05 (rough fields)

```python
gen = DarcyFlowGenerator(resolution=64, complexity='medium', seed=42)
X, U = gen.generate_dataset(n_samples=1000, noise_type='heteroscedastic')
```

### 2. Burgers Equation

**Equation**: `∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²`

**Input**: Initial condition `u₀(x)`
**Output**: Time evolution `u(x,t)`
**Data Format**:
- Input: `(n_samples, nx, 2)` = [x_coords, initial_condition]
- Output: `(n_samples, nt, nx, 1)` = [trajectory]

**Complexity Control**: Number of Fourier modes in initial condition
- Simple: 3 modes
- Medium: 5 modes
- Hard: 8 modes

```python
gen = BurgersGenerator(
    nx=256,          # Spatial resolution
    nt=100,          # Temporal resolution
    viscosity=0.01,  # Viscosity coefficient
    complexity='medium',
    seed=42
)
X, U = gen.generate_dataset(n_samples=1000, noise_type='mixed')
```

### 3. Poisson Equation

**Equation**: `-∇²u(x) = f(x)` in Ω=[0,1]²

**Input**: Source term `f(x,y)`
**Output**: Potential `u(x,y)`
**Data Format**:
- Input: `(n_samples, nx, ny, 3)` = [x_coords, y_coords, source_term]
- Output: `(n_samples, nx, ny, 1)` = [solution]

**Complexity Control**: GP length scale for source term
- Simple: 0.3 (smooth sources)
- Medium: 0.15
- Hard: 0.08 (rough sources)

```python
gen = PoissonGenerator(resolution=64, complexity='hard', seed=42)
X, U = gen.generate_dataset(n_samples=1000, noise_type='gaussian')
```

## Noise Models

### 1. Gaussian (Homoscedastic)

Constant noise level across the domain.

```python
noise_params = {
    'noise_level': 0.01  # σ = 0.01
}
```

**Use Case**: Simple baseline, sensor electronics noise

### 2. Heteroscedastic

Noise proportional to signal magnitude (realistic for most sensors).

```python
noise_params = {
    'base_noise': 0.001,    # Minimum noise floor
    'scale_factor': 0.02    # 2% relative noise
}
# σ(x) = base_noise + scale_factor * |u(x)|
```

**Use Case**: Most physical sensors, where larger values have larger absolute errors

### 3. Spatially Correlated

Smooth noise patterns (mimics environmental effects).

```python
noise_params = {
    'noise_level': 0.01,
    'correlation_length': 3.0  # In grid points
}
```

**Use Case**: Temperature gradients, vibrations, electromagnetic interference

### 4. Mixed

Combination of all three noise sources (most realistic).

```python
noise_params = {
    'gaussian_level': 0.005,       # White noise
    'hetero_scale': 0.01,          # Signal-dependent
    'spatial_level': 0.003,        # Correlated
    'correlation_length': 2.0
}
```

**Use Case**: Real-world measurements with multiple noise sources

## Configuration Parameters

Add these to `config.yaml`:

```yaml
# Dataset Generation Parameters
data_seed: 42
data_dir: './data'
save_data: true

# PDE Selection
pde_type: 'darcy'  # 'darcy', 'burgers', 'poisson'
complexity: 'medium'  # 'simple', 'medium', 'hard'

# Noise Configuration
noise_type: 'heteroscedastic'  # or 'gaussian', 'spatially_correlated', 'mixed'
base_noise: 0.001
scale_factor: 0.02

# Data Sizes
n_train: 1000
n_test: 200
batch_size: 16
grid_size: 64

# Burgers-specific
burgers_nx: 256
burgers_nt: 100
burgers_viscosity: 0.01
```

## File Structure

```
dataset/
├── __init__.py              # Package interface
├── darcy_flow.py            # Darcy flow generator
├── burgers.py               # Burgers equation generator
├── poisson.py               # Poisson equation generator
├── noise_models.py          # Noise addition functions
└── README.md                # This file

datautils.py                 # PyTorch DataLoader wrapper
generate_data.py             # Data generation script
config.yaml                  # Configuration file
```

## Generated File Naming

Files are automatically named based on parameters:

```
{pde_type}_{complexity}_{noise_type}_res{resolution}_{split}.npz

Examples:
- darcy_medium_heteroscedastic_res64_train.npz
- burgers_hard_mixed_nx256_nt100_train.npz
- poisson_simple_gaussian_res64_val.npz
```

## Advanced Usage

### Generate Multiple Configurations

```python
# Generate datasets with varying complexity
for complexity in ['simple', 'medium', 'hard']:
    gen = DarcyFlowGenerator(resolution=64, complexity=complexity, seed=42)
    X, U = gen.generate_dataset(1000, noise_type='heteroscedastic')
    gen.save_dataset(X, U, f'data/darcy_{complexity}_train.npz')

# Generate with varying noise levels
for noise_level in [0.01, 0.02, 0.05]:
    gen = PoissonGenerator(resolution=64, complexity='medium', seed=42)
    X, U = gen.generate_dataset(
        1000,
        noise_type='gaussian',
        noise_params={'noise_level': noise_level}
    )
    gen.save_dataset(X, U, f'data/poisson_noise{noise_level}_train.npz')
```

### Custom PDE Parameters

```python
# Darcy with custom domain
gen = DarcyFlowGenerator(
    resolution=128,
    domain=(0.0, 2.0, 0.0, 2.0),  # [0,2]×[0,2] instead of [0,1]×[0,1]
    complexity='hard',
    seed=42
)

# Burgers with custom viscosity
gen = BurgersGenerator(
    nx=512,
    nt=200,
    L=4*np.pi,     # Longer domain
    T=2.0,         # Longer time
    viscosity=0.001,  # Lower viscosity (more nonlinear)
    complexity='hard',
    seed=42
)
```

## Performance Notes

- **Darcy Flow**: ~0.05s per sample at 64×64 resolution
- **Burgers Equation**: ~0.02s per sample at 256×100 (nx×nt)
- **Poisson**: ~0.03s per sample at 64×64 resolution

Memory usage scales with resolution and batch size. For large datasets, use `save_data=true` to write to disk incrementally.

## Validation

Check generated data quality:

```python
from dataset import DarcyFlowGenerator
import matplotlib.pyplot as plt

gen = DarcyFlowGenerator(resolution=64, complexity='medium', seed=42)
x, u, metadata = gen.generate_sample(0, noise_type='heteroscedastic')

print(f"Permeability mean: {metadata['permeability_mean']:.3f}")
print(f"Solution mean: {metadata['solution_mean']:.3f}")
print(f"Noise std: {metadata['noise_std']:.5f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(x[..., 2], cmap='viridis')
axes[0].set_title('Permeability Field')
axes[1].imshow(u[..., 0], cmap='viridis')
axes[1].set_title('Solution (with noise)')
plt.show()
```

## Citation

If you use this dataset generation framework in your research, please cite:

```
@software{fno_ebm_dataset_2025,
  title={Synthetic PDE Dataset Generation for FNO-EBM},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/fno-ebm}
}
```

## References

1. Li et al. (2020). "Fourier Neural Operator for Parametric Partial Differential Equations." arXiv:2010.08895
2. Raissi et al. (2019). "Physics-informed neural networks." Journal of Computational Physics.
3. Lim et al. (2023). "Score-based diffusion models in function space." arXiv:2302.07400