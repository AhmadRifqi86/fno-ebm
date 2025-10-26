"""
dataset.py - PDE Dataset Generators (Stub)

This is a stub file created to satisfy imports in datautils.py.
The actual PDE generators are not implemented here.

If you need to generate data programmatically, you should either:
1. Implement these classes based on your PDE solvers
2. Use pre-generated .npz files with PDEDataset.from_file()
3. Create your own data generation pipeline

For the main training scripts (main_separate.py, main_noisy.py),
this stub is sufficient since they use pre-generated data files.
"""


class DarcyFlowGenerator:
    """
    Stub for Darcy flow PDE data generator.

    Darcy flow: -∇·(a(x,y)∇u) = f
    """
    def __init__(self, resolution=64, complexity='medium', seed=42):
        raise NotImplementedError(
            "DarcyFlowGenerator is not implemented. "
            "Please use pre-generated data files or implement your own generator."
        )

    def generate_dataset(self, n_samples, noise_type='heteroscedastic', noise_params=None):
        raise NotImplementedError(
            "DarcyFlowGenerator.generate_dataset is not implemented."
        )


class BurgersGenerator:
    """
    Stub for Burgers' equation data generator.

    Burgers' equation: u_t + u·∇u = ν∇²u
    """
    def __init__(self, nx=256, nt=100, viscosity=0.01, complexity='medium', seed=42):
        raise NotImplementedError(
            "BurgersGenerator is not implemented. "
            "Please use pre-generated data files or implement your own generator."
        )

    def generate_dataset(self, n_samples, noise_type='heteroscedastic', noise_params=None):
        raise NotImplementedError(
            "BurgersGenerator.generate_dataset is not implemented."
        )


class PoissonGenerator:
    """
    Stub for Poisson equation data generator.

    Poisson equation: -∇²u = f
    """
    def __init__(self, resolution=64, complexity='medium', seed=42):
        raise NotImplementedError(
            "PoissonGenerator is not implemented. "
            "Please use pre-generated data files or implement your own generator."
        )

    def generate_dataset(self, n_samples, noise_type='heteroscedastic', noise_params=None):
        raise NotImplementedError(
            "PoissonGenerator.generate_dataset is not implemented."
        )


# If someone tries to use these, they'll get a clear error message
if __name__ == "__main__":
    print("This is a stub file. PDE generators are not implemented.")
    print("Use pre-generated .npz files with PDEDataset.from_file() instead.")