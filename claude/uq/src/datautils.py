"""
datautils.py - Unified data loading for UQ paper (4 PDEs)

Supports:
- 1D PDEs: .hdf5 format (Burgers, Advection)
- Diffusion-Reaction: .h5 format
- Navier-Stokes: .pt format
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset


class PDEDataset(Dataset):
    """Memory-efficient dataset with in-place normalization and caching."""

    def __init__(self, X: np.ndarray, U: np.ndarray,
                 normalize_output: bool = True,
                 normalize_input: bool = True,
                 normalize_coords: bool = True,
                 cache_path: str = None):
        """
        Args:
            X: (n_samples, nx, ny, in_channels) - [x, y, input_field, ...]
            U: (n_samples, nx, ny, out_channels)
            normalize_output: Normalize U to N(0,1)
            normalize_input: Normalize input fields (beyond x,y) to N(0,1)
            normalize_coords: Normalize x,y to [-1, 1]
            cache_path: Path to save/load cached normalized dataset
        """
        self.n_samples, self.nx, self.ny, self.in_channels = X.shape
        self.normalize_coords = normalize_coords
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        # Store normalization stats
        self.x_min = self.x_max = self.y_min = self.y_max = None
        self.x_fields_mean = []
        self.x_fields_std = []
        self.u_mean = 0.0
        self.u_std = 1.0

        # IN-PLACE normalization to save memory
        # Coordinate normalization
        if normalize_coords:
            self.x_min, self.x_max = X[..., 0].min(), X[..., 0].max()
            self.y_min, self.y_max = X[..., 1].min(), X[..., 1].max()
            # Normalize IN-PLACE
            X[..., 0] = 2 * (X[..., 0] - self.x_min) / (self.x_max - self.x_min + 1e-8) - 1
            X[..., 1] = 2 * (X[..., 1] - self.y_min) / (self.y_max - self.y_min + 1e-8) - 1

        # Input field normalization (channels 2+) IN-PLACE
        if normalize_input and self.in_channels > 2:
            for ch in range(2, self.in_channels):
                field = X[..., ch]
                mean, std = field.mean(), field.std()
                self.x_fields_mean.append(mean)
                self.x_fields_std.append(std)
                # Normalize IN-PLACE
                X[..., ch] = (field - mean) / (std + 1e-8)

        # Output normalization IN-PLACE
        if normalize_output:
            self.u_mean = U.mean()
            self.u_std = U.std()
            # Normalize IN-PLACE
            U[:] = (U - self.u_mean) / self.u_std

        # Convert to PyTorch tensors (X and U are already modified in-place)
        self.X = torch.from_numpy(X).float()
        self.U = torch.from_numpy(U).float()

        # Save cache if requested
        if cache_path:
            self.save_cache(cache_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.U[idx]

    def denormalize(self, U_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original scale."""
        if self.normalize_output:
            return U_normalized * self.u_std + self.u_mean
        return U_normalized

    def save_cache(self, cache_path: str):
        """Save normalized dataset to disk."""
        cache_data = {
            'X': self.X,
            'U': self.U,
            'n_samples': self.n_samples,
            'nx': self.nx,
            'ny': self.ny,
            'in_channels': self.in_channels,
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'x_fields_mean': self.x_fields_mean,
            'x_fields_std': self.x_fields_std,
            'u_mean': self.u_mean,
            'u_std': self.u_std,
            'normalize_coords': self.normalize_coords,
            'normalize_input': self.normalize_input,
            'normalize_output': self.normalize_output,
        }
        torch.save(cache_data, cache_path)
        print(f"Dataset cached to: {cache_path}")

    @staticmethod
    def load_cache(cache_path: str):
        """Load normalized dataset from disk."""
        cache_data = torch.load(cache_path, weights_only=False)

        # Create empty instance
        dataset = PDEDataset.__new__(PDEDataset)

        # Restore attributes
        dataset.X = cache_data['X']
        dataset.U = cache_data['U']
        dataset.n_samples = cache_data['n_samples']
        dataset.nx = cache_data['nx']
        dataset.ny = cache_data['ny']
        dataset.in_channels = cache_data['in_channels']
        dataset.x_min = cache_data['x_min']
        dataset.x_max = cache_data['x_max']
        dataset.y_min = cache_data['y_min']
        dataset.y_max = cache_data['y_max']
        dataset.x_fields_mean = cache_data['x_fields_mean']
        dataset.x_fields_std = cache_data['x_fields_std']
        dataset.u_mean = cache_data['u_mean']
        dataset.u_std = cache_data['u_std']
        dataset.normalize_coords = cache_data['normalize_coords']
        dataset.normalize_input = cache_data['normalize_input']
        dataset.normalize_output = cache_data['normalize_output']

        print(f"Dataset loaded from cache: {cache_path}")
        return dataset


def load_hdf5_1d(filepath: str, nu_values: list = None, max_samples: int = None,
                 time_step_spacing: int = 10, max_pairs_per_sample: int = None) -> PDEDataset:
    """
    Load 1D PDE data (.hdf5): Burgers/Advection with multiple nu values.

    Args:
        filepath: Path to .hdf5 file
        nu_values: List of nu values to load (None = all)
        max_samples: Max samples per nu value
        time_step_spacing: Spacing between input and output time steps (default: 10)
        max_pairs_per_sample: Max number of pairs to extract per sample (None = all possible)

    Returns:
        PDEDataset
    """
    with h5py.File(filepath, 'r') as f:
        # PDEBench format: single 'tensor' with all nu values stacked
        if 'tensor' in f.keys():
            data = f['tensor'][:]  # (n_total, n_t, n_x)
            x_coords_data = f['x-coordinate'][:]

            n_total, n_t, n_x = data.shape

            # Assume 4 nu values stacked: split into equal parts
            n_nu = 4  # PDEBench has 4 nu values for Burgers
            samples_per_nu = n_total // n_nu

            X_list, U_list = [], []

            for nu_idx in range(n_nu):
                start_idx = nu_idx * samples_per_nu
                end_idx = (nu_idx + 1) * samples_per_nu

                nu_data = data[start_idx:end_idx]  # (samples_per_nu, n_t, n_x)

                n_samples = samples_per_nu
                if max_samples is not None:
                    n_samples = min(n_samples, max_samples)

                # Extract multiple pairs per sample with configurable spacing
                for i in range(n_samples):
                    # Generate all possible pairs with given spacing
                    possible_pairs = []
                    for t_start in range(0, n_t - time_step_spacing):
                        t_end = t_start + time_step_spacing
                        if t_end < n_t:
                            possible_pairs.append((t_start, t_end))

                    # Limit number of pairs if specified
                    if max_pairs_per_sample is not None:
                        # Sample uniformly across the trajectory
                        step = max(1, len(possible_pairs) // max_pairs_per_sample)
                        possible_pairs = possible_pairs[::step][:max_pairs_per_sample]

                    # Create training pairs
                    for t_in, t_out in possible_pairs:
                        input_field = nu_data[i, t_in, :]
                        output_field = nu_data[i, t_out, :]

                        # Create 2D grid: (n_x, 1) spatial
                        x_coords = np.linspace(0, 1, n_x)
                        y_coords = np.array([0.5])
                        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')

                        # X: (n_x, 1, 3) = [x, y, input_field]
                        X = np.zeros((n_x, 1, 3), dtype=np.float32)
                        X[:, :, 0] = grid_x
                        X[:, :, 1] = grid_y
                        X[:, :, 2] = input_field[:, np.newaxis]

                        # U: (n_x, 1, 1)
                        U = output_field[:, np.newaxis, np.newaxis]

                        X_list.append(X)
                        U_list.append(U)

        else:
            # Old format with separate nu keys
            all_nu = [k for k in f.keys() if k.startswith('nu_')]
            if nu_values is not None:
                all_nu = [k for k in all_nu if float(k.split('_')[1]) in nu_values]

            if not all_nu:
                raise ValueError(f"No nu keys found. Available: {list(f.keys())}")

            X_list, U_list = [], []

            for nu_key in all_nu:
                data = f[nu_key][:]  # (n_samples, n_t, n_x)
                n_samples, n_t, n_x = data.shape

                if max_samples is not None:
                    n_samples = min(n_samples, max_samples)

                for i in range(n_samples):
                    # Generate all possible pairs with given spacing
                    possible_pairs = []
                    for t_start in range(0, n_t - time_step_spacing):
                        t_end = t_start + time_step_spacing
                        if t_end < n_t:
                            possible_pairs.append((t_start, t_end))

                    # Limit number of pairs if specified
                    if max_pairs_per_sample is not None:
                        step = max(1, len(possible_pairs) // max_pairs_per_sample)
                        possible_pairs = possible_pairs[::step][:max_pairs_per_sample]

                    # Create training pairs
                    for t_in, t_out in possible_pairs:
                        input_field = data[i, t_in, :]
                        output_field = data[i, t_out, :]

                        x_coords = np.linspace(0, 1, n_x)
                        y_coords = np.array([0.5])
                        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')

                        X = np.zeros((n_x, 1, 3), dtype=np.float32)
                        X[:, :, 0] = grid_x
                        X[:, :, 1] = grid_y
                        X[:, :, 2] = input_field[:, np.newaxis]

                        U = output_field[:, np.newaxis, np.newaxis]

                        X_list.append(X)
                        U_list.append(U)

        X = np.stack(X_list)
        U = np.stack(U_list)

    return PDEDataset(X, U)


def load_h5_2d(filepath: str, max_samples: int = None) -> PDEDataset:
    """
    Load 2D diffusion-reaction data (.h5 format).

    Args:
        filepath: Path to .h5 file
        max_samples: Max samples to load

    Returns:
        PDEDataset
    """
    with h5py.File(filepath, 'r') as f:
        # Inspect keys
        keys = list(f.keys())

        # Find data key (usually simulation groups like '0000', '0001', ...)
        sim_keys = [k for k in keys if k.isdigit()]

        if not sim_keys:
            raise ValueError(f"No simulation keys found in {filepath}. Available: {keys}")

        X_list, U_list = [], []

        for sim_key in sim_keys:
            if max_samples and len(X_list) >= max_samples:
                break

            data = f[sim_key][:]  # (n_t, n_x, n_y, n_channels)
            n_t, n_x, n_y, _ = data.shape

            # Extract t=0 → t=-1
            input_field = data[0, :, :, 0]
            output_field = data[-1, :, :, 0]

            # Generate coordinate grid
            x_coords = np.linspace(0, 1, n_x)
            y_coords = np.linspace(0, 1, n_y)
            grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')

            # X: (n_x, n_y, 3)
            X = np.zeros((n_x, n_y, 3), dtype=np.float32)
            X[:, :, 0] = grid_x
            X[:, :, 1] = grid_y
            X[:, :, 2] = input_field

            # U: (n_x, n_y, 1)
            U = output_field[:, :, np.newaxis]

            X_list.append(X)
            U_list.append(U)

        X = np.stack(X_list)
        U = np.stack(U_list)

    return PDEDataset(X, U)


def load_pt_2d(filepath: str) -> PDEDataset:
    """
    Load Navier-Stokes data (.pt format).

    Args:
        filepath: Path to .pt file (expects dict with 'x' and 'y')

    Returns:
        PDEDataset
    """
    data = torch.load(filepath, weights_only=False)

    if not isinstance(data, dict) or 'x' not in data or 'y' not in data:
        raise ValueError(f"Expected dict with 'x' and 'y', got: {type(data)}")

    input_field = data['x'].numpy()  # (n_samples, n_x, n_y)
    output_field = data['y'].numpy()  # (n_samples, n_x, n_y)

    n_samples, n_x, n_y = input_field.shape

    # Generate coordinate grids
    x_coords = np.linspace(0, 1, n_x)
    y_coords = np.linspace(0, 1, n_y)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')

    # X: (n_samples, n_x, n_y, 3)
    X = np.zeros((n_samples, n_x, n_y, 3), dtype=np.float32)
    X[:, :, :, 0] = grid_x[np.newaxis, :, :]
    X[:, :, :, 1] = grid_y[np.newaxis, :, :]
    X[:, :, :, 2] = input_field

    # U: (n_samples, n_x, n_y, 1)
    U = output_field[..., np.newaxis]

    return PDEDataset(X, U)


def load_pde_data(filepath: str, pde_type: str, nu_values: list = None, max_samples: int = 8000,
                  time_step_spacing: int = 10, max_pairs_per_sample: int = 100) -> PDEDataset:
    """
    Unified loader for all 4 PDEs.

    Args:
        filepath: Path to data file
        pde_type: 'burgers', 'advection', 'diffusion_reaction', 'navier_stokes'
        nu_values: List of nu values (for 1D PDEs only)
        max_samples: Max samples to load
        time_step_spacing: Spacing between input and output time steps (1D PDEs only)
        max_pairs_per_sample: Max pairs to extract per sample (1D PDEs only)

    Returns:
        PDEDataset
    """
    filepath = Path(filepath)

    if pde_type in ['burgers', 'advection']:
        # 1D PDEs: .hdf5 format
        return load_hdf5_1d(str(filepath), nu_values, max_samples, time_step_spacing, max_pairs_per_sample)
    elif pde_type == 'diffusion_reaction':
        # 2D: .h5 format
        return load_h5_2d(str(filepath), max_samples)
    elif pde_type == 'navier_stokes':
        # 2D: .pt format
        return load_pt_2d(str(filepath))
    else:
        raise ValueError(f"Unknown pde_type: {pde_type}")


def create_dataloaders(dataset: PDEDataset, train_ratio: float = 0.8,
                       val_ratio: float = 0.1, batch_size: int = 32,
                       seed: int = 42) -> tuple:
    """
    Split dataset and create dataloaders.

    Args:
        dataset: PDEDataset instance
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        batch_size: Batch size
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    # Random split
    indices = np.random.RandomState(seed=seed).permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
