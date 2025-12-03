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
    """Unified dataset for all 4 PDEs with normalization."""

    def __init__(self, X: np.ndarray, U: np.ndarray,
                 normalize_output: bool = True,
                 normalize_input: bool = True,
                 normalize_coords: bool = True):
        """
        Args:
            X: (n_samples, nx, ny, in_channels) - [x, y, input_field, ...]
            U: (n_samples, nx, ny, out_channels)
            normalize_output: Normalize U to N(0,1)
            normalize_input: Normalize input fields (beyond x,y) to N(0,1)
            normalize_coords: Normalize x,y to [-1, 1]
        """
        self.n_samples, self.nx, self.ny, self.in_channels = X.shape

        # Coordinate normalization
        self.normalize_coords = normalize_coords
        if normalize_coords:
            X_coords = X[..., :2].copy()
            self.x_min, self.x_max = X_coords[..., 0].min(), X_coords[..., 0].max()
            self.y_min, self.y_max = X_coords[..., 1].min(), X_coords[..., 1].max()
            X_coords[..., 0] = 2 * (X_coords[..., 0] - self.x_min) / (self.x_max - self.x_min) - 1
            X_coords[..., 1] = 2 * (X_coords[..., 1] - self.y_min) / (self.y_max - self.y_min) - 1
            X = X.copy()
            X[..., :2] = X_coords

        # Input field normalization (channels 2+)
        self.normalize_input = normalize_input
        if normalize_input and self.in_channels > 2:
            X_coords = X[..., :2]
            X_fields = X[..., 2:]
            self.x_fields_mean = []
            self.x_fields_std = []
            X_fields_normalized = np.zeros_like(X_fields)

            for ch in range(X_fields.shape[-1]):
                field = X_fields[..., ch]
                mean, std = field.mean(), field.std()
                self.x_fields_mean.append(mean)
                self.x_fields_std.append(std)
                X_fields_normalized[..., ch] = (field - mean) / (std + 1e-8)

            X = np.concatenate([X_coords, X_fields_normalized], axis=-1)
            self.X = torch.from_numpy(X).float()
        else:
            self.x_fields_mean = []
            self.x_fields_std = []
            self.X = torch.from_numpy(X).float()

        # Output normalization
        self.normalize_output = normalize_output
        if normalize_output:
            self.u_mean = U.mean()
            self.u_std = U.std()
            U_normalized = (U - self.u_mean) / self.u_std
            self.U = torch.from_numpy(U_normalized).float()
        else:
            self.u_mean = 0.0
            self.u_std = 1.0
            self.U = torch.from_numpy(U).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.U[idx]

    def denormalize(self, U_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original scale."""
        if self.normalize_output:
            return U_normalized * self.u_std + self.u_mean
        return U_normalized


def load_hdf5_1d(filepath: str, nu_values: list = None, max_samples: int = None) -> PDEDataset:
    """
    Load 1D PDE data (.hdf5): Burgers/Advection with multiple nu values.

    Args:
        filepath: Path to .hdf5 file
        nu_values: List of nu values to load (None = all)
        max_samples: Max samples per nu value

    Returns:
        PDEDataset
    """
    with h5py.File(filepath, 'r') as f:
        # Get all nu keys
        all_nu = [k for k in f.keys() if k.startswith('nu_')]
        if nu_values is not None:
            # Filter by requested nu values
            all_nu = [k for k in all_nu if float(k.split('_')[1]) in nu_values]

        X_list, U_list = [], []

        for nu_key in all_nu:
            data = f[nu_key][:]  # (n_samples, n_t, n_x)
            n_samples, n_t, n_x = data.shape

            if max_samples is not None:
                n_samples = min(n_samples, max_samples)

            # Extract initial → final pairs
            for i in range(n_samples):
                input_field = data[i, 0, :]  # t=0
                output_field = data[i, -1, :]  # t=final

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


def load_pde_data(filepath: str, pde_type: str, nu_values: list = None, max_samples: int = None) -> PDEDataset:
    """
    Unified loader for all 4 PDEs.

    Args:
        filepath: Path to data file
        pde_type: 'burgers', 'advection', 'diffusion_reaction', 'navier_stokes'
        nu_values: List of nu values (for 1D PDEs only)
        max_samples: Max samples to load

    Returns:
        PDEDataset
    """
    filepath = Path(filepath)

    if pde_type in ['burgers', 'advection']:
        # 1D PDEs: .hdf5 format
        return load_hdf5_1d(str(filepath), nu_values, max_samples)
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
