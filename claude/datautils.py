
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dataset import DarcyFlowGenerator, BurgersGenerator, PoissonGenerator


class PDEDataset(Dataset):
    """
    PyTorch Dataset wrapper for synthetic PDE data with normalization.

    Handles loading from .npz files or in-memory numpy arrays.
    Automatically normalizes output data to zero mean and unit std.

    Example:
        >>> dataset = PDEDataset.from_file('data/darcy_train.npz')
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> # Denormalize predictions:
        >>> u_pred_real = u_pred * dataset.u_std + dataset.u_mean
    """

    def __init__(self, X: np.ndarray, U: np.ndarray, normalize_output: bool = True):
        """
        Initialize dataset from numpy arrays.

        Args:
            X: Input data, shape (n_samples, nx, ny, in_channels)
            U: Output data, shape (n_samples, nx, ny, out_channels)
            normalize_output: If True, normalize U to zero mean, unit std
        """
        # Convert to torch tensors but keep original shape
        # Our FNO model expects (batch, nx, ny, channels) format
        self.X = torch.from_numpy(X).float()

        # Normalize output data (critical for small-scale PDEs like Darcy flow)
        self.normalize_output = normalize_output
        if normalize_output:    #
            self.u_mean = U.mean()
            self.u_std = U.std()
            U_normalized = (U - self.u_mean) / self.u_std
            self.U = torch.from_numpy(U_normalized).float()

            print(f"Output normalization applied:")
            print(f"  Original: mean={self.u_mean:.6f}, std={self.u_std:.6f}")
            print(f"  Normalized: mean={U_normalized.mean():.6f}, std={U_normalized.std():.6f}")
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

    @classmethod
    def from_file(cls, filepath: str, normalize_output: bool = True):
        """Load dataset from .npz file."""
        data = np.load(filepath)
        X = data['X']
        U = data['U']
        return cls(X, U, normalize_output=normalize_output)


def create_dataloaders(config):
    """
    Create train/validation dataloaders from synthetic PDE data.

    This function either:
    1. Loads pre-generated datasets from disk (if available)
    2. Generates new datasets on-the-fly

    Args:
        config: Configuration object with data parameters

    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    data_dir = Path(config.data_dir) if hasattr(config, 'data_dir') else Path('data')
    data_dir.mkdir(exist_ok=True)

    # Construct filename based on config
    pde_type = config.pde_type if hasattr(config, 'pde_type') else 'darcy'
    complexity = config.complexity if hasattr(config, 'complexity') else 'medium'
    noise_type = config.noise_type if hasattr(config, 'noise_type') else 'heteroscedastic'
    resolution = config.grid_size if hasattr(config, 'grid_size') else 64

    train_file = data_dir / f"{pde_type}_{complexity}_{noise_type}_res{resolution}_train.npz"
    val_file = data_dir / f"{pde_type}_{complexity}_{noise_type}_res{resolution}_val.npz"

    # Try to load existing data
    if train_file.exists() and val_file.exists():
        print(f"Loading existing datasets from {data_dir}")
        train_dataset = PDEDataset.from_file(str(train_file))
        val_dataset = PDEDataset.from_file(str(val_file))
    else:
        print(f"Generating new {pde_type} dataset...")

        # Get number of samples
        n_train = config.n_train if hasattr(config, 'n_train') else 1000
        n_val = config.n_test if hasattr(config, 'n_test') else 200
        seed = config.data_seed if hasattr(config, 'data_seed') else 42

        # Get noise parameters
        noise_params = {}
        if hasattr(config, 'noise_level'):
            noise_params['noise_level'] = config.noise_level
        if hasattr(config, 'base_noise'):
            noise_params['base_noise'] = config.base_noise
        if hasattr(config, 'scale_factor'):
            noise_params['scale_factor'] = config.scale_factor

        # Generate data based on PDE type
        if pde_type == 'darcy':
            generator = DarcyFlowGenerator(
                resolution=resolution,
                complexity=complexity,
                seed=seed
            )
            X_train, U_train = generator.generate_dataset(
                n_samples=n_train,
                noise_type=noise_type,
                noise_params=noise_params
            )

            # Use different seed for validation
            generator_val = DarcyFlowGenerator(
                resolution=resolution,
                complexity=complexity,
                seed=seed + 10000
            )
            X_val, U_val = generator_val.generate_dataset(
                n_samples=n_val,
                noise_type=noise_type,
                noise_params=noise_params
            )

        elif pde_type == 'burgers':
            nx = config.burgers_nx if hasattr(config, 'burgers_nx') else 256
            nt = config.burgers_nt if hasattr(config, 'burgers_nt') else 100
            viscosity = config.burgers_viscosity if hasattr(config, 'burgers_viscosity') else 0.01

            generator = BurgersGenerator(
                nx=nx,
                nt=nt,
                viscosity=viscosity,
                complexity=complexity,
                seed=seed
            )
            X_train, U_train = generator.generate_dataset(
                n_samples=n_train,
                noise_type=noise_type,
                noise_params=noise_params
            )

            generator_val = BurgersGenerator(
                nx=nx,
                nt=nt,
                viscosity=viscosity,
                complexity=complexity,
                seed=seed + 10000
            )
            X_val, U_val = generator_val.generate_dataset(
                n_samples=n_val,
                noise_type=noise_type,
                noise_params=noise_params
            )

        elif pde_type == 'poisson':
            generator = PoissonGenerator(
                resolution=resolution,
                complexity=complexity,
                seed=seed
            )
            X_train, U_train = generator.generate_dataset(
                n_samples=n_train,
                noise_type=noise_type,
                noise_params=noise_params
            )

            generator_val = PoissonGenerator(
                resolution=resolution,
                complexity=complexity,
                seed=seed + 10000
            )
            X_val, U_val = generator_val.generate_dataset(
                n_samples=n_val,
                noise_type=noise_type,
                noise_params=noise_params
            )
        else:
            raise ValueError(f"Unknown PDE type: {pde_type}")

        # Save generated data
        if hasattr(config, 'save_data') and config.save_data:
            np.savez_compressed(train_file, X=X_train, U=U_train)
            np.savez_compressed(val_file, X=X_val, U=U_val)
            print(f"Saved datasets to {data_dir}")

        # Create datasets
        train_dataset = PDEDataset(X_train, U_train)
        val_dataset = PDEDataset(X_val, U_val)

    # Create dataloaders
    batch_size = config.batch_size if hasattr(config, 'batch_size') else 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers if hasattr(config, 'num_workers') else 0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers if hasattr(config, 'num_workers') else 0,
        pin_memory=True
    )

    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Input shape: {train_dataset.X[0].shape}")
    print(f"  Output shape: {train_dataset.U[0].shape}")

    return train_loader, val_loader


def dummy_dataloaders(config):
    """Creates dummy dataloaders for demonstration."""
    print("Creating dummy dataloaders. Implement datautils.py for real data.")
    s = config.grid_size

    # Create coordinate grids
    x_coords = torch.linspace(0, 1, s)
    y_coords = torch.linspace(0, 1, s)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # Prepare input data with 3 channels: (x_coord, y_coord, input_field)
    # Shape: (n_samples, grid_size, grid_size, 3)
    dummy_x_train = torch.zeros(config.n_train, s, s, 3)
    dummy_x_train[..., 0] = grid_x.unsqueeze(0)  # x coordinates
    dummy_x_train[..., 1] = grid_y.unsqueeze(0)  # y coordinates
    dummy_x_train[..., 2] = torch.randn(config.n_train, s, s)  # input field (e.g., forcing term)

    dummy_x_val = torch.zeros(config.n_test, s, s, 3)
    dummy_x_val[..., 0] = grid_x.unsqueeze(0)
    dummy_x_val[..., 1] = grid_y.unsqueeze(0)
    dummy_x_val[..., 2] = torch.randn(config.n_test, s, s)

    # Output: solution field with 1 channel
    dummy_y_train = torch.randn(config.n_train, s, s, 1)
    dummy_y_val = torch.randn(config.n_test, s, s, 1)

    train_dataset = TensorDataset(dummy_x_train, dummy_y_train)
    val_dataset = TensorDataset(dummy_x_val, dummy_y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader

def visualize_inference_results(y_true, y_fno, stats, config, num_samples=3):
    """Visualizes ground truth, FNO prediction, EBM mean, and EBM uncertainty."""
    print("\n--- Visualizing Inference Results ---")
    
    y_true = y_true.cpu().numpy()
    y_fno = y_fno.cpu().numpy()
    y_ebm_mean = stats['mean'].cpu().numpy()
    y_ebm_std = stats['std'].cpu().numpy()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        # Ground Truth
        im = axes[i, 0].imshow(y_true[i, ..., 0], cmap='viridis')
        axes[i, 0].set_title(f"Sample {i+1}: Ground Truth")
        fig.colorbar(im, ax=axes[i, 0])

        # FNO Deterministic
        im = axes[i, 1].imshow(y_fno[i, ..., 0], cmap='viridis')
        axes[i, 1].set_title(f"Sample {i+1}: FNO Prediction")
        fig.colorbar(im, ax=axes[i, 1])

        # EBM Probabilistic Mean
        im = axes[i, 2].imshow(y_ebm_mean[i, ..., 0], cmap='viridis')
        axes[i, 2].set_title(f"Sample {i+1}: EBM Mean")
        fig.colorbar(im, ax=axes[i, 2])

        # EBM Probabilistic Std Dev (Uncertainty)
        im = axes[i, 3].imshow(y_ebm_std[i, ..., 0], cmap='hot')
        axes[i, 3].set_title(f"Sample {i+1}: EBM Std Dev")
        fig.colorbar(im, ax=axes[i, 3])

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    
    output_dir = os.path.dirname(config.checkpoint_dir)
    fig_path = os.path.join(output_dir, "inference_results.png")
    plt.savefig(fig_path)
    print(f"Inference visualization saved to {fig_path}")
    plt.show()