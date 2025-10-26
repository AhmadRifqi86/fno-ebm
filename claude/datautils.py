
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

    **CRITICAL FIX:** Now normalizes ALL input channels beyond x,y coordinates!
    - Channels 0,1: x,y coordinates (kept as is, already in [0,1])
    - Channels 2+: Physical fields (permeability, forcing, etc.) - normalized to N(0,1)

    Example:
        >>> dataset = PDEDataset.from_file('data/darcy_train.npz')
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> # Denormalize predictions:
        >>> u_pred_real = u_pred * dataset.u_std + dataset.u_mean
    """

    def __init__(self, X: np.ndarray, U: np.ndarray, normalize_output: bool = True, normalize_input: bool = True):
        """
        Initialize dataset from numpy arrays.

        Args:
            X: Input data, shape (n_samples, nx, ny, in_channels)
               First 2 channels: x,y coordinates (kept unnormalized)
               Remaining channels: Physical fields (normalized to N(0,1))
            U: Output data, shape (n_samples, nx, ny, out_channels)
            normalize_output: If True, normalize U to zero mean, unit std
            normalize_input: If True, normalize input channels beyond x,y coords
        """
        # Store original X shape
        self.n_samples, self.nx, self.ny, self.in_channels = X.shape

        # ===== INPUT NORMALIZATION (CRITICAL FIX!) =====
        self.normalize_input = normalize_input

        if normalize_input and self.in_channels > 2:
            # Split coordinate channels from physical field channels
            X_coords = X[..., :2]  # (n_samples, nx, ny, 2) - x,y coordinates
            X_fields = X[..., 2:]  # (n_samples, nx, ny, n_fields) - permeability, forcing, etc.

            # Normalize each physical field channel independently
            self.x_fields_mean = []
            self.x_fields_std = []
            X_fields_normalized = np.zeros_like(X_fields)

            print(f"\nInput normalization applied:")
            print(f"  Keeping channels 0,1 (coordinates) unnormalized: range [0,1]")

            for ch in range(X_fields.shape[-1]):
                field = X_fields[..., ch]
                mean = field.mean()
                std = field.std()

                self.x_fields_mean.append(mean)
                self.x_fields_std.append(std)

                X_fields_normalized[..., ch] = (field - mean) / std

                print(f"  Channel {ch+2}: mean={mean:.6f}, std={std:.6f} → N(0,1)")

            # Concatenate coordinates + normalized fields
            X_normalized = np.concatenate([X_coords, X_fields_normalized], axis=-1)
            self.X = torch.from_numpy(X_normalized).float()

            print(f"  Final input shape: {self.X.shape}")
        else:
            # No normalization (or only 2 channels = coordinates only)
            self.x_fields_mean = []
            self.x_fields_std = []
            self.X = torch.from_numpy(X).float()

            if self.in_channels > 2:
                print(f"\nWARNING: Input normalization disabled, but you have {self.in_channels} channels!")
                print(f"  This may cause training issues if channels have different scales.")

        # ===== OUTPUT NORMALIZATION =====
        self.normalize_output = normalize_output
        if normalize_output:
            self.u_mean = U.mean()
            self.u_std = U.std()
            U_normalized = (U - self.u_mean) / self.u_std
            self.U = torch.from_numpy(U_normalized).float()

            print(f"\nOutput normalization applied:")
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

    def denormalize_input_fields(self, X_normalized: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized input fields back to original scale.
        Only affects channels 2+ (physical fields), keeps coordinates unchanged.

        Args:
            X_normalized: Normalized input (batch, nx, ny, channels)

        Returns:
            X_denormalized: Input with physical fields denormalized
        """
        if not self.normalize_input or len(self.x_fields_mean) == 0:
            return X_normalized

        X = X_normalized.clone()

        # Denormalize each physical field channel
        for ch in range(len(self.x_fields_mean)):
            X[..., ch + 2] = X[..., ch + 2] * self.x_fields_std[ch] + self.x_fields_mean[ch]

        return X

    @classmethod
    def from_file(cls, filepath: str, normalize_output: bool = True, normalize_input: bool = True):
        """
        Load dataset from .npz file.

        Args:
            filepath: Path to .npz file
            normalize_output: If True, normalize output to N(0,1)
            normalize_input: If True, normalize input channels beyond x,y coords to N(0,1)

        Returns:
            PDEDataset instance
        """
        data = np.load(filepath)
        X = data['X']
        U = data['U']
        return cls(X, U, normalize_output=normalize_output, normalize_input=normalize_input)


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