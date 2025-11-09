
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dataset import DarcyFlowGenerator, BurgersGenerator, PoissonGenerator
import h5py


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

    def __init__(self, X: np.ndarray, U: np.ndarray, normalize_output: bool = True,
                 normalize_input: bool = True, normalize_coords: bool = True):
        """
        Initialize dataset from numpy arrays.

        Args:
            X: Input data, shape (n_samples, nx, ny, in_channels)
               First 2 channels: x,y coordinates
               Remaining channels: Physical fields (normalized to N(0,1))
            U: Output data, shape (n_samples, nx, ny, out_channels)
            normalize_output: If True, normalize U to zero mean, unit std
            normalize_input: If True, normalize input channels beyond x,y coords
            normalize_coords: If True, normalize x,y coordinates to [-1, 1]
        """
        # Store original X shape
        self.n_samples, self.nx, self.ny, self.in_channels = X.shape

        # ===== COORDINATE NORMALIZATION (NEW!) =====
        self.normalize_coords = normalize_coords

        if normalize_coords:
            # Normalize first 2 channels (x, y coordinates) to [-1, 1]
            X_coords = X[..., :2].copy()  # (n_samples, nx, ny, 2)

            self.x_coord_min = X_coords[..., 0].min()
            self.x_coord_max = X_coords[..., 0].max()
            self.y_coord_min = X_coords[..., 1].min()
            self.y_coord_max = X_coords[..., 1].max()

            # Normalize to [-1, 1]
            X_coords[..., 0] = 2 * (X_coords[..., 0] - self.x_coord_min) / (self.x_coord_max - self.x_coord_min) - 1
            X_coords[..., 1] = 2 * (X_coords[..., 1] - self.y_coord_min) / (self.y_coord_max - self.y_coord_min) - 1

            print(f"\nCoordinate normalization applied:")
            print(f"  X coord: [{self.x_coord_min:.4f}, {self.x_coord_max:.4f}] → [-1, 1]")
            print(f"  Y coord: [{self.y_coord_min:.4f}, {self.y_coord_max:.4f}] → [-1, 1]")
            print(f"  Normalized X range: [{X_coords[..., 0].min():.4f}, {X_coords[..., 0].max():.4f}]")
            print(f"  Normalized Y range: [{X_coords[..., 1].min():.4f}, {X_coords[..., 1].max():.4f}]")

            # Replace coordinates in X
            X = X.copy()
            X[..., :2] = X_coords
        else:
            self.x_coord_min = None
            self.x_coord_max = None
            self.y_coord_min = None
            self.y_coord_max = None

        # ===== INPUT NORMALIZATION (CRITICAL FIX!) =====
        self.normalize_input = normalize_input

        print(f"\n[DEBUG] Input normalization check:")
        print(f"  normalize_input={normalize_input}, in_channels={self.in_channels}")
        print(f"  Condition (normalize_input and in_channels > 2) = {normalize_input and self.in_channels > 2}")

        if normalize_input and self.in_channels > 2:
            # Split coordinate channels from physical field channels
            X_coords = X[..., :2]  # (n_samples, nx, ny, 2) - x,y coordinates
            X_fields = X[..., 2:]  # (n_samples, nx, ny, n_fields) - permeability, forcing, etc.

            # Normalize each physical field channel independently
            self.x_fields_mean = []
            self.x_fields_std = []
            X_fields_normalized = np.zeros_like(X_fields)

            print(f"\nInput field normalization applied (channels 2+):")
            print(f"  Note: Channels 0,1 (coordinates) already normalized to [-1,1]")

            for ch in range(X_fields.shape[-1]):
                field = X_fields[..., ch]
                mean = field.mean()
                std = field.std()

                self.x_fields_mean.append(mean)
                self.x_fields_std.append(std)

                X_fields_normalized[..., ch] = (field - mean) / (std + 1e-8)

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
    def from_file(cls, filepath: str, normalize_output: bool = True,
                  normalize_input: bool = True, normalize_coords: bool = True,
                  h5_input_key: str = 'nu', h5_output_key: str = 'tensor'):
        """
        Load dataset from .npz or .h5/.hdf5 file.

        Args:
            filepath: Path to .npz, .h5, or .hdf5 file
            normalize_output: If True, normalize output to N(0,1)
            normalize_input: If True, normalize input channels beyond x,y coords to N(0,1)
            normalize_coords: If True, normalize x,y coordinates to [-1, 1]
            h5_input_key: Key for input data in HDF5 file (default: 'nu' for PDEBench)
            h5_output_key: Key for output data in HDF5 file (default: 'tensor' for PDEBench)

        Returns:
            PDEDataset instance

        Note:
            For HDF5 files, the function expects the structure:
            - Input: file[h5_input_key] with shape (n_samples, n_x, n_y) or (n_samples, n_x, n_y, n_channels)
            - Output: file[h5_output_key] with shape (n_samples, n_x, n_y, n_timesteps) or similar

            The function will automatically:
            1. Generate coordinate grids for channels 0,1
            2. Stack input data as channel 2
            3. Select the last timestep from output (for time-dependent PDEs)
        """
        filepath = str(filepath)  # Ensure string path
        file_ext = Path(filepath).suffix.lower()

        if file_ext == '.npz':
            # Load NPZ format
            print(f"Loading NPZ file: {filepath}")
            data = np.load(filepath)
            X = data['X']
            U = data['U']

        elif file_ext in ['.h5', '.hdf5']:
            # Load HDF5 format
            print(f"Loading HDF5 file: {filepath}")
            with h5py.File(filepath, 'r') as f:
                print(f"Available keys in HDF5 file: {list(f.keys())}")

                # Load input data
                if h5_input_key in f:
                    input_data = f[h5_input_key][:]  # Shape: (n_samples, n_x, n_y) or (n_samples, n_x, n_y, ...)
                    print(f"  Input data '{h5_input_key}' shape: {input_data.shape}")
                else:
                    raise KeyError(f"Input key '{h5_input_key}' not found in HDF5 file. Available keys: {list(f.keys())}")

                # Load output data
                if h5_output_key in f:
                    output_data = f[h5_output_key][:]  # Shape: (n_samples, n_x, n_y, n_t) for time-dependent
                    print(f"  Output data '{h5_output_key}' shape: {output_data.shape}")
                else:
                    raise KeyError(f"Output key '{h5_output_key}' not found in HDF5 file. Available keys: {list(f.keys())}")

            # Process input data
            if input_data.ndim == 3:
                # (n_samples, n_x, n_y) -> add channel dimension
                input_data = input_data[..., np.newaxis]  # (n_samples, n_x, n_y, 1)

            n_samples, n_x, n_y = input_data.shape[:3]

            # Generate coordinate grids
            x_coords = np.linspace(0, 1, n_x)
            y_coords = np.linspace(0, 1, n_y)
            grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')

            # Create X with shape (n_samples, n_x, n_y, 3): [x_coord, y_coord, input_field]
            X = np.zeros((n_samples, n_x, n_y, 3))
            X[:, :, :, 0] = grid_x[np.newaxis, :, :]  # x coordinates
            X[:, :, :, 1] = grid_y[np.newaxis, :, :]  # y coordinates
            X[:, :, :, 2] = input_data[..., 0]         # input field (first channel if multiple)

            print(f"  Generated X with coordinates: {X.shape}")

            # Process output data
            if output_data.ndim == 4:
                # (n_samples, n_x, n_y, n_t) -> take last timestep
                U = output_data[:, :, :, -1:] # (n_samples, n_x, n_y, 1)
                print(f"  Selected last timestep from output: {U.shape}")
            elif output_data.ndim == 3:
                # (n_samples, n_x, n_y) -> add channel dimension
                U = output_data[..., np.newaxis]  # (n_samples, n_x, n_y, 1)
            else:
                raise ValueError(f"Unexpected output data shape: {output_data.shape}")

            print(f"  Final U shape: {U.shape}")

        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .npz, .h5, .hdf5")

        return cls(X, U, normalize_output=normalize_output,
                   normalize_input=normalize_input, normalize_coords=normalize_coords)


class AugmentedPDEDataset(Dataset):
    """
    Data augmentation wrapper for PDE datasets.

    Applies random geometric transformations:
    - Horizontal/vertical flips
    - 90° rotations (0°, 90°, 180°, 270°)

    These augmentations are valid for PDEs with spatial symmetry (most physical systems).
    Effectively increases dataset size by 8× (2 flips × 4 rotations).

    Example:
        >>> base_dataset = PDEDataset.from_file('train.npz')
        >>> augmented_dataset = AugmentedPDEDataset(base_dataset)
        >>> # Now each sample is randomly flipped/rotated during training
    """
    def __init__(self, dataset, enable_flip=True, enable_rotation=True):
        """
        Initialize augmented dataset.

        Args:
            dataset: Base PDEDataset to wrap
            enable_flip: Enable horizontal/vertical flipping (default: True)
            enable_rotation: Enable 90° rotations (default: True)
        """
        self.dataset = dataset
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get augmented sample.

        Returns:
            x: Augmented input (n_x, n_y, 3)
            u: Augmented output (n_x, n_y, 1)
        """
        # Get base sample
        x, u = self.dataset[idx]

        # Apply augmentations
        if self.enable_flip:
            # Random horizontal flip (50% chance)
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [1])  # Flip along y-axis
                u = torch.flip(u, [1])

            # Random vertical flip (50% chance)
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [0])  # Flip along x-axis
                u = torch.flip(u, [0])

        if self.enable_rotation:
            # Random 90° rotation (0°, 90°, 180°, 270°)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:  # Only rotate if k > 0 (save computation)
                x = torch.rot90(x, k, [0, 1])
                u = torch.rot90(u, k, [0, 1])

        # NOTE: Coordinate channels are automatically transformed by flip/rot90
        # No need to regenerate - the transformations preserve spatial consistency
        return x, u

    def _update_coordinates(self, x, nx, ny):
        """
        Regenerate coordinate channels after spatial transformation.

        After flipping/rotating, the coordinate grid changes. We need to
        regenerate channels 0,1 to match the new spatial layout.

        Args:
            x: Transformed input (n_x, n_y, 3)
            nx, ny: Grid dimensions

        Returns:
            x: Input with updated coordinate channels
        """
        # Generate fresh coordinate grid
        # Note: Coordinates are already normalized to [-1, 1] by PDEDataset
        x_coords = torch.linspace(-1, 1, nx, device=x.device)
        y_coords = torch.linspace(-1, 1, ny, device=x.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')

        # Replace coordinate channels (0, 1) with fresh grid
        x_new = x.clone()
        x_new[..., 0] = grid_x
        x_new[..., 1] = grid_y
        # Channel 2 (input field) stays as is - it was correctly transformed

        return x_new

    # Expose base dataset attributes
    @property
    def u_mean(self):
        return self.dataset.u_mean

    @property
    def u_std(self):
        return self.dataset.u_std

    @property
    def normalize_output(self):
        return self.dataset.normalize_output

    def denormalize(self, U_normalized):
        """Delegate to base dataset"""
        return self.dataset.denormalize(U_normalized)


class PDEBenchH5Loader:
    """
    Utility class for loading and inspecting PDEBench HDF5 datasets.

    PDEBench datasets are stored as HDF5 files with temporal evolution data.
    This class helps:
    - Inspect file structure and shapes
    - Extract single-step problems (t=0 → t=final)
    - Convert to PDEDataset format for training

    Example:
        >>> loader = PDEBenchH5Loader('2d_diff_react_NA_NA.h5')
        >>> loader.print_info()  # Shows file structure
        >>> dataset = loader.to_dataset(input_t=0, output_t=-1)  # Extract initial→final
    """

    def __init__(self, filepath: str):
        """
        Initialize loader with HDF5 file path.

        Args:
            filepath: Path to PDEBench HDF5 file
        """
        self.filepath = filepath
        self.file = None
        self._open()

    def _open(self):
        """Open HDF5 file and cache reference."""
        if self.file is None:
            self.file = h5py.File(self.filepath, 'r')

    def close(self):
        """Close HDF5 file."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        """Context manager entry."""
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def print_info(self):
        """
        Print comprehensive information about the HDF5 file structure.
        Shows all keys, shapes, dtypes, and size estimates.
        """
        print("=" * 70)
        print(f"PDEBench HDF5 File: {self.filepath}")
        print("=" * 70)

        # Get file size
        file_size_gb = os.path.getsize(self.filepath) / (1024**3)
        print(f"\nFile size: {file_size_gb:.2f} GB")

        print("\nAvailable keys and shapes:")
        print("-" * 70)

        for key in self.file.keys():
            data = self.file[key]
            if isinstance(data, h5py.Dataset):
                shape = data.shape
                dtype = data.dtype
                size_gb = np.prod(shape) * np.dtype(dtype).itemsize / (1024**3)
                print(f"  '{key}':")
                print(f"    Shape: {shape}")
                print(f"    Dtype: {dtype}")
                print(f"    Size: {size_gb:.3f} GB")

                # Infer dimensions
                if len(shape) == 4:
                    print(f"    Inferred: (n_samples={shape[0]}, n_x={shape[1]}, n_y={shape[2]}, n_timesteps={shape[3]})")
                elif len(shape) == 3:
                    print(f"    Inferred: (n_samples={shape[0]}, n_x={shape[1]}, n_y={shape[2]})")
                elif len(shape) == 5:
                    print(f"    Inferred: (n_samples={shape[0]}, n_x={shape[1]}, n_y={shape[2]}, n_timesteps={shape[3]}, n_fields={shape[4]})")

        print("=" * 70)

    def get_shape(self, key: str = None):
        """
        Get shape of a specific key or the first dataset found.

        Args:
            key: HDF5 key name. If None, uses first dataset found.

        Returns:
            tuple: Shape of the dataset
        """
        if key is None:
            # Get first dataset key
            key = list(self.file.keys())[0]

        return self.file[key].shape

    def peek_sample(self, key: str = None, sample_idx: int = 0, timestep: int = 0):
        """
        Load and return a single sample for inspection.

        Args:
            key: HDF5 key name. If None, uses first dataset found.
            sample_idx: Which sample to load (default: 0)
            timestep: Which timestep to load (default: 0, or all if no time dimension)

        Returns:
            np.ndarray: The requested sample
        """
        if key is None:
            key = list(self.file.keys())[0]

        data = self.file[key]

        if len(data.shape) == 4:  # (n_samples, n_x, n_y, n_t)
            sample = data[sample_idx, :, :, timestep]
        elif len(data.shape) == 3:  # (n_samples, n_x, n_y)
            sample = data[sample_idx, :, :]
        else:
            sample = data[sample_idx]

        return sample

    def to_dataset(self,
                   key: str = None,
                   input_t: int = 0,
                   output_t: int = -1,
                   num_samples: int = None,
                   normalize_output: bool = True,
                   normalize_input: bool = True,
                   normalize_coords: bool = True):
        """
        Convert PDEBench HDF5 to PDEDataset for training.

        Extracts single-step problem: uses solution at time input_t as input field,
        and solution at time output_t as target output.

        Args:
            key: HDF5 key name. If None, uses first dataset found.
            input_t: Timestep index to use as input (default: 0 = initial condition)
            output_t: Timestep index to use as output (default: -1 = final state)
            num_samples: Number of samples to load. If None, loads all.
            normalize_output: Normalize output to N(0,1)
            normalize_input: Normalize input fields to N(0,1)
            normalize_coords: Normalize coordinates to [-1, 1]

        Returns:
            PDEDataset: Dataset ready for training
        """
        if key is None:
            key = list(self.file.keys())[0]

        data = self.file[key]
        shape = data.shape

        # Determine number of samples
        n_total = shape[0]
        if num_samples is None:
            num_samples = n_total
        else:
            num_samples = min(num_samples, n_total)

        print(f"\nLoading {num_samples} samples from '{key}'...")
        print(f"  Full shape: {shape}")

        # Load data based on dimensionality
        if len(shape) == 4:  # (n_samples, n_x, n_y, n_t)
            n_samples, n_x, n_y, n_t = shape

            print(f"  Temporal data detected: {n_t} timesteps")
            print(f"  Using t={input_t} as input, t={output_t} as output")

            # Load input and output timesteps
            input_field = data[:num_samples, :, :, input_t]   # (n_samples, n_x, n_y)
            output_field = data[:num_samples, :, :, output_t]  # (n_samples, n_x, n_y)

        elif len(shape) == 3:  # (n_samples, n_x, n_y) - steady state
            n_samples, n_x, n_y = shape

            print(f"  Steady-state data detected (no time dimension)")

            # For steady-state, we need separate input/output data
            # This typically means the file has different keys for input and output
            # For now, we'll just use the data as output and create dummy input
            # The user should provide proper input data separately
            raise ValueError(
                "Steady-state data detected. For Darcy flow and similar problems, "
                "PDEBench typically has separate input (e.g., 'nu', 'coeff') and "
                "output (e.g., 'tensor', 'sol') keys. Please specify both:\n"
                "  loader.to_dataset_steady_state(input_key='nu', output_key='tensor')"
            )

        else:
            raise ValueError(f"Unsupported data shape: {shape}")

        # Generate coordinate grids
        x_coords = np.linspace(0, 1, n_x)
        y_coords = np.linspace(0, 1, n_y)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')

        # Create X with shape (n_samples, n_x, n_y, 3): [x_coord, y_coord, input_field]
        X = np.zeros((num_samples, n_x, n_y, 3))
        X[:, :, :, 0] = grid_x[np.newaxis, :, :]  # x coordinates
        X[:, :, :, 1] = grid_y[np.newaxis, :, :]  # y coordinates
        X[:, :, :, 2] = input_field                # input field

        # Create U with shape (n_samples, n_x, n_y, 1)
        U = output_field[..., np.newaxis]

        print(f"  Created X: {X.shape}, U: {U.shape}")
        print(f"  Input field range: [{input_field.min():.4f}, {input_field.max():.4f}]")
        print(f"  Output field range: [{output_field.min():.4f}, {output_field.max():.4f}]")

        # Convert to PDEDataset
        return PDEDataset(X, U,
                         normalize_output=normalize_output,
                         normalize_input=normalize_input,
                         normalize_coords=normalize_coords)

    def to_dataset_steady_state(self,
                                input_key: str,
                                output_key: str,
                                num_samples: int = None,
                                normalize_output: bool = True,
                                normalize_input: bool = True,
                                normalize_coords: bool = True):
        """
        Convert steady-state PDEBench data (like Darcy flow) to PDEDataset.

        For steady-state problems, input and output are typically in separate keys.

        Args:
            input_key: HDF5 key for input field (e.g., 'nu' for permeability)
            output_key: HDF5 key for output field (e.g., 'tensor' for solution)
            num_samples: Number of samples to load. If None, loads all.
            normalize_output: Normalize output to N(0,1)
            normalize_input: Normalize input fields to N(0,1)
            normalize_coords: Normalize coordinates to [-1, 1]

        Returns:
            PDEDataset: Dataset ready for training
        """
        input_data = self.file[input_key]
        output_data = self.file[output_key]

        input_shape = input_data.shape
        output_shape = output_data.shape

        print(f"\nLoading steady-state data:")
        print(f"  Input '{input_key}': {input_shape}")
        print(f"  Output '{output_key}': {output_shape}")

        # Determine number of samples
        n_total = input_shape[0]
        if num_samples is None:
            num_samples = n_total
        else:
            num_samples = min(num_samples, n_total)

        # Load data
        if len(input_shape) == 3:  # (n_samples, n_x, n_y)
            n_samples, n_x, n_y = input_shape
            input_field = input_data[:num_samples, :, :]
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")

        if len(output_shape) == 4:  # (n_samples, n_x, n_y, 1) or (n_samples, n_x, n_y, n_t)
            output_field = output_data[:num_samples, :, :, -1]  # Take last timestep/channel
        elif len(output_shape) == 3:  # (n_samples, n_x, n_y)
            output_field = output_data[:num_samples, :, :]
        else:
            raise ValueError(f"Unexpected output shape: {output_shape}")

        # Generate coordinate grids
        x_coords = np.linspace(0, 1, n_x)
        y_coords = np.linspace(0, 1, n_y)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')

        # Create X with shape (n_samples, n_x, n_y, 3)
        X = np.zeros((num_samples, n_x, n_y, 3))
        X[:, :, :, 0] = grid_x[np.newaxis, :, :]
        X[:, :, :, 1] = grid_y[np.newaxis, :, :]
        X[:, :, :, 2] = input_field

        # Create U with shape (n_samples, n_x, n_y, 1)
        U = output_field[..., np.newaxis]

        print(f"  Created X: {X.shape}, U: {U.shape}")
        print(f"  Input field range: [{input_field.min():.4f}, {input_field.max():.4f}]")
        print(f"  Output field range: [{output_field.min():.4f}, {output_field.max():.4f}]")

        # Convert to PDEDataset
        return PDEDataset(X, U,
                         normalize_output=normalize_output,
                         normalize_input=normalize_input,
                         normalize_coords=normalize_coords)


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
        im = axes[i, 3].imshow(y_ebm_std[i, ..., 0], cmap='viridis')
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