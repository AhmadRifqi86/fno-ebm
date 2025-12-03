import torch
import torch.optim as optim
from typing import Dict, Any, Optional

class Factory:
    """
    Factory class for creating optimizers and schedulers from configuration.
    Supports dynamic instantiation based on string names and parameters.
    """

    # Mapping of optimizer names to their classes
    OPTIMIZER_REGISTRY = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta,
        'adamax': optim.Adamax,
    }

    # Mapping of scheduler names to their classes
    SCHEDULER_REGISTRY = {
        'step_lr': optim.lr_scheduler.StepLR,
        'multistep_lr': optim.lr_scheduler.MultiStepLR,
        'exponential_lr': optim.lr_scheduler.ExponentialLR,
        'cosine_annealing': optim.lr_scheduler.CosineAnnealingLR,
        'cosine_annealing_warm_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'reduce_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
        'cyclic_lr': optim.lr_scheduler.CyclicLR,
        'one_cycle_lr': optim.lr_scheduler.OneCycleLR,
    }

    @staticmethod
    def create_optimizer(optimizer_config: Dict[str, Any], model_parameters) -> optim.Optimizer:
        """
        Create an optimizer from configuration.

        Args:
            optimizer_config: Dictionary containing optimizer configuration
                Expected format: {
                    'type': 'adam',  # optimizer type (required)
                    'lr': 1e-4,      # learning rate (required)
                    'weight_decay': 0.01,  # optional parameters
                    ... other optimizer-specific parameters
                }
            model_parameters: Model parameters to optimize (from model.parameters())

        Returns:
            Initialized optimizer instance

        Raises:
            ValueError: If optimizer type is not supported or configuration is invalid
        """
        if not isinstance(optimizer_config, dict):
            raise ValueError("optimizer_config must be a dictionary")

        if 'type' not in optimizer_config:
            raise ValueError("optimizer_config must contain 'type' field")

        optimizer_type = optimizer_config['type'].lower()

        if optimizer_type not in Factory.OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unsupported optimizer type: {optimizer_type}. "
                f"Supported types: {list(Factory.OPTIMIZER_REGISTRY.keys())}"
            )

        # Get optimizer class
        optimizer_class = Factory.OPTIMIZER_REGISTRY[optimizer_type]

        # Extract parameters (exclude 'type' from config)
        optimizer_params = {k: v for k, v in optimizer_config.items() if k != 'type'}

        # Create and return optimizer
        return optimizer_class(model_parameters, **optimizer_params)

    @staticmethod
    def create_scheduler(
        scheduler_config: Dict[str, Any],
        optimizer: optim.Optimizer
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create a learning rate scheduler from configuration.

        Args:
            scheduler_config: Dictionary containing scheduler configuration
                Expected format: {
                    'type': 'cosine_annealing',  # scheduler type (required)
                    'T_max': 100,                # scheduler-specific parameters
                    ... other scheduler-specific parameters
                }
            optimizer: Optimizer instance to attach scheduler to

        Returns:
            Initialized scheduler instance, or None if scheduler_config is None/empty

        Raises:
            ValueError: If scheduler type is not supported or configuration is invalid
        """
        if not scheduler_config:
            return None

        if not isinstance(scheduler_config, dict):
            raise ValueError("scheduler_config must be a dictionary")

        if 'type' not in scheduler_config:
            raise ValueError("scheduler_config must contain 'type' field")

        scheduler_type = scheduler_config['type'].lower()

        # Check if it's a custom scheduler
        if scheduler_type == 'cosine_annealing_warm_restarts_with_decay':
            # Import custom scheduler
            from customs import CosineAnnealingWarmRestartsWithDecay
            scheduler_class = CosineAnnealingWarmRestartsWithDecay
        elif scheduler_type in Factory.SCHEDULER_REGISTRY:
            scheduler_class = Factory.SCHEDULER_REGISTRY[scheduler_type]
        else:
            raise ValueError(
                f"Unsupported scheduler type: {scheduler_type}. "
                f"Supported types: {list(Factory.SCHEDULER_REGISTRY.keys()) + ['cosine_annealing_warm_restarts_with_decay']}"
            )

        # Extract parameters (exclude 'type' from config)
        scheduler_params = {k: v for k, v in scheduler_config.items() if k != 'type'}

        # Create and return scheduler
        return scheduler_class(optimizer, **scheduler_params)

    @staticmethod
    def register_optimizer(name: str, optimizer_class):
        """
        Register a custom optimizer class.

        Args:
            name: Name to register the optimizer under
            optimizer_class: Optimizer class to register
        """
        Factory.OPTIMIZER_REGISTRY[name.lower()] = optimizer_class

    @staticmethod
    def register_scheduler(name: str, scheduler_class):
        """
        Register a custom scheduler class.

        Args:
            name: Name to register the scheduler under
            scheduler_class: Scheduler class to register
        """
        Factory.SCHEDULER_REGISTRY[name.lower()] = scheduler_class

class Config:
    """
    Configuration class for UQ paper experiments.
    Manages both FNO training and KAN-EBM training configurations.
    """
    def __init__(self, config_dict):
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Update attributes from the dictionary
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # For nested configs like schedulers
                setattr(self, key, value)
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"


# ============================================================================
# UQ Paper Configuration Templates
# ============================================================================

def get_fno_1d_burgers_config():
    """
    Configuration for FNO training on 1D Burgers equation.

    Dataset: 1D_Burgers_Sols_Nu{0.001, 0.004, 0.01, 0.1, 1.0}.hdf5
    Task: Single-step prediction u(t) -> u(t+1)
    """
    return {
        # Dataset
        'pde_type': '1d_burgers',
        'spatial_dim': 1,
        'nu_values': [0.001, 0.004, 0.01, 0.1, 1.0],
        'resolution': 256,
        'time_step': 1,
        'train_test_split': 0.8,

        # FNO Architecture
        'modes': 16,
        'width': 64,
        'n_layers': 4,
        'in_channels': 3,  # (x, t, u0)
        'out_channels': 1,  # u(t+1)

        # Training
        'batch_size': 32,
        'epochs': 500,
        'lr': 1e-3,
        'weight_decay': 1e-4,

        # Optimizer
        'optimizer': {
            'type': 'adam',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999)
        },

        # Scheduler
        'scheduler': {
            'type': 'reduce_on_plateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 20,
            'min_lr': 1e-6
        },

        # Loss
        'loss_type': 'mse',
        'gradient_penalty_weight': 0.1,  # For combating over-smoothing

        # Logging
        'log_interval': 10,
        'checkpoint_path': './checkpoints/fno_burgers.pt',
        'use_mlflow': True,
    }


def get_fno_1d_advection_config():
    """
    Configuration for FNO training on 1D Advection equation.

    Dataset: 1D_Advection_Sols_beta{0.1, 0.4, 1.0, 2.0, 4.0}.hdf5
    Task: Single-step prediction u(t) -> u(t+1)
    """
    return {
        # Dataset
        'pde_type': '1d_advection',
        'spatial_dim': 1,
        'beta_values': [0.1, 0.4, 1.0, 2.0, 4.0],
        'resolution': 256,
        'time_step': 1,
        'train_test_split': 0.8,

        # FNO Architecture
        'modes': 16,
        'width': 64,
        'n_layers': 4,
        'in_channels': 3,
        'out_channels': 1,

        # Training
        'batch_size': 32,
        'epochs': 500,
        'lr': 1e-3,
        'weight_decay': 1e-4,

        # Optimizer
        'optimizer': {
            'type': 'adam',
            'lr': 1e-3,
            'weight_decay': 1e-4
        },

        # Scheduler
        'scheduler': {
            'type': 'reduce_on_plateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 20,
            'min_lr': 1e-6
        },

        # Loss
        'loss_type': 'mse',
        'gradient_penalty_weight': 0.05,

        # Logging
        'log_interval': 10,
        'checkpoint_path': './checkpoints/fno_advection.pt',
        'use_mlflow': True,
    }


def get_fno_2d_diffusion_reaction_config():
    """
    Configuration for FNO training on 2D Diffusion-Reaction equation.

    Dataset: 2D_diff-react_NA_NA.h5
    Task: Single-step prediction u(t) -> u(t+1)
    Resolution: 128x128
    """
    return {
        # Dataset
        'pde_type': '2d_diffusion_reaction',
        'spatial_dim': 2,
        'resolution': 128,
        'time_step': 1,
        'train_test_split': 0.8,
        'n_channels': 2,  # Two species (u, v)

        # FNO Architecture
        'modes1': 12,
        'modes2': 12,
        'width': 32,
        'n_layers': 4,
        'in_channels': 5,  # (x, y, t, u0, v0)
        'out_channels': 1,  # u(t+1) only

        # Training
        'batch_size': 16,
        'epochs': 500,
        'lr': 1e-3,
        'weight_decay': 1e-4,

        # Optimizer
        'optimizer': {
            'type': 'adam',
            'lr': 1e-3,
            'weight_decay': 1e-4
        },

        # Scheduler
        'scheduler': {
            'type': 'reduce_on_plateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 20,
            'min_lr': 1e-6
        },

        # Loss
        'loss_type': 'mse',
        'gradient_penalty_weight': 0.15,  # Higher for sharp reaction fronts

        # Logging
        'log_interval': 10,
        'checkpoint_path': './checkpoints/fno_diff_react.pt',
        'use_mlflow': True,
    }


def get_fno_2d_navier_stokes_config():
    """
    Configuration for FNO training on 2D Navier-Stokes equation.

    Dataset: nsforcing_train_128.pt, nsforcing_test_128.pt
    Task: Single-step prediction x -> y
    Resolution: 128x128
    """
    return {
        # Dataset
        'pde_type': '2d_navier_stokes',
        'spatial_dim': 2,
        'resolution': 128,
        'train_samples': 10000,
        'test_samples': 2000,

        # FNO Architecture
        'modes1': 12,
        'modes2': 12,
        'width': 32,
        'n_layers': 4,
        'in_channels': 3,  # (x, y, forcing)
        'out_channels': 1,  # velocity field

        # Training
        'batch_size': 16,
        'epochs': 500,
        'lr': 1e-3,
        'weight_decay': 1e-4,

        # Optimizer
        'optimizer': {
            'type': 'adam',
            'lr': 1e-3,
            'weight_decay': 1e-4
        },

        # Scheduler
        'scheduler': {
            'type': 'reduce_on_plateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 20,
            'min_lr': 1e-6
        },

        # Loss
        'loss_type': 'mse',
        'gradient_penalty_weight': 0.2,  # High for turbulence
        'physics_loss_weight': 0.0,  # Can add physics loss if needed

        # Logging
        'log_interval': 10,
        'checkpoint_path': './checkpoints/fno_ns.pt',
        'use_mlflow': True,
    }


def get_kan_ebm_config(pde_type='1d_burgers'):
    """
    Configuration for KAN-EBM training on any PDE.

    Args:
        pde_type: One of ['1d_burgers', '1d_advection', '2d_diffusion_reaction', '2d_navier_stokes']

    Returns:
        Configuration dictionary for KAN-EBM training
    """
    # Base configuration
    config = {
        # Model
        'model_type': 'kan_ebm',
        'pde_type': pde_type,

        # KAN Architecture
        'kan_grid_size': 5,  # B-spline grid points
        'kan_spline_order': 3,  # Cubic splines
        'kan_layers': [64, 32, 16, 1],  # Hidden dims ending in scalar energy

        # Energy Model
        'condition_on_fno': True,  # E(u | x, û) vs E(u | x)
        'energy_reg_weight': 0.001,  # Regularization to prevent collapse

        # Score Matching
        'score_matching_type': 'weighted',  # 'standard' or 'weighted'
        'noise_levels': [0.01, 0.02, 0.05],
        'noise_weights': {0.01: 0.2, 0.02: 0.3, 0.05: 0.5},  # Inverse weighting

        # Contrastive Divergence (alternative to score matching)
        'use_contrastive_divergence': False,
        'langevin_steps_train': 20,
        'langevin_step_size_train': 0.01,

        # Calibration Loss
        'use_error_aware_loss': True,
        'calibration_weight': 0.5,  # Weight for error-aware loss

        # Training
        'batch_size': 16,
        'epochs': 200,
        'lr': 5e-4,
        'weight_decay': 1e-5,

        # Optimizer
        'optimizer': {
            'type': 'adam',
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'betas': (0.9, 0.999)
        },

        # Scheduler
        'scheduler': {
            'type': 'cosine_annealing',
            'T_max': 200,
            'eta_min': 1e-6
        },

        # Inference (Langevin sampling)
        'langevin_steps_inference': 50,
        'langevin_step_size_inference': 0.01,
        'n_samples': 100,  # Number of samples for uncertainty estimation

        # Logging
        'log_interval': 5,
        'checkpoint_path': f'./checkpoints/kan_ebm_{pde_type}.pt',
        'use_mlflow': True,
    }

    # PDE-specific adjustments
    if pde_type in ['1d_burgers', '1d_advection']:
        config['spatial_dim'] = 1
        config['input_encoding'] = 'patch'  # Patch-wise encoding for 1D
        config['patch_size'] = 16
    else:
        config['spatial_dim'] = 2
        config['input_encoding'] = 'global'  # Global pooling for 2D

    return config


def get_baseline_configs():
    """
    Configuration for baseline methods (MC Dropout, Ensemble, MLP-EBM).
    """
    return {
        # MC Dropout
        'mc_dropout': {
            'n_forward_passes': 30,
            'dropout_rate': 0.1,
            'dropout_positions': ['after_fourier', 'after_conv'],  # Where to place dropout
        },

        # Deep Ensemble
        'ensemble': {
            'n_models': 5,
            'seed_offset': 100,  # Different random seeds
            'aggregation': 'mean',  # How to combine predictions
        },

        # MLP-EBM (baseline for KAN-EBM)
        'mlp_ebm': {
            'mlp_layers': [128, 64, 32, 1],  # More parameters than KAN
            'activation': 'gelu',
            # Rest same as KAN-EBM
            'score_matching_type': 'weighted',
            'noise_levels': [0.01, 0.02, 0.05],
            'use_error_aware_loss': True,
            'calibration_weight': 0.5,
            'batch_size': 16,
            'epochs': 200,
            'lr': 5e-4,
        }
    }


def get_evaluation_config():
    """
    Configuration for evaluation metrics and test scenarios.
    """
    return {
        # Calibration Metrics
        'metrics': [
            'ece',  # Expected Calibration Error
            'nll',  # Negative Log-Likelihood
            'crps',  # Continuous Ranked Probability Score
            'coverage',  # Prediction interval coverage
            'sharpness',  # Prediction interval width
        ],

        # Calibration settings
        'confidence_levels': [0.68, 0.95, 0.99],  # 1σ, 2σ, 3σ
        'n_bins': 10,  # For ECE calculation

        # Test Scenarios
        'scenarios': {
            'in_distribution': True,
            'complexity_interpolation': True,  # 1D only
            'complexity_extrapolation': True,  # 1D only
            'spatial_uncertainty': True,  # 2D only
            'ood_detection': True,  # Corrupted inputs
        },

        # OOD Test
        'ood_noise_levels': [0.1, 0.5, 1.0],

        # Ablation Studies
        'ablations': {
            'kan_vs_mlp_iso_param': True,
            'kan_vs_mlp_iso_performance': True,
            'conditioning_ablation': True,  # E(u|x,û) vs E(u|x) vs E(u)
            'langevin_steps': [5, 10, 20, 50, 100],
        }
    }
