import torch
import torch.optim as optim
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

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
    def create_fno(config, pde_type=None):
        """
        Create FNO model from config based on spectral type and PDE type.

        Args:
            config: Configuration object
            pde_type: 'burgers', 'advection', 'diffusion_reaction', 'navier_stokes'
                     If None, will try to get from config
        """
        if pde_type is None:
            pde_type = getattr(config, 'pde_type', None)

        # For 1D PDEs (Burgers, Advection), use FNO1d
        if pde_type in ['burgers', 'advection']:
            from fno import FNO1d
            model = FNO1d(
                modes=config.fno_modes,
                width=config.fno_width,
                n_layers=config.fno_depth,
                in_channels=2,  # [x, input_field] for 1D
                out_channels=1
            )
            return model.to(config.device)

        # For 2D PDEs, use spectral type from config
        spectral = getattr(config, 'fno_spectral', 'Vanilla')

        if spectral == 'Factorized':
            from fno import FFNO2d
            model = FFNO2d(
                modes1=config.fno_modes,
                modes2=config.fno_modes,
                width=config.fno_width,
                n_layers=config.fno_depth,
                in_channels=3,
                out_channels=1
            )
        elif spectral == 'Vanilla':
            from fno import FNO2d
            model = FNO2d(
                modes1=config.fno_modes,
                modes2=config.fno_modes,
                width=config.fno_width,
                n_layers=config.fno_depth,
                in_channels=3,
                out_channels=1
            )
        elif spectral == 'Binned':
            # TODO: Implement BinnedFNO2d
            raise NotImplementedError("Binned spectral not yet implemented")
        elif spectral == 'Hybrid':
            # TODO: Implement HybridFNO2d
            raise NotImplementedError("Hybrid spectral not yet implemented")
        else:
            raise ValueError(f"Unknown spectral type: {spectral}")

        return model.to(config.device)

    @staticmethod
    def create_ebm(config):
        """Create EBM model from config based on architecture type."""
        from kanebm import EBM, ConvEnergyNet, MLPEnergyNet

        input_dim = getattr(config, 'ebm_input_dim', 4)
        use_cnn = getattr(config, 'ebm_use_cnn', False)
        use_kan = getattr(config, 'ebm_use_kan', False)

        print(f"[Factory] Creating EBM: use_cnn={use_cnn}, use_kan={use_kan}, input_dim={input_dim}")

        if use_cnn:
            # CNN-based EBM for 2D spatial data
            # Input shape will be (batch, channels, H, W) not flattened
            # Calculate input channels: u + x_coords + u_fno
            n_input_channels = getattr(config, 'ebm_n_input_channels', 3)  # x coordinate channels
            n_output_channels = getattr(config, 'ebm_n_output_channels', 1)  # u field channels

            # Total CNN input channels: u (1) + x (3) + u_fno (1) = 5 for typical 2D case
            in_channels = n_output_channels + n_input_channels + n_output_channels

            base_channels = getattr(config, 'ebm_base_channels', 64)
            num_blocks = getattr(config, 'ebm_num_blocks', 4)
            use_spatial_attn = getattr(config, 'ebm_use_spatial_attn', True)
            use_channel_attn = getattr(config, 'ebm_use_channel_attn', True)
            mlp_hidden = getattr(config, 'ebm_mlp_hidden', 256)

            print(f"[Factory] Building CNN-EBM: in_ch={in_channels} (u={n_output_channels} + x={n_input_channels} + u_fno={n_output_channels})")
            print(f"[Factory]   base_ch={base_channels}, blocks={num_blocks}")
            print(f"[Factory]   Spatial attn={use_spatial_attn}, Channel attn={use_channel_attn}")

            energy_net = ConvEnergyNet(
                in_channels=in_channels,
                base_channels=base_channels,
                num_blocks=num_blocks,
                use_spatial_attn=use_spatial_attn,
                use_channel_attn=use_channel_attn,
                dropout=0.1,
                mlp_hidden=mlp_hidden,
            )

            # Wrap in EBM with custom energy network
            model = EBM(
                energy_net=energy_net,
                input_dim=input_dim,  # Not used for CNN, but kept for compatibility
                condition_on_fno=True,
            )

        elif use_kan:
            # KAN-based EBM (legacy)
            from kanebm import KANEBM
            hidden_dim = getattr(config, 'ebm_hidden_dim', 64)
            num_layers = getattr(config, 'ebm_layers', 3)

            if isinstance(hidden_dim, int):
                hidden_dims = [hidden_dim] * (num_layers - 1)
            else:
                hidden_dims = hidden_dim

            print(f"[Factory] Building KAN-EBM: [{input_dim}] + {hidden_dims} + [1]")

            model = KANEBM(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                grid_size=getattr(config, 'grid_size', 5),
                spline_order=getattr(config, 'spline_order', 3),
            )

        else:
            # MLP-based EBM (default)
            hidden_dim = getattr(config, 'ebm_hidden_dim', 256)
            num_layers = getattr(config, 'ebm_layers', 4)

            if isinstance(hidden_dim, int):
                hidden_dims = [hidden_dim] * num_layers
            else:
                hidden_dims = hidden_dim

            print(f"[Factory] Building MLP-EBM: input={input_dim}, hidden={hidden_dims}")

            model = EBM(
                energy_net=None,  # Will create MLPEnergyNet internally
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                condition_on_fno=True,
                use_kan=False,
                use_attention=False,
                dropout=0.1,
                activation='gelu',
            )

        return model.to(config.device)

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

    @staticmethod
    def from_yaml(yaml_path: str, pde_type: str = None):
        """
        Load config from YAML file and flatten nested structure.

        Args:
            yaml_path: Path to YAML config file
            pde_type: Optional PDE type to apply PDE-specific overrides

        Returns:
            Config instance
        """
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Flatten nested structure
        flat_config = {}

        # FNO model
        if 'fno_model' in yaml_config:
            fno = yaml_config['fno_model']
            flat_config['fno_spectral'] = fno.get('spectral', 'Factorized')
            flat_config['fno_modes'] = fno.get('modes', 12)
            flat_config['fno_width'] = fno.get('width', 64)
            flat_config['fno_depth'] = fno.get('depth', 4)

        # EBM model
        if 'ebm_model' in yaml_config:
            ebm = yaml_config['ebm_model']
            flat_config['ebm_base'] = ebm.get('base', 'kan')
            flat_config['ebm_input_dim'] = ebm.get('input_dim', 4)
            flat_config['ebm_hidden_dim'] = ebm.get('hidden_dim', 64)
            flat_config['ebm_layers'] = ebm.get('num_layers', 3)

        # Optimizers (convert to format expected by Factory)
        if 'fno_optim' in yaml_config:
            opt = yaml_config['fno_optim']
            flat_config['fno_optimizer_config'] = {
                'type': opt.get('name', 'adamw'),
                'lr': opt.get('lr', 1e-3),
                'weight_decay': opt.get('weight_decay', 1e-4),
                'betas': opt.get('betas', [0.9, 0.999])
            }

        if 'ebm_optim' in yaml_config:
            opt = yaml_config['ebm_optim']
            flat_config['ebm_optimizer_config'] = {
                'type': opt.get('name', 'adam'),
                'lr': opt.get('lr', 1e-4),
                'weight_decay': opt.get('weight_decay', 1e-5),
                'betas': opt.get('betas', [0.9, 0.999])
            }

        # Schedulers
        if 'fno_scheduler' in yaml_config:
            sched = yaml_config['fno_scheduler']
            flat_config['fno_scheduler_config'] = {'type': sched.get('name', 'cosine_annealing')}
            for k, v in sched.items():
                if k != 'name':
                    flat_config['fno_scheduler_config'][k] = v

        if 'ebm_scheduler' in yaml_config:
            sched = yaml_config['ebm_scheduler']
            flat_config['ebm_scheduler_config'] = {'type': sched.get('name', 'cosine_annealing')}
            for k, v in sched.items():
                if k != 'name':
                    flat_config['ebm_scheduler_config'][k] = v

        # Training config
        if 'training' in yaml_config:
            train = yaml_config['training']
            flat_config.update({
                'device': train.get('device', 'cuda'),
                'seed': train.get('seed', 42),
                'fno_epochs': train.get('fno_epochs', 100),
                'ebm_epochs': train.get('ebm_epochs', 50),
                'batch_size': train.get('batch_size', 32),
                'train_ratio': train.get('train_ratio', 0.8),
                'val_ratio': train.get('val_ratio', 0.1),
                'max_samples': train.get('max_samples'),
                'enable_tracking': train.get('enable_tracking', True),
                'tracking_backend': train.get('tracking_backend', 'custom'),
                'log_dir': train.get('log_dir', './runs'),
                'checkpoint_dir': train.get('checkpoint_dir', './checkpoints'),
                'patience': train.get('fno_patience', 20)
            })

        # Data config
        if 'data' in yaml_config:
            data = yaml_config['data']
            flat_config.update({
                'pde_type': data.get('pde_type', 'burgers'),
                'data_path': data.get('data_path'),
                'nu_values': data.get('nu_values')
            })

        # Conformal Prediction config
        if 'conformal' in yaml_config:
            cp = yaml_config['conformal']
            flat_config.update({
                'conformal_enabled': cp.get('enabled', False),
                'conformal_alpha': cp.get('alpha', 0.05),
                'conformal_score_fn': cp.get('score_fn', 'l2'),
                'conformal_calibration_split': cp.get('calibration_split', 0.1)
            })
        else:
            # Default values
            flat_config.update({
                'conformal_enabled': False,
                'conformal_alpha': 0.05,
                'conformal_score_fn': 'l2',
                'conformal_calibration_split': 0.1
            })

        # Evidential Deep Learning config
        if 'evidential' in yaml_config:
            edl = yaml_config['evidential']
            flat_config.update({
                'evidential_enabled': edl.get('enabled', False),
                'evidential_nu_min': edl.get('nu_min', 0.01),
                'evidential_alpha_min': edl.get('alpha_min', 1.01),
                'evidential_beta_min': edl.get('beta_min', 0.01),
                'evidential_reg_weight': edl.get('reg_weight', 0.01),
                'evidential_reg_schedule': edl.get('reg_schedule', None)
            })
        else:
            # Default values
            flat_config.update({
                'evidential_enabled': False,
                'evidential_nu_min': 0.01,
                'evidential_alpha_min': 1.01,
                'evidential_beta_min': 0.01,
                'evidential_reg_weight': 0.01,
                'evidential_reg_schedule': None
            })

        # Apply PDE-specific overrides
        if pde_type and 'pde_configs' in yaml_config and pde_type in yaml_config['pde_configs']:
            overrides = yaml_config['pde_configs'][pde_type]
            if 'fno_model' in overrides:
                for k, v in overrides['fno_model'].items():
                    flat_config[f'fno_{k}'] = v
            if 'training' in overrides:
                flat_config.update(overrides['training'])

        return Config(flat_config)


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
    Configuration for baseline methods (MC Dropout, Ensemble, Bayesian, MLP-EBM).
    Cross-family comparison baselines.
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

        # Bayesian FNO
        'bayesian': {
            'prior_mean': 0.0,
            'prior_std': 1.0,
            'kl_weight': 0.01,  # Weight for KL divergence term in ELBO
            'n_samples': 1,  # Number of weight samples during training
            'n_samples_inference': 100,  # Number of samples during inference
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


def get_conformal_methods_configs():
    """
    Configuration for all 6 Conformal Prediction method variants.
    """
    return {
        # Method 1: Split Conformal (Baseline - Idea 5)
        'split_conformal': {
            'alpha': 0.05,  # 95% coverage
            'score_fn': 'l2',  # Options: 'l2', 'relative_l2', 'linf', 'pointwise'
            'calibration_split': 0.1,  # 10% of data for calibration
        },

        # Method 2: Full Conformal (Leave-One-Out)
        'full_conformal': {
            'alpha': 0.05,
            'score_fn': 'l2',
            'epochs': 50,  # Training epochs per LOO model
            'max_cal_samples': 100,  # Limit calibration set (expensive!)
        },

        # Method 3: Cross-Conformal (K-Fold)
        'cross_conformal': {
            'alpha': 0.05,
            'score_fn': 'l2',
            'k_folds': 5,  # Number of folds
            'epochs': 50,  # Training epochs per fold
        },

        # Method 4: CQR (Conformalized Quantile Regression)
        'cqr': {
            'alpha': 0.05,
            'quantiles': [0.025, 0.975],  # For 95% interval
            'calibration_split': 0.1,
            'epochs': 500,
            'lr': 1e-3,
        },

        # Method 5: Adaptive Conformal
        'adaptive_conformal': {
            'alpha': 0.05,
            'score_fn': 'l2',
            'kernel': 'rbf',  # Options: 'rbf', 'cosine'
            'bandwidth': 1.0,  # RBF kernel bandwidth
            'calibration_split': 0.1,
        },

        # Method 6: Mondrian Conformal
        'mondrian_conformal': {
            'alpha': 0.05,
            'score_fn': 'l2',
            'calibration_split': 0.1,
            'group_fn': 'input_magnitude',  # Options: 'input_magnitude', 'reynolds', 'custom'
            'n_groups': 3,  # Number of groups (e.g., low/medium/high)
        },
    }


def get_evidential_methods_configs():
    """
    Configuration for all 6 Evidential Deep Learning method variants.
    """
    return {
        # Method 1: DER (Deep Evidential Regression - NIG) - Baseline, Idea 13
        'der_nig': {
            'nu_min': 0.01,
            'alpha_min': 1.01,
            'beta_min': 0.01,
            'reg_weight': 0.01,
            'reg_schedule': {
                'start': 0.001,
                'end': 0.1,
                'warmup_epochs': 50,
            },
            'epochs': 500,
            'lr': 1e-3,
        },

        # Method 2: Improved DER (Tighter Evidence Bound)
        'improved_der': {
            'nu_min': 0.01,
            'alpha_min': 1.01,
            'beta_min': 0.01,
            'reg_weight': 0.01,  # For improved regularization
            'use_log_barrier': True,  # log(1 + error²) instead of linear
            'reg_schedule': {
                'start': 0.001,
                'end': 0.05,  # Lower than standard DER
                'warmup_epochs': 50,
            },
            'epochs': 500,
            'lr': 1e-3,
        },

        # Method 3: Prior Networks (Dirichlet)
        'prior_networks': {
            'n_bins': 50,  # Discretization bins
            'output_range': (-1, 1),  # Expected output range
            'use_reverse_kl': True,  # More robust
            'epochs': 500,
            'lr': 1e-3,
        },

        # Method 4: Posterior Networks (Normalizing Flow)
        'posterior_networks': {
            'n_flows': 4,  # Number of flow transformations
            'flow_hidden_dim': 64,  # Flow parameter dimension
            'n_samples': 100,  # Samples from flow
            'epochs': 500,
            'lr': 5e-4,  # Lower LR for stability
        },

        # Method 5: Natural Posterior Network (Natural NIG Parameterization)
        'natural_posterior': {
            'nu_min': 0.01,
            'alpha_min': 1.01,
            'beta_min': 0.01,
            'use_natural_params': True,  # η instead of (γ,ν,α,β)
            'reg_weight': 0.01,
            'epochs': 500,
            'lr': 1e-3,
        },

        # Method 6: Dirichlet Evidential Regression
        'dirichlet_evidential': {
            'n_bins': 50,  # Discretization bins
            'output_range': (-1, 1),
            'use_concentration': True,  # Predict concentrations
            'epochs': 500,
            'lr': 1e-3,
        },
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