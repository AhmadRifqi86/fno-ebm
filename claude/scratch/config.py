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
    Configuration class for all training and model parameters.
    Loads parameters from a dictionary.
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

