import torch

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

