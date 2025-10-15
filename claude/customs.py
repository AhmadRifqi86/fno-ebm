
import numpy as np
import math
import torch
from traitlets import Callable


PhysicsLossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def compute_pde_residual(u, x_coords, y_coords, pde_type='poisson'):
    """
    Compute PDE residual for physics loss using finite differences.

    This implementation uses finite difference approximations instead of autograd
    because the coordinates are input features, not differentiable variables.

    Args:
        u: solution field (batch, n_x, n_y, 1)
        x_coords: x coordinates (batch, n_x, n_y) - not used, assumes uniform grid
        y_coords: y coordinates (batch, n_x, n_y) - not used, assumes uniform grid
        pde_type: type of PDE to enforce

    Returns:
        residual: PDE residual (batch, n_x-2, n_y-2) - smaller due to boundary removal
    """
    u = u.squeeze(-1)  # (batch, n_x, n_y)

    # Compute grid spacing (assumes uniform grid from 0 to 1)
    n_x = u.shape[1]
    n_y = u.shape[2]
    dx = 1.0 / (n_x - 1)
    dy = 1.0 / (n_y - 1)

    # Compute second derivatives using central finite differences
    # u_xx = (u[i-1,j] - 2*u[i,j] + u[i+1,j]) / dx^2
    # u_yy = (u[i,j-1] - 2*u[i,j] + u[i,j+1]) / dy^2

    # Second derivative in x direction (batch, n_x-2, n_y)
    u_xx = (u[:, 2:, :] - 2*u[:, 1:-1, :] + u[:, :-2, :]) / (dx**2)

    # Second derivative in y direction (batch, n_x, n_y-2)
    u_yy = (u[:, :, 2:] - 2*u[:, :, 1:-1] + u[:, :, :-2]) / (dy**2)

    # Align dimensions by removing boundaries
    # u_xx has shape (batch, n_x-2, n_y), take middle n_y-2 in y
    u_xx = u_xx[:, :, 1:-1]  # (batch, n_x-2, n_y-2)

    # u_yy has shape (batch, n_x, n_y-2), take middle n_x-2 in x
    u_yy = u_yy[:, 1:-1, :]  # (batch, n_x-2, n_y-2)

    if pde_type == 'poisson':
        # Poisson equation: Δu = f
        # For now, we assume f=0 (Laplace equation) or minimize |Δu|
        laplacian = u_xx + u_yy
        residual = laplacian
    elif pde_type == 'heat':
        # Heat equation: u_t - Δu = 0
        # For steady state: -Δu = 0
        laplacian = u_xx + u_yy
        residual = laplacian
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")

    return residual

class CosineAnnealingWarmRestartsWithDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1.0, freq_mult=1.0, eta_min=0, decay=0.9, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay = decay          # Decay factor for max LR
        self.freq_mult = freq_mult  # Multiplier for cycle length (e.g., 0.9 for shorter cycles)
        self.base_lrs = None #[5e-5]#None  # lazy init
        self.current_max_lrs = None #[5e-5]#None
        self.T_i = T_0
        self.cycle = 0
        self.epoch_since_restart = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.base_lrs is None or self.current_max_lrs is None:
            self.base_lrs = [group['initial_lr'] if 'initial_lr' in group else group['lr']
                             for group in self.optimizer.param_groups]
            self.current_max_lrs = self.base_lrs.copy()
            #print("Initialized base_lrs:", self.base_lrs)
        # Standard cosine annealing formula, but with decaying max LR
        return [
            self.eta_min + (max_lr - self.eta_min) * (1 + torch.cos(torch.tensor(self.epoch_since_restart * math.pi/ self.T_i))) / 2
            for max_lr in self.current_max_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.epoch_since_restart += 1
        if self.epoch_since_restart >= self.T_i:
            self.cycle += 1
            self.epoch_since_restart = 0
            self.T_i = int(self.T_i * self.freq_mult) #max(1.0, self.T_i * self.freq_mult) #dipaksa turun berarti cycle nya
            self.current_max_lrs = [
                base_lr * (self.decay ** self.cycle)
                for base_lr in self.base_lrs
            ]
            for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, self.current_max_lrs)):
                name = group.get('name', f'group_{i}')
                print(f"[{name}] Decayed max LR: {lr}")
            print(f" T_i={self.T_i}")

        # Apply the new learning rates to param groups
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

        # ✅ Required for PyTorch's SequentialLR compatibility
        self._last_lr = lrs


class EarlyStoppingOld:
    """Early stopping utility to prevent overfitting"""
    def __init__(self, patience=20, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_best_model(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            return True
        return False


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            trace_func (function): function to use for printing messages.
                                   Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0


class GradientClipperWithNormTracking:
    """
    Advanced gradient clipping with gradient norm tracking
    """
    def __init__(self, max_norm=1.0, norm_type=2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.grad_norms = []
        
    def clip_gradients(self, model):
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, self.norm_type
        )
        self.grad_norms.append(total_norm.item())
        return total_norm
    
    def get_stats(self):
        if not self.grad_norms:
            return {}
        return {
            'grad_norm_mean': np.mean(self.grad_norms[-100:]),
            'grad_norm_std': np.std(self.grad_norms[-100:]),
            'grad_norm_max': max(self.grad_norms[-100:])
        }