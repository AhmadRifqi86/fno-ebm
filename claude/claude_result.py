import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SpectralConv2d(nn.Module):
    """Spectral convolution layer in Fourier space"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes (x-direction)
        self.modes2 = modes2  # Number of Fourier modes (y-direction)

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, 
                                   self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, 
                                   self.modes1, self.modes2, 2)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space"""
        # input: (batch, in_channel, x, y, 2), weights: (in_channel, out_channel, x, y, 2)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Compute FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels, 
                            x.size(-2), x.size(-1)//2 + 1, 2, 
                            device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] =             self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], 
                           self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] =             self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], 
                           self.weights2)
        
        # Convert back to complex
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        
        # Return to physical space
        x_out = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x_out


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D problems"""
    def __init__(self, modes1, modes2, width=64, num_layers=4):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        
        # Input projection
        self.fc0 = nn.Linear(3, self.width)  # (x, y, input_field)
        
        # Fourier layers
        self.conv_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(self.num_layers)
        ])
        
        # Local (non-spectral) connection
        self.w_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        """
        x: (batch, n_x, n_y, 3) where last dim is (x_coord, y_coord, input_field)
        returns: (batch, n_x, n_y, 1) - the solution field
        """
        # Lift to higher dimension
        x = self.fc0(x)  # (batch, n_x, n_y, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, n_x, n_y)
        
        # Fourier layers
        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        
        # Project to output
        x = x.permute(0, 2, 3, 1)  # (batch, n_x, n_y, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # (batch, n_x, n_y, 1)
        
        return x


def compute_pde_residual(u, x_coords, y_coords, pde_type='poisson'): #Ini masuk customs.py
    """
    Compute PDE residual for physics loss
    
    Args:
        u: solution field (batch, n_x, n_y, 1)
        x_coords: x coordinates (batch, n_x, n_y)
        y_coords: y coordinates (batch, n_x, n_y)
        pde_type: type of PDE to enforce
    
    Returns:
        residual: PDE residual (batch, n_x, n_y)
    """
    u = u.squeeze(-1)  # (batch, n_x, n_y)
    u.requires_grad_(True)
    
    # Compute gradients
    u_x = torch.autograd.grad(u.sum(), x_coords, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y_coords, create_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x.sum(), x_coords, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y_coords, create_graph=True)[0]
    
    if pde_type == 'poisson':
        # Poisson equation: Δu = f
        # Assuming f is encoded in the input
        laplacian = u_xx + u_yy
        residual = laplacian  # Should match forcing term
    elif pde_type == 'heat':
        # Heat equation: u_t - Δu = 0
        laplacian = u_xx + u_yy
        residual = laplacian
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
    
    return residual

class EBMPotential(nn.Module):
    """
    Energy-Based Model potential V(u, X) for uncertainty modeling
    """
    def __init__(self, input_dim=3, hidden_dims=[128, 256, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        
        # Output single energy value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, u, x):
        """
        Args:
            u: solution field (batch, n_x, n_y, 1)
            x: input coordinates (batch, n_x, n_y, 3)
        Returns:
            V: potential energy (batch,)
        """
        # Concatenate solution with coordinates
        combined = torch.cat([u, x], dim=-1)  # (batch, n_x, n_y, 4)
        
        # Flatten spatial dimensions
        batch_size = combined.shape[0]
        combined_flat = combined.reshape(batch_size, -1, combined.shape[-1])
        
        # Compute potential for each spatial point
        V_spatial = self.network(combined_flat)  # (batch, n_x*n_y, 1)
        
        # Aggregate over spatial dimensions
        V = V_spatial.mean(dim=1).squeeze(-1)  # (batch,)
        
        return V


class FNO_EBM(nn.Module):
    """
    Combined FNO-EBM model
    Total Energy: E(u, X) = 0.5 * ||u - u_FNO(X)||^2 + V(u, X)
    """
    def __init__(self, fno_model, ebm_potential):
        super().__init__()
        self.u_fno = fno_model
        self.V = ebm_potential
        
    def energy(self, u, x, u_fno=None):
        """
        Compute total energy E(u, X)
        
        Args:
            u: candidate solution (batch, n_x, n_y, 1)
            x: input coordinates (batch, n_x, n_y, 3)
            u_fno: pre-computed FNO solution (optional)
        Returns:
            E: total energy (batch,)
        """
        if u_fno is None:
            with torch.no_grad():
                u_fno = self.u_fno(x)
        
        # Quadratic term: anchors to FNO solution
        quadratic_term = 0.5 * torch.mean((u - u_fno)**2, dim=[1, 2, 3])
        
        # Potential term: captures uncertainty structure
        potential_term = self.V(u, x)
        
        return quadratic_term + potential_term
    
    def forward(self, x):
        """Direct FNO prediction"""
        return self.u_fno(x)


def train_fno_stage(fno_model, train_loader, num_epochs=100, 
                    lr=1e-3, lambda_phys=1.0, device='cuda'):
    """
    Stage 1: Train FNO for deterministic solution
    
    Args:
        fno_model: FNO network
        train_loader: DataLoader with (x, u_data) pairs
        num_epochs: number of training epochs
        lr: learning rate
        lambda_phys: weight for physics loss
        device: cuda or cpu
    """
    fno_model.to(device)
    optimizer = optim.Adam(fno_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    for epoch in range(num_epochs):
        fno_model.train()
        epoch_loss = 0.0
        
        for batch_idx, (x, u_data) in enumerate(train_loader):
            x = x.to(device)
            u_data = u_data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            u_pred = fno_model(x)
            
            # Data loss
            loss_data = torch.mean((u_pred - u_data)**2)
            
            # Physics loss (PDE residual)
            x_coords = x[..., 0].requires_grad_(True)
            y_coords = x[..., 1].requires_grad_(True)
            residual = compute_pde_residual(u_pred, x_coords, y_coords)
            loss_phys = torch.mean(residual**2)
            
            # Total loss
            loss = loss_data + lambda_phys * loss_phys
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # Freeze FNO parameters
    for param in fno_model.parameters():
        param.requires_grad = False
    
    return fno_model


# def langevin_dynamics(model, x, num_steps=100, step_size=0.01, 
#                      noise_scale=0.01, device='cuda'):
#     """
#     Langevin dynamics for sampling from energy-based model
    
#     Args:
#         model: FNO_EBM model
#         x: input coordinates (batch, n_x, n_y, 3)
#         num_steps: number of MCMC steps
#         step_size: step size epsilon
#         noise_scale: noise scale sqrt(2*epsilon)
    
#     Returns:
#         u_samples: samples from p(u|X)
#     """
#     # Initialize near positive samples or FNO solution
#     with torch.no_grad():
#         u = model.u_fno(x).clone()
    
#     u.requires_grad_(True)
    
#     for k in range(num_steps):
#         # Compute energy gradient
#         energy = model.energy(u, x).sum()
#         grad_u = torch.autograd.grad(energy, u, create_graph=False)[0]
        
#         with torch.no_grad():
#             # Langevin update
#             noise = torch.randn_like(u) * noise_scale
#             u = u - step_size * grad_u + noise
#             u.requires_grad_(True)
    
#     return u.detach()

def langevin_dynamics(model, x, num_steps=200, step_size=0.005, 
                     noise_scale=None, device='cuda'):
    """
    Langevin dynamics for inference sampling
    """
    if noise_scale is None:
        noise_scale = np.sqrt(2 * step_size)
    
    # Initialize from FNO solution
    with torch.no_grad():
        u = model.u_fno(x).clone()
    
    u.requires_grad_(True)
    
    for k in range(num_steps):
        # Compute energy gradient
        energy = model.energy(u, x).sum()
        grad_u = torch.autograd.grad(energy, u)[0]
        
        with torch.no_grad():
            # Langevin update
            noise = torch.randn_like(u) * noise_scale
            u = u - step_size * grad_u + noise
            
            # Optional: project to valid range if needed
            # u = torch.clamp(u, min_val, max_val)
            
            u.requires_grad_(True)
    
    return u.detach()


def train_ebm_stage(fno_ebm_model, train_loader, num_epochs=100, 
                   lr=1e-4, num_mcmc_steps=60, device='cuda'):
    """
    Stage 2: Train EBM potential V(u, X)
    
    Args:
        fno_ebm_model: Combined FNO-EBM model (FNO frozen)
        train_loader: DataLoader with (x, u_data) pairs
        num_epochs: number of training epochs
        lr: learning rate
        num_mcmc_steps: MCMC steps for negative sampling
        device: cuda or cpu
    """
    fno_ebm_model.to(device)
    
    # Only optimize EBM potential parameters
    optimizer = optim.Adam(fno_ebm_model.V.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        fno_ebm_model.train()
        epoch_loss = 0.0
        
        for batch_idx, (x, u_pos) in enumerate(train_loader):
            x = x.to(device)
            u_pos = u_pos.to(device)
            
            optimizer.zero_grad()
            
            # Positive energy (data)
            E_pos = fno_ebm_model.energy(u_pos, x).mean()
            
            # Generate negative samples via Langevin dynamics
            u_neg = langevin_dynamics(fno_ebm_model, x, 
                                     num_steps=num_mcmc_steps, 
                                     device=device)
            
            # Negative energy (generated samples)
            E_neg = fno_ebm_model.energy(u_neg, x).mean()
            
            # Contrastive divergence loss
            loss = E_pos - E_neg
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"EBM Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    return fno_ebm_model


def compute_pde_residual(u, x_coords, y_coords):
    """Simplified residual computation for training"""
    # This is a placeholder - implement your specific PDE
    return torch.zeros_like(u.squeeze(-1))


def inference_deterministic(model, x, device='cuda'):
    """
    Deterministic inference: return FNO prediction
    
    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
    
    Returns:
        u_mean: deterministic prediction (batch, n_x, n_y, 1)
    """
    model.eval()
    x = x.to(device)
    
    with torch.no_grad():
        u_mean = model.u_fno(x)
    
    return u_mean


def inference_probabilistic(model, x, num_samples=100, num_mcmc_steps=200,
                           step_size=0.005, device='cuda'):
    """
    Probabilistic inference: sample from p(u|X) using Langevin dynamics
    
    Args:
        model: FNO_EBM model
        x: input coordinates (batch, n_x, n_y, 3)
        num_samples: number of samples to draw
        num_mcmc_steps: MCMC steps per sample
        step_size: Langevin step size
    
    Returns:
        samples: ensemble of predictions (num_samples, batch, n_x, n_y, 1)
        stats: dictionary with mean, std, quantiles
    """
    model.eval()
    x = x.to(device)
    
    samples = []
    
    for i in range(num_samples):
        # Generate sample via Langevin dynamics
        u_sample = langevin_dynamics(
            model, x, 
            num_steps=num_mcmc_steps,
            step_size=step_size,
            device=device
        )
        samples.append(u_sample.cpu())
    
    samples = torch.stack(samples, dim=0)  # (num_samples, batch, n_x, n_y, 1)
    
    # Compute statistics
    stats = {
        'mean': samples.mean(dim=0),
        'std': samples.std(dim=0),
        'q05': samples.quantile(0.05, dim=0),
        'q95': samples.quantile(0.95, dim=0),
        'median': samples.median(dim=0).values
    }
    
    return samples, stats





def visualize_uncertainty(samples, x_test, save_path='uncertainty.png'):
    """
    Visualize uncertainty quantification results
    
    Args:
        samples: (num_samples, batch, n_x, n_y, 1)
        x_test: test coordinates
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt
    
    mean = samples.mean(dim=0)[0, ..., 0].numpy()
    std = samples.std(dim=0)[0, ..., 0].numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(mean, cmap='viridis')
    axes[0].set_title('Mean Prediction')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(std, cmap='hot')
    axes[1].set_title('Uncertainty (Std Dev)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_synthetic_data(n_samples=1000, grid_size=64):
    """
    Create synthetic PDE data for demonstration
    For example: Poisson equation with random forcing
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    X_data = []
    U_data = []
    
    for _ in range(n_samples):
        # Random forcing term
        forcing = np.random.randn(grid_size, grid_size) * 0.1
        
        # Solve Poisson equation (simplified - use actual solver in practice)
        u_solution = forcing + np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
        
        # Create input: (x, y, forcing)
        x_sample = np.stack([X, Y, forcing], axis=-1)
        
        X_data.append(x_sample)
        U_data.append(u_solution[..., np.newaxis])
    
    X_data = torch.FloatTensor(np.array(X_data))
    U_data = torch.FloatTensor(np.array(U_data))
    
    return X_data, U_data


def main():
    # Hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data parameters
    n_train = 800
    n_test = 200
    grid_size = 64
    batch_size = 16
    
    # Model parameters
    modes = 12  # Fourier modes
    width = 64  # FNO width
    
    # Training parameters
    fno_epochs = 100
    ebm_epochs = 50
    fno_lr = 1e-3
    ebm_lr = 1e-4
    
    print("\n=== Creating synthetic dataset ===")
    X_train, U_train = create_synthetic_data(n_train, grid_size)
    X_test, U_test = create_synthetic_data(n_test, grid_size)
    
    train_dataset = TensorDataset(X_train, U_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Grid size: {grid_size}x{grid_size}")
    
    # Initialize models
    print("\n=== Initializing models ===")
    fno_model = FNO2d(modes1=modes, modes2=modes, width=width, num_layers=4)
    ebm_potential = EBMPotential(input_dim=4, hidden_dims=[128, 256, 256, 128])
    
    print(f"FNO parameters: {sum(p.numel() for p in fno_model.parameters()):,}")
    print(f"EBM parameters: {sum(p.numel() for p in ebm_potential.parameters()):,}")
    
    # Stage 1: Train FNO
    print("\n=== Stage 1: Training FNO (Physics-Informed) ===")
    fno_model = train_fno_stage(
        fno_model, 
        train_loader, 
        num_epochs=fno_epochs,
        lr=fno_lr,
        lambda_phys=1.0,
        device=device
    )
    
    # Create combined model
    fno_ebm_model = FNO_EBM(fno_model, ebm_potential)
    
    # Stage 2: Train EBM
    print("\n=== Stage 2: Training EBM (Uncertainty) ===")
    fno_ebm_model = train_ebm_stage(
        fno_ebm_model,
        train_loader,
        num_epochs=ebm_epochs,
        lr=ebm_lr,
        num_mcmc_steps=60,
        device=device
    )
    
    # Inference
    print("\n=== Running inference on test data ===")
    
    # Deterministic prediction
    x_test_sample = X_test[:4].to(device)
    u_det = inference_deterministic(fno_ebm_model, x_test_sample, device)
    print(f"Deterministic prediction shape: {u_det.shape}")
    
    # Probabilistic prediction
    print("Generating probabilistic samples (this may take a while)...")
    samples, stats = inference_probabilistic(
        fno_ebm_model,
        x_test_sample,
        num_samples=50,
        num_mcmc_steps=100,
        device=device
    )
    
    print(f"Samples shape: {samples.shape}")
    print(f"Mean prediction shape: {stats['mean'].shape}")
    print(f"Std prediction shape: {stats['std'].shape}")
    
    # Compute metrics
    mse_det = torch.mean((u_det.cpu() - U_test[:4])**2).item()
    mse_prob = torch.mean((stats['mean'] - U_test[:4])**2).item()
    
    print(f"\nDeterministic MSE: {mse_det:.6f}")
    print(f"Probabilistic MSE: {mse_prob:.6f}")
    print(f"Mean uncertainty (std): {stats['std'].mean().item():.6f}")
    
    print("\n=== Training complete! ===")
    
    # Save models
    torch.save({
        'fno_state_dict': fno_model.state_dict(),
        'ebm_state_dict': ebm_potential.state_dict(),
    }, 'fno_ebm_model.pt')
    print("Model saved to 'fno_ebm_model.pt'")


if __name__ == '__main__':
    main()



# File to be created: Model.py, trainer.py,
# inference.py, datautils.py, customs.py, factory.py
