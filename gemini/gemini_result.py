import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# --- 1. Model Definitions ---

class SpectralConv1d(nn.Module):
    """1D Fourier layer for FNO"""
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat))
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        # Apply FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", 
            x_ft[:, :, :self.modes], self.weights)
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    """Improved FNO architecture for 1D problems"""
    def __init__(self, modes=16, width=64):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Input lifting
        self.fc0 = nn.Linear(2, self.width)  # 2 = (x,t)
        
        # Fourier layers
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        
        # Regular layers
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        
        # Output layer
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.activation = torch.nn.GELU()
    
    def forward(self, x, t):
        # Combine inputs
        grid = torch.cat([x, t], dim=-1)
        batch_size = grid.shape[0]
        
        # Lift to higher dimension
        x = self.fc0(grid)
        x = x.permute(0, 2, 1)
        
        # Fourier layers
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activation(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.activation(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.activation(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        # Project back
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x

class FNO_Placeholder(nn.Module):
    """
    Placeholder for the FNO Network (Replaces u_PINN).
    Input: (x, t) -> Output: u_FNO (The deterministic mean)
    """
    def __init__(self, in_features=2, out_features=1, num_layers=5, hidden_units=50):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_units), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.Tanh()])
        layers.append(nn.Linear(hidden_units, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

class EBMPotential(nn.Module):
    """Improved EBM Potential Network"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # Deep architecture for energy modeling
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 3 = (u, x, t)
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.SiLU(),
            
            nn.Linear(hidden_dim//2, 1),
            nn.Softplus()  # Ensures positive energy output
        )
        
    def forward(self, u, x, t):
        inputs = torch.cat([u, x, t], dim=1)
        return self.net(inputs)

# --- 2. Core Functions ---

def physics_loss(fno_model, X_coll, alpha=0.01, boundary_conditions=None):
    """Enhanced physics loss for 1D Heat Equation with stability improvements"""
    x = X_coll[:, 0:1].requires_grad_(True)
    t = X_coll[:, 1:2].requires_grad_(True)
    u = fno_model(x, t)

    # Compute derivatives
    u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Heat equation residual: ∂u/∂t - α∂²u/∂x²
    residual = u_t - alpha * u_xx
    
    # Normalize residual for better training stability
    residual = residual / (torch.mean(torch.abs(u_t)) + 1e-8)
    
    # Base PDE loss
    pde_loss = torch.mean(residual**2)
    
    # Add boundary conditions if provided
    if boundary_conditions is not None:
        bc_loss = boundary_conditions(u, x, t)
        return pde_loss + bc_loss
        
    return pde_loss

def total_energy(u, x, t, u_fno_mean, v_ebm_potential, beta=1.0):
    """Enhanced total energy function with temperature and regularization
    
    Args:
        u: Current solution samples
        x: Spatial coordinates
        t: Time coordinates
        u_fno_mean: FNO predicted mean
        v_ebm_potential: EBM potential network
        beta: Inverse temperature (default=1.0)
    """
    # 1. Quadratic anchor term (data attachment)
    # Normalized to prevent numerical instability
    u_diff = u - u_fno_mean
    E_anchor = 0.5 * torch.sum(u_diff**2, dim=1, keepdim=True)
    E_anchor = E_anchor / (torch.mean(torch.abs(u_fno_mean)) + 1e-8)
    
    # 2. EBM potential correction
    E_potential = v_ebm_potential(u, x, t)
    
    # 3. Optional regularization term for smoothness
    if u.requires_grad:
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        E_smooth = 0.01 * torch.sum(u_x**2, dim=1, keepdim=True)
    else:
        E_smooth = 0.0
    
    # Combine terms with temperature scaling
    return beta * (E_anchor + E_potential + E_smooth)

def langevin_dynamics(u_init, X, fno_model, v_ebm_potential, steps=100, step_size=0.01):
    """Performs Langevin Dynamics to sample from p_model(u|X)."""
    u_k = u_init.clone().detach().requires_grad_(True)
    
    for _ in range(steps):
        # 1. Get the current FNO mean (fixed during EBM sampling)
        x, t = X[:, 0:1], X[:, 1:2]
        u_fno_mean = fno_model(x, t).detach()

        # 2. Calculate the total energy E(u, X)
        E = total_energy(u_k, x, t, u_fno_mean, v_ebm_potential)
        
        # 3. Calculate the gradient (Force)
        # Note: retain_graph=False is appropriate here as we're not backpropagating through MCMC steps
        grad_E = autograd.grad(E.sum(), u_k, retain_graph=False)[0]
        
        # 4. Langevin Update: u_k+1 = u_k - step_size * grad_E + noise
        noise = torch.randn_like(u_k) * np.sqrt(2 * step_size)
        u_k = u_k.detach() - step_size * grad_E + noise
        u_k.requires_grad_(True)

    return u_k.detach() # Return the final negative sample u^-

# --- 3. Training Loop ---

def train_approach2(fno_model, v_ebm_potential, data_loader, X_coll, alpha, config):
    
    # --- Stage 1: FNO Training (PINN Component) ---
    print("--- Stage 1: FNO (PINN) Training Started ---")
    optimizer_fno = optim.Adam(fno_model.parameters(), lr=config['lr_fno'])
    
    for epoch in range(config['epochs_fno']):
        for X_data, u_data in data_loader:
            optimizer_fno.zero_grad()
            
            loss_phys = physics_loss(fno_model, X_coll, alpha)
            u_pred = fno_model(X_data[:, 0:1], X_data[:, 1:2])
            loss_data = torch.mean((u_pred - u_data)**2)
            
            loss_total = loss_data + config['lambda_phys'] * loss_phys
            loss_total.backward()
            optimizer_fno.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"FNO Epoch {epoch+1}, Loss: {loss_total.item():.6f}")

    # Freeze FNO parameters after training
    for param in fno_model.parameters():
        param.requires_grad = False
    print("--- Stage 1: FNO Training Complete (Parameters Frozen) ---")

    # --- Stage 2: EBM Training (Uncertainty Component) ---
    print("--- Stage 2: EBM Potential V Training Started ---")
    optimizer_ebm = optim.Adam(v_ebm_potential.parameters(), lr=config['lr_ebm'])
    
    for epoch in range(config['epochs_ebm']):
        for X_data, u_data in data_loader:
            optimizer_ebm.zero_grad()
            x, t = X_data[:, 0:1], X_data[:, 1:2]
            
            # 1. Positive Samples (u^+ = u_data)
            u_fno_mean_fixed = fno_model(x, t).detach()
            E_pos = total_energy(u_data, x, t, u_fno_mean_fixed, v_ebm_potential)
            
            # 2. Negative Samples (u^-) via Langevin Dynamics
            # Initialize samples near the true data for Contrastive Divergence
            u_init = u_data + config['noise_init'] * torch.randn_like(u_data) 
            
            u_neg = langevin_dynamics(u_init, X_data, fno_model, v_ebm_potential, 
                                      steps=config['mcmc_steps'], step_size=config['mcmc_step_size'])
            
            E_neg = total_energy(u_neg, x, t, u_fno_mean_fixed, v_ebm_potential)
            
            # 3. EBM Contrastive Loss
            loss_ebm = torch.mean(E_pos) - torch.mean(E_neg)
            
            loss_ebm.backward()
            optimizer_ebm.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"EBM Epoch {epoch+1}, Loss: {loss_ebm.item():.6f}")
            
    print("--- Stage 2: EBM Training Complete ---")
    
    return fno_model, v_ebm_potential

# --- 4. Inference (Prediction) ---

def inference_approach2(X_test, fno_model, v_ebm_potential, N_samples, K, epsilon):
    """Generates probabilistic samples using Langevin Dynamics."""
    print(f"\n--- Inference Started (Sampling {N_samples} times) ---")
    
    # 1. Calculate the deterministic mean anchor (Fixed)
    x_test, t_test = X_test[:, 0:1], X_test[:, 1:2]
    u_mean_anchor = fno_model(x_test, t_test).detach()
    
    # Repeat the input for batch sampling
    X_test_repeated = X_test.repeat(N_samples, 1)
    
    # Initialize u near the anchor for batch MCMC
    u_init_batch = u_mean_anchor.repeat(N_samples, 1) + 0.01 * torch.randn(N_samples, 1)
    
    # 2. Run Langevin Dynamics (Inference Engine)
    # The 'langevin_dynamics' function already performs the full MCMC run
    u_samples = langevin_dynamics(u_init_batch, X_test_repeated, fno_model, v_ebm_potential, 
                                  steps=K, step_size=epsilon)
    
    print("--- Inference Complete ---")
    
    return u_samples, u_mean_anchor


# --- 5. Execution Example (Mock Data) ---

if __name__ == '__main__':
    # Configuration
    CONFIG = {
        'alpha': 0.01,
        'lr_fno': 1e-3, 'lr_ebm': 1e-4,
        'epochs_fno': 500, 'epochs_ebm': 500,
        'lambda_phys': 1.0, # PINN weight
        'mcmc_steps': 100, 'mcmc_step_size': 0.01,
        'noise_init': 0.1,
        'N_DATA': 1000, 'N_COLL': 10000, 'BATCH_SIZE': 64
    }
    
    # Generate mock data (Example: True solution + noise)
    X_data_mock = torch.rand(CONFIG['N_DATA'], 2) * 2.0
    u_true = torch.sin(np.pi * X_data_mock[:, 0:1]) * torch.exp(-CONFIG['alpha'] * np.pi**2 * X_data_mock[:, 1:2])
    u_data_mock = u_true + 0.1 * torch.randn(CONFIG['N_DATA'], 1) 
    
    X_coll_mock = torch.rand(CONFIG['N_COLL'], 2) * 2.0 
    
    data_set = TensorDataset(X_data_mock, u_data_mock)
    data_loader = DataLoader(data_set, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)

    # Initialize models
    fno_model = FNO_Placeholder()
    v_ebm_potential = EBMPotential()

    # 1. Training
    fno_model, v_ebm_potential = train_approach2(
        fno_model, v_ebm_potential, data_loader, X_coll_mock, CONFIG['alpha'], CONFIG
    )

    # 2. Inference
    X_test_point = torch.tensor([[0.5, 0.5]]) # Test point (x=0.5, t=0.5)
    N_INF_SAMPLES = 5000
    
    u_samples, u_anchor = inference_approach2(
        X_test_point, fno_model, v_ebm_potential, 
        N_INF_SAMPLES, K=200, epsilon=0.005 # Use more steps/smaller step size for stable inference
    )
    
    # 3. Result Interpretation
    print("\n--- Final Prediction Results ---")
    print(f"Test Point X = {X_test_point.tolist()[0]}")
    print(f"Deterministic Anchor (u_FNO): {u_anchor.item():.4f}")
    
    u_mean = u_samples.mean().item()
    u_std = u_samples.std().item()
    
    print(f"Probabilistic Mean (Sample Average): {u_mean:.4f}")
    print(f"Uncertainty (Sample Std Dev): {u_std:.4f}")
    
    # Interpretation: The true prediction is u_mean +/- 2*u_std
    print(f"95% Confidence Interval: ({u_mean - 2*u_std:.4f}, {u_mean + 2*u_std:.4f})")