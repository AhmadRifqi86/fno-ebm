import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("Using official Mamba implementation from mamba-ssm")
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not available, falling back to custom implementation")
    print("Install with: pip install mamba-ssm")

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


class SpatialTransformerBlock(nn.Module):
    """Transformer block for spatial attention over 2D fields"""
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, embed_dim)
        Returns:
            x: (batch, num_patches, embed_dim)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerFNO2d(nn.Module):
    """
    Transformer-based Fourier Neural Operator for 2D problems.
    Replaces SpectralConv2d with spatial transformer attention.
    """
    def __init__(self, width=64, num_layers=4, num_heads=4, patch_size=4, dropout=0.0):
        super().__init__()

        self.width = width
        self.num_layers = num_layers
        self.patch_size = patch_size

        # Input projection
        self.fc0 = nn.Linear(3, self.width)  # (x, y, input_field)

        # Positional embedding parameters (learned)
        # Will be initialized based on input size in forward pass
        self.pos_embed = None
        self.grid_size = None

        # Transformer blocks (replacing SpectralConv2d layers)
        self.transformer_blocks = nn.ModuleList([
            SpatialTransformerBlock(
                embed_dim=self.width,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout
            )
            for _ in range(self.num_layers)
        ])

        # Local (non-attention) connection for residual
        self.w_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def initialize_pos_embedding(self, grid_h, grid_w, device):
        """Initialize positional embeddings if needed"""
        num_patches = grid_h * grid_w

        if self.pos_embed is None or self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)
            # Create learnable positional embeddings
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches, self.width, device=device) * 0.02
            )

    def forward(self, x):
        """
        x: (batch, n_x, n_y, 3) where last dim is (x_coord, y_coord, input_field)
        returns: (batch, n_x, n_y, 1) - the solution field
        """
        batch_size, n_x, n_y, _ = x.shape

        # Lift to higher dimension
        x = self.fc0(x)  # (batch, n_x, n_y, width)

        # Initialize positional embeddings if needed
        self.initialize_pos_embedding(n_x, n_y, x.device)

        # For transformer processing: flatten spatial dimensions
        x_flat = x.reshape(batch_size, n_x * n_y, self.width)  # (batch, n_x*n_y, width)

        # Add positional embeddings
        x_flat = x_flat + self.pos_embed

        # For residual connection: prepare in conv format
        x_conv = x.permute(0, 3, 1, 2)  # (batch, width, n_x, n_y)

        # Apply transformer blocks
        for i in range(self.num_layers):
            # Transformer path (global attention)
            x_attn = self.transformer_blocks[i](x_flat)  # (batch, n_x*n_y, width)

            # Reshape for residual
            x_attn_2d = x_attn.reshape(batch_size, n_x, n_y, self.width)
            x_attn_2d = x_attn_2d.permute(0, 3, 1, 2)  # (batch, width, n_x, n_y)

            # Local path (pointwise convolution)
            x_local = self.w_layers[i](x_conv)

            # Combine paths
            x_conv = x_attn_2d + x_local

            # Activation (except last layer)
            if i < self.num_layers - 1:
                x_conv = F.gelu(x_conv)
                # Update flat representation
                x_flat = x_conv.permute(0, 2, 3, 1).reshape(batch_size, n_x * n_y, self.width)

        # Project to output
        x = x_conv.permute(0, 2, 3, 1)  # (batch, n_x, n_y, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # (batch, n_x, n_y, 1)

        return x


class EBMPotential(nn.Module):
    """
    Energy-Based Model potential V(u, X) for uncertainty modeling
    """
    def __init__(self, input_dim=4, hidden_dims=[128, 256, 256, 128]):
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
    def __init__(self, fno_model, ebm_model):
        super().__init__()
        self.u_fno = fno_model
        self.V_ebm = ebm_model
        
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
        potential_term = self.V_ebm(u, x)
        
        return quadratic_term + potential_term
    
    def forward(self, x):
        """Direct FNO prediction"""
        return self.u_fno(x)