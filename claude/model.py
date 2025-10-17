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


class FourierTransformerBlock(nn.Module):
    """
    Hybrid block that combines Fourier filtering with transformer attention
    """
    def __init__(self, channels, modes1, modes2, num_heads=4, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Fourier weights for spectral filtering
        self.scale = 1 / (channels * channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(channels, channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(channels, channels, modes1, modes2, 2)
        )

        # Transformer attention in Fourier space
        self.norm = nn.LayerNorm(channels * 2)  # *2 for real and imag parts
        self.attn = nn.MultiheadAttention(
            channels * 2, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels * 2),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(channels * 2)

    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) - spatial domain
        Returns:
            (batch, channels, height, width) - spatial domain
        """
        batch_size = x.shape[0]

        # Transform to Fourier domain
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        # Spectral filtering (similar to SpectralConv2d)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Flatten Fourier coefficients for transformer attention
        # (batch, channels, freq_h, freq_w, 2) -> (batch, freq_h*freq_w, channels*2)
        freq_h, freq_w = out_ft.shape[2], out_ft.shape[3]
        out_ft_flat = out_ft.permute(0, 2, 3, 1, 4).reshape(
            batch_size, freq_h * freq_w, self.channels * 2
        )

        # Apply transformer attention in Fourier space
        out_ft_norm = self.norm(out_ft_flat)
        attn_out, _ = self.attn(out_ft_norm, out_ft_norm, out_ft_norm)
        out_ft_flat = out_ft_flat + attn_out
        out_ft_flat = out_ft_flat + self.mlp(self.norm2(out_ft_flat))

        # Reshape back to Fourier space
        out_ft = out_ft_flat.reshape(batch_size, freq_h, freq_w, self.channels, 2)
        out_ft = out_ft.permute(0, 3, 1, 2, 4)

        # Transform back to spatial domain
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x_out = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)), norm='ortho')

        return x_out


class FourierTransformerFNO2d(nn.Module):
    """
    Fourier Neural Operator with Transformer attention in Fourier space.
    Combines spectral filtering with self-attention on Fourier coefficients.
    """
    def __init__(self, modes1, modes2, width=64, num_layers=4, num_heads=4, dropout=0.0):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers

        # Input projection
        self.fc0 = nn.Linear(3, self.width)

        # Fourier-Transformer hybrid layers
        self.fourier_transformer_layers = nn.ModuleList([
            FourierTransformerBlock(
                self.width, self.modes1, self.modes2,
                num_heads=num_heads, dropout=dropout
            )
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

        # Fourier-Transformer layers
        for i in range(self.num_layers):
            x1 = self.fourier_transformer_layers[i](x)
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


class MambaBlock2D(nn.Module):
    """
    Mamba-based block for 2D spatial processing
    Applies Mamba along both spatial dimensions
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        if MAMBA_AVAILABLE:
            # Use official Mamba implementation
            self.mamba_h = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.mamba_w = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # Fallback to simple convolution if Mamba not available
            print("Warning: Using Conv2d fallback instead of Mamba")
            self.mamba_h = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            self.mamba_w = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            (batch, channels, height, width)
        """
        batch, channels, height, width = x.shape

        if MAMBA_AVAILABLE:
            # Process along height dimension
            x_h = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
            x_h = x_h.reshape(batch * height, width, channels)
            x_h = self.mamba_h(x_h)
            x_h = x_h.reshape(batch, height, width, channels)
            x_h = x_h.permute(0, 3, 1, 2)  # (batch, channels, height, width)

            # Process along width dimension
            x_w = x.permute(0, 3, 2, 1)  # (batch, width, height, channels)
            x_w = x_w.reshape(batch * width, height, channels)
            x_w = self.mamba_w(x_w)
            x_w = x_w.reshape(batch, width, height, channels)
            x_w = x_w.permute(0, 3, 2, 1)  # (batch, channels, height, width)

            # Combine both directions
            return x_h + x_w
        else:
            # Fallback: simple convolution
            return self.mamba_h(x) + self.mamba_w(x)


class MambaFNO2d(nn.Module):   #Operate on Spatial Domain
    """
    Mamba-based Fourier Neural Operator for 2D problems.
    Uses Mamba state space models for efficient spatial processing.
    """
    def __init__(self, modes1, modes2, width=64, num_layers=4, d_state=16):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers

        # Input projection
        self.fc0 = nn.Linear(3, self.width)

        # Spectral convolution layers (for Fourier processing)
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(self.num_layers)
        ])

        # Mamba layers (for spatial processing)
        self.mamba_layers = nn.ModuleList([
            MambaBlock2D(self.width, d_state=d_state)
            for _ in range(self.num_layers)
        ])

        # Local connection
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

        # Hybrid Fourier-Mamba layers
        for i in range(self.num_layers):
            # Spectral path (global Fourier)
            x1 = self.spectral_layers[i](x)

            # Mamba path (sequential spatial processing)
            x2 = self.mamba_layers[i](x)

            # Local path (pointwise)
            x3 = self.w_layers[i](x)

            # Combine all paths
            x = x1 + x2 + x3

            if i < self.num_layers - 1:
                x = F.gelu(x)

        # Project to output
        x = x.permute(0, 2, 3, 1)  # (batch, n_x, n_y, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # (batch, n_x, n_y, 1)

        return x


class FourierMambaBlock(nn.Module): #Operate on Frequency Domain
    """
    Hybrid block that combines Fourier filtering with Mamba in frequency domain
    """
    def __init__(self, channels, modes1, modes2, d_state=16, d_conv=4):
        super().__init__()
        self.channels = channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Fourier weights for spectral filtering
        self.scale = 1 / (channels * channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(channels, channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(channels, channels, modes1, modes2, 2)
        )

        # Mamba for processing in Fourier space
        if MAMBA_AVAILABLE:
            # Process Fourier coefficients as sequences
            # Input will be flattened frequency domain (real + imag)
            self.mamba_freq = Mamba(
                d_model=channels * 2,  # *2 for real and imag
                d_state=d_state,
                d_conv=d_conv,
                expand=2
            )
        else:
            # Fallback: 1D convolution over frequency sequence
            print("Warning: Using Conv1d fallback for frequency domain Mamba")
            self.mamba_freq = nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1)

        self.norm = nn.LayerNorm(channels * 2)

    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) - spatial domain
        Returns:
            (batch, channels, height, width) - spatial domain
        """
        batch_size = x.shape[0]

        # Transform to Fourier domain
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)  # (batch, channels, freq_h, freq_w, 2)

        # Spectral filtering (similar to SpectralConv2d)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Flatten Fourier coefficients for Mamba processing
        # (batch, channels, freq_h, freq_w, 2) -> (batch, freq_h*freq_w, channels*2)
        freq_h, freq_w = out_ft.shape[2], out_ft.shape[3]
        out_ft_flat = out_ft.permute(0, 2, 3, 1, 4).reshape(
            batch_size, freq_h * freq_w, self.channels * 2
        )

        # Apply Mamba in Fourier space
        if MAMBA_AVAILABLE:
            out_ft_norm = self.norm(out_ft_flat)
            out_ft_mamba = self.mamba_freq(out_ft_norm)
            out_ft_flat = out_ft_flat + out_ft_mamba  # Residual connection
        else:
            # Fallback: Conv1d
            out_ft_norm = self.norm(out_ft_flat)
            out_ft_conv = out_ft_norm.permute(0, 2, 1)  # (batch, channels*2, freq_h*freq_w)
            out_ft_conv = self.mamba_freq(out_ft_conv)
            out_ft_flat = out_ft_flat + out_ft_conv.permute(0, 2, 1)

        # Reshape back to Fourier space
        # (batch, freq_h*freq_w, channels*2) -> (batch, channels, freq_h, freq_w, 2)
        out_ft = out_ft_flat.reshape(batch_size, freq_h, freq_w, self.channels, 2)
        out_ft = out_ft.permute(0, 3, 1, 2, 4)

        # Transform back to spatial domain
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x_out = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)), norm='ortho')

        return x_out


class FourierMambaFNO2d(nn.Module):  # Operate on Frequency Domain
    """
    Fourier Neural Operator with Mamba processing in frequency domain.
    Combines FFT spectral filtering with Mamba state space models on Fourier coefficients.

    This architecture:
    1. Transforms input to Fourier space (FFT)
    2. Applies spectral filtering on selected modes
    3. Uses Mamba to process Fourier coefficients as sequences
    4. Transforms back to spatial domain (IFFT)
    """
    def __init__(self, modes1, modes2, width=64, num_layers=4, d_state=16):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers

        # Input projection
        self.fc0 = nn.Linear(3, self.width)

        # Fourier-Mamba hybrid layers
        self.fourier_mamba_layers = nn.ModuleList([
            FourierMambaBlock(
                self.width, self.modes1, self.modes2,
                d_state=d_state
            )
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

        # Fourier-Mamba layers
        for i in range(self.num_layers):
            # Fourier-Mamba path (frequency domain processing)
            x1 = self.fourier_mamba_layers[i](x)

            # Local path (spatial domain)
            x2 = self.w_layers[i](x)

            # Combine paths
            x = x1 + x2

            if i < self.num_layers - 1:
                x = F.gelu(x)

        # Project to output
        x = x.permute(0, 2, 3, 1)  # (batch, n_x, n_y, width)
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

# Class that called by trainer
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


# ============================================================================
# KAN-based EBM Architectures
# ============================================================================

class KANLayer(nn.Module):  #KAN from scratch, consider to use pykan library
    """
    Kolmogorov-Arnold Network Layer
    Uses learnable spline-based activation functions on edges instead of nodes.

    Based on the KAN paper: https://arxiv.org/abs/2404.19756
    This is a simplified implementation using B-splines.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Learnable spline coefficients for each input-output pair
        # Shape: (out_features, in_features, grid_size + spline_order)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * 0.1
        )

        # Base weights (linear transformation as residual)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # Grid points for spline basis (fixed, not learned)
        grid = torch.linspace(-1, 1, grid_size + 1)
        # Extend grid for spline order
        grid = torch.cat([
            grid[0].repeat(spline_order),
            grid,
            grid[-1].repeat(spline_order)
        ])
        self.register_buffer('grid', grid)

    def b_splines(self, x):
        """
        Compute B-spline basis functions
        Args:
            x: input tensor (..., in_features)
        Returns:
            basis: B-spline basis (..., in_features, grid_size + spline_order)
        """
        # Clamp x to grid range
        x = torch.clamp(x, -1.0, 1.0)

        # Compute B-spline basis using Cox-de Boor recursion
        # Simplified version: using linear interpolation for efficiency
        grid_size = self.grid_size
        spline_order = self.spline_order

        # Find the interval
        # x: (..., in_features) -> (..., in_features, 1)
        x_expanded = x.unsqueeze(-1)

        # grid: (grid_size + 2*spline_order + 1)
        # Create basis functions
        basis = torch.relu(1 - torch.abs(
            (x_expanded - self.grid) * grid_size / 2
        ))

        return basis

    def forward(self, x):
        """
        Args:
            x: (..., in_features)
        Returns:
            (..., out_features)
        """
        # Base linear transformation
        base_output = F.linear(x, self.base_weight)  # (..., out_features)

        # Spline transformation
        # Compute B-spline basis
        basis = self.b_splines(x)  # (..., in_features, grid_size + spline_order)

        # Apply spline weights
        # spline_weight: (out_features, in_features, grid_size + spline_order)
        # basis: (..., in_features, grid_size + spline_order)
        # We want: (..., out_features)

        spline_output = torch.einsum(
            '...ik,oik->...o',
            basis,
            self.spline_weight
        )

        return base_output + spline_output


class KAN_EBM(nn.Module):
    """
    Pure KAN-based Energy Model
    Uses Kolmogorov-Arnold Networks with learnable spline activations
    for modeling energy landscapes.

    Architecture:
        Input field → Flatten → KAN layers → Scalar energy

    Advantages:
    - Learnable activation functions (splines) for flexible energy surfaces
    - Potentially more interpretable than black-box MLPs
    - Efficient parameterization with fewer parameters
    """
    def __init__(
        self,
        input_dim=4,
        hidden_dims=[64, 128, 64],
        grid_size=5,
        spline_order=3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build KAN layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(KANLayer(
                prev_dim,
                hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order
            ))
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        # Final layer to scalar energy
        layers.append(KANLayer(
            prev_dim,
            1,
            grid_size=grid_size,
            spline_order=spline_order
        ))

        self.kan_network = nn.Sequential(*layers)

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
        combined_flat = combined.reshape(batch_size, -1, self.input_dim)

        # Compute energy for each spatial point
        V_spatial = self.kan_network(combined_flat)  # (batch, n_x*n_y, 1)

        # Aggregate over spatial dimensions (mean pooling)
        V = V_spatial.mean(dim=1).squeeze(-1)  # (batch,)

        return V


class FNO_KAN_EBM(nn.Module):
    """
    Hybrid FNO Encoder + KAN Energy Head

    Uses FNO's strength for learning spatial field features,
    then feeds compressed features to KAN for energy prediction.

    Architecture:
        Input field → FNO encoder → Adaptive pooling → KAN → Scalar energy

    This is the recommended approach combining:
    - FNO's ability to capture spatial/spectral structure
    - KAN's flexible learnable activations for energy modeling
    """
    def __init__(
        self,
        fno_modes1=12,
        fno_modes2=12,
        fno_width=64,
        fno_layers=3,
        pool_size=4,
        kan_hidden_dims=[128, 64],
        grid_size=5,
        spline_order=3
    ):
        super().__init__()

        self.fno_width = fno_width
        self.pool_size = pool_size

        # FNO encoder (feature extraction)
        self.fc0 = nn.Linear(4, fno_width)  # (u, x, y, input_field) = 4

        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(fno_width, fno_width, fno_modes1, fno_modes2)
            for _ in range(fno_layers)
        ])

        self.w_layers = nn.ModuleList([
            nn.Conv2d(fno_width, fno_width, 1)
            for _ in range(fno_layers)
        ])

        # Adaptive pooling to reduce spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # KAN energy head
        kan_input_dim = fno_width * pool_size * pool_size

        kan_layers = []
        prev_dim = kan_input_dim

        for hidden_dim in kan_hidden_dims:
            kan_layers.append(KANLayer(
                prev_dim,
                hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order
            ))
            kan_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        # Final energy output
        kan_layers.append(KANLayer(prev_dim, 1, grid_size=grid_size, spline_order=spline_order))

        self.kan_head = nn.Sequential(*kan_layers)

    def forward(self, u, x):
        """
        Args:
            u: solution field (batch, n_x, n_y, 1)
            x: input coordinates (batch, n_x, n_y, 3)
        Returns:
            V: potential energy (batch,)
        """
        # Combine solution and coordinates
        combined = torch.cat([u, x], dim=-1)  # (batch, n_x, n_y, 4)

        # FNO encoding
        features = self.fc0(combined)  # (batch, n_x, n_y, fno_width)
        features = features.permute(0, 3, 1, 2)  # (batch, fno_width, n_x, n_y)

        # Apply FNO layers to extract spatial features
        for i, (spectral, w) in enumerate(zip(self.spectral_layers, self.w_layers)):
            x1 = spectral(features)
            x2 = w(features)
            features = x1 + x2
            features = F.gelu(features)

        # Pool to fixed size
        features = self.adaptive_pool(features)  # (batch, fno_width, pool_size, pool_size)

        # Flatten for KAN
        features_flat = features.flatten(1)  # (batch, fno_width * pool_size^2)

        # KAN energy prediction
        energy = self.kan_head(features_flat)  # (batch, 1)
        energy = energy.squeeze(-1)  # (batch,)

        return energy


# ============================================================================
# Graph Neural Network-based EBM
# ============================================================================

class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolution Layer for spatial fields
    Implements message passing: h_i' = σ(W_self * h_i + Σ_j W_neighbor * h_j)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Separate weights for self and neighbor aggregation
        self.weight_self = nn.Linear(in_features, out_features)
        self.weight_neighbor = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, node_features, adjacency):
        """
        Args:
            node_features: (batch, num_nodes, in_features)
            adjacency: (batch, num_nodes, num_nodes) adjacency matrix
        Returns:
            (batch, num_nodes, out_features)
        """
        # Self transformation
        h_self = self.weight_self(node_features)

        # Neighbor aggregation
        h_neighbor = self.weight_neighbor(node_features)

        # Message passing: aggregate from neighbors
        # adjacency @ h_neighbor sums features from neighbors
        h_agg = torch.bmm(adjacency, h_neighbor)

        # Combine self and neighbor information
        out = h_self + h_agg
        out = self.norm(out)
        out = F.gelu(out)

        return out


class GNN_EBM(nn.Module):
    """
    Graph Neural Network-based Energy Model

    Treats spatial field as a graph where each grid point is a node.
    Uses message passing to capture local and global spatial interactions.

    Architecture:
        Field → Graph (nodes=grid points) → GNN layers → Global pooling → Energy

    Advantages:
    - Natural representation of spatial structure
    - Permutation invariance (with proper pooling)
    - Proven effective for molecular energy prediction (analogous problem)
    - Captures multi-scale interactions through message passing
    """
    def __init__(
        self,
        node_features=4,  # (u, x, y, input)
        hidden_dims=[64, 128, 128, 64],
        use_8_connected=False
    ):
        super().__init__()

        self.node_features = node_features
        self.hidden_dims = hidden_dims
        self.use_8_connected = use_8_connected

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU()
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gnn_layers.append(
                GraphConvLayer(hidden_dims[i], hidden_dims[i + 1])
            )

        # Global pooling and energy head
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )

        # Cache for adjacency matrix
        self.adjacency_cache = {}

    def create_grid_adjacency(self, n_x, n_y, device):
        """
        Create adjacency matrix for grid graph

        Args:
            n_x, n_y: grid dimensions
            device: torch device
        Returns:
            adjacency: (n_x*n_y, n_x*n_y) adjacency matrix
        """
        cache_key = (n_x, n_y, device)
        if cache_key in self.adjacency_cache:
            return self.adjacency_cache[cache_key]

        num_nodes = n_x * n_y
        adjacency = torch.zeros(num_nodes, num_nodes, device=device)

        # Create 4-connected or 8-connected grid
        for i in range(n_x):
            for j in range(n_y):
                node_idx = i * n_y + j

                # 4-connected neighbors (up, down, left, right)
                neighbors = []
                if i > 0:  # up
                    neighbors.append((i - 1) * n_y + j)
                if i < n_x - 1:  # down
                    neighbors.append((i + 1) * n_y + j)
                if j > 0:  # left
                    neighbors.append(i * n_y + (j - 1))
                if j < n_y - 1:  # right
                    neighbors.append(i * n_y + (j + 1))

                # 8-connected (add diagonals)
                if self.use_8_connected:
                    if i > 0 and j > 0:  # top-left
                        neighbors.append((i - 1) * n_y + (j - 1))
                    if i > 0 and j < n_y - 1:  # top-right
                        neighbors.append((i - 1) * n_y + (j + 1))
                    if i < n_x - 1 and j > 0:  # bottom-left
                        neighbors.append((i + 1) * n_y + (j - 1))
                    if i < n_x - 1 and j < n_y - 1:  # bottom-right
                        neighbors.append((i + 1) * n_y + (j + 1))

                # Set adjacency
                for neighbor in neighbors:
                    adjacency[node_idx, neighbor] = 1.0

        # Normalize by degree (optional: helps with gradient flow)
        degree = adjacency.sum(dim=1, keepdim=True)
        adjacency = adjacency / (degree + 1e-8)

        # Cache it
        self.adjacency_cache[cache_key] = adjacency

        return adjacency

    def forward(self, u, x):
        """
        Args:
            u: solution field (batch, n_x, n_y, 1)
            x: input coordinates (batch, n_x, n_y, 3)
        Returns:
            V: potential energy (batch,)
        """
        batch_size, n_x, n_y, _ = u.shape

        # Combine solution and coordinates
        combined = torch.cat([u, x], dim=-1)  # (batch, n_x, n_y, 4)

        # Reshape to graph nodes
        node_features = combined.reshape(batch_size, n_x * n_y, self.node_features)

        # Encode node features
        node_features = self.node_encoder(node_features)  # (batch, num_nodes, hidden_dims[0])

        # Create adjacency matrix
        adjacency = self.create_grid_adjacency(n_x, n_y, u.device)

        # Expand adjacency for batch
        adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, adjacency)

        # Global pooling (mean over all nodes)
        global_features = node_features.mean(dim=1)  # (batch, hidden_dims[-1])

        # Predict energy
        energy = self.energy_head(global_features)  # (batch, 1)
        energy = energy.squeeze(-1)  # (batch,)

        return energy