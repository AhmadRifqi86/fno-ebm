"""
Fourier Neural Operator (FNO) Models for UQ Paper

This module contains FNO architectures for 1D and 2D PDEs:
- FNO1d: 1D Fourier Neural Operator for Burgers/Advection equations
- FNO2d: 2D Fourier Neural Operator for Diffusion-Reaction/Navier-Stokes
- FFNO2d: Factorized FNO for parameter efficiency
- FNOWrapper: Training wrapper with train_step() and train() methods

Based on:
- Li et al., "Fourier Neural Operator for Parametric PDEs" (ICLR 2021)
- Tran et al., "Factorized Fourier Neural Operators" (ICLR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from tqdm import tqdm
import logging


# ============================================================================
# 1D Spectral Convolution
# ============================================================================

class SpectralConv1d(nn.Module):
    """1D Fourier layer for 1D PDEs"""
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep

        # Xavier initialization scaled by 1/sqrt(modes)
        scale = 1 / (in_channels * out_channels * modes) ** 0.5
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def compl_mul1d(self, input, weights):
        """Complex multiplication in Fourier space"""
        # input: (batch, in_channel, x, 2), weights: (in_channel, out_channel, x, 2)
        real = torch.einsum("bix,iox->box", input[..., 0], weights[..., 0]) - \
               torch.einsum("bix,iox->box", input[..., 1], weights[..., 1])
        imag = torch.einsum("bix,iox->box", input[..., 0], weights[..., 1]) + \
               torch.einsum("bix,iox->box", input[..., 1], weights[..., 0])
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        """
        x: (batch, in_channels, n_x)
        Returns: (batch, out_channels, n_x)
        """
        batch = x.shape[0]

        # Compute FFT
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)  # (batch, in_channels, n_x//2+1, 2)

        # Truncate to self.modes
        x_ft = x_ft[:, :, :self.modes, :]

        # Multiply with learnable weights
        out_ft = self.compl_mul1d(x_ft, self.weights)

        # Pad back to original size
        x_padded = torch.zeros(batch, self.out_channels, x.shape[-1]//2 + 1, 2,
                               device=x.device)
        x_padded[:, :, :self.modes, :] = out_ft

        # Inverse FFT
        out_ft_complex = torch.complex(x_padded[..., 0], x_padded[..., 1])
        x_out = torch.fft.irfft(out_ft_complex, n=x.shape[-1], dim=-1, norm='ortho')

        return x_out


# ============================================================================
# 2D Spectral Convolution
# ============================================================================

class SpectralConv2d(nn.Module):
    """2D Fourier layer for 2D PDEs"""
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes (x-direction)
        self.modes2 = modes2  # Number of Fourier modes (y-direction)

        # Xavier initialization scaled by 1/sqrt(modes)
        scale = 1 / (in_channels * out_channels * modes1 * modes2) ** 0.5
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space"""
        # input: (batch, in_channel, x, y, 2), weights: (in_channel, out_channel, x, y, 2)
        real = torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 0]) - \
               torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 1])
        imag = torch.einsum("bixy,ioxy->boxy", input[..., 0], weights[..., 1]) + \
               torch.einsum("bixy,ioxy->boxy", input[..., 1], weights[..., 0])
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        """
        x: (batch, in_channels, n_x, n_y)
        Returns: (batch, out_channels, n_x, n_y)
        """
        batch_size = x.shape[0]

        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels,
                            x.size(-2), x.size(-1)//2 + 1, 2,
                            device=x.device)

        # Upper-left quadrant
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Lower-left quadrant
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Convert back to complex and inverse FFT
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x_out = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)), norm='ortho')

        return x_out


# ============================================================================
# Factorized Spectral Convolution (for FFNO2d)
# ============================================================================

class FactorizedSpectralConv2d(nn.Module):
    """
    Factorized 2D spectral convolution using separable kernels.
    Reduces parameters from O(modes1 * modes2) to O(modes1 + modes2).
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Xavier init scaled for factorization
        scale = (1 / (in_channels * out_channels * modes1 * modes2)) ** 0.5 * (1 / 2**0.5)

        # Factorized weights: separate for each dimension
        self.weights_x1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, 2)
        )
        self.weights_x2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, 2)
        )
        self.weights_y = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes2, 2)
        )

    def compl_mul1d(self, input, weights):
        """Complex multiplication for 1D factorized weights"""
        # input: (batch, in_channel, x, 2), weights: (in_channel, out_channel, x, 2)
        real = torch.einsum("bix,iox->box", input[..., 0], weights[..., 0]) - \
               torch.einsum("bix,iox->box", input[..., 1], weights[..., 1])
        imag = torch.einsum("bix,iox->box", input[..., 0], weights[..., 1]) + \
               torch.einsum("bix,iox->box", input[..., 1], weights[..., 0])
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        """
        Factorized spectral convolution forward pass.

        Process:
        1. FFT to frequency domain
        2. Apply separable 1D convolutions in x and y (vectorized)
        3. IFFT back to spatial domain
        """
        batch_size = x.shape[0]

        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        # Initialize output
        out_ft = torch.zeros(batch_size, self.out_channels,
                            x.size(-2), x.size(-1)//2 + 1, 2,
                            device=x.device)

        # Upper-left quadrant - vectorized factorization
        x_modes = x_ft[:, :, :self.modes1, :self.modes2]

        # Step 1: Convolve along x-dimension for all y-modes at once
        x_reshaped = x_modes.permute(0, 3, 1, 2, 4).reshape(-1, self.in_channels, self.modes1, 2)
        temp = self.compl_mul1d(x_reshaped, self.weights_x1)
        temp = temp.reshape(batch_size, self.modes2, self.out_channels, self.modes1, 2)
        temp = temp.permute(0, 2, 3, 1, 4)

        # Step 2: Convolve along y-dimension for all x-modes at once
        temp_reshaped = temp.permute(0, 2, 1, 3, 4).reshape(-1, self.out_channels, self.modes2, 2)
        out_modes = self.compl_mul1d(temp_reshaped, self.weights_y)
        out_modes = out_modes.reshape(batch_size, self.modes1, self.out_channels, self.modes2, 2)
        out_modes = out_modes.permute(0, 2, 1, 3, 4)

        out_ft[:, :, :self.modes1, :self.modes2, :] = out_modes

        # Lower-left quadrant
        x_modes = x_ft[:, :, -self.modes1:, :self.modes2]

        # Step 1: x-dimension (vectorized)
        x_reshaped = x_modes.permute(0, 3, 1, 2, 4).reshape(-1, self.in_channels, self.modes1, 2)
        temp = self.compl_mul1d(x_reshaped, self.weights_x2)
        temp = temp.reshape(batch_size, self.modes2, self.out_channels, self.modes1, 2)
        temp = temp.permute(0, 2, 3, 1, 4)

        # Step 2: y-dimension (vectorized)
        temp_reshaped = temp.permute(0, 2, 1, 3, 4).reshape(-1, self.out_channels, self.modes2, 2)
        out_modes = self.compl_mul1d(temp_reshaped, self.weights_y)
        out_modes = out_modes.reshape(batch_size, self.modes1, self.out_channels, self.modes2, 2)
        out_modes = out_modes.permute(0, 2, 1, 3, 4)

        out_ft[:, :, -self.modes1:, :self.modes2, :] = out_modes

        # Convert back to complex and IFFT
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x_out = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)), norm='ortho')

        return x_out

import torch
import torch.nn as nn
import torch.fft

class BinnedSpectralConv2d(nn.Module):
    """HFNO Spectral Convolution with Wavenumber Binning"""
    def __init__(self, in_channels, out_channels, size, num_bins=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.num_bins = num_bins
        
        # Define wavenumber bin boundaries
        # e.g., for size=64, num_bins=3: [0-21, 21-42, 42-64]
        self.bin_boundaries = self._create_bin_boundaries(size, num_bins)
        
        # Separate FCNN (MLP) for each frequency bin
        self.bin_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels * 2, in_channels * 4),  # *2 for complex (real, imag)
                nn.GELU(),
                nn.Linear(in_channels * 4, out_channels * 2)  # *2 for complex output
            )
            for _ in range(num_bins)
        ])
    
    def _create_bin_boundaries(self, size, num_bins):
        """Create boundaries for wavenumber bins"""
        max_k = size // 2 + 1  # For rfft2
        boundaries = []
        bin_size = max_k // num_bins
        
        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else max_k
            boundaries.append((start, end))
        
        return boundaries
    
    def _get_bin_mask(self, kx, ky, bin_idx):
        """Create mask for a specific wavenumber bin"""
        start, end = self.bin_boundaries[bin_idx]
        # Compute radial wavenumber: k = sqrt(kx^2 + ky^2)
        k_radial = torch.sqrt(kx**2 + ky**2)
        mask = (k_radial >= start) & (k_radial < end)
        return mask
    
    def forward(self, x):
        batch, channels, H, W = x.shape
        
        # FFT to frequency domain
        x_ft = torch.fft.rfft2(x, norm='ortho')  # (batch, channels, H, W//2+1)
        
        # Create wavenumber grids
        kx = torch.arange(0, H, device=x.device).view(-1, 1).float()
        ky = torch.arange(0, W//2 + 1, device=x.device).view(1, -1).float()
        kx = kx.expand(H, W//2 + 1)
        ky = ky.expand(H, W//2 + 1)
        
        # Initialize output in frequency domain
        out_ft = torch.zeros_like(x_ft, dtype=torch.complex64)
        
        # Process each frequency bin separately
        for bin_idx in range(self.num_bins):
            # Get mask for current bin
            bin_mask = self._get_bin_mask(kx, ky, bin_idx)  # (H, W//2+1)
            
            # Extract frequencies in this bin
            # Convert complex to real representation (real, imag)
            x_ft_real = torch.view_as_real(x_ft)  # (batch, channels, H, W//2+1, 2)
            
            # Flatten spatial dimensions for the bin
            bin_indices = bin_mask.nonzero(as_tuple=False)  # (num_points, 2)
            
            if bin_indices.shape[0] > 0:
                # Extract data at bin locations
                bin_data = []
                for b in range(batch):
                    points = []
                    for idx in bin_indices:
                        h_idx, w_idx = idx[0], idx[1]
                        point = x_ft_real[b, :, h_idx, w_idx, :].flatten()  # (channels*2,)
                        points.append(point)
                    bin_data.append(torch.stack(points))  # (num_points, channels*2)
                bin_data = torch.stack(bin_data)  # (batch, num_points, channels*2)
                
                # Process through bin-specific network
                bin_output = self.bin_networks[bin_idx](bin_data)  # (batch, num_points, out_channels*2)
                
                # Reshape back to complex
                bin_output = bin_output.view(batch, -1, self.out_channels, 2)  # (batch, num_points, out_channels, 2)
                bin_output_complex = torch.view_as_complex(bin_output.contiguous())  # (batch, num_points, out_channels)
                
                # Place back into output
                for b in range(batch):
                    for point_idx, idx in enumerate(bin_indices):
                        h_idx, w_idx = idx[0], idx[1]
                        out_ft[b, :, h_idx, w_idx] = bin_output_complex[b, point_idx, :]
        
        # IFFT back to spatial domain
        out = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        
        return out
    
# ============================================================================
# FNO1d: 1D Fourier Neural Operator
# ============================================================================

class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator for 1D PDEs (Burgers, Advection).

    Architecture:
        Input (n_x, in_channels) � Lift � [Fourier Layers] � Project � Output (n_x, 1)
    """
    def __init__(self, modes: int = 16, width: int = 64, n_layers: int = 4,
                 in_channels: int = 2, out_channels: int = 1):
        """
        Args:
            modes: Number of Fourier modes to keep
            width: Hidden dimension (channel width)
            n_layers: Number of Fourier layers
            in_channels: Input channels (e.g., 2 for [x, u0])
            out_channels: Output channels (default: 1)
        """
        super().__init__()

        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # Lift: project input to hidden dimension
        self.lift = nn.Linear(in_channels, width)

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])

        # Local (pointwise) layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])

        # Projection: hidden � output
        self.project = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, out_channels)
        )

    def forward(self, x):
        """
        x: (batch, n_x, in_channels) OR (batch, n_x, 1, in_channels) for compatibility
        Returns: (batch, n_x, out_channels) OR (batch, n_x, 1, out_channels) to match input
        """
        # Handle 4D input from dataloader (batch, n_x, 1, in_channels)
        input_is_4d = (x.dim() == 4)
        if input_is_4d:
            x = x.squeeze(2)  # (batch, n_x, in_channels)

        # Lift
        x = self.lift(x)  # (batch, n_x, width)
        x = x.permute(0, 2, 1)  # (batch, width, n_x) for conv

        # Fourier layers
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)  # Fourier convolution
            x2 = self.conv_layers[i](x)     # Local convolution
            x = x1 + x2                     # Combine
            if i < self.n_layers - 1:
                x = F.gelu(x)

        # Project
        x = x.permute(0, 2, 1)  # (batch, n_x, width)
        x = self.project(x)     # (batch, n_x, out_channels)

        # Restore 4D shape if input was 4D
        if input_is_4d:
            x = x.unsqueeze(2)  # (batch, n_x, 1, out_channels)

        return x


# ============================================================================
# FNO2d: 2D Fourier Neural Operator
# ============================================================================

class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for 2D PDEs (Diffusion-Reaction, Navier-Stokes).

    Architecture:
        Input (n_x, n_y, in_channels) � Lift � [Fourier Layers] � Project � Output (n_x, n_y, 1)
    """
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32,
                 n_layers: int = 4, in_channels: int = 3, out_channels: int = 1):
        """
        Args:
            modes1, modes2: Number of Fourier modes in x and y directions
            width: Hidden dimension (channel width)
            n_layers: Number of Fourier layers
            in_channels: Input channels (e.g., 3 for [x, y, u0])
            out_channels: Output channels (default: 1)
        """
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        # Lift
        self.lift = nn.Conv2d(in_channels, width, 1)

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])

        # Local layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])

        # Project
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        """
        x: (batch, n_x, n_y, in_channels)
        Returns: (batch, n_x, n_y, out_channels)
        """
        # Permute to (batch, channels, n_x, n_y)
        x = x.permute(0, 3, 1, 2)

        # Lift
        x = self.lift(x)

        # Fourier layers
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        # Project
        x = self.project(x)

        # Permute back
        x = x.permute(0, 2, 3, 1)

        return x


# ============================================================================
# FFNO2d: Factorized FNO for Parameter Efficiency
# ============================================================================

class FFNO2d(nn.Module):
    """
    Factorized 2D Fourier Neural Operator with parameter efficiency.

    Key improvements over FNO2d:
    - Factorized spectral layers (30-50% parameter reduction)
    - Layer normalization for better gradient flow
    - Suitable for limited data scenarios
    """
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32,
                 n_layers: int = 4, in_channels: int = 3, out_channels: int = 1,
                 dropout: float = 0.0):
        """
        Args:
            modes1, modes2: Number of Fourier modes in x and y directions
            width: Hidden dimension (channel width)
            n_layers: Number of Fourier layers
            in_channels: Input channels (e.g., 3 for [x, y, u0])
            out_channels: Output channels (default: 1)
            dropout: Dropout rate (default: 0.0)
        """
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.dropout = dropout

        # Input projection
        self.fc0 = nn.Linear(in_channels, width)
        self.dropout0 = nn.Dropout(dropout)

        # Factorized Fourier layers
        self.conv_layers = nn.ModuleList([
            FactorizedSpectralConv2d(width, width, modes1, modes2)
            for _ in range(n_layers)
        ])

        # Local (non-spectral) connection
        self.w_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])

        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(width) for _ in range(n_layers)
        ])

        # Dropout after each layer
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(n_layers)
        ])

        # Output projection
        self.fc1 = nn.Linear(width, width)
        self.dropout_out = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, out_channels)

    def forward(self, x):
        """
        x: (batch, n_x, n_y, in_channels)
        Returns: (batch, n_x, n_y, out_channels)
        """
        # Lift to higher dimension
        x = self.fc0(x)  # (batch, n_x, n_y, width)
        x = self.dropout0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, n_x, n_y)

        # Factorized Fourier layers with improved residuals
        for i in range(self.n_layers):
            x_res = x

            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2

            if i < self.n_layers - 1:
                x = F.gelu(x)

            x = x + x_res

            x = x.permute(0, 2, 3, 1)  # (batch, n_x, n_y, width)
            x = self.norm_layers[i](x)
            x = x.permute(0, 3, 1, 2)  # (batch, width, n_x, n_y)

            if i < self.n_layers - 1:
                x = self.dropout_layers[i](x)

        # Project to output
        x = x.permute(0, 2, 3, 1)  # (batch, n_x, n_y, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout_out(x)
        x = self.fc2(x)  # (batch, n_x, n_y, out_channels)

        return x


class ConvResidualBlock(nn.Module):
    """Convolutional residual block for capturing local high-frequency details"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.gelu(out)
        return out


class ChannelAttention(nn.Module):
    """Channel attention for adaptive feature selection"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention for focusing on important regions"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class EquivariantAttention(nn.Module):
    """Combined channel and spatial attention (translation equivariant)"""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class HierarchicalFourierBlock(nn.Module):
    """Single hierarchical block combining Fourier, Conv, and Attention"""
    def __init__(self, width, modes1, modes2):
        super().__init__()
        
        # Fourier component (global, low-frequency)
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        
        # Convolutional component (local, high-frequency)
        self.conv_residual = ConvResidualBlock(width)
        
        # Attention mechanism (adaptive weighting)
        self.attention = EquivariantAttention(width)
        
        # Skip connection
        self.w = nn.Conv2d(width, width, 1)
        
    def forward(self, x):
        # Fourier branch (global)
        x_fourier = self.spectral_conv(x)
        
        # Convolutional branch (local)
        x_conv = self.conv_residual(x)
        
        # Skip connection
        x_skip = self.w(x)
        
        # Combine all branches
        x_combined = x_fourier + x_conv + x_skip
        
        # Apply attention
        x_out = self.attention(x_combined)
        
        return x_out


class HFNO2d(nn.Module):
    """Hierarchical Fourier Neural Operator with Conv-Residual and Attention"""
    def __init__(self, modes1, modes2, width, in_channels=3, out_channels=1, n_layers=4):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        # Lifting layer
        self.fc0 = nn.Linear(in_channels, self.width)
        
        # Hierarchical Fourier blocks
        self.hf_blocks = nn.ModuleList([
            HierarchicalFourierBlock(self.width, self.modes1, self.modes2)
            for _ in range(self.n_layers)
        ])
        
        # Projection layer
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (batch, H, W, channels)
        x = self.fc0(x)  # Lift to width channels
        x = x.permute(0, 3, 1, 2)  # (batch, width, H, W)
        
        # Process through hierarchical blocks
        for i in range(self.n_layers):
            x = self.hf_blocks[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# ============================================================================
# FNOWrapper: Training Wrapper for FNO Models
# ============================================================================

class FNOTrainer:
    """
    Training wrapper for FNO models with train_step() and train() methods.

    Supports:
    - Standard MSE loss
    - Gradient penalty for combating over-smoothing
    - Physics-informed loss (optional)
    - Learning rate scheduling
    - Checkpointing
    """
    def __init__(self, model: nn.Module, config: Dict,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Args:
            model: FNO model (FNO1d, FNO2d, or FFNO2d)
            config: Configuration dictionary with training hyperparameters
            optimizer: Optional custom optimizer (otherwise Adam with config.lr)
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            lr = getattr(config, 'lr', 1e-3)
            weight_decay = getattr(config, 'weight_decay', 1e-4)
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Loss function
        self.criterion = nn.MSELoss()

        # Loss weights
        self.gradient_penalty_weight = getattr(config, 'gradient_penalty_weight', 0.0)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Checkpoint configuration
        self.save_checkpoints = getattr(config, 'save_checkpoints', True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Gradient/weight tracking
        self.enable_tracking = getattr(config, 'enable_tracking', True)
        self.tracking_backend = getattr(config, 'tracking_backend', 'custom')  # 'custom' or 'tensorboard'
        self.gradient_tracker = None
        self.writer = None

        if self.enable_tracking:
            log_dir = getattr(config, 'log_dir', './runs')
            experiment_name = getattr(config, 'experiment_name', 'fno_training')

            if self.tracking_backend == 'custom':
                try:
                    from track import GradientTracker
                    self.gradient_tracker = GradientTracker(
                        model=self.model,
                        log_dir=log_dir,
                        experiment_name=experiment_name,
                        track_interval=getattr(config, 'track_interval', 10),
                        histogram_interval=getattr(config, 'histogram_interval', 100),
                        gradient_clip_threshold=getattr(config, 'gradient_clip_threshold', 10.0),
                    )
                    self.logger.info("Custom gradient tracking enabled")
                except ImportError:
                    self.logger.warning("track.py not found, falling back to built-in TensorBoard")
                    self.tracking_backend = 'tensorboard'

            if self.tracking_backend == 'tensorboard':
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    import os
                    log_path = os.path.join(log_dir, experiment_name)
                    self.writer = SummaryWriter(log_dir=log_path)
                    self.global_step = 0
                    self.logger.info(f"Built-in TensorBoard logging enabled at {log_path}")
                except ImportError:
                    self.logger.warning("TensorBoard not available, disabling tracking")
                    self.enable_tracking = False

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step.

        Args:
            x: Input batch (batch, n_x, [n_y], in_channels)
            y: Target batch (batch, n_x, [n_y], out_channels)

        Returns:
            loss: Total loss
            loss_dict: Dictionary with loss components for logging
        """
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        # Permute from (batch, nx, ny, channels) to (batch, channels, nx, ny)
        # x = x.permute(0, 3, 1, 2)
        # y = y.permute(0, 3, 1, 2)

        # Forward pass
        pred = self.model(x)

        # MSE loss
        mse_loss = self.criterion(pred, y)

        # Gradient penalty (if enabled)
        grad_loss = torch.tensor(0.0, device=self.device)
        if self.gradient_penalty_weight > 0:
            grad_loss = self._gradient_penalty_loss(pred, y)

        # Total loss
        total_loss = mse_loss + self.gradient_penalty_weight * grad_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Track gradients (before optimizer step)
        if self.enable_tracking:
            if self.gradient_tracker is not None:
                # Custom tracking with anomaly detection
                self.gradient_tracker.track(loss=total_loss, custom_metrics={'mse': mse_loss.item(), 'grad_penalty': grad_loss.item() if isinstance(grad_loss, torch.Tensor) else grad_loss})
            elif self.writer is not None:
                # Built-in TensorBoard logging
                self.writer.add_scalar('train/loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('train/mse', mse_loss.item(), self.global_step)
                self.writer.add_scalar('train/grad_penalty', grad_loss.item() if isinstance(grad_loss, torch.Tensor) else grad_loss, self.global_step)

                # Log gradients manually
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'gradients/{name}', param.grad, self.global_step)
                        self.writer.add_scalar(f'gradient_norm/{name}', param.grad.norm().item(), self.global_step)

                self.global_step += 1

        self.optimizer.step()

        loss_dict = {
            'mse': mse_loss.item(),
            'grad': grad_loss.item() if isinstance(grad_loss, torch.Tensor) else grad_loss,
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def _gradient_penalty_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Gradient penalty loss to combat over-smoothing.

        Penalizes predictions where spatial gradients don't match target gradients.
        """
        # Squeeze if needed
        if pred.dim() == 4 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        if target.dim() == 4 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        # Handle 1D case
        if pred.dim() == 2:
            pred_grad = torch.diff(pred, dim=1)
            target_grad = torch.diff(target, dim=1)
            return F.l1_loss(pred_grad, target_grad)

        # Handle 2D case
        # X-direction
        pred_grad_x = torch.diff(pred, dim=1)
        target_grad_x = torch.diff(target, dim=1)

        # Y-direction
        pred_grad_y = torch.diff(pred, dim=2)
        target_grad_y = torch.diff(target, dim=2)

        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return (loss_x + loss_y) / 2

    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, float]:
        """
        Validation step.

        Args:
            val_loader: Validation data loader

        Returns:
            avg_val_loss: Average validation loss
            avg_rel_l2: Average relative L2 error
        """
        self.model.eval()
        total_val_loss = 0
        total_rel_l2 = 0

        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)

            # Permute from (batch, nx, ny, channels) to (batch, channels, nx, ny)
            # x = x.permute(0, 3, 1, 2)
            # y = y.permute(0, 3, 1, 2)

            pred = self.model(x)

            # Validation loss
            val_loss = self.criterion(pred, y)
            total_val_loss += val_loss.item()

            # Relative L2 error
            rel_l2 = torch.norm(pred - y) / torch.norm(y)
            total_rel_l2 += rel_l2.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_rel_l2 = total_rel_l2 / len(val_loader)

        return avg_val_loss, avg_rel_l2

    def train(self, train_loader, val_loader, epochs: int):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
        """
        self.logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Training phase
            self.model.train()
            epoch_loss = 0
            epoch_mse = 0
            epoch_grad = 0

            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
            for x, y in loop:
                loss, loss_dict = self.train_step(x, y)

                epoch_loss += loss.item()
                epoch_mse += loss_dict['mse']
                epoch_grad += loss_dict['grad']

                loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mse=f"{loss_dict['mse']:.4f}",
                    grad=f"{loss_dict['grad']:.4f}"
                )

            avg_train_loss = epoch_loss / len(train_loader)
            avg_mse = epoch_mse / len(train_loader)
            avg_grad = epoch_grad / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation phase
            avg_val_loss, avg_rel_l2 = self.validate(val_loader)
            self.val_losses.append(avg_val_loss)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={avg_train_loss:.6f} (MSE={avg_mse:.6f}, Grad={avg_grad:.6f}), "
                f"Val Loss={avg_val_loss:.6f}, "
                f"Rel L2={avg_rel_l2:.6f}, "
                f"LR={current_lr:.2e}"
            )

            # TensorBoard logging for epoch-level metrics
            if self.enable_tracking:
                if self.gradient_tracker is not None:
                    # Custom tracker
                    epoch_metrics = {
                        'train_loss': avg_train_loss,
                        'train_mse': avg_mse,
                        'train_grad_penalty': avg_grad,
                        'val_loss': avg_val_loss,
                        'val_rel_l2': avg_rel_l2,
                        'learning_rate': current_lr,
                    }
                    self.gradient_tracker.log_scalars('epoch', epoch_metrics, epoch)
                elif self.writer is not None:
                    # Built-in TensorBoard
                    self.writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
                    self.writer.add_scalar('epoch/train_mse', avg_mse, epoch)
                    self.writer.add_scalar('epoch/train_grad_penalty', avg_grad, epoch)
                    self.writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
                    self.writer.add_scalar('epoch/val_rel_l2', avg_rel_l2, epoch)
                    self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)

            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                if self.save_checkpoints:
                    self.save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"  � Best model saved (val_loss={avg_val_loss:.6f})")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = getattr(self.config, 'checkpoint_path', './checkpoints')
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(checkpoint_dir, 'best_fno.pt')
        else:
            path = os.path.join(checkpoint_dir, f'fno_epoch_{epoch}.pt')

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


# ============================================================================
# PART 2: UQ METHOD IMPLEMENTATIONS
# ============================================================================
# This section contains implementations for:
# 1. Evidential Deep Learning (Idea 13)
# 2. MC Dropout (Cross-family comparison)
# 3. Deep Ensemble (Cross-family comparison)
# 4. Bayesian FNO (Cross-family comparison)
# 5. Conformalized Quantile Regression (Conformal methods comparison)
# 6. Prior Networks (Evidential methods comparison)
# 7. Dirichlet Evidential Regression (Evidential methods comparison)
# ============================================================================


# ============================================================================
# 1. EVIDENTIAL DEEP LEARNING (Idea 13)
# ============================================================================

class EvidentialFNO1d(nn.Module):
    """
    Evidential 1D FNO that outputs NIG parameters (γ, ν, α, β).
    
    For Idea 13: Evidential Deep Learning for Neural Operators
    
    Architecture:
        Input → Shared FNO Backbone → 4 Separate Heads → (γ, ν, α, β)
        
    Returns Normal-Inverse-Gamma (NIG) parameters:
        - γ: mean prediction
        - ν: evidence for mean (pseudo-observations)
        - α: evidence strength (shape parameter)
        - β: scale parameter
    """
    def __init__(self, modes: int = 16, width: int = 64, n_layers: int = 4,
                 in_channels: int = 2, nu_min: float = 0.01, alpha_min: float = 1.01,
                 beta_min: float = 0.01):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        
        # Minimum values for stability
        self.nu_min = nu_min
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        
        # Shared FNO backbone
        self.lift = nn.Linear(in_channels, width)
        
        self.fourier_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Evidential heads (4 separate output branches)
        self.gamma_head = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.nu_head = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.alpha_head = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.beta_head = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_x, in_channels) or (batch, n_x, 1, in_channels)
        
        Returns:
            gamma: (batch, n_x, 1) - mean prediction
            nu: (batch, n_x, 1) - evidence for mean
            alpha: (batch, n_x, 1) - evidence strength
            beta: (batch, n_x, 1) - scale parameter
        """
        # Handle 4D input
        input_is_4d = (x.dim() == 4)
        if input_is_4d:
            x = x.squeeze(2)
        
        # Shared backbone
        x = self.lift(x)  # (batch, n_x, width)
        x = x.permute(0, 2, 1)  # (batch, width, n_x)
        
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        x = x.permute(0, 2, 1)  # (batch, n_x, width)
        
        # Four evidential output heads
        gamma = self.gamma_head(x)  # Unconstrained
        
        # Apply constraints: softplus + minimum
        nu = F.softplus(self.nu_head(x)) + self.nu_min
        alpha = F.softplus(self.alpha_head(x)) + self.alpha_min
        beta = F.softplus(self.beta_head(x)) + self.beta_min
        
        # Restore 4D shape if needed
        if input_is_4d:
            gamma = gamma.unsqueeze(2)
            nu = nu.unsqueeze(2)
            alpha = alpha.unsqueeze(2)
            beta = beta.unsqueeze(2)
        
        return gamma, nu, alpha, beta


class EvidentialFNO2d(nn.Module):
    """
    Evidential 2D FNO that outputs NIG parameters (γ, ν, α, β).
    
    For Idea 13: Evidential Deep Learning for Neural Operators
    """
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32,
                 n_layers: int = 4, in_channels: int = 3,
                 nu_min: float = 0.01, alpha_min: float = 1.01, beta_min: float = 0.01):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        self.nu_min = nu_min
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        
        # Shared backbone
        self.lift = nn.Conv2d(in_channels, width, 1)
        
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Evidential heads
        self.gamma_head = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )
        
        self.nu_head = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )
        
        self.alpha_head = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )
        
        self.beta_head = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_x, n_y, in_channels)
        
        Returns:
            gamma, nu, alpha, beta: Each (batch, n_x, n_y, 1)
        """
        # Permute to (batch, channels, n_x, n_y)
        x = x.permute(0, 3, 1, 2)
        
        # Shared backbone
        x = self.lift(x)
        
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        # Evidential heads
        gamma = self.gamma_head(x)
        nu = F.softplus(self.nu_head(x)) + self.nu_min
        alpha = F.softplus(self.alpha_head(x)) + self.alpha_min
        beta = F.softplus(self.beta_head(x)) + self.beta_min
        
        # Permute back to (batch, n_x, n_y, 1)
        gamma = gamma.permute(0, 2, 3, 1)
        nu = nu.permute(0, 2, 3, 1)
        alpha = alpha.permute(0, 2, 3, 1)
        beta = beta.permute(0, 2, 3, 1)
        
        return gamma, nu, alpha, beta


# ============================================================================
# 2. MC DROPOUT (Cross-Family Comparison)
# ============================================================================

class MCDropoutFNO1d(nn.Module):
    """
    FNO with Monte Carlo Dropout for uncertainty quantification.
    
    Key idea: Keep dropout active during inference, sample T times.
    """
    def __init__(self, modes: int = 16, width: int = 64, n_layers: int = 4,
                 in_channels: int = 2, out_channels: int = 1,
                 dropout_rate: float = 0.1, n_samples: int = 20):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        
        # Lift
        self.lift = nn.Linear(in_channels, width)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])
        
        # Local layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Dropout layers (applied after each block)
        self.dropout_layers = nn.ModuleList([
            nn.Dropout1d(dropout_rate) for _ in range(n_layers)
        ])
        
        # Project
        self.project = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, out_channels)
        )
    
    def forward_single(self, x):
        """Single forward pass with dropout active."""
        input_is_4d = (x.dim() == 4)
        if input_is_4d:
            x = x.squeeze(2)
        
        x = self.lift(x)
        x = x.permute(0, 2, 1)
        
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            x = self.dropout_layers[i](x)  # Dropout active!
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        x = x.permute(0, 2, 1)
        x = self.project(x)
        
        if input_is_4d:
            x = x.unsqueeze(2)
        
        return x
    
    def forward(self, x, return_samples=False, return_uncertainty=True):
        """
        Forward pass with MC Dropout sampling.
        
        Args:
            x: Input
            return_samples: If True, return all T samples
            return_uncertainty: If True, return mean and std
        
        Returns:
            If return_samples: samples (T, batch, ...)
            If return_uncertainty: mean, std
            Else: mean only
        """
        if not return_uncertainty:
            return self.forward_single(x)
        
        # Enable dropout for sampling
        self.train()  # Sets model to train mode (dropout active)
        
        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                sample = self.forward_single(x)
                samples.append(sample)
        
        samples = torch.stack(samples, dim=0)  # (T, batch, ...)
        
        if return_samples:
            return samples
        
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        return mean, std


class MCDropoutFNO2d(nn.Module):
    """2D version of MC Dropout FNO."""
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32,
                 n_layers: int = 4, in_channels: int = 3, out_channels: int = 1,
                 dropout_rate: float = 0.1, n_samples: int = 20):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        
        # Lift
        self.lift = nn.Conv2d(in_channels, width, 1)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])
        
        # Local layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout2d(dropout_rate) for _ in range(n_layers)
        ])
        
        # Project
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(128, out_channels, 1)
        )
    
    def forward_single(self, x):
        """Single forward pass with dropout active."""
        x = x.permute(0, 3, 1, 2)
        x = self.lift(x)
        
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            x = self.dropout_layers[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        x = self.project(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def forward(self, x, return_samples=False, return_uncertainty=True):
        """Forward with MC Dropout sampling."""
        if not return_uncertainty:
            return self.forward_single(x)
        
        self.train()
        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                samples.append(self.forward_single(x))
        
        samples = torch.stack(samples, dim=0)
        
        if return_samples:
            return samples
        
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std


# ============================================================================
# 3. DEEP ENSEMBLE (Cross-Family Comparison)
# ============================================================================

class FNOEnsemble:
    """
    Ensemble of M independently trained FNO models.
    
    Training: Train M models with different random seeds
    Inference: Average predictions from all M models
    Uncertainty: Standard deviation across M predictions
    """
    def __init__(self, n_models: int = 5, model_class=FNO2d,
                 device: str = 'cuda', models: list = None, **model_kwargs):
        """
        Args:
            n_models: Number of ensemble members (default: 5)
            model_class: FNO class to use (FNO1d, FNO2d, FFNO2d)
            device: Device to use
            models: Optional list of pre-trained models (if None, creates new models)
            **model_kwargs: Arguments for model initialization
        """
        self.device = device

        if models is not None:
            # Use pre-trained models
            self.models = models
            self.n_models = len(models)
            self.model_class = type(models[0]) if models else model_class
            self.model_kwargs = model_kwargs
        else:
            # Create M models with different random initializations
            self.n_models = n_models
            self.model_class = model_class
            self.model_kwargs = model_kwargs

            self.models = []
            for i in range(n_models):
                torch.manual_seed(42 + i)
                model = model_class(**model_kwargs).to(device)
                self.models.append(model)
    
    def train_ensemble(self, train_loader, val_loader, config, 
                      save_dir: str = './ensemble_checkpoints'):
        """
        Train all ensemble members independently.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for i, model in enumerate(self.models):
            print(f"\n{'='*60}")
            print(f"Training Ensemble Member {i+1}/{self.n_models}")
            print(f"{'='*60}")
            
            trainer = FNOTrainer(model, config)
            trainer.train(train_loader, val_loader, epochs=config.epochs)
            
            checkpoint_path = os.path.join(save_dir, f'ensemble_model_{i}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_kwargs': self.model_kwargs,
            }, checkpoint_path)
            
            print(f"Saved model {i+1} to {checkpoint_path}")
    
    def load_ensemble(self, save_dir: str = './ensemble_checkpoints'):
        """Load pre-trained ensemble from disk."""
        import os
        
        for i, model in enumerate(self.models):
            checkpoint_path = os.path.join(save_dir, f'ensemble_model_{i}.pt')
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded ensemble member {i+1} from {checkpoint_path}")
    
    def predict(self, x, return_samples=False, return_uncertainty=True):
        """
        Predict with ensemble.
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        if return_samples:
            return predictions
        
        mean = predictions.mean(dim=0)
        
        if not return_uncertainty:
            return mean
        
        std = predictions.std(dim=0)
        return mean, std
    
    def __call__(self, x, **kwargs):
        """Make ensemble callable like a model."""
        return self.predict(x, **kwargs)


# ============================================================================
# 4. BAYESIAN FNO (Cross-Family Comparison)
# ============================================================================

class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with variational inference.
    """
    def __init__(self, in_features: int, out_features: int, 
                 prior_sigma: float = 1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Variational parameters for weights
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Variational parameters for bias
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)
        
        # Prior
        self.prior_sigma = prior_sigma
    
    def forward(self, x, sample=True):
        """Forward pass with reparameterization trick."""
        if sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_sigma * weight_eps
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """KL divergence KL(q(w) || p(w)) for ELBO."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        kl_weight = 0.5 * (
            (self.weight_mu ** 2 + weight_sigma ** 2) / (self.prior_sigma ** 2) -
            torch.log(weight_sigma ** 2) +
            torch.log(torch.tensor(self.prior_sigma ** 2)) - 1
        ).sum()
        
        kl_bias = 0.5 * (
            (self.bias_mu ** 2 + bias_sigma ** 2) / (self.prior_sigma ** 2) -
            torch.log(bias_sigma ** 2) +
            torch.log(torch.tensor(self.prior_sigma ** 2)) - 1
        ).sum()
        
        return kl_weight + kl_bias


class BayesianFNO2d(nn.Module):
    """
    Bayesian FNO with variational inference.
    
    Replace projection layers with Bayesian layers.
    Keep Fourier layers deterministic (too many parameters).
    """
    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32,
                 n_layers: int = 4, in_channels: int = 3, out_channels: int = 1,
                 n_samples: int = 10, prior_sigma: float = 1.0):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.n_samples = n_samples
        
        # Deterministic backbone
        self.lift = nn.Conv2d(in_channels, width, 1)
        
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Bayesian projection head
        self.fc1 = BayesianLinear(width, 128, prior_sigma)
        self.fc2 = BayesianLinear(128, out_channels, prior_sigma)
    
    def forward_single(self, x, sample=True):
        """Single forward pass."""
        x = x.permute(0, 3, 1, 2)
        x = self.lift(x)
        
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x, sample=sample)
        x = F.gelu(x)
        x = self.fc2(x, sample=sample)
        
        return x
    
    def forward(self, x, return_samples=False, return_uncertainty=True):
        """Forward with sampling."""
        if not return_uncertainty:
            return self.forward_single(x, sample=False)
        
        samples = []
        for _ in range(self.n_samples):
            sample = self.forward_single(x, sample=True)
            samples.append(sample)
        
        samples = torch.stack(samples, dim=0)
        
        if return_samples:
            return samples
        
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std
    
    def kl_divergence(self):
        """Total KL divergence for ELBO."""
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()


# ============================================================================
# 5. QUANTILE FNO FOR CQR (Conformal Methods Comparison)
# ============================================================================

class QuantileFNO2d(nn.Module):
    """
    FNO that predicts quantiles instead of mean.
    
    For Conformalized Quantile Regression (CQR).
    Outputs two quantiles: q_low (α/2) and q_high (1-α/2).
    """
    def __init__(self, modes1=12, modes2=12, width=32, n_layers=4,
                 in_channels=3, quantiles=[0.025, 0.975]):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.quantiles = quantiles
        
        # Shared backbone
        self.lift = nn.Conv2d(in_channels, width, 1)
        
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Two output heads for two quantiles
        self.project_low = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )
        
        self.project_high = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )
    
    def forward(self, x):
        """
        Returns:
            q_low, q_high: Lower and upper quantiles
        """
        x = x.permute(0, 3, 1, 2)
        x = self.lift(x)
        
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        q_low = self.project_low(x)
        q_high = self.project_high(x)
        
        q_low = q_low.permute(0, 2, 3, 1)
        q_high = q_high.permute(0, 2, 3, 1)
        
        # Enforce q_low < q_high
        q_high = q_low + F.softplus(q_high - q_low)
        
        return q_low, q_high


# ============================================================================
# 6. PRIOR NETWORK FNO (Evidential Methods Comparison)
# ============================================================================

class PriorNetworkFNO2d(nn.Module):
    """
    Prior Networks using Dirichlet distribution for regression.
    
    Discretize output space, predict Dirichlet over bins.
    Uses reverse KL divergence (more robust to misspecification).
    """
    def __init__(self, modes1=12, modes2=12, width=32, n_layers=4,
                 in_channels=3, n_bins=50, output_range=(-1, 1)):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.n_bins = n_bins
        self.output_range = output_range
        
        # Register bin centers
        self.register_buffer(
            'bin_centers',
            torch.linspace(output_range[0], output_range[1], n_bins)
        )
        
        # Shared backbone
        self.lift = nn.Conv2d(in_channels, width, 1)
        
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Project to Dirichlet concentrations
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, n_bins, 1)
        )
    
    def forward(self, x):
        """
        Returns:
            alphas: Dirichlet concentrations
            mean: Expected value
            uncertainty: Total uncertainty
        """
        x = x.permute(0, 3, 1, 2)
        x = self.lift(x)
        
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        alphas = F.softplus(self.project(x)) + 1
        alphas = alphas.permute(0, 2, 3, 1)
        
        # Expected value
        alpha_0 = alphas.sum(dim=-1, keepdim=True)
        mean = (alphas * self.bin_centers).sum(dim=-1, keepdim=True) / alpha_0
        
        # Uncertainty
        expected_p = alphas / alpha_0
        variance = (expected_p * (1 - expected_p) / (alpha_0 + 1)).sum(dim=-1, keepdim=True)
        epistemic = 1 / alpha_0
        uncertainty = variance + epistemic
        
        return alphas, mean, uncertainty


# ============================================================================
# 7. DIRICHLET EVIDENTIAL FNO (Evidential Methods Comparison)
# ============================================================================

class DirichletEvidentialFNO2d(nn.Module):
    """
    Evidential regression using Dirichlet prior (alternative to NIG).
    
    Discretizes output space and predicts Dirichlet over bins.
    """
    def __init__(self, modes1=12, modes2=12, width=32, n_layers=4,
                 in_channels=3, n_bins=50, output_range=(-1, 1)):
        super().__init__()
        
        self.n_bins = n_bins
        self.output_range = output_range
        
        # Register bin centers
        self.register_buffer(
            'bin_centers',
            torch.linspace(output_range[0], output_range[1], n_bins)
        )
        
        # Backbone
        self.backbone = FNO2d(
            modes1=modes1, modes2=modes2, width=width,
            n_layers=n_layers, in_channels=in_channels,
            out_channels=n_bins
        )
    
    def forward(self, x):
        """
        Returns:
            mean, aleatoric, epistemic
        """
        # Get Dirichlet concentrations
        alphas = F.softplus(self.backbone(x)) + 1
        
        # Expected value
        alpha_0 = alphas.sum(dim=-1, keepdim=True)
        mean = (alphas * self.bin_centers).sum(dim=-1, keepdim=True) / alpha_0
        
        # Aleatoric uncertainty
        probs = alphas / alpha_0
        variance = (probs * (self.bin_centers - mean) ** 2).sum(dim=-1, keepdim=True)
        aleatoric = variance
        
        # Epistemic uncertainty
        epistemic = 1 / alpha_0
        
        return mean, aleatoric, epistemic


# ============================================================================
# Posterior Networks (EDL Variant 4 of 6)
# ============================================================================

class PosteriorNetworkFNO2d(nn.Module):
    """
    Posterior Networks using normalizing flows.

    Most flexible evidential method, but also most complex.
    Uses learned flow to represent flexible posterior distribution.

    Note: This is a simplified implementation. Full Posterior Networks
    would require proper normalizing flow library (e.g., nflows).
    """
    def __init__(self, modes1=12, modes2=12, width=32, n_layers=4,
                 in_channels=3, n_flows=4, flow_hidden_dim=64):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.flow_hidden_dim = flow_hidden_dim

        # Shared FNO backbone
        self.lift = nn.Conv2d(in_channels, width, 1)

        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        ])

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])

        # Project to flow parameters
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, flow_hidden_dim, 1)
        )

        # Normalizing flow components (simplified)
        # In full implementation, use proper flows like MAF, RealNVP, etc.
        self.flow_mean = nn.Sequential(
            nn.Conv2d(flow_hidden_dim, flow_hidden_dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(flow_hidden_dim // 2, 1, 1)
        )

        self.flow_scale = nn.Sequential(
            nn.Conv2d(flow_hidden_dim, flow_hidden_dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(flow_hidden_dim // 2, 1, 1)
        )

    def forward(self, x, n_samples=100):
        """
        Args:
            x: (batch, nx, ny, in_channels)
            n_samples: Number of samples from posterior flow

        Returns:
            mean: (batch, nx, ny, 1) - Posterior mean
            aleatoric: (batch, nx, ny, 1) - Aleatoric uncertainty
            epistemic: (batch, nx, ny, 1) - Epistemic uncertainty
        """
        # FNO backbone
        x = x.permute(0, 3, 1, 2)  # (batch, in_channels, nx, ny)
        x = self.lift(x)

        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        # Get flow parameters
        flow_params = self.project(x)  # (batch, flow_hidden_dim, nx, ny)

        # Sample from flow (simplified Gaussian approximation)
        mean = self.flow_mean(flow_params)  # (batch, 1, nx, ny)
        scale = F.softplus(self.flow_scale(flow_params))  # (batch, 1, nx, ny)

        # Monte Carlo sampling from learned distribution
        samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(mean)
            sample = mean + scale * eps
            samples.append(sample)

        samples = torch.stack(samples, dim=0)  # (n_samples, batch, 1, nx, ny)

        # Permute back to (batch, nx, ny, 1)
        mean = mean.permute(0, 2, 3, 1)
        scale = scale.permute(0, 2, 3, 1)

        # Uncertainty decomposition
        sample_mean = samples.mean(dim=0).permute(0, 2, 3, 1)  # Expected value
        aleatoric = scale  # Learned scale (irreducible uncertainty)
        epistemic = samples.std(dim=0).permute(0, 2, 3, 1)  # Sample variance (model uncertainty)

        return sample_mean, aleatoric, epistemic


# ============================================================================
# TRAINER MODIFICATION FOR EVIDENTIAL FNO
# ============================================================================

# Extend FNOTrainer to handle evidential models
# Add this to the existing FNOTrainer.train_step method:

def is_evidential_model(model):
    """Check if model is evidential (returns 4 outputs)."""
    return isinstance(model, (EvidentialFNO1d, EvidentialFNO2d))

