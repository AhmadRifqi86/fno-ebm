"""
Advanced FNO Architectures for Complex PDEs

This module contains state-of-the-art FNO variants:
- F-FNO (Factorized FNO): Separable spectral layers for parameter efficiency
- U-FNO (U-shaped FNO): Multi-scale encoder-decoder for complex patterns

Based on:
- F-FNO: "Factorized Fourier Neural Operators" (ICLR 2023)
- U-FNO: "Equivariant U-shaped Neural Operators" (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Factorized Spectral Convolution (F-FNO Core Component)
# ============================================================================

class FactorizedSpectralConv2d(nn.Module):
    """
    Factorized spectral convolution using separable kernels.

    Instead of full 2D kernel W(kx, ky), uses W_x(kx) ⊗ W_y(ky).
    This reduces parameters from O(modes1 * modes2) to O(modes1 + modes2).

    Key for data-limited scenarios (e.g., 1000 training samples).
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)

        # Factorized weights: separate for each dimension
        # W_x: weights for x-direction modes
        self.weights_x1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2)
        )
        self.weights_x2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2)
        )

        # W_y: weights for y-direction modes
        self.weights_y = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes2, 2)
        )

    def compl_mul1d(self, input, weights):
        """Complex multiplication for 1D factorized weights"""
        # input: (batch, in_channel, x, 2)
        # weights: (in_channel, out_channel, x, 2)
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
        2. Apply separable 1D convolutions in x and y
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

        # Apply factorized convolution
        # Upper-left quadrant
        x_modes = x_ft[:, :, :self.modes1, :self.modes2]  # (batch, in_ch, modes1, modes2, 2)

        # Factorize: convolve x-modes first, then y-modes
        # Step 1: Convolve along x-dimension
        temp = torch.zeros(batch_size, self.out_channels, self.modes1, self.modes2, 2,
                          device=x.device)
        for j in range(self.modes2):
            temp[:, :, :, j, :] = self.compl_mul1d(x_modes[:, :, :, j, :], self.weights_x1)

        # Step 2: Convolve along y-dimension
        for i in range(self.modes1):
            out_ft[:, :, i, :self.modes2, :] = self.compl_mul1d(
                temp[:, :, i, :, :], self.weights_y
            )

        # Lower-left quadrant
        x_modes = x_ft[:, :, -self.modes1:, :self.modes2]
        temp = torch.zeros(batch_size, self.out_channels, self.modes1, self.modes2, 2,
                          device=x.device)
        for j in range(self.modes2):
            temp[:, :, :, j, :] = self.compl_mul1d(x_modes[:, :, :, j, :], self.weights_x2)

        for i in range(self.modes1):
            out_ft[:, :, -self.modes1 + i, :self.modes2, :] = self.compl_mul1d(
                temp[:, :, i, :, :], self.weights_y
            )

        # Convert back to complex and IFFT
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x_out = torch.fft.irfft2(out_ft_complex, s=(x.size(-2), x.size(-1)), norm='ortho')

        return x_out


# ============================================================================
# Channel Attention (Squeeze-and-Excitation)
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention for Fourier modes.

    Learns which frequency channels (Fourier modes) are important for the current input.
    - Low modes (1-8): Global topology
    - Medium modes (8-16): Periodic structures
    - High modes (16-24): Fine boundaries

    Only ~1% parameter overhead: O(width²/reduction) vs O(width * spatial²)
    """
    def __init__(self, channels, reduction=4):
        """
        Args:
            channels: Number of input channels (FNO width)
            reduction: Bottleneck reduction factor (higher = fewer params)
        """
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # Squeeze: global average pooling (done in forward)
        # Excitation: MLP with bottleneck
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            x_attended: (batch, channels, height, width) with channel attention applied
        """
        batch, channels, height, width = x.shape

        # Squeeze: global average pooling across spatial dimensions
        # (batch, channels, H, W) -> (batch, channels)
        squeeze = F.adaptive_avg_pool2d(x, 1).view(batch, channels)

        # Excitation: MLP to learn channel importance
        excitation = self.fc1(squeeze)  # (batch, channels//reduction)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)  # (batch, channels)
        excitation = torch.sigmoid(excitation)  # (batch, channels)

        # Reshape for broadcasting: (batch, channels, 1, 1)
        excitation = excitation.view(batch, channels, 1, 1)

        # Scale features by learned attention weights
        return x * excitation


# ============================================================================
# F-FNO (Factorized Fourier Neural Operator)
# ============================================================================

class FFNO2d(nn.Module):
    """
    Factorized Fourier Neural Operator for 2D problems.

    Key improvements over standard FNO:
    - Factorized spectral layers (5% of parameters vs dense FNO)
    - Improved residual connections (enables deeper networks)
    - Better for data-limited scenarios
    - Optional channel attention (learns which Fourier modes matter)

    Performance:
    - 83% error reduction on Navier-Stokes
    - 57% improvement on airfoil flow
    - Can scale to 24 layers (vs 4 in standard FNO)
    """
    def __init__(self, modes1, modes2, width=64, num_layers=4, dropout=0.0,
                 use_attention=False, attention_reduction=4):
        """
        Args:
            modes1, modes2: Number of Fourier modes
            width: Hidden dimension (channel width)
            num_layers: Number of FNO layers
            dropout: Dropout rate
            use_attention: Enable channel attention (Squeeze-and-Excitation)
            attention_reduction: Bottleneck reduction for attention (higher = fewer params)
        """
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention

        # Input projection
        self.fc0 = nn.Linear(3, self.width)
        self.dropout0 = nn.Dropout(dropout)

        # Factorized Fourier layers
        self.conv_layers = nn.ModuleList([
            FactorizedSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(self.num_layers)
        ])

        # Local (non-spectral) connection
        self.w_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.num_layers)
        ])

        # Channel attention (optional, for learning mode importance)
        if self.use_attention:
            self.attention_layers = nn.ModuleList([
                ChannelAttention(self.width, reduction=attention_reduction)
                for _ in range(self.num_layers)
            ])
        else:
            self.attention_layers = None

        # Improved residual connections (layer normalization)
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.width)
            for _ in range(self.num_layers)
        ])

        # Dropout after each layer
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(self.num_layers)
        ])

        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.dropout_out = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        x: (batch, n_x, n_y, 3) - input with coordinates
        returns: (batch, n_x, n_y, 1) - predicted solution
        """
        # Lift to higher dimension
        x = self.fc0(x)  # (batch, n_x, n_y, width)
        x = self.dropout0(x)

        x = x.permute(0, 3, 1, 2)  # (batch, width, n_x, n_y)

        # Factorized Fourier layers with improved residuals
        for i in range(self.num_layers):
            x_res = x

            # Factorized spectral convolution
            x1 = self.conv_layers[i](x)

            # Local convolution
            x2 = self.w_layers[i](x)

            # Combine paths
            x = x1 + x2

            # Channel attention (optional): learn which Fourier modes are important
            if self.use_attention:
                x = self.attention_layers[i](x)

            # Activation
            if i < self.num_layers - 1:
                x = F.gelu(x)

            # Improved residual: add back input with layer norm
            x = x + x_res

            # Normalize for stable deep training
            x = x.permute(0, 2, 3, 1)  # (batch, n_x, n_y, width)
            x = self.norm_layers[i](x)
            x = x.permute(0, 3, 1, 2)  # (batch, width, n_x, n_y)

            # Dropout
            if i < self.num_layers - 1:
                x = self.dropout_layers[i](x)

        # Project to output
        x = x.permute(0, 2, 3, 1)  # (batch, n_x, n_y, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout_out(x)
        x = self.fc2(x)  # (batch, n_x, n_y, 1)

        return x


# ============================================================================
# U-FNO (U-shaped Fourier Neural Operator)
# ============================================================================

class FNOBlock(nn.Module):
    """Basic FNO block for U-FNO encoder/decoder"""
    def __init__(self, in_channels, out_channels, modes1, modes2, activation=True):
        super().__init__()
        from fno import SpectralConv2d  # Import from main fno.py

        self.activation = activation
        self.conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.w = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (batch, channels, n_x, n_y)
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if self.activation:
            x = F.gelu(x)
        return x


class UFNO2d(nn.Module):
    """
    U-shaped Fourier Neural Operator for multi-scale problems.

    Architecture:
    - Encoder path: progressively downsample and increase channels
    - Decoder path: progressively upsample with skip connections
    - Combines global Fourier modes with local multi-scale features

    Key advantages:
    - Order of magnitude better than FNO on multi-scale patterns
    - Captures both coarse topology and fine boundaries
    - Super-resolution capability

    Ideal for: Reaction-diffusion, phase-field, turbulence
    """
    def __init__(self, modes1, modes2, width=64, depth=3, dropout=0.0):
        """
        Args:
            modes1, modes2: Fourier modes (halved at each downsampling level)
            width: Base channel width (doubled at each downsampling level)
            depth: Number of downsampling levels (U-Net depth)
            dropout: Dropout rate
        """
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.dropout = dropout

        # Input projection
        self.fc0 = nn.Linear(3, width)
        self.dropout0 = nn.Dropout(dropout)

        # Encoder path (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for i in range(depth):
            in_ch = width * (2 ** i)
            out_ch = width * (2 ** (i + 1))
            modes_x = max(modes1 // (2 ** i), 4)  # Don't go below 4 modes
            modes_y = max(modes2 // (2 ** i), 4)

            # FNO block at this scale
            self.encoder_blocks.append(
                FNOBlock(in_ch, in_ch, modes_x, modes_y, activation=True)
            )

            # Downsample: strided conv to reduce spatial resolution
            self.downsample_layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )

        # Bottleneck (deepest level)
        bottleneck_ch = width * (2 ** depth)
        bottleneck_modes_x = max(modes1 // (2 ** depth), 4)
        bottleneck_modes_y = max(modes2 // (2 ** depth), 4)
        self.bottleneck = FNOBlock(bottleneck_ch, bottleneck_ch,
                                   bottleneck_modes_x, bottleneck_modes_y,
                                   activation=True)

        # Decoder path (upsampling with skip connections)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            in_ch = width * (2 ** (i + 1))
            out_ch = width * (2 ** i)
            modes_x = max(modes1 // (2 ** i), 4)
            modes_y = max(modes2 // (2 ** i), 4)

            # Upsample: transposed conv to increase spatial resolution
            self.upsample_layers.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                  padding=1, output_padding=1)
            )

            # Skip connection: concatenate encoder features
            # Input: out_ch (from upsample) + out_ch (from skip) = 2*out_ch
            self.skip_connections.append(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=1)
            )

            # FNO block at this scale
            self.decoder_blocks.append(
                FNOBlock(out_ch, out_ch, modes_x, modes_y, activation=True)
            )

        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(depth * 2 + 1)
        ])

        # Output projection
        self.fc1 = nn.Linear(width, 128)
        self.dropout_out = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        x: (batch, n_x, n_y, 3) - input with coordinates
        returns: (batch, n_x, n_y, 1) - predicted solution
        """
        # Lift to higher dimension
        x = self.fc0(x)  # (batch, n_x, n_y, width)
        x = self.dropout0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, n_x, n_y)

        # Encoder path with skip connection storage
        encoder_features = []

        for i in range(self.depth):
            # FNO processing at this scale
            x = self.encoder_blocks[i](x)
            x = self.dropout_layers[i](x)

            # Store for skip connection
            encoder_features.append(x)

            # Downsample to next scale
            x = self.downsample_layers[i](x)
            x = F.gelu(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout_layers[self.depth](x)

        # Decoder path with skip connections
        for i in range(self.depth):
            # Upsample
            x = self.upsample_layers[i](x)
            x = F.gelu(x)

            # Skip connection: concatenate with encoder features
            skip_feat = encoder_features[self.depth - 1 - i]

            # Handle potential size mismatch from downsampling/upsampling
            if x.shape[2:] != skip_feat.shape[2:]:
                x = F.interpolate(x, size=skip_feat.shape[2:],
                                mode='bilinear', align_corners=False)

            x = torch.cat([x, skip_feat], dim=1)
            x = self.skip_connections[i](x)

            # FNO processing at this scale
            x = self.decoder_blocks[i](x)
            x = self.dropout_layers[self.depth + 1 + i](x)

        # Project to output
        x = x.permute(0, 2, 3, 1)  # (batch, n_x, n_y, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout_out(x)
        x = self.fc2(x)  # (batch, n_x, n_y, 1)

        return x