"""
KAN-EBM: Kolmogorov-Arnold Network Energy-Based Model for UQ Paper

This module contains:
- KANLinear: Efficient B-spline based KAN layer (from efficient-kan)
- KAN: Multi-layer KAN network
- KANEBM: KAN-based energy model for uncertainty quantification
- EBMWrapper: Training wrapper with score matching and Langevin sampling

Based on:
- Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
- Blealtan's efficient-kan: https://github.com/Blealtan/efficient-kan
- Song & Ermon, "Generative Modeling by Estimating Gradients" (NeurIPS 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List
from tqdm import tqdm
import logging
import numpy as np


# ============================================================================
# Efficient KAN Implementation (B-spline basis)
# ============================================================================

class KANLinear(nn.Module):
    """
    Efficient KAN layer using B-spline basis functions.

    Instead of fixed activations σ(Wx), uses learnable univariate functions:
        h_out[j] = Σ_i φ_{i,j}(h_in[i])
    where φ_{i,j} are B-spline basis combinations.

    Reference: https://github.com/Blealtan/efficient-kan
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            grid_size: Number of intervals for B-spline grid (G in paper)
            spline_order: Order of B-spline (k in paper, typically 3 for cubic)
            scale_noise: Initial noise scale for spline coefficients
            scale_base: Scaling for base activation
            scale_spline: Scaling for spline activation
            enable_standalone_scale_spline: Use separate learnable scale for splines
            base_activation: Base activation function (default: SiLU)
            grid_eps: Grid adaptivity factor (0=adaptive, 1=uniform)
            grid_range: Initial grid range [min, max]
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create B-spline grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Learnable parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # Configuration
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            # Initialize spline weights with small noise
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions for input x.

        Uses Cox-de Boor recursion formula:
        B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0
        B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x) +
                     (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)

        Args:
            x: Input tensor (batch, in_features)

        Returns:
            bases: B-spline basis values (batch, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)

        # Order 0: indicator functions
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Recursive computation for higher orders
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Fit B-spline coefficients to data points using least squares.

        Solves: A @ coeff = y, where A is B-spline basis matrix

        Args:
            x: Input points (n_points, in_features)
            y: Target values (n_points, in_features, out_features)

        Returns:
            coeff: Spline coefficients (out_features, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # Compute B-spline basis
        A = self.b_splines(x).transpose(0, 1)  # (in_features, n_points, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, n_points, out_features)

        # Solve least squares
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """Apply scaling to spline weights."""
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass: output = base_activation(x) @ W_base + B-spline(x) @ W_spline

        Args:
            x: Input tensor (..., in_features)

        Returns:
            output: Output tensor (..., out_features)
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # Base output: standard linear layer with activation
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Spline output: B-spline basis @ spline weights
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )

        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Update B-spline grid based on input distribution.

        Uses mixture of uniform and adaptive grids:
        grid = eps * grid_uniform + (1 - eps) * grid_adaptive

        Args:
            x: Input samples (batch, in_features)
            margin: Margin to add beyond data range
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # Compute current spline outputs
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        # Adaptive grid: percentiles of input distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # Uniform grid: evenly spaced over data range
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # Blend grids
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # Extend grid for B-spline order
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # Update buffers and parameters
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation: float = 1.0,
                           regularize_entropy: float = 1.0):
        """
        Compute regularization loss for sparsity and smoothness.

        L_reg = λ_act * ||W_spline||_1 + λ_ent * H(p)
        where p is normalized spline weight distribution

        Args:
            regularize_activation: Weight for L1 sparsity
            regularize_entropy: Weight for entropy regularization

        Returns:
            loss: Regularization loss scalar
        """
        # L1 norm (encourages sparsity)
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()

        # Entropy (encourages diversity)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())

        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(nn.Module):
    """
    Multi-layer Kolmogorov-Arnold Network.

    Stacks multiple KANLinear layers to form a deep network with
    learnable activation functions.
    """
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
    ):
        """
        Args:
            layers_hidden: List of hidden dimensions [in_dim, h1, h2, ..., out_dim]
            grid_size: Number of B-spline grid intervals
            spline_order: B-spline order (3 for cubic)
            scale_noise: Initialization noise scale
            scale_base: Base activation scaling
            scale_spline: Spline activation scaling
            base_activation: Base activation function
            grid_eps: Grid adaptivity (0=adaptive, 1=uniform)
            grid_range: Initial grid range
        """
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid: bool = False):
        """
        Forward pass through all KAN layers.

        Args:
            x: Input tensor
            update_grid: Whether to update B-spline grids based on input

        Returns:
            output: Network output
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation: float = 1.0,
                           regularize_entropy: float = 1.0):
        """Sum regularization losses from all layers."""
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


# ============================================================================
# Self-Attention Module for MLP
# ============================================================================

class SelfAttention(nn.Module):
    """
    Self-Attention mechanism for MLP-based energy models.

    Uses scaled dot-product attention to allow features to attend to each other.
    This helps the energy model capture feature dependencies and interactions.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)

        Returns:
            output: Attended features (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)

        return out


# ============================================================================
# MLP-based Energy Network with Optional Attention
# ============================================================================

class MLPEnergyNet(nn.Module):
    """
    MLP-based energy network with optional self-attention.

    Architecture:
        Input → [Attention] → MLP Blocks → Output (scalar energy)

    Each MLP block consists of: Linear → LayerNorm → Activation → Dropout
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            use_attention: Whether to use self-attention
            num_heads: Number of attention heads (if use_attention=True)
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', 'silu')
        """
        super().__init__()
        self.input_dim = input_dim
        self.use_attention = use_attention

        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Optional self-attention (applied after input projection)
        if use_attention:
            self.input_proj = nn.Linear(input_dim, hidden_dims[0])
            self.attention = SelfAttention(hidden_dims[0], num_heads, dropout)
            self.attn_norm = nn.LayerNorm(hidden_dims[0])
            start_idx = 0
        else:
            start_idx = 0

        # MLP blocks
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer
        if not use_attention:
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.norms.append(nn.LayerNorm(hidden_dims[i + 1]))
            self.dropouts.append(nn.Dropout(dropout))

        # Output layer (to scalar energy)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            energy: Scalar energy (batch,)
        """
        # Apply attention if enabled
        if self.use_attention:
            x = self.input_proj(x)  # (batch, hidden_dims[0])
            x = x.unsqueeze(1)  # (batch, 1, hidden_dims[0]) - treat as seq_len=1
            x = self.attention(x)  # (batch, 1, hidden_dims[0])
            x = self.attn_norm(x)
            x = x.squeeze(1)  # (batch, hidden_dims[0])

            # MLP blocks (skip first layer since we already projected)
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                if i < len(self.norms):
                    x = self.norms[i](x)
                x = self.activation(x)
                if i < len(self.dropouts):
                    x = self.dropouts[i](x)
        else:
            # Standard MLP forward
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.norms):
                    x = self.norms[i](x)
                x = self.activation(x)
                if i < len(self.dropouts):
                    x = self.dropouts[i](x)

        # Output
        energy = self.output_layer(x).squeeze(-1)  # (batch,)
        return energy


# ============================================================================
# CNN-based Energy Network with Attention
# ============================================================================

class SpatialAttention(nn.Module):
    """Spatial attention module for highlighting important spatial regions."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (batch, channels, H, W)
        Returns:
            x_attended: (batch, channels, H, W)
        """
        # Aggregate channel information
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (batch, 1, H, W)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, 2, H, W)

        # Compute attention map
        attention = self.sigmoid(self.conv(pooled))  # (batch, 1, H, W)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention (Squeeze-and-Excitation) for feature recalibration."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, H, W)
        Returns:
            x_attended: (batch, channels, H, W)
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """Convolutional block with optional attention."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        use_spatial_attn=False,
        use_channel_attn=False,
        dropout=0.1
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

        # Attention modules
        self.use_spatial_attn = use_spatial_attn
        self.use_channel_attn = use_channel_attn
        if use_spatial_attn:
            self.spatial_attn = SpatialAttention()
        if use_channel_attn:
            self.channel_attn = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # Apply attention
        if self.use_spatial_attn:
            x = self.spatial_attn(x)
        if self.use_channel_attn:
            x = self.channel_attn(x)

        x = self.dropout(x)
        return x


class ConvEnergyNet(nn.Module):
    """
    CNN-based energy network with attention for 2D spatial data.

    Architecture:
        Input (batch, in_channels, H, W)
        → Conv Blocks with attention (gradually increase channels, reduce spatial)
        → Global pooling
        → MLP head → scalar energy

    Much more efficient than flattened MLP for 2D PDEs (128×128 grid).
    """
    def __init__(
        self,
        in_channels: int = 3,  # [u, x, u_fno] for 1D, or channels for 2D
        base_channels: int = 64,
        num_blocks: int = 4,
        use_spatial_attn: bool = True,
        use_channel_attn: bool = True,
        dropout: float = 0.1,
        mlp_hidden: int = 256,
    ):
        """
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels (doubles each downsampling)
            num_blocks: Number of convolutional blocks
            use_spatial_attn: Use spatial attention in blocks
            use_channel_attn: Use channel attention (SE) in blocks
            dropout: Dropout rate
            mlp_hidden: Hidden dimension for final MLP head
        """
        super().__init__()
        self.in_channels = in_channels

        # Build convolutional blocks
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_blocks):
            out_channels = base_channels * (2 ** (i // 2))  # Double channels every 2 blocks
            stride = 2 if i % 2 == 1 else 1  # Downsample every other block

            # Add attention to later blocks (more abstract features)
            use_attn_here = (use_spatial_attn or use_channel_attn) and i >= num_blocks // 2

            self.blocks.append(ConvBlock(
                current_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                use_spatial_attn=use_spatial_attn and use_attn_here,
                use_channel_attn=use_channel_attn and use_attn_here,
                dropout=dropout
            ))
            current_channels = out_channels

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Simple linear head for energy output (channels → 1)
        self.mlp_head = nn.Linear(current_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, H, W) for 2D
               or (batch, in_channels, N, 1) for 1D (will squeeze)

        Returns:
            energy: Scalar energy (batch,)
        """
        # Handle 1D case: (batch, channels, N, 1) → treat as (batch, channels, N, N)
        # or just process as-is with 2D convs

        # Apply convolutional blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling: (batch, channels, H, W) → (batch, channels, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch, channels)

        # MLP head to scalar energy
        energy = self.mlp_head(x).squeeze(-1)  # (batch,)
        return energy


# ============================================================================
# Flexible EBM: Accepts Any Energy Network (KAN or MLP)
# ============================================================================

class EBM(nn.Module):
    """
    Flexible Energy-Based Model for Uncertainty Quantification.

    Accepts any neural network (KAN, MLP, etc.) as the energy function.

    Learns an energy function E(u | x) or E(u | x, û) where:
    - u: predicted field (to be refined)
    - x: input coordinates/conditions
    - û: FNO prediction (optional conditioning)

    Lower energy = more likely prediction.
    """
    def __init__(
        self,
        energy_net: Optional[nn.Module] = None,
        input_dim: Optional[int] = None,
        condition_dim: int = 0,
        hidden_dims: List[int] = [128, 64, 32],
        condition_on_fno: bool = True,
        # KAN-specific parameters
        use_kan: bool = False,
        grid_size: int = 5,
        spline_order: int = 3,
        # MLP-specific parameters
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        """
        Args:
            energy_net: Pre-built energy network (KAN, MLP, etc.)
                       If None, builds network based on use_kan flag
            input_dim: Dimension of predicted field u (flattened)
                      Required if energy_net is None
            condition_dim: Dimension of conditioning input x (coordinates, etc.)
            hidden_dims: Hidden layer dimensions
            condition_on_fno: Whether to condition on FNO prediction

            # KAN-specific
            use_kan: If True and energy_net is None, builds KAN network
            grid_size: B-spline grid size (for KAN)
            spline_order: B-spline order (for KAN)

            # MLP-specific
            use_attention: Whether to use self-attention (for MLP)
            num_heads: Number of attention heads (for MLP)
            dropout: Dropout rate (for MLP)
            activation: Activation function (for MLP)
        """
        super().__init__()
        self.condition_on_fno = condition_on_fno

        # Compute total input dimension
        if energy_net is None and input_dim is None:
            raise ValueError("Must provide either energy_net or input_dim")

        if input_dim is not None:
            self.input_dim = input_dim
            self.condition_dim = condition_dim
            # input_dim is ALREADY the total flattened size (u + x + u_fno)
            total_input_dim = input_dim

        # Build or assign energy network
        if energy_net is not None:
            # Use provided network
            self.energy_net = energy_net
        elif use_kan:
            # Build KAN network
            layers = [total_input_dim] + hidden_dims + [1]
            self.energy_net = KAN(
                layers_hidden=layers,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
            )
        else:
            # Build MLP network
            self.energy_net = MLPEnergyNet(
                input_dim=total_input_dim,
                hidden_dims=hidden_dims,
                use_attention=use_attention,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )

    def forward(self, u: torch.Tensor, x: torch.Tensor,
                u_fno: Optional[torch.Tensor] = None):
        """
        Compute energy E(u | x, û).

        Args:
            u: Predicted field (batch, n_x, [n_y], channels)
            x: Input coordinates (batch, n_x, [n_y], coord_channels)
            u_fno: FNO prediction (optional, same shape as u)

        Returns:
            energy: Scalar energy for each sample (batch,)
        """
        batch_size = u.shape[0]

        # Check if using CNN-based energy network
        is_cnn = isinstance(self.energy_net, ConvEnergyNet)

        if is_cnn:
            # CNN path: Keep spatial structure, stack channels
            # Input shapes: u (batch, H, W, 1), x (batch, H, W, coord_ch), u_fno (batch, H, W, 1)
            # Need to convert to (batch, channels, H, W)

            if len(u.shape) == 4:  # 2D: (batch, H, W, channels)
                # Permute to (batch, channels, H, W)
                u_spatial = u.permute(0, 3, 1, 2)  # (batch, 1, H, W)
                x_spatial = x.permute(0, 3, 1, 2)  # (batch, coord_ch, H, W)

                if self.condition_on_fno and u_fno is not None:
                    u_fno_spatial = u_fno.permute(0, 3, 1, 2)  # (batch, 1, H, W)
                    inputs = torch.cat([u_spatial, x_spatial, u_fno_spatial], dim=1)
                else:
                    inputs = torch.cat([u_spatial, x_spatial], dim=1)

            elif len(u.shape) == 3:  # 1D: (batch, N, channels)
                # Convert to (batch, channels, N, 1) for 2D conv
                u_spatial = u.permute(0, 2, 1).unsqueeze(-1)  # (batch, 1, N, 1)
                x_spatial = x.permute(0, 2, 1).unsqueeze(-1)  # (batch, coord_ch, N, 1)

                if self.condition_on_fno and u_fno is not None:
                    u_fno_spatial = u_fno.permute(0, 2, 1).unsqueeze(-1)  # (batch, 1, N, 1)
                    inputs = torch.cat([u_spatial, x_spatial, u_fno_spatial], dim=1)
                else:
                    inputs = torch.cat([u_spatial, x_spatial], dim=1)
            else:
                raise ValueError(f"Unexpected input shape for CNN: {u.shape}")

            # Debug
            if not hasattr(self, '_dim_checked'):
                print(f"[CNN-EBM Debug] u shape: {u.shape} → {u_spatial.shape}")
                print(f"[CNN-EBM Debug] x shape: {x.shape} → {x_spatial.shape}")
                if u_fno is not None:
                    print(f"[CNN-EBM Debug] u_fno shape: {u_fno.shape} → {u_fno_spatial.shape}")
                print(f"[CNN-EBM Debug] CNN input shape: {inputs.shape}")
                self._dim_checked = True

        else:
            # MLP path: Flatten everything
            u_flat = u.reshape(batch_size, -1)
            x_flat = x.reshape(batch_size, -1)

            if self.condition_on_fno and u_fno is not None:
                u_fno_flat = u_fno.reshape(batch_size, -1)
                inputs = torch.cat([u_flat, x_flat, u_fno_flat], dim=-1)
            else:
                inputs = torch.cat([u_flat, x_flat], dim=-1)

            # Debug
            if not hasattr(self, '_dim_checked'):
                print(f"[MLP-EBM Debug] u shape: {u.shape}, u_flat: {u_flat.shape}")
                print(f"[MLP-EBM Debug] x shape: {x.shape}, x_flat: {x_flat.shape}")
                if u_fno is not None:
                    print(f"[MLP-EBM Debug] u_fno shape: {u_fno.shape}, u_fno_flat: {u_fno_flat.shape}")
                print(f"[MLP-EBM Debug] MLP input shape: {inputs.shape}")
                print(f"[MLP-EBM Debug] Expected input_dim: {self.input_dim}")
                self._dim_checked = True

        # Compute energy through network
        energy = self.energy_net(inputs)

        # Handle different output formats
        if energy.dim() > 1:
            energy = energy.squeeze(-1)  # (batch,)

        return energy


# Backward compatibility: KANEBM is now an alias for EBM with use_kan=True
class KANEBM(EBM):
    """
    KAN-based Energy Model (backward compatibility wrapper).

    This is now a wrapper around the flexible EBM class with use_kan=True.
    """
    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 0,
        hidden_dims: List[int] = [64, 32, 16],
        grid_size: int = 5,
        spline_order: int = 3,
        condition_on_fno: bool = True,
    ):
        super().__init__(
            energy_net=None,
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dims=hidden_dims,
            condition_on_fno=condition_on_fno,
            use_kan=True,
            grid_size=grid_size,
            spline_order=spline_order,
        )


# ============================================================================
# EBMWrapper: Training Wrapper for Energy-Based Models
# ============================================================================

class EBMTrainer:
    """
    Training wrapper for Energy-Based Models (KAN-EBM, MLP-EBM, etc.) with score matching and Langevin sampling.

    Supports two training objectives:
    1. Score Matching: Match ∇_u E(u) to denoising score
    2. Contrastive Divergence: Minimize E(u_data) - E(u_neg)

    Compatible with any EBM architecture (KAN, MLP, custom networks).
    """
    def __init__(
        self,
        model: nn.Module,  # Can be EBM, KANEBM, or any custom energy model
        config: Dict,
        fno_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Args:
            model: Energy-based model (EBM, KANEBM, or custom)
            config: Configuration dictionary
            fno_model: Optional FNO model for conditioning
            optimizer: Optional custom optimizer
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.fno_model = fno_model
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        if fno_model is not None:
            self.fno_model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            lr = getattr(config, 'lr', 5e-4)
            weight_decay = getattr(config, 'weight_decay', 1e-5)
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Training configuration
        self.score_matching_type = getattr(config, 'score_matching_type', 'weighted')
        self.noise_levels = getattr(config, 'noise_levels', [0.01, 0.02, 0.05])
        self.noise_weights = getattr(config, 'noise_weights', {0.01: 0.2, 0.02: 0.3, 0.05: 0.5})

        # Langevin sampling config
        self.langevin_steps_train = getattr(config, 'langevin_steps_train', 20)
        self.langevin_step_size_train = getattr(config, 'langevin_step_size_train', 0.01)
        self.langevin_steps_inference = getattr(config, 'langevin_steps_inference', 50)
        self.langevin_step_size_inference = getattr(config, 'langevin_step_size_inference', 0.01)

        # Loss weights
        self.energy_reg_weight = getattr(config, 'energy_reg_weight', 0.001)
        self.calibration_weight = getattr(config, 'calibration_weight', 0.5)
        self.use_error_aware_loss = getattr(config, 'use_error_aware_loss', True)

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
            experiment_name = getattr(config, 'experiment_name', 'ebm_training')

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

    def score_matching_loss(
        self,
        u_clean: torch.Tensor,
        x: torch.Tensor,
        u_fno: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Weighted score matching loss with multiple noise levels.

        Loss = Σ_σ w_σ · ||∇_u E(u+ε) - (-ε/σ²)||²

        Args:
            u_clean: Clean field samples
            x: Input coordinates
            u_fno: FNO predictions (optional)

        Returns:
            loss: Score matching loss
            diagnostics: Dictionary with per-level losses
        """
        total_loss = 0.0
        diagnostics = {}

        for sigma in self.noise_levels:
            # Add noise
            noise = torch.randn_like(u_clean) * sigma
            u_noisy = u_clean + noise
            u_noisy.requires_grad_(True)

            # Compute energy
            energy = self.model(u_noisy, x, u_fno)

            # Compute score: ∇_u E(u)
            predicted_score = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=u_noisy,
                create_graph=True
            )[0]

            # Target score: -ε/σ²
            target_score = -noise / (sigma ** 2)

            # MSE loss
            level_loss = F.mse_loss(predicted_score, target_score)

            # Apply weight
            weight = self.noise_weights.get(sigma, 1.0 / len(self.noise_levels))
            weighted_loss = weight * level_loss
            total_loss += weighted_loss

            # Diagnostics
            diagnostics[f'loss_sigma_{sigma}'] = level_loss.item()
            diagnostics[f'score_norm_sigma_{sigma}'] = predicted_score.norm().item()

        diagnostics['total_score_loss'] = total_loss.item()
        return total_loss, diagnostics

    def contrastive_divergence_loss(
        self,
        u_pos: torch.Tensor,
        x: torch.Tensor,
        u_fno: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Contrastive divergence loss with Langevin sampling.

        Loss = E(u_pos) - E(u_neg)
        where u_neg is sampled via Langevin dynamics

        Args:
            u_pos: Positive samples (ground truth)
            x: Input coordinates
            u_fno: FNO predictions (optional)

        Returns:
            loss: CD loss
            u_neg: Negative samples
            diagnostics: Dictionary with energy values
        """
        # Positive energy
        pos_energy = self.model(u_pos, x, u_fno)

        # Negative sampling via Langevin
        u_neg = u_pos + 0.2 * torch.randn_like(u_pos)  # Initialize near data
        u_neg = u_neg.detach()

        noise_scale = np.sqrt(2 * self.langevin_step_size_train)

        for _ in range(self.langevin_steps_train):
            u_neg.requires_grad_(True)
            neg_energy = self.model(u_neg, x, u_fno)

            neg_grad = torch.autograd.grad(
                outputs=neg_energy.sum(),
                inputs=u_neg,
                create_graph=True
            )[0]

            with torch.no_grad():
                noise = torch.randn_like(u_neg) * noise_scale
                u_neg = u_neg - self.langevin_step_size_train * neg_grad + noise

            u_neg = u_neg.detach()

        # Final negative energy
        neg_energy = self.model(u_neg, x, u_fno)

        # CD loss
        cd_loss = pos_energy.mean() - neg_energy.mean()

        diagnostics = {
            'pos_energy': pos_energy.mean().item(),
            'neg_energy': neg_energy.mean().item(),
            'cd_loss': cd_loss.item()
        }

        return cd_loss, u_neg, diagnostics

    def train_step(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        u_gt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step for EBM.

        Args:
            u: Input field (clean or FNO prediction)
            x: Input coordinates
            u_gt: Ground truth (optional, for error-aware loss)

        Returns:
            loss: Total loss
            loss_dict: Dictionary with loss components
        """
        self.model.train()
        u, x = u.to(self.device), x.to(self.device)
        if u_gt is not None:
            u_gt = u_gt.to(self.device)

        # Get FNO prediction if available
        u_fno = None
        if self.fno_model is not None:
            with torch.no_grad():
                u_fno = self.fno_model(x)

        # Score matching loss
        score_loss, score_diag = self.score_matching_loss(u, x, u_fno)

        # Energy regularization
        energy = self.model(u, x, u_fno)
        energy_reg = torch.mean(energy ** 2)

        # Combined loss
        total_loss = score_loss + self.energy_reg_weight * energy_reg

        # Error-aware calibration loss (if ground truth available)
        if self.use_error_aware_loss and u_gt is not None and u_fno is not None:
            # Import from customs
            from customs import error_aware_ebm_loss

            # Compute EBM uncertainty (from score norm)
            u_temp = u.detach().clone()
            u_temp.requires_grad_(True)
            energy_temp = self.model(u_temp, x, u_fno)
            ebm_score = torch.autograd.grad(
                outputs=energy_temp.sum(),
                inputs=u_temp,
                create_graph=False
            )[0]
            ebm_std = torch.norm(ebm_score, dim=-1)

            #calib_loss = error_aware_ebm_loss(ebm_std, u_fno, u_gt)
            #total_loss += self.calibration_weight * calib_loss
            #score_diag['calibration'] = calib_loss.item()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Track gradients (after clipping, before optimizer step)
        if self.enable_tracking:
            if self.gradient_tracker is not None:
                # Custom tracking with anomaly detection
                custom_metrics = {
                    'score_loss': score_loss.item(),
                    'energy_reg': energy_reg.item(),
                }
                # Add per-noise-level diagnostics
                for key, val in score_diag.items():
                    if key != 'total_score_loss':
                        custom_metrics[key] = val
                self.gradient_tracker.track(loss=total_loss, custom_metrics=custom_metrics)
            elif self.writer is not None:
                # Built-in TensorBoard logging
                self.writer.add_scalar('train/loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('train/score_loss', score_loss.item(), self.global_step)
                self.writer.add_scalar('train/energy_reg', energy_reg.item(), self.global_step)

                # Log per-noise-level diagnostics
                for key, val in score_diag.items():
                    if key != 'total_score_loss':
                        self.writer.add_scalar(f'train/{key}', val, self.global_step)

                # Log gradients manually
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'gradients/{name}', param.grad, self.global_step)
                        self.writer.add_scalar(f'gradient_norm/{name}', param.grad.norm().item(), self.global_step)

                self.global_step += 1

        self.optimizer.step()

        # Compile loss dict
        loss_dict = {
            'total': total_loss.item(),
            'score': score_loss.item(),
            'energy_reg': energy_reg.item(),
            **score_diag
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def validate(self, val_loader) -> float:
        """
        Validation step using score matching loss.

        Args:
            val_loader: Validation data loader

        Returns:
            avg_val_loss: Average validation loss
        """
        self.model.eval()
        total_val_loss = 0

        for x, u in val_loader:
            x, u = x.to(self.device), u.to(self.device)

            # Get FNO prediction
            u_fno = None
            if self.fno_model is not None:
                u_fno = self.fno_model(x)

            # Score matching loss (enable gradients temporarily for validation)
            batch_loss = 0.0
            for sigma in self.noise_levels:
                noise = torch.randn_like(u)
                u_noisy = u + sigma * noise

                # Enable gradients for u_noisy within this scope
                with torch.enable_grad():
                    u_noisy.requires_grad_(True)
                    energy = self.model(u_noisy, x, u_fno)
                    score = torch.autograd.grad(
                        outputs=energy.sum(),
                        inputs=u_noisy,
                        create_graph=False
                    )[0]

                target_score = -noise / sigma
                score_loss = torch.mean((score - target_score) ** 2)
                batch_loss += score_loss.item()

            batch_loss /= len(self.noise_levels)
            total_val_loss += batch_loss

        avg_val_loss = total_val_loss / len(val_loader)
        return avg_val_loss

    def train(self, train_loader, val_loader, epochs: int):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
        """
        self.logger.info(f"Starting EBM training for {epochs} epochs...")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Training phase
            self.model.train()
            epoch_loss = 0
            epoch_score = 0
            epoch_energy_reg = 0

            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
            for batch_data in loop:
                if len(batch_data) == 2:
                    x, u = batch_data
                    u_gt = None
                else:
                    x, u, u_gt = batch_data

                loss, loss_dict = self.train_step(u, x, u_gt)

                epoch_loss += loss.item()
                epoch_score += loss_dict['score']
                epoch_energy_reg += loss_dict['energy_reg']

                loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    score=f"{loss_dict['score']:.4f}",
                    energy_reg=f"{loss_dict['energy_reg']:.6f}"
                )

            avg_train_loss = epoch_loss / len(train_loader)
            avg_score = epoch_score / len(train_loader)
            avg_energy_reg = epoch_energy_reg / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation phase
            avg_val_loss = self.validate(val_loader)
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
                f"Train Loss={avg_train_loss:.6f} (Score={avg_score:.6f}, Reg={avg_energy_reg:.6f}), "
                f"Val Loss={avg_val_loss:.6f}, "
                f"LR={current_lr:.2e}"
            )

            # TensorBoard logging for epoch-level metrics
            if self.enable_tracking:
                if self.gradient_tracker is not None:
                    # Custom tracker
                    epoch_metrics = {
                        'train_loss': avg_train_loss,
                        'train_score': avg_score,
                        'train_energy_reg': avg_energy_reg,
                        'val_loss': avg_val_loss,
                        'learning_rate': current_lr,
                    }
                    self.gradient_tracker.log_scalars('epoch', epoch_metrics, epoch)
                elif self.writer is not None:
                    # Built-in TensorBoard
                    self.writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
                    self.writer.add_scalar('epoch/train_score', avg_score, epoch)
                    self.writer.add_scalar('epoch/train_energy_reg', avg_energy_reg, epoch)
                    self.writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
                    self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)

            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                if self.save_checkpoints:
                    self.save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"  → Best EBM model saved (val_loss={avg_val_loss:.6f})")

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
            path = os.path.join(checkpoint_dir, 'best_ebm.pt')
        else:
            path = os.path.join(checkpoint_dir, f'ebm_epoch_{epoch}.pt')

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

        self.logger.info(f"Loaded EBM checkpoint from epoch {self.current_epoch}")

    @torch.no_grad()
    def sample_posterior(
        self,
        x: torch.Tensor,
        u_fno: torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """
        Sample from posterior using Langevin dynamics.

        Starts from FNO prediction and refines via:
        u_{t+1} = u_t - ε∇_u E(u_t) + √(2ε) z_t

        Args:
            x: Input coordinates
            u_fno: FNO prediction (initial state)
            n_samples: Number of samples to generate

        Returns:
            samples: Posterior samples (n_samples, batch, n_x, [n_y], channels)
        """
        self.model.eval()
        x = x.to(self.device)
        u_fno = u_fno.to(self.device)

        batch_size = u_fno.shape[0]
        samples = []

        noise_scale = np.sqrt(2 * self.langevin_step_size_inference)

        for _ in range(n_samples):
            # Initialize from FNO + small noise
            u_sample = u_fno + 0.01 * torch.randn_like(u_fno)

            # Langevin refinement
            for step in range(self.langevin_steps_inference):
                u_sample.requires_grad_(True)
                energy = self.model(u_sample, x, u_fno)

                grad = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=u_sample,
                    create_graph=False
                )[0]

                with torch.no_grad():
                    noise = torch.randn_like(u_sample) * noise_scale
                    u_sample = u_sample - self.langevin_step_size_inference * grad + noise

                u_sample = u_sample.detach()

            samples.append(u_sample.unsqueeze(0))

        # Stack samples
        samples = torch.cat(samples, dim=0)  # (n_samples, batch, ...)
        return samples