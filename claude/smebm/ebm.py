"""
Energy-Based Models (EBM) for Uncertainty Quantification

This module contains various EBM architectures:
- EBMPotential: Standard MLP-based energy model
- KAN_EBM: Pure KAN-based energy model
- FNO_KAN_EBM: Hybrid FNO encoder + KAN head (recommended)
- GNN_EBM: Graph neural network-based energy model

All EBM models extend from BaseEnergyFunction (torchebm library) when available,
or fallback to nn.Module if torchebm is not installed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fno import SpectralConv2d

# Import BaseEnergyFunction from torchebm if available
try:
    from torchebm.core import BaseEnergyFunction
    TORCHEBM_AVAILABLE = True
    print("Using BaseEnergyFunction from torchebm library")
except ImportError:
    # Fallback: use nn.Module if torchebm not available
    BaseEnergyFunction = nn.Module
    TORCHEBM_AVAILABLE = False
    print("Warning: torchebm not available, using nn.Module as base class")


# ============================================================================
# Standard MLP-based EBM
# ============================================================================

class EBMPotential(BaseEnergyFunction):
    """
    Energy-Based Model potential V(u, X) for uncertainty modeling
    Standard MLP-based implementation.

    Extends BaseEnergyFunction from torchebm for compatibility with torchebm library.
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


# ============================================================================
# Convolutional EBM (RECOMMENDED FOR SPATIAL STRUCTURE)
# ============================================================================

class ConvEBM(BaseEnergyFunction):
    """
    Convolutional Energy-Based Model for spatially-structured uncertainty.

    Unlike MLP-based EBM that processes each pixel independently,
    ConvEBM uses convolutions to capture spatial correlations.

    This is CRITICAL for learning structured uncertainty maps!

    Extends BaseEnergyFunction from torchebm for compatibility with torchebm library.
    """
    # Consider use spectral normalization later
    def __init__(self, in_channels=4, hidden_channels=[64, 128, 128, 64]):
        super().__init__()

        layers = []
        prev_channels = in_channels

        for hidden_ch in hidden_channels:
            layers.append(nn.Conv2d(prev_channels, hidden_ch, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(8, hidden_ch))  # GroupNorm for stability
            layers.append(nn.GELU())
            prev_channels = hidden_ch

        # Final conv to energy map
        layers.append(nn.Conv2d(prev_channels, 1, kernel_size=1))

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, u, x):
        """
        Args:
            u: solution field (batch, n_x, n_y, 1)
            x: input coordinates (batch, n_x, n_y, 3)
        Returns:
            energy: scalar energy (batch,)

        Note: Returns -E(u,x) following standard EBM convention where p(u) ∝ exp(f(u))
        and f(u) = -E(u). This prevents sign confusion and aligns with literature.
        """
        # Concatenate solution with coordinates
        combined = torch.cat([u, x], dim=-1)  # (batch, n_x, n_y, 4)

        # Reshape to (batch, channels, height, width) for conv
        combined = combined.permute(0, 3, 1, 2)  # (batch, 4, n_x, n_y)

        # Apply convolutional network to get f(u,x)
        f_map = self.network(combined)  # (batch, 1, n_x, n_y)

        # Global average pooling to get scalar f(u,x)
        f = self.pool(f_map).squeeze(-1).squeeze(-1).squeeze(-1)  # (batch,)

        # Standard EBM convention: network outputs f(u,x) = -E(u,x)
        # So energy E = -f
        # But for compatibility and following standard practice:
        # We return f directly (which is -E in standard notation)
        # The loss function will work with this convention
        return f


class ConditionalEnergyWrapper(BaseEnergyFunction):
    """
    Wrapper for conditional EBMs that separates sampling (u) from conditioning (x).

    This wrapper ensures that when MCMC sampling is performed, only the solution field u
    is updated, while the conditioning information x remains fixed.

    Compatible with torchebm library's samplers and loss functions.
    """
    def __init__(self, energy_fn, condition):
        """
        Args:
            energy_fn: The base energy function E(u, x)
            condition: The conditioning information x (fixed during sampling)
        """
        super().__init__()
        self.energy_fn = energy_fn
        # Store condition - clone and detach to avoid gradient issues
        self._condition = condition.clone().detach()

    @property
    def condition(self):
        return self._condition

    def forward(self, u):
        """
        Args:
            u: solution field to be sampled (batch, n_x, n_y, 1)
        Returns:
            energy: scalar energy (batch,)
        """
        # Always ensure condition is on same device as u
        cond = self._condition.to(u.device)
        return self.energy_fn(u, cond)

    def update_condition(self, new_condition):
        """Update the conditioning information"""
        self._condition = new_condition.clone().detach()

class SimpleFNO_EBM(BaseEnergyFunction):
    """
    FNO-based Energy Model with GLOBAL receptive field.

    KEY ADVANTAGE: Spectral convolutions see the ENTIRE spatial field at once!
    - Solves the receptive field problem (ConvEBM only sees 17x17 pixels)
    - Can learn global patterns like radial uncertainty
    - More stable gradients via spectral parameterization

    Architecture:
        Input (u, x) → FNO encoder layers → Global pooling → MLP head → Energy

    This is SIMPLER than FNO_KAN_EBM (no KAN complexity) but with same
    global receptive field advantage.
    """
    def __init__(
        self,
        in_channels=1,
        fno_modes1=12,
        fno_modes2=12,
        fno_width=32,
        fno_layers=3
    ):
        super().__init__()

        self.fno_width = fno_width
        self.fno_modes1 = fno_modes1
        self.fno_modes2 = fno_modes2

        # Initial lifting: map from input channels to FNO width
        self.fc0 = nn.Linear(in_channels, fno_width)

        # FNO spectral convolution layers (GLOBAL receptive field!)
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(fno_width, fno_width, fno_modes1, fno_modes2)
            for _ in range(fno_layers)
        ])

        # Local skip connections
        self.w_layers = nn.ModuleList([
            nn.Conv2d(fno_width, fno_width, 1)
            for _ in range(fno_layers)
        ])

        # Final projection to energy map
        self.final_conv = nn.Conv2d(fno_width, 1, kernel_size=1)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MLP head for energy prediction (removed to simplify - just use pooled features directly)
        # Simpler is better for initial testing

    def forward(self, u):
        """
        Args:
            u: solution field (batch, n_x, n_y, 1)
            x: input coordinates (batch, n_x, n_y, 3)
        Returns:
            energy: scalar energy (batch,)
        """
        # Concatenate solution with coordinates
        #combined = torch.cat([u, x], dim=-1)  # (batch, n_x, n_y, 4)

        # Lift to FNO width
        features = self.fc0(u)  # (batch, n_x, n_y, fno_width)
        features = features.permute(0, 3, 1, 2)  # (batch, fno_width, n_x, n_y)

        # Apply FNO layers (each has GLOBAL receptive field via FFT)
        for spectral, w in zip(self.spectral_layers, self.w_layers):
            x1 = spectral(features)  # Spectral convolution (GLOBAL)
            x2 = w(features)         # Local skip connection
            features = x1 + x2
            features = F.gelu(features)

        # Final conv to energy map
        energy_map = self.final_conv(features)  # (batch, 1, n_x, n_y)

        # Global pooling to scalar energy
        energy = self.pool(energy_map).squeeze(-1).squeeze(-1).squeeze(-1)  # (batch,)

        return energy


# ============================================================================
# KAN-based EBM
# ============================================================================

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer
    Uses learnable spline-based activation functions on edges instead of nodes.

    Based on the KAN paper: https://arxiv.org/abs/2404.19756
    This is a simplified implementation using B-splines.

    Note: Consider using pykan library for production use.
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


class KAN_EBM(BaseEnergyFunction):
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

    Extends BaseEnergyFunction from torchebm for compatibility with torchebm library.
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


# ============================================================================
# Hybrid FNO + KAN EBM (Recommended)
# ============================================================================

class FNO_KAN_EBM(BaseEnergyFunction):
    """
    Hybrid FNO Encoder + KAN Energy Head

    Uses FNO's strength for learning spatial field features,
    then feeds compressed features to KAN for energy prediction.

    Architecture:
        Input field → FNO encoder → Adaptive pooling → KAN → Scalar energy

    This is the recommended approach combining:
    - FNO's ability to capture spatial/spectral structure
    - KAN's flexible learnable activations for energy modeling

    Extends BaseEnergyFunction from torchebm for compatibility with torchebm library.
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


class GNN_EBM(BaseEnergyFunction):
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

    Extends BaseEnergyFunction from torchebm for compatibility with torchebm library.
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