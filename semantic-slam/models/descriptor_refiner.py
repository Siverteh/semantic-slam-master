"""
Descriptor Refiner for Fused Semantic+Geometric Features
Handles concatenated DINO (384-dim) + CNN (64-dim) = 448-dim input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorRefiner(nn.Module):
    """
    Refine fused semantic+geometric features into compact descriptors.

    Input: 256-dim fused features (from FeatureFusion module)
    Output: 128-dim L2-normalized descriptors

    Architecture: Residual MLP with careful normalization
    """

    def __init__(
        self,
        input_dim: int = 256,  # After fusion
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for maximum diversity"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Refine features into descriptors.

        Args:
            fused_features: (B, N, input_dim) fused semantic+geometric features

        Returns:
            descriptors: (B, N, output_dim) L2-normalized descriptors
        """
        B, N, C = fused_features.shape

        # Flatten
        x = fused_features.reshape(B * N, C)

        # Input projection
        x = F.relu(self.input_proj(x))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output projection
        descriptors = self.output_proj(x)

        # L2 normalize (CRITICAL for descriptor matching)
        descriptors = F.normalize(descriptors, p=2, dim=-1)

        # Reshape
        descriptors = descriptors.reshape(B, N, self.output_dim)

        return descriptors


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm"""

    def __init__(self, dim: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.norm1(x)
        out = F.relu(self.fc1(out))

        out = self.norm2(out)
        out = self.fc2(out)

        out = out + identity
        out = F.relu(out)

        return out