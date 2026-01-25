"""
Improved Descriptor Refiner with Residual Connections
Following DINO-VO and R2D2 best practices for discriminative descriptors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorRefiner(nn.Module):
    """
    Refine DINOv3 384-dim features → compact 128-dim descriptors.

    IMPROVEMENTS over v1:
    1. Residual connections for better gradient flow
    2. Slightly more capacity (384 hidden instead of 256)
    3. LayerNorm for stability
    4. Still L2 norm only at the end (per R2D2)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 384,  # Increased from 256
        output_dim: int = 128,
        num_layers: int = 4     # Increased from 3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks (with LayerNorm for stability)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers - 2)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization for maximum diversity.
        Critical for preventing descriptor collapse.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Refine features into descriptors.

        CRITICAL: L2 normalization is ONLY at the end (per R2D2).

        Args:
            dino_features: (B, N, C) features at keypoints

        Returns:
            descriptors: (B, N, output_dim) L2-normalized descriptors
        """
        B, N, C = dino_features.shape

        # Flatten
        x = dino_features.reshape(B * N, C)

        # Input projection
        x = F.relu(self.input_proj(x))

        # Residual blocks (better gradient flow)
        for block in self.residual_blocks:
            x = block(x)

        # Output projection
        descriptors = self.output_proj(x)

        # L2 normalize ONLY at the very end (per R2D2/DINO-VO)
        descriptors = F.normalize(descriptors, p=2, dim=-1)

        # Reshape back
        descriptors = descriptors.reshape(B, N, self.output_dim)

        return descriptors


class ResidualBlock(nn.Module):
    """
    Simple residual block with LayerNorm.
    Helps with gradient flow and training stability.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Residual block: x + F(x)
        """
        identity = x

        # First transform
        out = self.norm1(x)
        out = F.relu(self.fc1(out))

        # Second transform
        out = self.norm2(out)
        out = self.fc2(out)

        # Residual connection
        out = out + identity
        out = F.relu(out)

        return out