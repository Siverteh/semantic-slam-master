"""
SOTA 2025 Descriptor Refiner with Residual Connections
Based on DINO-VO, R2D2, and DeDoDe v2 best practices (2024-2025)

Architecture:
- 4-layer MLP with residual connections
- LayerNorm for stability
- Orthogonal initialization to prevent descriptor collapse
- L2-normalize output (critical!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorRefiner(nn.Module):
    """
    Refine DINOv3 384-dim features → compact 128-dim descriptors.

    Architecture (as specified):
    - Linear(384 → 384) + ReLU
    - ResidualBlock(384) with LayerNorm - repeat 2 times
    - Linear(384 → 128)
    - L2-normalize output (critical!)

    Output: Descriptors (B, N, 128)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 384,
        output_dim: int = 128,
        num_layers: int = 4  # Total layers including input/output projections
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection: Linear(384 → 384) + ReLU
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Residual blocks with LayerNorm (2 blocks as specified)
        num_res_blocks = max(num_layers - 2, 2)  # At least 2 residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])

        # Output projection: Linear(384 → 128)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization for maximum descriptor diversity.
        Critical for preventing descriptor collapse!
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal initialization (as specified)
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Refine DINOv3 features into discriminative descriptors.

        CRITICAL: L2 normalization is ONLY at the end (per R2D2/DINO-VO).

        Args:
            dino_features: (B, N, C) features at keypoints where C=384

        Returns:
            descriptors: (B, N, output_dim) L2-normalized descriptors
        """
        B, N, C = dino_features.shape

        assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got {C}"

        # Flatten for MLP processing
        x = dino_features.reshape(B * N, C)

        # Input projection with ReLU
        x = self.input_proj(x)

        # Residual blocks (better gradient flow)
        for block in self.residual_blocks:
            x = block(x)

        # Output projection
        descriptors = self.output_proj(x)

        # L2 normalize ONLY at the very end (critical per R2D2/DINO-VO)
        descriptors = F.normalize(descriptors, p=2, dim=-1)

        # Reshape back to batch format
        descriptors = descriptors.reshape(B, N, self.output_dim)

        return descriptors


class ResidualBlock(nn.Module):
    """
    Residual block with LayerNorm for training stability.

    Structure:
    - LayerNorm → Linear → ReLU
    - LayerNorm → Linear
    - Residual connection + ReLU
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

        # First transform with LayerNorm
        out = self.norm1(x)
        out = F.relu(self.fc1(out))

        # Second transform with LayerNorm
        out = self.norm2(out)
        out = self.fc2(out)

        # Residual connection
        out = out + identity
        out = F.relu(out)

        return out
