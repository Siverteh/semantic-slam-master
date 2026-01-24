"""
Simplified Descriptor Refiner following R2D2 best practices
FIXED: Proper architecture, L2 norm at the END only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorRefiner(nn.Module):
    """
    Refine DINOv3 384-dim features → compact 128-dim descriptors.

    Key principles from R2D2/SuperPoint:
    1. Simple MLP (3 layers, no batchnorm inside)
    2. L2 normalization ONLY at the very end
    3. Orthogonal initialization for diversity
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Simple 3-layer MLP (no normalization inside!)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization for maximum diversity.
        This prevents descriptor collapse.
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

        # Apply MLP (no normalization inside!)
        descriptors = self.mlp(x)

        # L2 normalize ONLY at the very end
        descriptors = F.normalize(descriptors, p=2, dim=-1)

        # Reshape back
        descriptors = descriptors.reshape(B, N, self.output_dim)

        return descriptors