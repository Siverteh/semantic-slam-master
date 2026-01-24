"""
Descriptor Refiner Head
FIXED: Removed BatchNorm, L2 normalization as FINAL step (per R2D2/SuperPoint)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorRefiner(nn.Module):
    """
    Refines DINOv3 384-dim features into compact 128-dim descriptors.

    CRITICAL FIXES:
    1. REMOVED BatchNorm - it was killing descriptor variance!
    2. L2 normalization is now the LAST step (per R2D2/SuperPoint architecture)
    3. Proper initialization for descriptor diversity

    Architecture based on R2D2 paper (Figure 2):
    - MLP layers WITHOUT normalization
    - L2 normalization at the very end
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,  # REDUCED from 3 - simpler is better!
        dropout: float = 0.0  # REMOVED dropout - was hurting diversity
    ):
        """
        Args:
            input_dim: DINOv3 feature dimension (384 for ViT-S)
            hidden_dim: Hidden layer dimension
            output_dim: Final descriptor dimension (128)
            num_layers: Number of MLP layers (REDUCED to 2)
            dropout: Dropout rate (REMOVED - was hurting diversity)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build SIMPLE MLP - 2 layers only!
        # Too many layers cause mode collapse
        if num_layers == 2:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Fallback for 3 layers (not recommended)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )

        # Initialize with EXTRA diversity
        self._init_weights()

    def _init_weights(self):
        """
        Initialize with STRONG diversity to prevent collapse.

        Use orthogonal initialization for maximum initial diversity.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal initialization - maximizes initial diversity
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    # Random uniform bias for additional diversity
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Refine DINOv3 features into descriptors.

        CRITICAL: L2 normalization is applied at the VERY END (per R2D2/SuperPoint)

        Args:
            dino_features: (B, N, C) features at N keypoint locations
                          OR (B, H, W, C) dense feature map

        Returns:
            descriptors: (B, N, output_dim) L2-normalized descriptors
        """
        input_shape = dino_features.shape
        is_dense = len(input_shape) == 4

        if is_dense:
            B, H, W, C = input_shape
            x = dino_features.reshape(B * H * W, C)
        else:
            B, N, C = input_shape
            x = dino_features.reshape(B * N, C)

        # Apply MLP (NO normalization inside!)
        descriptors = self.mlp(x)

        # L2 normalize ONLY at the very end (per R2D2 paper)
        # This is CRITICAL - normalizing before MLP was causing collapse!
        descriptors = F.normalize(descriptors, p=2, dim=-1)

        # Reshape back
        if is_dense:
            descriptors = descriptors.reshape(B, H, W, self.output_dim)
        else:
            descriptors = descriptors.reshape(B, N, self.output_dim)

        return descriptors