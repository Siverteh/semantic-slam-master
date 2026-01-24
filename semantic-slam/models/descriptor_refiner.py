"""
Descriptor Refiner Head
FIXED: Added variance regularization to prevent mode collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorRefiner(nn.Module):
    """
    Refines DINOv3 384-dim features into compact 128-dim descriptors.
    Uses variance regularization to prevent descriptor collapse.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: DINOv3 feature dimension (384 for ViT-S)
            hidden_dim: Hidden layer dimension
            output_dim: Final descriptor dimension (128)
            num_layers: Number of MLP layers
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build MLP layers WITHOUT BatchNorm
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # Initialize with higher variance to prevent immediate collapse
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization with higher gain for diversity"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use higher gain to spread descriptors initially
                nn.init.xavier_uniform_(m.weight, gain=1.5)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)  # Non-zero bias for diversity

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Refine DINOv3 features into descriptors.

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

        # Apply MLP (before normalization to preserve variance)
        descriptors = self.mlp(x)

        # L2 normalize
        descriptors = F.normalize(descriptors, p=2, dim=-1)

        # Reshape back
        if is_dense:
            descriptors = descriptors.reshape(B, H, W, self.output_dim)
        else:
            descriptors = descriptors.reshape(B, N, self.output_dim)

        return descriptors

    def compute_variance_loss(self, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Compute variance regularization loss to prevent descriptor collapse.
        Encourages high variance across descriptor dimensions.

        Args:
            descriptors: (B, N, D) descriptors

        Returns:
            loss: Negative log variance (lower = more diverse)
        """
        # Flatten batch and keypoint dimensions
        B, N, D = descriptors.shape
        desc_flat = descriptors.reshape(B * N, D)  # (B*N, D)

        # Compute variance per dimension
        desc_var = desc_flat.var(dim=0)  # (D,)

        # Mean variance across dimensions (want this HIGH)
        mean_var = desc_var.mean()

        # Loss: negative log variance (minimize to maximize variance)
        # Add small epsilon for numerical stability
        variance_loss = -torch.log(mean_var + 1e-6)

        return variance_loss