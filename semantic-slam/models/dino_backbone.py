"""
Fixed DINOv3 Backbone with Grid Alignment
CRITICAL FIXES from DINO-VO and DINOv3 papers:
1. Proper 16x16 patch grid alignment
2. Feature normalization (layer norm + batch norm) to suppress outliers
3. All coordinates in PATCH space (0-27 for 448x448 input)
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple


class DinoBackbone(nn.Module):
    """
    Grid-aligned DINOv3 backbone following DINO-VO best practices.

    Key design principles:
    - 16x16 patches → 28x28 grid for 448x448 input
    - All coordinates in PATCH space (0-27)
    - Proper feature normalization per DINOv3 paper
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        input_size: int = 448,
        freeze: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.input_size = input_size
        self.patch_size = 16  # DINOv3 uses 16x16 patches

        # Grid dimensions
        self.grid_h = input_size // self.patch_size  # 28
        self.grid_w = input_size // self.patch_size  # 28
        self.num_patches = self.grid_h * self.grid_w  # 784

        # Load pretrained DINOv3
        print(f"Loading {model_name}...")
        self.dino = timm.create_model(
            model_name,
            pretrained=True,
            dynamic_img_size=True
        )

        self.embed_dim = self.dino.embed_dim  # 384 for ViT-S
        self.n_storage_tokens = 4  # DINOv3 has 4 storage/register tokens

        # CRITICAL FIX: Add BatchNorm to suppress feature dimension outliers
        # Per DINOv3 paper Section A.2: "applying batch normalization can suppress
        # these feature dimension outliers"
        self.feature_norm = nn.BatchNorm1d(self.embed_dim, affine=True)

        # Freeze backbone
        if freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()

        print(f"✓ DINOv3 Backbone:")
        print(f"  - Feature dim: {self.embed_dim}")
        print(f"  - Grid size: {self.grid_h}x{self.grid_w}")
        print(f"  - Patch size: {self.patch_size}x{self.patch_size}")
        print(f"  - Coordinate space: PATCH (0-{self.grid_h-1})")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract grid-aligned patch features.

        Args:
            images: (B, 3, H, W) RGB images

        Returns:
            patch_features: (B, grid_h, grid_w, embed_dim)
                           Features in PATCH coordinate space
        """
        B = images.shape[0]

        # Extract features (with gradient flow only if training unfrozen)
        with torch.set_grad_enabled(self.training and not self._is_frozen()):
            features = self.dino.forward_features(images)

        # features: (B, num_tokens, embed_dim)
        # num_tokens = 1 (CLS) + 4 (storage) + num_patches

        # Extract patch tokens (skip CLS and storage tokens)
        patch_tokens = features[:, 1 + self.n_storage_tokens:, :]

        # Verify shape
        assert patch_tokens.shape[1] == self.num_patches, \
            f"Expected {self.num_patches} patches, got {patch_tokens.shape[1]}"

        # Apply BatchNorm to suppress outlier dimensions (CRITICAL!)
        B, N, C = patch_tokens.shape
        patch_tokens_flat = patch_tokens.reshape(B * N, C)
        patch_tokens_normed = self.feature_norm(patch_tokens_flat)
        patch_tokens = patch_tokens_normed.reshape(B, N, C)

        # Reshape to spatial grid (PATCH coordinates)
        patch_features = patch_tokens.reshape(
            B, self.grid_h, self.grid_w, self.embed_dim
        )

        return patch_features

    def _is_frozen(self) -> bool:
        """Check if backbone is frozen"""
        return not next(self.dino.parameters()).requires_grad

    def extract_at_keypoints(
        self,
        patch_features: torch.Tensor,
        keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features at keypoint locations using bilinear interpolation.

        CRITICAL: Keypoints MUST be in PATCH coordinates (0 to grid_h-1).

        Args:
            patch_features: (B, H, W, C) feature map in patch space
            keypoints: (B, N, 2) in PATCH coordinates [0, grid_h-1] x [0, grid_w-1]

        Returns:
            sampled_features: (B, N, C) features at keypoints
        """
        B, H, W, C = patch_features.shape

        # Normalize to [-1, 1] for grid_sample
        norm_coords = keypoints.clone()
        norm_coords[:, :, 0] = 2.0 * keypoints[:, :, 0] / (W - 1) - 1.0  # x
        norm_coords[:, :, 1] = 2.0 * keypoints[:, :, 1] / (H - 1) - 1.0  # y

        # Prepare for grid_sample
        grid = norm_coords.unsqueeze(1)  # (B, 1, N, 2)
        features_bchw = patch_features.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Sample features
        sampled = torch.nn.functional.grid_sample(
            features_bchw, grid,
            mode='bilinear',
            align_corners=True
        )

        # Reshape: (B, C, 1, N) → (B, N, C)
        sampled_features = sampled.squeeze(2).permute(0, 2, 1)

        return sampled_features

    def patch_to_pixel(self, patch_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert PATCH coordinates to PIXEL coordinates.

        Args:
            patch_coords: (B, N, 2) in [0, grid_h-1] x [0, grid_w-1]

        Returns:
            pixel_coords: (B, N, 2) in [0, H-1] x [0, W-1]
        """
        pixel_coords = patch_coords * self.patch_size + self.patch_size / 2
        return pixel_coords

    def pixel_to_patch(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert PIXEL coordinates to PATCH coordinates.

        Args:
            pixel_coords: (B, N, 2) in [0, H-1] x [0, W-1]

        Returns:
            patch_coords: (B, N, 2) in [0, grid_h-1] x [0, grid_w-1]
        """
        patch_coords = (pixel_coords - self.patch_size / 2) / self.patch_size
        return patch_coords