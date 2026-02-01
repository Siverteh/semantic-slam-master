"""
Lightweight Geometric CNN (FinerCNN-style)
Following DINO-VO architecture for fine-grained geometric features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricCNN(nn.Module):
    """
    Lightweight CNN for extracting fine-grained geometric features.

    Architecture follows DINO-VO's FinerCNN:
    - Input: RGB image (H, W, 3)
    - Output: Multi-scale geometric features (H/4, W/4, 64)

    Key properties:
    - ~2M parameters (lightweight!)
    - 4× downsampling (maintains good resolution)
    - Multi-scale fusion for robustness
    """

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 32,
        output_channels: int = 64
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        # Initial conv (no downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Downsample to H/2, W/2
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )

        # Downsample to H/4, W/4
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, output_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        # Lateral connections for multi-scale fusion
        self.lateral1 = nn.Conv2d(base_channels, output_channels, 1)
        self.lateral2 = nn.Conv2d(base_channels*2, output_channels, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract fine-grained geometric features.

        Args:
            image: (B, 3, H, W) RGB image

        Returns:
            features: (B, output_channels, H/4, W/4) geometric features
        """
        # Initial features
        x1 = self.conv1(image)  # (B, 32, H, W)

        # Downsample
        x2 = self.down1(x1)     # (B, 64, H/2, W/2)
        x3 = self.down2(x2)     # (B, 64, H/4, W/4)

        # Multi-scale fusion (FPN-style)
        # Upsample and add
        lat1 = self.lateral1(x1)  # (B, 64, H, W)
        lat1 = F.interpolate(lat1, size=x3.shape[2:], mode='bilinear', align_corners=False)

        lat2 = self.lateral2(x2)  # (B, 64, H/2, W/2)
        lat2 = F.interpolate(lat2, size=x3.shape[2:], mode='bilinear', align_corners=False)

        # Fuse
        features = x3 + lat1 + lat2

        return features

    def extract_at_keypoints(
        self,
        features: torch.Tensor,
        keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample geometric features at keypoint locations.

        Args:
            features: (B, C, H, W) feature map at H/4, W/4 resolution
            keypoints: (B, N, 2) keypoints in PIXEL coordinates (full resolution)

        Returns:
            sampled: (B, N, C) features at keypoints
        """
        B, C, H, W = features.shape

        # Keypoints are in full resolution, scale to feature map resolution
        # Feature map is H/4, W/4 of input image
        keypoints_scaled = keypoints / 4.0

        # Normalize to [-1, 1] for grid_sample
        norm_coords = keypoints_scaled.clone()
        norm_coords[:, :, 0] = 2.0 * keypoints_scaled[:, :, 0] / (W - 1) - 1.0  # x
        norm_coords[:, :, 1] = 2.0 * keypoints_scaled[:, :, 1] / (H - 1) - 1.0  # y

        # Sample
        grid = norm_coords.unsqueeze(1)  # (B, 1, N, 2)
        sampled = F.grid_sample(
            features, grid,
            mode='bilinear',
            align_corners=True
        )

        # Reshape: (B, C, 1, N) → (B, N, C)
        sampled = sampled.squeeze(2).permute(0, 2, 1)

        return sampled


class FeatureFusion(nn.Module):
    """
    Fuse semantic (DINO) and geometric (CNN) features.
    Simple concatenation + 1×1 conv.
    """

    def __init__(
        self,
        semantic_dim: int = 384,
        geometric_dim: int = 64,
        output_dim: int = 256
    ):
        super().__init__()

        self.semantic_dim = semantic_dim
        self.geometric_dim = geometric_dim
        self.output_dim = output_dim

        # Simple fusion: concatenate + project
        self.fusion = nn.Sequential(
            nn.Linear(semantic_dim + geometric_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

    def forward(
        self,
        semantic_features: torch.Tensor,
        geometric_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse semantic and geometric features.

        Args:
            semantic_features: (B, N, semantic_dim) from DINO
            geometric_features: (B, N, geometric_dim) from CNN

        Returns:
            fused: (B, N, output_dim) fused features
        """
        # Concatenate
        combined = torch.cat([semantic_features, geometric_features], dim=-1)

        # Fuse
        fused = self.fusion(combined)

        return fused