"""
Keypoint Selector Head
FIXED: Removed BatchNorm on final output, improved NMS, better peakiness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KeypointSelector(nn.Module):
    """
    Predicts saliency heatmap indicating which locations are good keypoints.

    FIXES:
    - Removed BatchNorm from final saliency output (was interfering with peak detection)
    - Improved non-maximum suppression
    - Better initialization
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        """
        Args:
            input_dim: DINOv3 feature dimension (384 for ViT-S)
            hidden_dim: Hidden layer dimension
            num_layers: Number of convolutional layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Build convolutional layers
        layers = []
        in_channels = input_dim

        for i in range(num_layers):
            out_channels = hidden_dim if i < num_layers - 1 else 1

            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

            if i < num_layers - 1:
                # BatchNorm ONLY on intermediate layers (not on final output)
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Final activation to produce probabilities
        self.activation = nn.Sigmoid()

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for gradual learning"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dino_features: (B, H, W, C) DINOv3 patch features

        Returns:
            saliency_map: (B, H, W, 1) keypoint probability map in [0, 1]
        """
        # Permute to (B, C, H, W) for conv layers
        x = dino_features.permute(0, 3, 1, 2)

        # Apply convolutions
        x = self.conv_layers(x)

        # Apply sigmoid activation
        x = self.activation(x)

        # Permute back to (B, H, W, 1)
        saliency_map = x.permute(0, 2, 3, 1)

        return saliency_map

    def select_keypoints(
        self,
        saliency_map: torch.Tensor,
        num_keypoints: int = 500,
        nms_radius: int = 2,
        threshold: float = 0.01  # Lower threshold to get more candidates
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top keypoints from saliency map with non-maximum suppression.

        IMPROVED: Better NMS and selection strategy

        Args:
            saliency_map: (B, H, W, 1) saliency heatmap
            num_keypoints: Maximum number of keypoints to select
            nms_radius: Radius for non-maximum suppression
            threshold: Minimum saliency threshold

        Returns:
            keypoints: (B, N, 2) keypoint coordinates in [0, H-1] x [0, W-1]
            scores: (B, N) saliency scores for each keypoint
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        # Squeeze last dimension
        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        # Apply NMS if requested
        if nms_radius > 0:
            saliency = self._apply_nms(saliency, nms_radius)

        # Threshold (use lower threshold to get more candidates)
        saliency = torch.where(saliency > threshold, saliency, torch.zeros_like(saliency))

        # Select top-k keypoints per batch
        keypoints_list = []
        scores_list = []

        for b in range(B):
            sal_b = saliency[b]  # (H, W)

            # Get valid coordinates
            valid_mask = sal_b > 0
            valid_scores = sal_b[valid_mask]
            valid_coords = torch.nonzero(valid_mask)  # (N_valid, 2) as (y, x)

            if len(valid_scores) == 0:
                # No valid keypoints, use grid sampling for robustness
                y_grid = torch.linspace(0, H-1, int(num_keypoints**0.5), device=device)
                x_grid = torch.linspace(0, W-1, int(num_keypoints**0.5), device=device)
                yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
                kpts = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()
                scrs = torch.ones(len(kpts), device=device) * 0.1

                # Pad to num_keypoints
                if len(kpts) < num_keypoints:
                    pad_size = num_keypoints - len(kpts)
                    kpts = torch.cat([kpts, kpts[-1:].repeat(pad_size, 1)], dim=0)
                    scrs = torch.cat([scrs, torch.zeros(pad_size, device=device)], dim=0)
                else:
                    kpts = kpts[:num_keypoints]
                    scrs = scrs[:num_keypoints]
            else:
                # Select top-k
                k = min(num_keypoints, len(valid_scores))
                top_scores, top_indices = torch.topk(valid_scores, k)
                top_coords = valid_coords[top_indices]  # (k, 2) as (y, x)

                # Convert to (x, y) format
                kpts = torch.stack([top_coords[:, 1], top_coords[:, 0]], dim=1).float()
                scrs = top_scores

                # Pad if necessary
                if k < num_keypoints:
                    pad_size = num_keypoints - k
                    # Use grid sampling for padding instead of duplicating last point
                    pad_y = torch.randint(0, H, (pad_size,), device=device).float()
                    pad_x = torch.randint(0, W, (pad_size,), device=device).float()
                    pad_kpts = torch.stack([pad_x, pad_y], dim=1)
                    pad_scrs = torch.zeros(pad_size, device=device)

                    kpts = torch.cat([kpts, pad_kpts], dim=0)
                    scrs = torch.cat([scrs, pad_scrs], dim=0)

            keypoints_list.append(kpts)
            scores_list.append(scrs)

        keypoints = torch.stack(keypoints_list, dim=0)  # (B, N, 2)
        scores = torch.stack(scores_list, dim=0)  # (B, N)

        return keypoints, scores

    def _apply_nms(self, saliency: torch.Tensor, radius: int) -> torch.Tensor:
        """
        Apply non-maximum suppression to saliency map.

        Args:
            saliency: (B, H, W) saliency scores
            radius: NMS radius

        Returns:
            nms_saliency: (B, H, W) after NMS
        """
        # Max pooling to find local maxima
        kernel_size = 2 * radius + 1
        max_pooled = F.max_pool2d(
            saliency.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=radius
        ).squeeze(1)

        # Keep only local maxima
        nms_mask = (saliency == max_pooled)
        nms_saliency = saliency * nms_mask.float()

        return nms_saliency