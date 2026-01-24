"""
Keypoint Selector Head
FIXED: Removed broken softmax, using sigmoid instead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KeypointSelector(nn.Module):
    """
    Predicts saliency heatmap using spatial attention.

    CRITICAL FIX: Uses sigmoid (not softmax!) to allow MULTIPLE peaks.
    Peakiness loss handles making it focused.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Step 1: Reduce dimensionality
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Step 2: Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Step 3: Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        # Step 4: Final saliency prediction
        self.saliency_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # SIGMOID, not softmax!
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights"""
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
        # Permute to (B, C, H, W)
        x = dino_features.permute(0, 3, 1, 2)

        # Project to lower dimension
        x = self.proj(x)

        # Compute spatial attention
        attn = self.spatial_attention(x)

        # Apply attention
        x = x * attn

        # Refine features
        x = self.refine(x)

        # Predict saliency (sigmoid gives independent probabilities per location)
        saliency = self.saliency_head(x)  # (B, 1, H, W) in [0, 1]

        # Permute back to (B, H, W, 1)
        saliency_map = saliency.permute(0, 2, 3, 1)

        return saliency_map

    def select_keypoints(
        self,
        saliency_map: torch.Tensor,
        num_keypoints: int = 500,
        nms_radius: int = 2,
        threshold: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top keypoints from saliency map.

        Args:
            saliency_map: (B, H, W, 1) saliency heatmap
            num_keypoints: Maximum number of keypoints to select
            nms_radius: Radius for non-maximum suppression
            threshold: Minimum saliency threshold

        Returns:
            keypoints: (B, N, 2) keypoint coordinates
            scores: (B, N) saliency scores
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        # Apply NMS
        if nms_radius > 0:
            saliency = self._apply_nms(saliency, nms_radius)

        # Threshold
        saliency = torch.where(saliency > threshold, saliency, torch.zeros_like(saliency))

        keypoints_list = []
        scores_list = []

        for b in range(B):
            sal_b = saliency[b]

            # Get valid coordinates
            valid_mask = sal_b > 0
            valid_scores = sal_b[valid_mask]
            valid_coords = torch.nonzero(valid_mask)  # (N_valid, 2) as (y, x)

            if len(valid_scores) == 0:
                # Fallback: uniform grid
                grid_size = int(num_keypoints ** 0.5)
                y_grid = torch.linspace(2, H-3, grid_size, device=device)
                x_grid = torch.linspace(2, W-3, grid_size, device=device)
                yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
                kpts = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()
                scrs = torch.ones(len(kpts), device=device) * 0.1
            else:
                # Select top-k by score
                k = min(num_keypoints, len(valid_scores))
                top_scores, top_indices = torch.topk(valid_scores, k)
                top_coords = valid_coords[top_indices]

                # Convert to (x, y) format
                kpts = torch.stack([top_coords[:, 1], top_coords[:, 0]], dim=1).float()
                scrs = top_scores

            # Padding if needed
            if len(kpts) < num_keypoints:
                pad_size = num_keypoints - len(kpts)

                # Use grid sampling for padding (not random, not duplicates)
                grid_size = int(pad_size ** 0.5) + 1
                y_pad = torch.linspace(2, H-3, grid_size, device=device)
                x_pad = torch.linspace(2, W-3, grid_size, device=device)
                yy_pad, xx_pad = torch.meshgrid(y_pad, x_pad, indexing='ij')
                pad_kpts = torch.stack([xx_pad.flatten(), yy_pad.flatten()], dim=1).float()[:pad_size]
                pad_scrs = torch.ones(pad_size, device=device) * 1e-6

                kpts = torch.cat([kpts, pad_kpts], dim=0)
                scrs = torch.cat([scrs, pad_scrs], dim=0)

            keypoints_list.append(kpts)
            scores_list.append(scrs)

        keypoints = torch.stack(keypoints_list, dim=0)
        scores = torch.stack(scores_list, dim=0)

        return keypoints, scores

    def _apply_nms(self, saliency: torch.Tensor, radius: int) -> torch.Tensor:
        """Apply non-maximum suppression"""
        kernel_size = 2 * radius + 1
        max_pooled = F.max_pool2d(
            saliency.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=radius
        ).squeeze(1)

        nms_mask = (saliency == max_pooled)
        nms_saliency = saliency * nms_mask.float()

        return nms_saliency