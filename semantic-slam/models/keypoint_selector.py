"""
Fixed Keypoint Selector - CORRECTED VERSION
CRITICAL FIX: Use sigmoid (not softmax!) for independent per-patch scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KeypointSelector(nn.Module):
    """
    Minimal keypoint selector following SuperPoint principles.

    CRITICAL: Use SIGMOID for independent per-location scores,
    NOT softmax which forces single-peak collapse!
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128
    ):
        super().__init__()

        # SIMPLIFIED: Just 2 conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            # NO activation here - we'll apply sigmoid explicitly
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Small init
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  # Start neutral

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Predict per-patch saliency scores.

        Args:
            dino_features: (B, H, W, C) in PATCH space

        Returns:
            saliency_map: (B, H, W, 1) scores in [0, 1]
        """
        # Convert to (B, C, H, W)
        x = dino_features.permute(0, 3, 1, 2)

        # Predict logits
        logits = self.conv(x)  # (B, 1, H, W)

        # Apply SIGMOID (not softmax!)
        # Each location is INDEPENDENT
        saliency = torch.sigmoid(logits)

        # Convert back to (B, H, W, 1)
        saliency_map = saliency.permute(0, 2, 3, 1)

        return saliency_map

    def select_keypoints(
        self,
        saliency_map: torch.Tensor,
        num_keypoints: int = 500,
        nms_radius: int = 2,
        score_threshold: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select keypoints with NMS.

        Args:
            saliency_map: (B, H, W, 1) independent scores
            num_keypoints: Target number of keypoints
            nms_radius: NMS radius
            score_threshold: Minimum score

        Returns:
            keypoints: (B, N, 2) in PATCH coordinates
            scores: (B, N) saliency scores
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        # Apply NMS to get local maxima
        if nms_radius > 0:
            saliency = self._apply_nms(saliency, nms_radius)

        # Threshold
        saliency = torch.where(
            saliency > score_threshold,
            saliency,
            torch.zeros_like(saliency)
        )

        keypoints_list = []
        scores_list = []

        for b in range(B):
            sal_b = saliency[b]

            # Get valid points
            valid_mask = sal_b > 0
            valid_scores = sal_b[valid_mask]

            if len(valid_scores) == 0:
                # Fallback: Use top-k from raw saliency
                sal_flat = sal_b.flatten()
                top_scores, top_indices = torch.topk(sal_flat, min(num_keypoints, len(sal_flat)))

                y_coords = top_indices // W
                x_coords = top_indices % W

                kpts = torch.stack([x_coords, y_coords], dim=1).float()
                scrs = top_scores
            else:
                # Get coordinates
                valid_coords = torch.nonzero(valid_mask, as_tuple=False)

                # Select top-k by score
                k = min(num_keypoints, len(valid_scores))
                top_scores, top_indices = torch.topk(valid_scores, k)
                top_coords = valid_coords[top_indices]

                # Convert to (x, y)
                kpts = torch.stack([top_coords[:, 1], top_coords[:, 0]], dim=1).float()
                scrs = top_scores

            # Pad if needed
            if len(kpts) < num_keypoints:
                pad_size = num_keypoints - len(kpts)
                # Use low-score fallback points
                pad_kpts, pad_scrs = self._uniform_grid_keypoints(
                    H, W, pad_size, device
                )
                kpts = torch.cat([kpts, pad_kpts], dim=0)
                scrs = torch.cat([scrs, pad_scrs], dim=0)

            keypoints_list.append(kpts[:num_keypoints])
            scores_list.append(scrs[:num_keypoints])

        keypoints = torch.stack(keypoints_list, dim=0)
        scores = torch.stack(scores_list, dim=0)

        return keypoints, scores

    def _apply_nms(self, saliency: torch.Tensor, radius: int) -> torch.Tensor:
        """Non-maximum suppression"""
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

    def _uniform_grid_keypoints(
        self,
        H: int,
        W: int,
        num_keypoints: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform grid fallback"""
        grid_size = int(num_keypoints ** 0.5) + 1
        y_grid = torch.linspace(1, H-2, grid_size, device=device)
        x_grid = torch.linspace(1, W-2, grid_size, device=device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')

        kpts = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:num_keypoints]
        scores = torch.ones(len(kpts), device=device) * 0.001  # Very low score

        return kpts, scores