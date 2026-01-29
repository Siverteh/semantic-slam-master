"""
Keypoint Selector with Sub-Pixel Refinement (R2D2-style)

Key improvements:
1. Predicts sub-pixel offsets for accurate localization
2. Percentile-based thresholding (no uniform fallback)
3. Clean, focused implementation

Reference: R2D2 (CVPR 2019), SuperPoint (CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KeypointSelector(nn.Module):
    """
    Learns to select stable keypoints with sub-pixel accuracy.

    Architecture:
    - Saliency head: predicts per-patch importance scores
    - Offset head: refines location within patch (R2D2 style)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128
    ):
        super().__init__()

        # Saliency prediction (which patches are interesting)
        self.saliency_head = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # Sub-pixel offset prediction (where exactly in the patch)
        # Outputs (dx, dy) in range [-1, 1] (normalized to patch size)
        self.offset_head = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),  # 2 channels: (dx, dy)
            nn.Tanh()  # Bound to [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, dino_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict saliency and sub-pixel offsets.

        Args:
            dino_features: (B, H, W, C) in PATCH space

        Returns:
            saliency_map: (B, H, W, 1) scores in [0, 1]
            offset_map: (B, H, W, 2) sub-pixel offsets in [-1, 1]
        """
        # Convert to (B, C, H, W) for conv layers
        x = dino_features.permute(0, 3, 1, 2)

        # Predict saliency
        saliency_logits = self.saliency_head(x)
        saliency = torch.sigmoid(saliency_logits)

        # Predict offsets
        offsets = self.offset_head(x)

        # Convert back to (B, H, W, C) format
        saliency_map = saliency.permute(0, 2, 3, 1)  # (B, H, W, 1)
        offset_map = offsets.permute(0, 2, 3, 1)     # (B, H, W, 2)

        return saliency_map, offset_map

    def select_keypoints(
        self,
        saliency_map: torch.Tensor,
        offset_map: torch.Tensor,
        num_keypoints: int = 500,
        nms_radius: int = 2,
        min_score_percentile: float = 0.50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select keypoints with SUB-PIXEL accuracy.

        Process:
        1. NMS on saliency to find local maxima
        2. Percentile-based thresholding (adaptive)
        3. Refine locations with predicted offsets

        Args:
            saliency_map: (B, H, W, 1) saliency scores
            offset_map: (B, H, W, 2) sub-pixel offsets
            num_keypoints: Target number of keypoints
            nms_radius: NMS radius in patches
            min_score_percentile: Minimum score percentile (0.5 = top 50%)

        Returns:
            keypoints: (B, N, 2) in PATCH coordinates (with sub-pixel precision)
            scores: (B, N) saliency scores
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        keypoints_list = []
        scores_list = []

        for b in range(B):
            sal_b = saliency[b]
            offset_b = offset_map[b]  # (H, W, 2)

            # Step 1: Adaptive thresholding
            sal_flat = sal_b.flatten()
            threshold = torch.quantile(sal_flat, min_score_percentile)
            threshold = max(threshold.item(), 0.1)  # Minimum safety threshold

            # Step 2: NMS
            sal_nms = self._apply_nms(sal_b.unsqueeze(0), nms_radius).squeeze(0)

            # Step 3: Threshold
            valid_mask = sal_nms > threshold
            valid_coords = torch.nonzero(valid_mask, as_tuple=False)  # (M, 2) in (y, x)
            valid_scores = sal_nms[valid_mask]

            # Step 4: Select top-k
            if len(valid_scores) >= num_keypoints:
                k = num_keypoints
                top_scores, top_indices = torch.topk(valid_scores, k)
                top_coords = valid_coords[top_indices]  # (K, 2)
            else:
                # Not enough candidates - lower threshold iteratively
                for percentile in [0.40, 0.30, 0.20, 0.10]:
                    lower_threshold = torch.quantile(sal_flat, percentile)
                    lower_mask = sal_nms > max(lower_threshold.item(), 0.05)
                    lower_coords = torch.nonzero(lower_mask, as_tuple=False)
                    lower_scores = sal_nms[lower_mask]

                    if len(lower_scores) >= num_keypoints:
                        top_scores, top_indices = torch.topk(lower_scores, num_keypoints)
                        top_coords = lower_coords[top_indices]
                        break
                else:
                    # Last resort: use all available
                    top_coords = valid_coords
                    top_scores = valid_scores

                    # Pad if needed
                    if len(top_coords) < num_keypoints:
                        # Duplicate highest score point
                        pad_size = num_keypoints - len(top_coords)
                        best_idx = top_scores.argmax()
                        pad_coords = top_coords[best_idx:best_idx+1].repeat(pad_size, 1)
                        pad_scores = top_scores[best_idx:best_idx+1].repeat(pad_size)

                        top_coords = torch.cat([top_coords, pad_coords], dim=0)
                        top_scores = torch.cat([top_scores, pad_scores], dim=0)

            # Step 5: SUB-PIXEL REFINEMENT (R2D2 style)
            y_coords = top_coords[:, 0]  # (K,)
            x_coords = top_coords[:, 1]  # (K,)

            # Get offsets at these integer locations
            offsets_at_kpts = offset_b[y_coords, x_coords]  # (K, 2) in (dx, dy)

            # FIXED: Clamp offsets first to prevent saturation
            offsets_at_kpts = torch.clamp(offsets_at_kpts, -1.0, 1.0)

            # Refine: integer coords + fractional offsets
            # Offsets are in [-1, 1], scale to [-0.5, 0.5] for sub-pixel precision
            x_refined = x_coords.float() + 0.5 * offsets_at_kpts[:, 0]
            y_refined = y_coords.float() + 0.5 * offsets_at_kpts[:, 1]

            # Clamp to valid range
            x_refined = torch.clamp(x_refined, 0, W - 1)
            y_refined = torch.clamp(y_refined, 0, H - 1)

            kpts = torch.stack([x_refined, y_refined], dim=1)  # (K, 2) in (x, y)
            scrs = top_scores

            keypoints_list.append(kpts)
            scores_list.append(scrs)

        keypoints = torch.stack(keypoints_list, dim=0)
        scores = torch.stack(scores_list, dim=0)

        return keypoints, scores

    def _apply_nms(self, saliency: torch.Tensor, radius: int) -> torch.Tensor:
        """Non-maximum suppression"""
        if radius == 0:
            return saliency

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