"""
SOTA 2025 Keypoint Selector with Sub-Pixel Offset Prediction
Based on DeDoDe v2, ALIKE, DINO-VO, XFeat, and Keypt2Subpx (2024-2025)

Key Innovation: Sub-pixel offset head solves the grid constraint issue
by predicting (dx, dy) offsets in [-1, 1] for each patch location.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class KeypointSelector(nn.Module):
    """
    Keypoint selector with sub-pixel offset prediction (SOTA 2025).

    Architecture:
    1. Saliency Head (3-layer CNN): Predicts per-patch saliency scores [0, 1]
    2. Offset Head (R2D2/ALIKE-style): Predicts sub-pixel offsets (dx, dy) in [-1, 1]

    The offset head solves the grid constraint issue - keypoints can now be
    located at sub-pixel positions within each 16x16 patch.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Saliency Head (3-layer CNN)
        # Input: (B, C, H, W) where H=W=28 and C=384
        # Output: (B, 1, H, W) saliency scores in [0, 1]
        self.saliency_head = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Sub-Pixel Offset Head (R2D2/ALIKE-style)
        # Input: (B, C, H, W) where H=W=28 and C=384
        # Output: (B, 2, H, W) offsets (dx, dy) in [-1, 1]
        self.offset_head = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh()  # Output in [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization with gain=0.5 (as specified)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, dino_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict per-patch saliency scores AND sub-pixel offsets.

        Args:
            dino_features: (B, H, W, C) in PATCH space (BHWC format)

        Returns:
            saliency_map: (B, H, W, 1) scores in [0, 1]
            offset_map: (B, H, W, 2) offsets (dx, dy) in [-1, 1]
        """
        # Convert BHWC to BCHW for convolutions
        x = dino_features.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Predict saliency
        saliency = self.saliency_head(x)  # (B, 1, H, W)

        # Predict sub-pixel offsets
        offsets = self.offset_head(x)  # (B, 2, H, W)

        # Convert back to BHWC format
        saliency_map = saliency.permute(0, 2, 3, 1)  # (B, H, W, 1)
        offset_map = offsets.permute(0, 2, 3, 1)  # (B, H, W, 2)

        return saliency_map, offset_map

    def select_keypoints(
        self,
        saliency_map: torch.Tensor,
        offset_map: torch.Tensor,
        num_keypoints: int = 500,
        nms_radius: int = 2,
        min_score_percentile: float = 0.50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select keypoints with sub-pixel refinement.

        Process:
        1. Apply NMS on saliency map with radius=2 patches (kernel_size=5)
        2. Use percentile-based thresholding (top 50-70% of saliency values)
        3. Select top K=500 keypoints by saliency score
        4. Sub-pixel refinement: kpt_refined = kpt_integer + 0.5 * offset[kpt_integer]
        5. Clamp offsets to [-1, 1] before applying

        Args:
            saliency_map: (B, H, W, 1) independent scores
            offset_map: (B, H, W, 2) sub-pixel offsets
            num_keypoints: Target number of keypoints (default 500)
            nms_radius: NMS radius in patches (default 2, kernel_size=5)
            min_score_percentile: Minimum percentile threshold (default 0.5 = top 50%)

        Returns:
            keypoints: (B, N, 2) in PATCH coordinates with sub-pixel refinement
            scores: (B, N) saliency scores
            offsets: (B, N, 2) applied offsets for each keypoint
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        keypoints_list = []
        scores_list = []
        offsets_list = []

        for b in range(B):
            sal_b = saliency[b]  # (H, W)
            off_b = offset_map[b]  # (H, W, 2)

            # Step 1: Compute adaptive threshold (percentile-based)
            sal_flat = sal_b.flatten()
            threshold = torch.quantile(sal_flat, min_score_percentile)
            threshold = max(threshold.item(), 0.1)  # Minimum threshold to avoid noise

            # Step 2: Apply NMS
            sal_nms = self._apply_nms(sal_b.unsqueeze(0), nms_radius).squeeze(0)

            # Step 3: Threshold
            valid_mask = sal_nms > threshold
            valid_coords = torch.nonzero(valid_mask, as_tuple=False)  # (M, 2) [y, x]
            valid_scores = sal_nms[valid_mask]

            # Step 4: Select top-k by score
            if len(valid_scores) >= num_keypoints:
                k = num_keypoints
                top_scores, top_indices = torch.topk(valid_scores, k)
                top_coords = valid_coords[top_indices]
            elif len(valid_scores) > 0:
                # Not enough - try lower thresholds progressively
                kpts, scrs = self._get_additional_keypoints(
                    sal_b, sal_nms, sal_flat, valid_mask, valid_coords, valid_scores,
                    num_keypoints, W
                )
                top_coords = kpts
                top_scores = scrs
            else:
                # Fallback: select from raw saliency
                top_scores, top_indices = torch.topk(sal_flat, num_keypoints)
                y_coords = top_indices // W
                x_coords = top_indices % W
                top_coords = torch.stack([y_coords, x_coords], dim=1)

            # Ensure exactly num_keypoints
            top_coords, top_scores = self._ensure_num_keypoints(
                top_coords, top_scores, num_keypoints
            )

            # Step 5: Extract offsets at keypoint locations
            # top_coords is (N, 2) with [y, x]
            y_idx = top_coords[:, 0].long()
            x_idx = top_coords[:, 1].long()

            # Clamp indices to valid range
            y_idx = y_idx.clamp(0, H - 1)
            x_idx = x_idx.clamp(0, W - 1)

            kpt_offsets = off_b[y_idx, x_idx]  # (N, 2)

            # Clamp offsets to [-1, 1] (safety) - use non-inplace clone
            kpt_offsets_clamped = kpt_offsets.clone().clamp(-1, 1)

            # Step 6: Apply sub-pixel refinement
            # kpt_refined = kpt_integer + 0.5 * offset
            # This gives sub-pixel accuracy within ±0.5 patch units
            kpts_int = torch.stack([x_idx.float(), y_idx.float()], dim=1)  # (N, 2) [x, y]
            kpts_refined = kpts_int + 0.5 * kpt_offsets_clamped  # Apply scaled offset

            # Clamp to valid patch coordinate range [0, H-1] x [0, W-1]
            # Use non-inplace operations to avoid gradient issues
            kpts_x = kpts_refined[:, 0].clamp(0, W - 1)
            kpts_y = kpts_refined[:, 1].clamp(0, H - 1)
            kpts_refined = torch.stack([kpts_x, kpts_y], dim=1)

            keypoints_list.append(kpts_refined)
            scores_list.append(top_scores.detach())  # Detach scores (not needed for grad)
            offsets_list.append(kpt_offsets_clamped)

        keypoints = torch.stack(keypoints_list, dim=0)  # (B, N, 2)
        scores = torch.stack(scores_list, dim=0)  # (B, N)
        offsets = torch.stack(offsets_list, dim=0)  # (B, N, 2)

        return keypoints, scores, offsets

    def _apply_nms(self, saliency: torch.Tensor, radius: int) -> torch.Tensor:
        """Non-maximum suppression using max pooling"""
        if radius == 0:
            return saliency

        kernel_size = 2 * radius + 1  # radius=2 -> kernel_size=5
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

    def _get_additional_keypoints(
        self,
        sal_b: torch.Tensor,
        sal_nms: torch.Tensor,
        sal_flat: torch.Tensor,
        valid_mask: torch.Tensor,
        valid_coords: torch.Tensor,
        valid_scores: torch.Tensor,
        num_keypoints: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get additional keypoints when not enough above threshold"""
        device = sal_b.device

        existing_kpts = valid_coords  # (M, 2) [y, x]
        existing_scrs = valid_scores  # (M,)

        remaining = num_keypoints - len(valid_scores)

        # Try progressively lower thresholds
        for percentile in [0.40, 0.30, 0.20, 0.10]:
            lower_threshold = torch.quantile(sal_flat, percentile)
            lower_threshold = max(lower_threshold.item(), 0.05)

            additional_mask = (sal_nms > lower_threshold) & (~valid_mask)
            additional_coords = torch.nonzero(additional_mask, as_tuple=False)
            additional_scores = sal_nms[additional_mask]

            if len(additional_scores) >= remaining:
                top_scores, top_indices = torch.topk(additional_scores, remaining)
                top_coords = additional_coords[top_indices]

                kpts = torch.cat([existing_kpts, top_coords], dim=0)
                scrs = torch.cat([existing_scrs, top_scores], dim=0)
                return kpts, scrs

        # Use what we have + pad with top remaining
        if len(existing_kpts) < num_keypoints:
            remaining = num_keypoints - len(existing_kpts)
            top_remaining, top_idx = torch.topk(sal_flat, remaining)

            y_coords = top_idx // W
            x_coords = top_idx % W
            add_kpts = torch.stack([y_coords, x_coords], dim=1)

            kpts = torch.cat([existing_kpts, add_kpts], dim=0)
            scrs = torch.cat([existing_scrs, top_remaining], dim=0)
            return kpts, scrs

        return existing_kpts, existing_scrs

    def _ensure_num_keypoints(
        self,
        coords: torch.Tensor,
        scores: torch.Tensor,
        num_keypoints: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensure exactly num_keypoints are returned"""
        if len(coords) > num_keypoints:
            return coords[:num_keypoints], scores[:num_keypoints]
        elif len(coords) < num_keypoints:
            # Pad with duplicates of highest score point
            pad_size = num_keypoints - len(coords)
            best_idx = scores.argmax()

            pad_coords = coords[best_idx:best_idx+1].repeat(pad_size, 1)
            pad_scores = scores[best_idx:best_idx+1].repeat(pad_size)

            coords = torch.cat([coords, pad_coords], dim=0)
            scores = torch.cat([scores, pad_scores], dim=0)

        return coords, scores

    def get_offset_map(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Get only the offset map (useful for loss computation).

        Args:
            dino_features: (B, H, W, C) in PATCH space

        Returns:
            offset_map: (B, H, W, 2) offsets in [-1, 1]
        """
        x = dino_features.permute(0, 3, 1, 2)
        offsets = self.offset_head(x)
        return offsets.permute(0, 2, 3, 1)
