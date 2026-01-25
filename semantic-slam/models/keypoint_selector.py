"""
FIXED Keypoint Selector - Actually Uses Saliency Properly!
CRITICAL FIXES:
1. Better thresholding (percentile-based, not absolute)
2. Smart fallback (samples from high-saliency regions only)
3. No uniform grid fallback (that was causing dark monitor selections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KeypointSelector(nn.Module):
    """
    Keypoint selector with PROPER saliency-based selection.

    Key fix: Never falls back to uniform sampling!
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128
    ):
        super().__init__()

        # Simple 2-layer CNN
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

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
        logits = self.conv(x)

        # Apply SIGMOID (not softmax!)
        saliency = torch.sigmoid(logits)

        # Convert back to (B, H, W, 1)
        saliency_map = saliency.permute(0, 2, 3, 1)

        return saliency_map

    def select_keypoints(
        self,
        saliency_map: torch.Tensor,
        num_keypoints: int = 500,
        nms_radius: int = 2,
        min_score_percentile: float = 0.50  # NEW: Use top 50% of scores
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select keypoints using PERCENTILE-BASED thresholding.

        CRITICAL FIX: Never uses uniform fallback!
        Always selects from high-saliency regions.

        Args:
            saliency_map: (B, H, W, 1) independent scores
            num_keypoints: Target number of keypoints
            nms_radius: NMS radius
            min_score_percentile: Minimum percentile (0.5 = top 50%)

        Returns:
            keypoints: (B, N, 2) in PATCH coordinates
            scores: (B, N) saliency scores
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        keypoints_list = []
        scores_list = []

        for b in range(B):
            sal_b = saliency[b]

            # STEP 1: Compute ADAPTIVE threshold (percentile-based)
            # This ensures we always select from high-saliency regions!
            sal_flat = sal_b.flatten()
            threshold = torch.quantile(sal_flat, min_score_percentile)

            # Ensure minimum threshold (avoid selecting noise)
            threshold = max(threshold.item(), 0.1)

            # STEP 2: Apply NMS
            sal_nms = self._apply_nms(sal_b.unsqueeze(0), nms_radius).squeeze(0)

            # STEP 3: Threshold with adaptive value
            valid_mask = sal_nms > threshold
            valid_coords = torch.nonzero(valid_mask, as_tuple=False)
            valid_scores = sal_nms[valid_mask]

            # STEP 4: Select top-k by score
            if len(valid_scores) >= num_keypoints:
                # Have enough candidates - select top-k
                k = num_keypoints
                top_scores, top_indices = torch.topk(valid_scores, k)
                top_coords = valid_coords[top_indices]

                # Convert to (x, y) format
                kpts = torch.stack([top_coords[:, 1], top_coords[:, 0]], dim=1).float()
                scrs = top_scores

            elif len(valid_scores) > 0:
                # Have some candidates - use all of them + sample more from lower threshold
                existing_kpts = torch.stack([valid_coords[:, 1], valid_coords[:, 0]], dim=1).float()
                existing_scrs = valid_scores

                # Need more keypoints - lower threshold gradually
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

                        add_kpts = torch.stack([top_coords[:, 1], top_coords[:, 0]], dim=1).float()
                        add_scrs = top_scores

                        kpts = torch.cat([existing_kpts, add_kpts], dim=0)
                        scrs = torch.cat([existing_scrs, add_scrs], dim=0)
                        break
                else:
                    # Use what we have
                    kpts = existing_kpts
                    scrs = existing_scrs

                    # Pad with highest remaining scores
                    if len(kpts) < num_keypoints:
                        all_scores = sal_b.flatten()
                        remaining = num_keypoints - len(kpts)
                        top_remaining, top_idx = torch.topk(all_scores, remaining)

                        y_coords = top_idx // W
                        x_coords = top_idx % W
                        add_kpts = torch.stack([x_coords, y_coords], dim=1).float()

                        kpts = torch.cat([kpts, add_kpts], dim=0)
                        scrs = torch.cat([scrs, top_remaining], dim=0)
            else:
                # LAST RESORT: No valid candidates above threshold
                # Select top-k from raw saliency (but still from high scores!)
                sal_flat = sal_b.flatten()
                top_scores, top_indices = torch.topk(sal_flat, num_keypoints)

                y_coords = top_indices // W
                x_coords = top_indices % W

                kpts = torch.stack([x_coords, y_coords], dim=1).float()
                scrs = top_scores

            # Ensure exactly num_keypoints
            if len(kpts) > num_keypoints:
                kpts = kpts[:num_keypoints]
                scrs = scrs[:num_keypoints]
            elif len(kpts) < num_keypoints:
                # Pad with duplicates of highest score point
                pad_size = num_keypoints - len(kpts)
                best_idx = scrs.argmax()

                pad_kpts = kpts[best_idx:best_idx+1].repeat(pad_size, 1)
                pad_scrs = scrs[best_idx:best_idx+1].repeat(pad_size)

                kpts = torch.cat([kpts, pad_kpts], dim=0)
                scrs = torch.cat([scrs, pad_scrs], dim=0)

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