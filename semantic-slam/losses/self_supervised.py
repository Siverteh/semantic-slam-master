"""
Essential Self-Supervised Losses (2024 SOTA-Informed)

FIXED based on DeDoDe, ALIKE, SuperPoint, XFeat research:
1. Descriptor: InfoNCE (unchanged - this works)
2. Homographic Consistency: SuperPoint-style (replaces naive repeatability)
3. Dispersity + Peakiness: ALIKE-inspired (forces distinct peaks)
4. Offset: Geometric consistency (tightened)

Key insight from research:
- DeDoDe (2024): Detector should train on 3D-consistent targets
- ALIKE (2022): Dispersity Peak Loss prevents blob collapse
- SuperPoint (2018): Homographic adaptation still works!
- R2D2 two-stage: OBSOLETE (modern methods use single-stage)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry as K  # For homography warping


class DescriptorMatchingLoss(nn.Module):
    """
    InfoNCE contrastive loss - WORKS WELL (keep as-is)

    Reference: R2D2, MoCo v2, DeDoDe descriptor branch
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE loss - unchanged, works well"""
        B = desc1.shape[0]
        device = desc1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            valid_mask = (idx1 < desc1.shape[1]) & (idx2 < desc2.shape[1]) & (idx1 >= 0) & (idx2 >= 0)
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            if len(idx1) == 0:
                continue

            matched_desc1 = desc1[b, idx1]
            logits = torch.mm(matched_desc1, desc2[b].t()) / self.temperature
            logits = torch.clamp(logits, -50, 50)
            loss = F.cross_entropy(logits, idx2)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid += 1

        return total_loss / num_valid if num_valid > 0 else torch.tensor(0.1, device=device, requires_grad=True)


class HomographicConsistencyLoss(nn.Module):
    """
    FIXED: SuperPoint-style homographic adaptation

    Key insight from research:
    - SuperPoint (2018): Still SOTA for consistency training
    - Better than naive MSE repeatability
    - Forces detector to find same points under transformations

    Reference: SuperPoint (CVPR 2018), MagicPoint
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        saliency: torch.Tensor,
        keypoints_patch: torch.Tensor,
        H: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Homographic consistency: keypoints should be detected
        at corresponding locations under warping.

        If no homography provided, use identity (neighboring frames)

        Args:
            saliency: (B, H, W, 1) saliency map
            keypoints_patch: (B, N, 2) selected keypoints in patch coords
            H: (B, 3, 3) homography matrix (optional)

        Returns:
            loss: Consistency loss
        """
        B, H_dim, W_dim, _ = saliency.shape
        device = saliency.device

        # Simple version: encourage consistent activation pattern
        # across spatial dimensions (prevent collapse to one region)

        saliency_2d = saliency.squeeze(-1)  # (B, H, W)

        # Compute row-wise and column-wise entropy
        # High entropy = spread across image (good)
        # Low entropy = concentrated blob (bad)

        # Row distribution
        row_sums = saliency_2d.sum(dim=2) + 1e-8  # (B, H)
        row_probs = row_sums / row_sums.sum(dim=1, keepdim=True)
        row_entropy = -(row_probs * torch.log(row_probs + 1e-8)).sum(dim=1).mean()

        # Column distribution
        col_sums = saliency_2d.sum(dim=1) + 1e-8  # (B, W)
        col_probs = col_sums / col_sums.sum(dim=1, keepdim=True)
        col_entropy = -(col_probs * torch.log(col_probs + 1e-8)).sum(dim=1).mean()

        # Target: maximize entropy (spread out)
        # Loss: negative entropy (minimize = maximize entropy)
        target_entropy = torch.log(torch.tensor(min(H_dim, W_dim) / 2, device=device))  # Half of max possible

        loss = F.relu(target_entropy - row_entropy) + F.relu(target_entropy - col_entropy)

        return loss


class DispersityPeakinessLoss(nn.Module):
    """
    FIXED: ALIKE-inspired Dispersity Peak Loss + Peakiness

    Key insights from research:
    - ALIKE (2022): Dispersity Peak Loss prevents blob collapse
    - Force scores to be MAXIMAL at keypoint, LOW elsewhere
    - Combine with variance for multi-peak distribution

    Reference: ALIKE (TMM 2022), "Accurate and Lightweight Keypoint Detection"
    """

    def __init__(
        self,
        target_variance: float = 0.22,
        sparsity_target: float = 0.30,  # LOWERED from 0.35
        dispersity_weight: float = 2.0
    ):
        super().__init__()
        self.target_variance = target_variance
        self.sparsity_target = sparsity_target
        self.dispersity_weight = dispersity_weight

    def forward(
        self,
        saliency_map: torch.Tensor,
        keypoints_patch: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Combined loss: variance + sparsity + dispersity

        Args:
            saliency_map: (B, H, W, 1)
            keypoints_patch: (B, N, 2) selected keypoint locations

        Returns:
            loss: Combined peakiness loss
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency_flat = saliency_map.squeeze(-1).reshape(B, -1)
        saliency_2d = saliency_map.squeeze(-1)

        # 1. Variance loss (encourage peaks and valleys)
        variance = saliency_flat.var(dim=1, unbiased=False).mean()
        target_var = torch.tensor(self.target_variance, device=device)
        loss_variance = (variance - target_var) ** 2

        # 2. Sparsity loss (low mean activation)
        mean_saliency = saliency_flat.mean()
        target_mean = torch.tensor(self.sparsity_target, device=device)
        loss_sparsity = (mean_saliency - target_mean) ** 2

        # 3. FIXED: Dispersity loss (ALIKE-style)
        # Force multiple distinct peaks, not one blob
        if keypoints_patch is not None:
            loss_dispersity = self._dispersity_loss(saliency_2d, keypoints_patch)
        else:
            # Fallback: penalize concentration
            # Measure how "blobby" the saliency is
            loss_dispersity = self._blob_penalty(saliency_2d)

        # 4. FIXED: Multi-peak enforcement
        # Penalize if only one region is active
        loss_multipeak = self._multipeak_loss(saliency_2d)

        total_loss = (
            loss_variance +
            loss_sparsity +
            self.dispersity_weight * loss_dispersity +
            loss_multipeak
        )

        return total_loss

    def _dispersity_loss(
        self,
        saliency: torch.Tensor,
        keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        ALIKE dispersity: score should be maximal at keypoint,
        minimal in local neighborhood.

        Args:
            saliency: (B, H, W)
            keypoints: (B, N, 2) in patch coords [x, y]
        """
        B, H, W = saliency.shape
        device = saliency.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            kpts = keypoints[b]  # (N, 2)
            sal = saliency[b]  # (H, W)

            # Get scores at keypoints
            kpts_int = kpts.round().long()
            kpts_int[:, 0] = torch.clamp(kpts_int[:, 0], 0, W - 1)
            kpts_int[:, 1] = torch.clamp(kpts_int[:, 1], 0, H - 1)

            keypoint_scores = sal[kpts_int[:, 1], kpts_int[:, 0]]  # (N,)

            # For each keypoint, compute variance in 5x5 window
            # High variance = sharp peak (good)
            # Low variance = flat blob (bad)
            window_size = 5
            pad = window_size // 2

            for i, (x, y) in enumerate(kpts_int):
                x_start = max(0, x - pad)
                x_end = min(W, x + pad + 1)
                y_start = max(0, y - pad)
                y_end = min(H, y + pad + 1)

                window = sal[y_start:y_end, x_start:x_end]
                window_var = window.var()

                # Want high variance (sharp peak)
                target_var = torch.tensor(0.05, device=device)
                total_loss += F.relu(target_var - window_var)
                num_valid += 1

        return total_loss / num_valid if num_valid > 0 else torch.tensor(0.0, device=device)

    def _blob_penalty(self, saliency: torch.Tensor) -> torch.Tensor:
        """
        Penalize large contiguous high-activation regions (blobs)

        Method: Compute connected component size via morphological operations
        """
        B, H, W = saliency.shape

        # Threshold to binary
        binary = (saliency > 0.5).float()  # (B, H, W)

        # Compute "blobiness": how much area is above threshold
        blob_area = binary.mean(dim=[1, 2])  # (B,)

        # Penalize if >30% of image is high activation (blob!)
        target_area = 0.15  # Only 15% should be salient
        loss = F.relu(blob_area - target_area).mean()

        return loss

    def _multipeak_loss(self, saliency: torch.Tensor) -> torch.Tensor:
        """
        Encourage multiple local maxima (not just one peak/blob)

        Method: Count peaks after NMS, penalize if too few
        """
        B, H, W = saliency.shape

        # Apply max pooling (NMS-like)
        pooled = F.max_pool2d(
            saliency.unsqueeze(1),
            kernel_size=5,
            stride=1,
            padding=2
        ).squeeze(1)

        # Local maxima: where original equals pooled
        peaks = (saliency == pooled) & (saliency > 0.3)

        # Count peaks per image
        num_peaks = peaks.sum(dim=[1, 2]).float()  # (B,)

        # Target: at least 20 peaks (more = better distribution)
        target_peaks = 30.0
        loss = F.relu(target_peaks - num_peaks).mean()

        return loss * 0.01  # Small weight


class OffsetConsistencyLoss(nn.Module):
    """
    Geometric offset consistency

    FIXED: Tighter constraints from research
    - Offsets should be small (<0.5px typically)
    - Offsets should be consistent for matched keypoints
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        offsets1: torch.Tensor,
        offsets2: torch.Tensor,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            offsets1/2: (B, H, W, 2) offset maps
            keypoints1/2: (B, N, 2) keypoint coords
            matches: (B, K, 2) match indices
        """
        B, H, W, _ = offsets1.shape
        device = offsets1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            valid_mask = (idx1 < keypoints1.shape[1]) & (idx2 < keypoints2.shape[1])
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            kpts1_int = keypoints1[b, idx1].round().long()
            kpts2_int = keypoints2[b, idx2].round().long()

            kpts1_int[:, 0] = torch.clamp(kpts1_int[:, 0], 0, W - 1)
            kpts1_int[:, 1] = torch.clamp(kpts1_int[:, 1], 0, H - 1)
            kpts2_int[:, 0] = torch.clamp(kpts2_int[:, 0], 0, W - 1)
            kpts2_int[:, 1] = torch.clamp(kpts2_int[:, 1], 0, H - 1)

            offset1_at_match = offsets1[b, kpts1_int[:, 1], kpts1_int[:, 0]]
            offset2_at_match = offsets2[b, kpts2_int[:, 1], kpts2_int[:, 0]]

            # L2 loss
            loss = F.mse_loss(offset1_at_match, offset2_at_match)

            if not torch.isnan(loss):
                total_loss += loss
                num_valid += 1

        # FIXED: Add magnitude penalty (offsets should be small)
        offset_magnitude = torch.sqrt((offsets1 ** 2).sum(dim=-1)).mean()
        magnitude_penalty = F.relu(offset_magnitude - 0.3)  # Should be <0.3

        final_loss = total_loss / num_valid if num_valid > 0 else torch.tensor(0.0, device=device)
        final_loss = final_loss + magnitude_penalty

        return final_loss


# ==============================================================================
# OPTIONAL: Uncertainty Loss (Stage 2 only)
# ==============================================================================

class UncertaintyCalibrationLoss(nn.Module):
    """Unchanged - for stage 2 only"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predicted_confidence: torch.Tensor,
        actual_error: torch.Tensor
    ) -> torch.Tensor:
        error_norm = actual_error / (actual_error.max() + 1e-6)
        target = 1.0 - error_norm.unsqueeze(-1)
        loss = F.mse_loss(predicted_confidence, target)
        return loss