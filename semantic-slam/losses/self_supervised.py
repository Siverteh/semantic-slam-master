"""
SOTA 2025 Self-Supervised Losses for Semantic Keypoint Detection
Based on DeDoDe v2, ALIKE, DINO-VO, XFeat, and Keypt2Subpx (2024-2025)

EXACTLY 4 LOSSES (NO MORE):
1. Descriptor Matching Loss (weight=10.0) - InfoNCE with temperature=0.07
2. Homographic Consistency Loss (weight=1.0) - Spatial entropy maximization
3. Dispersity Peakiness Loss (weight=5.0) - Variance + sparsity + window variance
4. Offset Consistency Loss (weight=0.15) - Offset matching + magnitude regularization

Total loss = 10.0*descriptor + 1.0*homographic + 5.0*peakiness + 0.15*offset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DescriptorMatchingLoss(nn.Module):
    """
    Loss 1: Descriptor Matching Loss (weight=10.0)

    InfoNCE / cross-entropy on similarity matrix.
    Temperature = 0.07 (lower = harder negatives)

    For matched keypoint pairs between frame1 and frame2:
    - similarity_matrix = descriptors1 @ descriptors2.T / temperature
    - loss = CrossEntropy(similarity_matrix, match_indices)
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
        """
        InfoNCE loss: pull matched descriptors together, push others apart.

        Args:
            desc1: (B, N, D) L2-normalized descriptors from frame 1
            desc2: (B, N, D) L2-normalized descriptors from frame 2
            matches: (B, M, 2) match indices where matches[b, i] = [idx1, idx2]

        Returns:
            loss: Scalar InfoNCE loss
        """
        B = desc1.shape[0]
        device = desc1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            # Filter valid indices
            valid_mask = (idx1 >= 0) & (idx2 >= 0) & \
                        (idx1 < desc1.shape[1]) & (idx2 < desc2.shape[1])

            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            if len(idx1) < 2:  # Need at least 2 matches for contrastive
                continue

            # Get matched descriptors
            matched_desc1 = desc1[b, idx1]  # (M, D)

            # Compute similarity to ALL descriptors in frame 2
            # similarity_matrix = descriptors1 @ descriptors2.T / temperature
            logits = torch.mm(matched_desc1, desc2[b].t()) / self.temperature

            # Clamp for numerical stability
            logits = torch.clamp(logits, -50, 50)

            # Cross-entropy: want the matched descriptor (idx2) to have highest similarity
            loss = F.cross_entropy(logits, idx2)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            # Return small positive loss to maintain gradient flow
            return torch.tensor(0.1, device=device, requires_grad=True)


class HomographicConsistencyLoss(nn.Module):
    """
    Loss 2: Homographic Consistency Loss (weight=1.0)

    SuperPoint-style: maximize spatial entropy of saliency distribution.
    - Compute row-wise and column-wise entropy of saliency map
    - Target: entropy should be high (spread across image, not concentrated)
    - Formula: loss = max(0, target_entropy - actual_entropy)
    - Target entropy = log(grid_size / 2) for 28×28 grid
    """

    def __init__(self, grid_size: int = 28):
        super().__init__()
        self.grid_size = grid_size
        # Target entropy = log(grid_size / 2) = log(14) ≈ 2.64
        self.target_entropy = torch.log(torch.tensor(grid_size / 2.0))

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        Maximize spatial entropy - encourage spread keypoints.

        Args:
            saliency_map: (B, H, W, 1) saliency scores in [0, 1]

        Returns:
            loss: Scalar entropy loss
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        # Normalize to probability distribution
        saliency_flat = saliency.reshape(B, -1)  # (B, H*W)
        saliency_prob = saliency_flat / (saliency_flat.sum(dim=1, keepdim=True) + 1e-8)

        # Row-wise entropy: marginal distribution over rows
        row_marginal = saliency.sum(dim=2)  # (B, H) - sum over columns
        row_marginal = row_marginal / (row_marginal.sum(dim=1, keepdim=True) + 1e-8)
        row_entropy = -(row_marginal * torch.log(row_marginal + 1e-8)).sum(dim=1)

        # Column-wise entropy: marginal distribution over columns
        col_marginal = saliency.sum(dim=1)  # (B, W) - sum over rows
        col_marginal = col_marginal / (col_marginal.sum(dim=1, keepdim=True) + 1e-8)
        col_entropy = -(col_marginal * torch.log(col_marginal + 1e-8)).sum(dim=1)

        # Average entropy
        avg_entropy = (row_entropy + col_entropy) / 2.0  # (B,)

        # Loss: penalize if entropy is below target
        # loss = max(0, target_entropy - actual_entropy)
        target = self.target_entropy.to(device)
        loss = F.relu(target - avg_entropy).mean()

        return loss


class DispersityPeakinessLoss(nn.Module):
    """
    Loss 3: Dispersity Peakiness Loss (weight=5.0)

    Three components combined:
    A) Variance: encourage peaked vs flat distribution
       - Target variance = 0.22
       - loss = (actual_variance - 0.22)^2

    B) Sparsity: low mean activation (only 30% of pixels active)
       - Target mean = 0.30
       - loss = (actual_mean - 0.30)^2

    C) Dispersity (ALIKE-style): penalize flat blobs
       - For each selected keypoint, get 5×5 window around it
       - Window should have high variance (sharp peak at center)
       - Loss = ReLU(0.05 - window_variance) summed over keypoints
    """

    def __init__(
        self,
        target_variance: float = 0.22,
        target_mean: float = 0.30,
        window_size: int = 5,
        min_window_variance: float = 0.05
    ):
        super().__init__()
        self.target_variance = target_variance
        self.target_mean = target_mean
        self.window_size = window_size
        self.min_window_variance = min_window_variance

    def forward(
        self,
        saliency_map: torch.Tensor,
        keypoint_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combined dispersity + peakiness loss.

        Args:
            saliency_map: (B, H, W, 1) saliency scores
            keypoint_coords: (B, N, 2) keypoint locations in [x, y] patch coords
                            If None, uses top saliency locations

        Returns:
            loss: Scalar combined loss
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency = saliency_map.squeeze(-1)  # (B, H, W)
        saliency_flat = saliency.reshape(B, -1)

        # Component A: Variance loss
        variance = saliency_flat.var(dim=1, unbiased=False)  # (B,)
        mean_variance = variance.mean()
        target_var = torch.tensor(self.target_variance, device=device)
        loss_variance = (mean_variance - target_var) ** 2

        # Component B: Sparsity loss (target mean activation)
        mean_activation = saliency.mean()
        target_m = torch.tensor(self.target_mean, device=device)
        loss_sparsity = (mean_activation - target_m) ** 2

        # Component C: Dispersity loss (window variance around keypoints)
        loss_dispersity = self._compute_dispersity(saliency, keypoint_coords)

        # Combine: all components equally weighted within this loss
        total_loss = loss_variance + loss_sparsity + loss_dispersity

        return total_loss

    def _compute_dispersity(
        self,
        saliency: torch.Tensor,
        keypoint_coords: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute dispersity loss: penalize flat blobs around keypoints.

        For each keypoint, extract 5x5 window and check if it has high variance
        (indicating a sharp peak at the center).
        """
        B, H, W = saliency.shape
        device = saliency.device

        if keypoint_coords is None:
            # Use top-k saliency locations as keypoints
            k = min(100, H * W // 10)  # Sample 100 points
            sal_flat = saliency.reshape(B, -1)
            _, top_indices = torch.topk(sal_flat, k, dim=1)

            y_coords = top_indices // W
            x_coords = top_indices % W
            keypoint_coords = torch.stack([x_coords, y_coords], dim=-1).float()

        B_kpt, N, _ = keypoint_coords.shape
        half_w = self.window_size // 2

        total_dispersity = 0.0
        num_valid = 0

        for b in range(B):
            sal_b = saliency[b]  # (H, W)

            # Pad saliency for safe window extraction
            sal_padded = F.pad(sal_b.unsqueeze(0).unsqueeze(0),
                              (half_w, half_w, half_w, half_w),
                              mode='reflect').squeeze()

            window_variances = []

            for n in range(min(N, 100)):  # Limit to 100 keypoints for efficiency
                x = int(keypoint_coords[b, n, 0].item())
                y = int(keypoint_coords[b, n, 1].item())

                # Clamp to valid range
                x = max(0, min(x, W - 1))
                y = max(0, min(y, H - 1))

                # Extract window (accounting for padding offset)
                window = sal_padded[y:y + self.window_size, x:x + self.window_size]

                if window.numel() == self.window_size ** 2:
                    window_var = window.var()
                    window_variances.append(window_var)

            if len(window_variances) > 0:
                window_variances = torch.stack(window_variances)
                # Penalize windows with low variance (flat blobs)
                # Loss = ReLU(min_variance - actual_variance)
                min_var = torch.tensor(self.min_window_variance, device=device)
                dispersity_loss = F.relu(min_var - window_variances).sum()
                total_dispersity += dispersity_loss
                num_valid += len(window_variances)

        if num_valid > 0:
            return total_dispersity / num_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class OffsetConsistencyLoss(nn.Module):
    """
    Loss 4: Offset Consistency Loss (weight=0.15)

    Two components:
    1. For matched keypoint pairs, offsets should be similar
       - loss_consistency = MSE(offset1[matches], offset2[matches])

    2. Penalize large offsets (should be < 0.3 in normalized coordinates)
       - loss_magnitude = ReLU(offset_magnitude - 0.3)

    Total = loss_consistency + loss_magnitude
    """

    def __init__(self, max_offset_magnitude: float = 0.3):
        super().__init__()
        self.max_offset_magnitude = max_offset_magnitude

    def forward(
        self,
        offset_map1: torch.Tensor,
        offset_map2: torch.Tensor,
        matches: torch.Tensor,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor
    ) -> torch.Tensor:
        """
        Offset consistency loss for matched keypoints.

        Args:
            offset_map1: (B, H, W, 2) offset predictions for frame 1
            offset_map2: (B, H, W, 2) offset predictions for frame 2
            matches: (B, M, 2) match indices
            keypoints1: (B, N, 2) keypoint integer coords in frame 1 [x, y]
            keypoints2: (B, N, 2) keypoint integer coords in frame 2 [x, y]

        Returns:
            loss: Scalar offset consistency loss
        """
        B, H, W, _ = offset_map1.shape
        device = offset_map1.device

        total_consistency_loss = 0.0
        total_magnitude_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            # Filter valid indices
            valid_mask = (idx1 >= 0) & (idx2 >= 0) & \
                        (idx1 < keypoints1.shape[1]) & (idx2 < keypoints2.shape[1])

            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            # Get keypoint integer coordinates (detach since we just use them for indexing)
            kpts1_int = keypoints1[b, idx1].detach().long()  # (M, 2) [x, y]
            kpts2_int = keypoints2[b, idx2].detach().long()  # (M, 2) [x, y]

            # Clamp to valid range (non-inplace)
            kpts1_x = kpts1_int[:, 0].clamp(0, W - 1)
            kpts1_y = kpts1_int[:, 1].clamp(0, H - 1)
            kpts2_x = kpts2_int[:, 0].clamp(0, W - 1)
            kpts2_y = kpts2_int[:, 1].clamp(0, H - 1)

            # Extract offsets at matched keypoint locations
            offsets1 = offset_map1[b, kpts1_y, kpts1_x]  # (M, 2)
            offsets2 = offset_map2[b, kpts2_y, kpts2_x]  # (M, 2)

            # Component 1: Offset consistency - matched pairs should have similar offsets
            consistency_loss = F.mse_loss(offsets1, offsets2)
            total_consistency_loss += consistency_loss

            # Component 2: Magnitude regularization - penalize large offsets
            magnitude1 = torch.sqrt((offsets1 ** 2).sum(dim=1))  # (M,)
            magnitude2 = torch.sqrt((offsets2 ** 2).sum(dim=1))  # (M,)

            max_mag = torch.tensor(self.max_offset_magnitude, device=device)
            magnitude_loss = F.relu(magnitude1 - max_mag).mean() + \
                           F.relu(magnitude2 - max_mag).mean()
            total_magnitude_loss += magnitude_loss

            num_valid += 1

        if num_valid > 0:
            return (total_consistency_loss + total_magnitude_loss) / num_valid
        else:
            # Return small loss to maintain gradient flow
            return torch.tensor(0.01, device=device, requires_grad=True)


# ============================================================================
# Combined Loss Function for Easy Use
# ============================================================================

class SOTAKeypointLoss(nn.Module):
    """
    Combined SOTA loss with exactly 4 components.

    Total loss = 10.0*descriptor + 1.0*homographic + 5.0*peakiness + 0.15*offset
    """

    def __init__(
        self,
        desc_weight: float = 10.0,
        homographic_weight: float = 1.0,
        peakiness_weight: float = 5.0,
        offset_weight: float = 0.15,
        temperature: float = 0.07,
        grid_size: int = 28
    ):
        super().__init__()

        self.weights = {
            'descriptor': desc_weight,
            'homographic': homographic_weight,
            'peakiness': peakiness_weight,
            'offset': offset_weight
        }

        self.descriptor_loss = DescriptorMatchingLoss(temperature=temperature)
        self.homographic_loss = HomographicConsistencyLoss(grid_size=grid_size)
        self.peakiness_loss = DispersityPeakinessLoss()
        self.offset_loss = OffsetConsistencyLoss()

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        saliency1: torch.Tensor,
        saliency2: torch.Tensor,
        offset1: torch.Tensor,
        offset2: torch.Tensor,
        matches: torch.Tensor,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute all 4 losses and return weighted sum.

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss values
        """
        # Loss 1: Descriptor matching
        loss_desc = self.descriptor_loss(desc1, desc2, matches)

        # Loss 2: Homographic consistency (spatial entropy)
        loss_homo = self.homographic_loss(saliency1)

        # Loss 3: Dispersity + Peakiness
        loss_peak = self.peakiness_loss(saliency1, keypoints1)

        # Loss 4: Offset consistency
        loss_offset = self.offset_loss(
            offset1, offset2, matches, keypoints1, keypoints2
        )

        # Weighted sum
        total = (
            self.weights['descriptor'] * loss_desc +
            self.weights['homographic'] * loss_homo +
            self.weights['peakiness'] * loss_peak +
            self.weights['offset'] * loss_offset
        )

        loss_dict = {
            'descriptor': loss_desc.item(),
            'homographic': loss_homo.item(),
            'peakiness': loss_peak.item(),
            'offset': loss_offset.item(),
            'total': total.item()
        }

        return total, loss_dict
