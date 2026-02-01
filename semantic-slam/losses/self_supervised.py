"""
Self-Supervised Losses with R2D2's AP Loss
Better than triplet loss for descriptor learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class APLoss(nn.Module):
    """
    Average Precision loss from R2D2.

    Better than triplet loss because:
    - Optimizes global ranking metric (AP) directly
    - Uses all pairs in batch, not just triplets
    - Better gradient signal
    """

    def __init__(self, kappa: float = 0.5):
        super().__init__()
        self.kappa = kappa  # Minimum expected AP

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute AP loss for descriptor matching.

        Args:
            desc1: (B, N, D) descriptors from frame 1
            desc2: (B, M, D) descriptors from frame 2
            matches: (B, K, 2) ground truth matches [idx1, idx2]

        Returns:
            loss: Scalar AP loss
        """
        B, N, D = desc1.shape
        M = desc2.shape[1]
        device = desc1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            # Compute similarity matrix
            sim_matrix = torch.mm(desc1[b], desc2[b].t())  # (N, M)

            # Create ground truth matrix
            gt_matrix = torch.zeros(N, M, device=device)

            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            # Filter valid indices
            valid_mask = (idx1 < N) & (idx2 < M) & (idx1 >= 0) & (idx2 >= 0)
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            gt_matrix[idx1, idx2] = 1.0

            # Compute AP for each query
            for i in range(N):
                # Skip if no ground truth
                if gt_matrix[i].sum() == 0:
                    continue

                # Similarities for query i
                sims = sim_matrix[i]  # (M,)
                gt = gt_matrix[i]  # (M,)

                # Sort by similarity (descending)
                sorted_indices = torch.argsort(sims, descending=True)
                sorted_gt = gt[sorted_indices]

                # Compute precision at each rank
                cumsum = torch.cumsum(sorted_gt, dim=0)
                ranks = torch.arange(1, M + 1, device=device, dtype=torch.float32)
                precision = cumsum / ranks

                # Average precision
                ap = (precision * sorted_gt).sum() / (sorted_gt.sum() + 1e-8)

                # Loss: 1 - AP
                loss_i = 1.0 - ap

                if not torch.isnan(loss_i):
                    total_loss += loss_i
                    num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.1, device=device, requires_grad=True)


class DescriptorVarianceLoss(nn.Module):
    """Prevent descriptor collapse"""

    def __init__(self, min_variance: float = 0.005):
        super().__init__()
        self.min_variance = min_variance

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        B, N, D = descriptors.shape

        # Reshape
        desc_flat = descriptors.reshape(B * N, D)

        # Variance per dimension
        variance_per_dim = desc_flat.var(dim=0)
        mean_variance = variance_per_dim.mean()

        # Penalize if too low
        min_var = torch.tensor(self.min_variance, device=descriptors.device)
        loss = F.relu(min_var - mean_variance)

        return loss


class RepeatabilityLoss(nn.Module):
    """Keypoint repeatability loss"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        saliency1: torch.Tensor,
        saliency2: torch.Tensor
    ) -> torch.Tensor:
        """Compare saliency maps directly"""
        sal1_flat = saliency1.reshape(saliency1.shape[0], -1)
        sal2_flat = saliency2.reshape(saliency2.shape[0], -1)
        loss = F.mse_loss(sal1_flat, sal2_flat)
        return loss


class SemanticEdgeLoss(nn.Module):
    """
    Loss to encourage saliency at semantic edges.
    Your idea! Keypoints should align with object boundaries.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        saliency_map: torch.Tensor,
        semantic_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage saliency to correlate with semantic edges.

        Args:
            saliency_map: (B, H, W, 1) predicted saliency
            semantic_edges: (B, H, W, 1) semantic edge strength

        Returns:
            loss: Correlation loss
        """
        B, H, W, _ = saliency_map.shape

        sal_flat = saliency_map.reshape(B, -1)
        edge_flat = semantic_edges.reshape(B, -1)

        # Normalize
        sal_norm = sal_flat - sal_flat.mean(dim=1, keepdim=True)
        edge_norm = edge_flat - edge_flat.mean(dim=1, keepdim=True)

        # Pearson correlation
        correlation = (sal_norm * edge_norm).sum(dim=1) / (
            torch.sqrt((sal_norm ** 2).sum(dim=1) * (edge_norm ** 2).sum(dim=1)) + 1e-8
        )

        # Maximize correlation (minimize negative)
        # Add 1 to shift to [0, 2] range, then divide by 2 to get [0, 1]
        loss = -(correlation.mean() + 1.0) / 2.0

        return loss