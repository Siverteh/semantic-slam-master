"""
InfoNCE Contrastive Loss for Descriptor Learning
FIXED: Spatial regularization now operates on saliency map (trainable!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for descriptor learning."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE loss for matched descriptors."""
        B, N, D = desc1.shape
        M = desc2.shape[1]
        device = desc1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            valid_mask = (idx1 < N) & (idx2 < M) & (idx1 >= 0) & (idx2 >= 0)
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            anchors = desc1[b, idx1]
            positives = desc2[b, idx2]

            logits = torch.mm(anchors, desc2[b].t()) / self.temperature
            labels = idx2

            loss_b = F.cross_entropy(logits, labels)

            if not torch.isnan(loss_b) and not torch.isinf(loss_b):
                total_loss += loss_b
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class TripletMarginLoss(nn.Module):
    """Triplet loss with hard negative mining."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
        """Compute triplet loss with hard negative mining."""
        B, N, D = desc1.shape
        M = desc2.shape[1]
        device = desc1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            valid_mask = (idx1 < N) & (idx2 < M) & (idx1 >= 0) & (idx2 >= 0)
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            anchors = desc1[b, idx1]
            positives = desc2[b, idx2]

            pos_dist = torch.norm(anchors - positives, dim=1)

            sim_matrix = torch.mm(anchors, desc2[b].t())

            mask = torch.ones_like(sim_matrix)
            mask[torch.arange(len(idx1)), idx2] = 0

            masked_sim = sim_matrix * mask + (1 - mask) * -1e9
            hard_neg_idx = masked_sim.argmax(dim=1)
            hard_negatives = desc2[b, hard_neg_idx]

            neg_dist = torch.norm(anchors - hard_negatives, dim=1)

            loss_b = F.relu(pos_dist - neg_dist + self.margin).mean()

            if not torch.isnan(loss_b) and not torch.isinf(loss_b):
                total_loss += loss_b
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class SpatialRegularizationLoss(nn.Module):
    """
    Spatial regularization to prevent keypoint clustering.

    CRITICAL FIX: Operates on SALIENCY MAP (before selection) so gradients flow!

    Encourages uniform distribution of high-saliency regions across image.
    """

    def __init__(self, grid_size: int = 4):
        super().__init__()
        self.grid_size = grid_size

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        Penalize uneven saliency distribution across grid.

        FIXED: Operates on saliency map (differentiable) instead of
        selected keypoints (non-differentiable).

        Args:
            saliency_map: (B, H, W, 1) predicted saliency scores [0, 1]

        Returns:
            loss: Spatial regularization loss (encourages uniform saliency)
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        # Remove channel dimension
        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        # Divide into grid cells
        cell_h = H // self.grid_size
        cell_w = W // self.grid_size

        total_loss = 0.0

        for b in range(B):
            sal_b = saliency[b]  # (H, W)

            # Compute total saliency per grid cell
            cell_saliencies = []

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Extract cell
                    y_start = i * cell_h
                    y_end = (i + 1) * cell_h if i < self.grid_size - 1 else H
                    x_start = j * cell_w
                    x_end = (j + 1) * cell_w if j < self.grid_size - 1 else W

                    cell = sal_b[y_start:y_end, x_start:x_end]

                    # Sum of saliency in this cell
                    cell_sal = cell.sum()
                    cell_saliencies.append(cell_sal)

            cell_saliencies = torch.stack(cell_saliencies)  # (grid_size^2,)

            # Ideal: uniform distribution of saliency
            # Total saliency should be evenly split across cells
            total_saliency = cell_saliencies.sum()
            ideal_per_cell = total_saliency / (self.grid_size ** 2)

            # Variance from ideal (lower is better)
            # Using L2 loss instead of variance for better gradients
            loss_b = ((cell_saliencies - ideal_per_cell) ** 2).mean()

            total_loss += loss_b

        return total_loss / B

    def forward_with_stats(self, saliency_map: torch.Tensor) -> tuple:
        """
        Compute loss and return distribution stats for debugging.

        Returns:
            loss: Spatial regularization loss
            stats: Dict with distribution statistics
        """
        B, H, W, _ = saliency_map.shape
        saliency = saliency_map.squeeze(-1)

        cell_h = H // self.grid_size
        cell_w = W // self.grid_size

        total_loss = 0.0
        all_distributions = []

        for b in range(B):
            sal_b = saliency[b]

            cell_saliencies = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    y_start = i * cell_h
                    y_end = (i + 1) * cell_h if i < self.grid_size - 1 else H
                    x_start = j * cell_w
                    x_end = (j + 1) * cell_w if j < self.grid_size - 1 else W

                    cell = sal_b[y_start:y_end, x_start:x_end]
                    cell_saliencies.append(cell.sum())

            cell_saliencies = torch.stack(cell_saliencies)
            total_saliency = cell_saliencies.sum()
            ideal_per_cell = total_saliency / (self.grid_size ** 2)

            loss_b = ((cell_saliencies - ideal_per_cell) ** 2).mean()
            total_loss += loss_b

            # Store for stats
            all_distributions.append(cell_saliencies.detach())

        loss = total_loss / B

        # Compute statistics
        all_dists = torch.stack(all_distributions, dim=0)  # (B, grid_size^2)
        stats = {
            'loss': loss.item(),
            'mean_per_cell': all_dists.mean().item(),
            'std_per_cell': all_dists.std().item(),
            'min_cell': all_dists.min().item(),
            'max_cell': all_dists.max().item(),
            'grid_distributions': all_dists.reshape(B, self.grid_size, self.grid_size).cpu().numpy()
        }

        return loss, stats


class PeakinessRegularizationLoss(nn.Module):
    """
    Discourage single-point peaks in saliency map.
    Encourages smoother, more distributed saliency.

    This helps prevent all keypoints clustering at a few "super-salient" points.
    """

    def __init__(self):
        super().__init__()

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        Penalize extremely peaked saliency distributions.

        Args:
            saliency_map: (B, H, W, 1) saliency scores

        Returns:
            loss: Peakiness penalty
        """
        B, H, W, _ = saliency_map.shape
        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        total_loss = 0.0

        for b in range(B):
            sal_b = saliency[b]  # (H, W)

            # Compute entropy of saliency distribution
            # High entropy = more spread out (good)
            # Low entropy = peaked at few points (bad)

            # Normalize to probability distribution
            sal_norm = sal_b / (sal_b.sum() + 1e-8)

            # Entropy: -sum(p * log(p))
            entropy = -(sal_norm * torch.log(sal_norm + 1e-8)).sum()

            # Maximum possible entropy (uniform distribution)
            max_entropy = torch.log(torch.tensor(H * W, dtype=torch.float32, device=saliency.device))

            # Loss: encourage high entropy (closer to uniform)
            loss_b = max_entropy - entropy

            total_loss += loss_b

        return total_loss / B