"""
Self-Supervised Losses - DESCRIPTOR QUALITY FOCUSED
Key addition: Proper descriptor variance regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorMatchingLoss(nn.Module):
    """
    InfoNCE contrastive loss with proper negative sampling.
    IMPROVED: Better temperature and more negatives.
    """

    def __init__(self, temperature: float = 0.10, num_negatives: int = 40):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE loss: pull matched descriptors together, push others apart.
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

            valid_mask = (idx1 < desc1.shape[1]) & (idx2 < desc2.shape[1]) & (idx1 >= 0) & (idx2 >= 0)
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            if len(idx1) == 0:
                continue

            matched_desc1 = desc1[b, idx1]
            matched_desc2 = desc2[b, idx2]

            # Compute similarity to ALL descriptors in frame 2
            logits = torch.mm(matched_desc1, desc2[b].t()) / self.temperature
            logits = torch.clamp(logits, -50, 50)

            # Cross-entropy: want matched_desc2 to be most similar
            loss = F.cross_entropy(logits, idx2)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.1, device=device, requires_grad=True)


class DescriptorVarianceLoss(nn.Module):
    """
    FIXED: Descriptor variance regularization with correct target.
    Prevents descriptor collapse while allowing discriminative learning.

    Key insight: L2-normalized D-dim descriptors have expected variance ≈ 1/D
    For 128-dim: expected variance ≈ 0.0078
    """

    def __init__(self, min_variance: float = 0.005):
        """
        Args:
            min_variance: Minimum acceptable variance (default: 0.005 for 128-dim)
                         Should be slightly below 1/D to allow some specialization
        """
        super().__init__()
        self.min_variance = min_variance

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Prevent descriptor collapse by ensuring minimum variance.

        Args:
            descriptors: (B, N, D) L2-normalized descriptors

        Returns:
            loss: Scalar variance regularization loss
        """
        B, N, D = descriptors.shape

        # Reshape to (B*N, D)
        desc_flat = descriptors.reshape(B * N, D)

        # Variance per dimension
        variance_per_dim = desc_flat.var(dim=0)  # (D,)
        mean_variance = variance_per_dim.mean()

        # Only penalize if variance is TOO LOW (collapse)
        # Don't penalize high variance (that's good!)
        min_var = torch.tensor(self.min_variance, device=descriptors.device)

        # Loss: only if variance drops below minimum
        loss = F.relu(min_var - mean_variance)

        return loss


class DescriptorDecorrelationLoss(nn.Module):
    """
    OPTIONAL: Encourage descriptor dimensions to be decorrelated.
    Helps prevent redundant dimensions.

    Similar to Barlow Twins / VICReg approach.
    """

    def __init__(self):
        super().__init__()

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Encourage off-diagonal elements of correlation matrix to be zero.

        Args:
            descriptors: (B, N, D) descriptors

        Returns:
            loss: Decorrelation loss
        """
        B, N, D = descriptors.shape

        # Flatten to (B*N, D)
        desc_flat = descriptors.reshape(B * N, D)

        # Normalize each dimension to zero mean, unit variance
        desc_centered = desc_flat - desc_flat.mean(dim=0, keepdim=True)
        desc_std = desc_centered.std(dim=0, keepdim=True) + 1e-6
        desc_normalized = desc_centered / desc_std

        # Compute correlation matrix
        corr_matrix = (desc_normalized.T @ desc_normalized) / (B * N)

        # Loss: minimize off-diagonal elements
        # Want identity matrix (decorrelated dimensions)
        eye = torch.eye(D, device=descriptors.device)
        off_diagonal = (corr_matrix - eye) ** 2

        # Only penalize off-diagonal
        mask = 1 - eye
        loss = (off_diagonal * mask).sum() / (D * (D - 1))

        return loss


class RepeatabilityLoss(nn.Module):
    """Repeatability loss - compare saliency maps directly"""

    def __init__(self, distance_threshold: float = 2.0):
        super().__init__()
        self.distance_threshold = distance_threshold

    def forward(
        self,
        saliency1: torch.Tensor,
        saliency2: torch.Tensor
    ) -> torch.Tensor:
        """Force same saliency patterns across frames"""
        sal1_flat = saliency1.reshape(saliency1.shape[0], -1)
        sal2_flat = saliency2.reshape(saliency2.shape[0], -1)
        loss = F.mse_loss(sal1_flat, sal2_flat)
        return loss


class PeakinessLoss(nn.Module):
    """Encourage variance in saliency map"""

    def __init__(self, target_variance: float = 0.22):
        super().__init__()
        self.target_variance = target_variance

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = saliency_map.shape
        saliency_flat = saliency_map.squeeze(-1).reshape(B, -1)
        variance = saliency_flat.var(dim=1, unbiased=False)
        mean_variance = variance.mean()
        target = torch.tensor(self.target_variance, device=saliency_map.device)
        loss = (mean_variance - target) ** 2
        return loss


class ActivationLoss(nn.Module):
    """Ensure selector activates"""

    def __init__(self, target_mean: float = 0.35):
        super().__init__()
        self.target_mean = target_mean

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        mean_saliency = saliency_map.mean()
        target = torch.tensor(self.target_mean, device=saliency_map.device)
        loss = F.mse_loss(mean_saliency, target)
        return loss


class EdgeAwarenessLoss(nn.Module):
    """Encourage saliency to align with image edges"""

    def __init__(self, edge_threshold: float = 0.1):
        super().__init__()
        self.edge_threshold = edge_threshold

        # Sobel filters
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))

        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0))

    def forward(self, saliency_map: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Encourage saliency to correlate with edge strength.

        CORRECT APPROACH: Use negative correlation loss (maximize positive correlation)
        The key is using a LOWER WEIGHT (0.3) to avoid dominating other losses.
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        # Convert to grayscale
        gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        gray = gray.unsqueeze(1)

        # Edge detection with Sobel
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Normalize to [0, 1]
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)

        # Downsample to saliency resolution (28x28)
        edge_map_downsampled = F.adaptive_avg_pool2d(edge_magnitude, (H, W))

        # Prepare saliency for correlation
        saliency_2d = saliency_map.squeeze(-1).unsqueeze(1)  # (B, 1, H, W)

        # Flatten for correlation computation
        edge_flat = edge_map_downsampled.reshape(B, -1)
        saliency_flat = saliency_2d.reshape(B, -1)

        # Pearson correlation
        edge_centered = edge_flat - edge_flat.mean(dim=1, keepdim=True)
        sal_centered = saliency_flat - saliency_flat.mean(dim=1, keepdim=True)

        correlation = (edge_centered * sal_centered).sum(dim=1) / (
            torch.sqrt((edge_centered ** 2).sum(dim=1) * (sal_centered ** 2).sum(dim=1)) + 1e-8
        )

        # CORRECT: Negative correlation loss (maximize positive correlation)
        # Goal: Make correlation → +1.0
        # If correlation is negative (bad), loss is positive (high penalty)
        # If correlation is positive (good), loss is negative (reward)
        # Gradient descent minimizes loss → pushes toward positive correlation
        loss = -correlation.mean()

        return loss


class SpatialSparsityLoss(nn.Module):
    """Encourage spatial sparsity - prevent blobs"""

    def __init__(self, sparsity_target: float = 0.35, penalty_weight: float = 2.0):
        super().__init__()
        self.sparsity_target = sparsity_target
        self.penalty_weight = penalty_weight

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = saliency_map.shape
        saliency_2d = saliency_map.squeeze(-1)

        # Spatial gradients (high = sparse peaks)
        grad_x = saliency_2d[:, :, 1:] - saliency_2d[:, :, :-1]
        grad_y = saliency_2d[:, 1:, :] - saliency_2d[:, :-1, :]
        spatial_variation = (grad_x.abs().mean() + grad_y.abs().mean()) / 2

        # Want high spatial variation
        target_variation = torch.tensor(0.15, device=saliency_map.device)
        sparsity_loss = F.relu(target_variation - spatial_variation)

        # Penalize too many high-saliency locations
        high_saliency_ratio = (saliency_2d > 0.6).float().mean()
        ratio_penalty = F.relu(high_saliency_ratio - 0.20) * self.penalty_weight

        total_loss = sparsity_loss + ratio_penalty

        return total_loss