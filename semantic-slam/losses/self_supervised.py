"""
Self-Supervised Losses - FIXED with InfoNCE
Following DINO-VO: InfoNCE is better than AP loss for preserving semantic structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) Loss

    Used in DINO-VO instead of AP loss because:
    1. Better preserves semantic structure
    2. More stable gradients
    3. Naturally handles hard negatives
    4. Doesn't degrade edge features in joint training

    For each positive pair (matched keypoints), treat all other keypoints as negatives.
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
        Compute InfoNCE loss for descriptor matching.

        Args:
            desc1: (B, N, D) descriptors from frame 1
            desc2: (B, M, D) descriptors from frame 2
            matches: (B, K, 2) ground truth matches [idx1, idx2]

        Returns:
            loss: Scalar InfoNCE loss
        """
        B, N, D = desc1.shape
        M = desc2.shape[1]
        device = desc1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            # Extract match indices
            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            # Filter valid indices
            valid_mask = (idx1 >= 0) & (idx1 < N) & (idx2 >= 0) & (idx2 < M)
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            # Get matched descriptors
            query_desc = desc1[b, idx1]  # (K, D)
            positive_desc = desc2[b, idx2]  # (K, D)

            # Compute similarity matrix: query × all desc2
            # (K, D) × (M, D).T = (K, M)
            sim_matrix = torch.mm(query_desc, desc2[b].t()) / self.temperature

            # For each query, positive is at idx2[i]
            # Create labels: positive index for each query
            labels = idx2  # (K,)

            # Cross-entropy loss: log(exp(pos) / sum(exp(all)))
            loss = F.cross_entropy(sim_matrix, labels, reduction='mean')

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class DescriptorVarianceLoss(nn.Module):
    """
    Prevent descriptor collapse

    IMPROVED: Per-batch normalization for stability
    """

    def __init__(self, min_variance: float = 0.005):
        super().__init__()
        self.min_variance = min_variance

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            descriptors: (B, N, D)
        """
        B, N, D = descriptors.shape

        # Compute variance per dimension across batch
        desc_flat = descriptors.reshape(B * N, D)

        # Normalize first (remove mean)
        desc_centered = desc_flat - desc_flat.mean(dim=0, keepdim=True)

        # Variance per dimension
        variance_per_dim = (desc_centered ** 2).mean(dim=0)
        mean_variance = variance_per_dim.mean()

        # Penalize if too low
        min_var = torch.tensor(self.min_variance, device=descriptors.device)
        loss = F.relu(min_var - mean_variance)

        return loss


class RepeatabilityLoss(nn.Module):
    """
    Keypoint repeatability loss

    IMPROVED: Use correlation instead of MSE
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        saliency1: torch.Tensor,
        saliency2: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage consistent saliency maps across frames

        Use correlation (better than MSE for this task)
        """
        B = saliency1.shape[0]

        sal1_flat = saliency1.reshape(B, -1)
        sal2_flat = saliency2.reshape(B, -1)

        # Normalize (zero mean, unit variance)
        sal1_norm = (sal1_flat - sal1_flat.mean(dim=1, keepdim=True))
        sal2_norm = (sal2_flat - sal2_flat.mean(dim=1, keepdim=True))

        sal1_norm = sal1_norm / (sal1_norm.std(dim=1, keepdim=True) + 1e-8)
        sal2_norm = sal2_norm / (sal2_norm.std(dim=1, keepdim=True) + 1e-8)

        # Pearson correlation
        correlation = (sal1_norm * sal2_norm).mean(dim=1)

        # Loss: 1 - correlation (we want high positive correlation)
        loss = (1.0 - correlation).mean()

        return loss


class SemanticEdgeLoss(nn.Module):
    """
    Loss to encourage saliency at semantic edges.

    IMPROVED: Better correlation formula
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
            loss: Correlation loss in [0, 1] where 0 is best
        """
        B, H, W, _ = saliency_map.shape

        sal_flat = saliency_map.reshape(B, -1)
        edge_flat = semantic_edges.reshape(B, -1)

        # Normalize (zero mean, unit variance) per image
        sal_norm = sal_flat - sal_flat.mean(dim=1, keepdim=True)
        edge_norm = edge_flat - edge_flat.mean(dim=1, keepdim=True)

        sal_norm = sal_norm / (sal_norm.std(dim=1, keepdim=True) + 1e-8)
        edge_norm = edge_norm / (edge_norm.std(dim=1, keepdim=True) + 1e-8)

        # Pearson correlation coefficient
        correlation = (sal_norm * edge_norm).mean(dim=1)  # (B,)

        # Average across batch
        avg_correlation = correlation.mean()

        # Convert to loss: want high positive correlation
        # correlation in [-1, 1], map to loss in [0, 1]
        loss = (1.0 - avg_correlation) / 2.0

        return torch.clamp(loss, 0.0, 1.0)


class GeometricConsistencyLoss(nn.Module):
    """
    NEW: Regularize geometric features to maintain edge structure in Stage 2

    This prevents degradation when training with descriptor losses!
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        geo_features: torch.Tensor,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure geometric features remain edge-responsive

        Args:
            geo_features: (B, C, H, W) geometric CNN features
            image: (B, 3, H, W) RGB image

        Returns:
            loss: Correlation between geometric features and image edges
        """
        B, C, H_geo, W_geo = geo_features.shape

        # Compute image gradients
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        gray = gray.unsqueeze(1)  # (B, 1, H, W)

        # Downsample to match geometric feature resolution
        gray_down = F.interpolate(
            gray,
            size=(H_geo, W_geo),
            mode='bilinear',
            align_corners=False
        )

        # Image gradients
        grad_x = torch.abs(gray_down[:, :, :, 1:] - gray_down[:, :, :, :-1])
        grad_y = torch.abs(gray_down[:, :, 1:, :] - gray_down[:, :, :-1, :])

        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        image_edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Geometric feature magnitude
        geo_mag = torch.norm(geo_features, dim=1, keepdim=True)

        # Normalize both
        image_edges_norm = image_edges / (image_edges.max() + 1e-8)
        geo_mag_norm = geo_mag / (geo_mag.max() + 1e-8)

        # Flatten
        img_flat = image_edges_norm.reshape(B, -1)
        geo_flat = geo_mag_norm.reshape(B, -1)

        # Correlation
        img_mean = img_flat - img_flat.mean(dim=1, keepdim=True)
        geo_mean = geo_flat - geo_flat.mean(dim=1, keepdim=True)

        img_std = img_mean.std(dim=1, keepdim=True) + 1e-8
        geo_std = geo_mean.std(dim=1, keepdim=True) + 1e-8

        img_norm = img_mean / img_std
        geo_norm = geo_mean / geo_std

        correlation = (img_norm * geo_norm).mean(dim=1)

        # Loss: 1 - correlation
        loss = 1.0 - correlation.mean()

        return loss