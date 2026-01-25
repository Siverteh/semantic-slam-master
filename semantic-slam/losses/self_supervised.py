"""
Self-Supervised Losses for Semantic SLAM
FIXED: Added edge awareness and spatial sparsity to learn corners/edges, not blobs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorMatchingLoss(nn.Module):
    """InfoNCE contrastive loss - unchanged"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
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

            logits = torch.mm(matched_desc1, desc2[b].t()) / self.temperature
            logits = torch.clamp(logits, -50, 50)

            loss = F.cross_entropy(logits, idx2)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.1, device=device, requires_grad=True)


class RepeatabilityLoss(nn.Module):
    """
    FIXED: Proper repeatability loss that encourages selecting THE SAME keypoints.

    Key insight: We don't want keypoints1 ≈ keypoints2 (all close together),
    we want CORRESPONDING keypoints to be in the same locations!
    """

    def __init__(self, distance_threshold: float = 2.0):
        super().__init__()
        self.distance_threshold = distance_threshold

    def forward(
        self,
        saliency1: torch.Tensor,
        saliency2: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage consistent saliency patterns across frames.

        FIXED: Instead of comparing keypoint coordinates (which might be different),
        we compare the SALIENCY MAPS themselves. This forces the network to
        produce similar saliency patterns → same keypoints selected.

        Args:
            saliency1: (B, H, W, 1) saliency map for frame 1
            saliency2: (B, H, W, 1) saliency map for frame 2

        Returns:
            loss: Scalar repeatability loss
        """
        # Flatten saliency maps
        sal1_flat = saliency1.reshape(saliency1.shape[0], -1)  # (B, H*W)
        sal2_flat = saliency2.reshape(saliency2.shape[0], -1)  # (B, H*W)

        # L2 loss between saliency maps
        # This encourages the same patches to be salient in both frames
        loss = F.mse_loss(sal1_flat, sal2_flat)

        # Alternative: Cosine similarity (encourages same pattern)
        # cos_sim = F.cosine_similarity(sal1_flat, sal2_flat, dim=1).mean()
        # loss = 1.0 - cos_sim

        return loss


class PeakinessLoss(nn.Module):
    """
    FIXED: Encourage variance in saliency map.
    Higher variance = more peaked = clear distinction between features.
    """

    def __init__(self, target_variance: float = 0.22):
        super().__init__()
        self.target_variance = target_variance

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        Encourage variance around target (e.g., 0.22 for 0.1-0.85 range).

        Args:
            saliency_map: (B, H, W, 1) sigmoid outputs

        Returns:
            loss: Scalar peakiness loss
        """
        B, H, W, _ = saliency_map.shape
        saliency_flat = saliency_map.squeeze(-1).reshape(B, -1)

        # Compute variance for each sample
        variance = saliency_flat.var(dim=1, unbiased=False)
        mean_variance = variance.mean()

        # Target variance (higher = more dynamic range)
        target = torch.tensor(self.target_variance, device=saliency_map.device)

        # MSE loss to target variance
        loss = (mean_variance - target) ** 2

        return loss


class ActivationLoss(nn.Module):
    """
    Ensures the selector activates (doesn't stay near 0).
    REDUCED weight to allow more dynamic range.
    """

    def __init__(self, target_mean: float = 0.35):
        super().__init__()
        self.target_mean = target_mean

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        Ensure mean saliency is around target.

        Args:
            saliency_map: (B, H, W, 1) sigmoid outputs

        Returns:
            loss: Scalar activation loss
        """
        mean_saliency = saliency_map.mean()
        target = torch.tensor(self.target_mean, device=saliency_map.device)
        loss = F.mse_loss(mean_saliency, target)
        return loss


class EdgeAwarenessLoss(nn.Module):
    """
    NEW: Encourage saliency to align with image edges and corners.

    This is the KEY to learning feature detectors, not semantic segmentation!
    We compute image gradients and encourage high saliency at edge locations.
    """

    def __init__(self, edge_threshold: float = 0.1):
        super().__init__()
        self.edge_threshold = edge_threshold

        # Sobel filters for edge detection
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

        Args:
            saliency_map: (B, H, W, 1) saliency predictions (28x28)
            images: (B, 3, 448, 448) original RGB images

        Returns:
            loss: Edge alignment loss (lower = better alignment)
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        # Convert to grayscale for edge detection
        gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        gray = gray.unsqueeze(1)  # (B, 1, 448, 448)

        # Compute gradients
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        # Edge magnitude
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Normalize edge magnitude to [0, 1]
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)

        # Downsample edge map to match saliency resolution (28x28)
        edge_map_downsampled = F.adaptive_avg_pool2d(edge_magnitude, (H, W))

        # Threshold: only care about strong edges
        edge_map_binary = (edge_map_downsampled > self.edge_threshold).float()

        # Convert saliency to match format
        saliency_2d = saliency_map.squeeze(-1).unsqueeze(1)  # (B, 1, H, W)

        # Loss: Encourage high saliency at edge locations
        # We want: saliency[edge_locations] to be high

        # Option 1: Correlation-based (encourage positive correlation)
        edge_flat = edge_map_downsampled.reshape(B, -1)
        saliency_flat = saliency_2d.reshape(B, -1)

        # Normalize both to zero mean for correlation
        edge_centered = edge_flat - edge_flat.mean(dim=1, keepdim=True)
        sal_centered = saliency_flat - saliency_flat.mean(dim=1, keepdim=True)

        # Pearson correlation (we want positive correlation)
        correlation = (edge_centered * sal_centered).sum(dim=1) / (
            torch.sqrt((edge_centered ** 2).sum(dim=1) * (sal_centered ** 2).sum(dim=1)) + 1e-8
        )

        # Loss: negative correlation (maximize positive correlation)
        loss = -correlation.mean()

        # Option 2: Direct supervision on edge locations
        # Encourage saliency to be high where edges are strong
        edge_supervision_loss = F.mse_loss(
            saliency_2d * edge_map_binary,
            edge_map_downsampled * edge_map_binary
        )

        # Combine both approaches
        total_loss = loss + 0.5 * edge_supervision_loss

        return total_loss


class SpatialSparsityLoss(nn.Module):
    """
    NEW: Encourage spatial sparsity to prevent uniform blobs.

    Penalizes large contiguous regions of high saliency.
    Forces the network to select specific locations (corners/edges), not entire objects.
    """

    def __init__(self, sparsity_target: float = 0.35, penalty_weight: float = 2.0):
        super().__init__()
        self.sparsity_target = sparsity_target
        self.penalty_weight = penalty_weight

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        Penalize lack of spatial sparsity.

        Args:
            saliency_map: (B, H, W, 1) saliency predictions

        Returns:
            loss: Sparsity penalty (higher = less sparse)
        """
        B, H, W, _ = saliency_map.shape

        saliency_2d = saliency_map.squeeze(-1)  # (B, H, W)

        # Compute spatial gradients of saliency
        # High gradients = sparse, localized peaks
        # Low gradients = smooth, blob-like regions
        grad_x = saliency_2d[:, :, 1:] - saliency_2d[:, :, :-1]
        grad_y = saliency_2d[:, 1:, :] - saliency_2d[:, :-1, :]

        # Mean absolute gradient (higher = more sparse/peaky)
        spatial_variation = (grad_x.abs().mean() + grad_y.abs().mean()) / 2

        # We want HIGH spatial variation (peaks, not blobs)
        # So we penalize LOW spatial variation
        target_variation = torch.tensor(0.15, device=saliency_map.device)

        # Loss: encourage spatial variation to be high
        sparsity_loss = F.relu(target_variation - spatial_variation)

        # Also penalize if too many locations are highly salient
        # Count percentage of locations with saliency > threshold
        high_saliency_ratio = (saliency_2d > 0.6).float().mean()

        # Penalize if more than 20% of locations are very salient
        ratio_penalty = F.relu(high_saliency_ratio - 0.20) * self.penalty_weight

        total_loss = sparsity_loss + ratio_penalty

        return total_loss


class CornerResponseLoss(nn.Module):
    """
    OPTIONAL: Explicitly encourage corner-like responses.

    Uses Harris corner detection as supervision signal.
    This is more advanced - only use if edge loss isn't enough.
    """

    def __init__(self):
        super().__init__()

    def forward(self, saliency_map: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Encourage saliency to correlate with corner strength.

        Args:
            saliency_map: (B, H, W, 1) saliency predictions
            images: (B, 3, 448, 448) RGB images

        Returns:
            loss: Corner alignment loss
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        # Convert to grayscale
        gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        gray = gray.unsqueeze(1)  # (B, 1, 448, 448)

        # Compute gradients (Sobel)
        sobel_x = torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        Ix = F.conv2d(gray, sobel_x, padding=1)
        Iy = F.conv2d(gray, sobel_y, padding=1)

        # Harris corner response (simplified)
        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy

        # Gaussian smoothing (approximate with average pooling)
        Ix2 = F.avg_pool2d(Ix2, kernel_size=5, stride=1, padding=2)
        Iy2 = F.avg_pool2d(Iy2, kernel_size=5, stride=1, padding=2)
        Ixy = F.avg_pool2d(Ixy, kernel_size=5, stride=1, padding=2)

        # Harris response: det(M) - k * trace(M)^2
        k = 0.04
        det = Ix2 * Iy2 - Ixy ** 2
        trace = Ix2 + Iy2
        corner_response = det - k * (trace ** 2)

        # Normalize
        corner_response = F.relu(corner_response)
        corner_response = corner_response / (corner_response.max() + 1e-8)

        # Downsample to saliency resolution
        corner_map = F.adaptive_avg_pool2d(corner_response, (H, W))

        # Correlation loss (like EdgeAwarenessLoss)
        saliency_2d = saliency_map.squeeze(-1).unsqueeze(1)
        corner_flat = corner_map.reshape(B, -1)
        saliency_flat = saliency_2d.reshape(B, -1)

        # Pearson correlation
        corner_centered = corner_flat - corner_flat.mean(dim=1, keepdim=True)
        sal_centered = saliency_flat - saliency_flat.mean(dim=1, keepdim=True)

        correlation = (corner_centered * sal_centered).sum(dim=1) / (
            torch.sqrt((corner_centered ** 2).sum(dim=1) * (sal_centered ** 2).sum(dim=1)) + 1e-8
        )

        loss = -correlation.mean()

        return loss