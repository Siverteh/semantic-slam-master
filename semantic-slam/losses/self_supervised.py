"""
Self-Supervised Losses for Semantic SLAM
FIXED: Better descriptor loss with more negatives, removed variance regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PhotometricLoss(nn.Module):
    """Photometric consistency loss using depth warping"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        rgb1: torch.Tensor,
        rgb2: torch.Tensor,
        depth1: torch.Tensor,
        relative_pose: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, _, H, W = rgb1.shape
        device = rgb1.device

        if intrinsics is None:
            fx = fy = 525.0 * (H / 480.0)
            cx = cy = H / 2.0
            intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=device).unsqueeze(0).repeat(B, 1, 1)

        rgb1_warped = self._warp_image(rgb1, depth1, relative_pose, intrinsics)
        error = torch.abs(rgb1_warped - rgb2) / 2.0

        valid_mask = (depth1 > 0) & (depth1 < 10.0)
        valid_mask = valid_mask.expand_as(error)

        if self.reduction == 'mean':
            loss = error[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0, device=device)
        else:
            loss = error[valid_mask].sum() if valid_mask.sum() > 0 else torch.tensor(0.0, device=device)

        return loss

    def _warp_image(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        B, _, H, W = image.shape
        device = image.device

        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        x = x.reshape(1, -1).expand(B, -1)
        y = y.reshape(1, -1).expand(B, -1)
        z = depth.reshape(B, -1)

        K_inv = torch.inverse(intrinsics)
        pixels = torch.stack([x, y, torch.ones_like(x)], dim=2)
        rays = torch.bmm(pixels, K_inv.transpose(1, 2))
        points_3d = rays * z.unsqueeze(2)

        points_3d_hom = torch.cat([points_3d, torch.ones(B, H*W, 1, device=device)], dim=2)
        points_3d_target = torch.bmm(points_3d_hom, pose.transpose(1, 2))[:, :, :3]

        pixels_target = torch.bmm(points_3d_target, intrinsics.transpose(1, 2))
        pixels_target = pixels_target[:, :, :2] / (pixels_target[:, :, 2:3] + 1e-7)

        pixels_target[:, :, 0] = 2.0 * pixels_target[:, :, 0] / (W - 1) - 1.0
        pixels_target[:, :, 1] = 2.0 * pixels_target[:, :, 1] / (H - 1) - 1.0

        grid = pixels_target.reshape(B, H, W, 2)
        warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped


class RepeatabilityLoss(nn.Module):
    """
    Keypoint repeatability loss with proper scaling.

    FIXED: Better distance threshold and scaling
    """

    def __init__(self, distance_threshold: float = 3.0):
        super().__init__()
        self.distance_threshold = distance_threshold

    def forward(
        self,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor,
        depth1: torch.Tensor,
        relative_pose: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, _ = keypoints1.shape
        device = keypoints1.device
        H, W = depth1.shape[2:]

        if intrinsics is None:
            fx = fy = 525.0 * (H / 480.0)
            cx = cy = H / 2.0
            intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=device).unsqueeze(0).repeat(B, 1, 1)

        projected_kpts = self._project_keypoints(
            keypoints1, depth1, relative_pose, intrinsics, H, W
        )

        distances = torch.norm(projected_kpts - keypoints2, dim=2)

        # Smooth L1 loss (Huber loss)
        smooth_l1 = torch.where(
            distances < self.distance_threshold,
            0.5 * distances ** 2 / self.distance_threshold,
            distances - 0.5 * self.distance_threshold
        )

        loss = smooth_l1.mean()

        # Scale by grid size for stability
        grid_size = (H / 16)
        loss = loss / grid_size

        return loss

    def _project_keypoints(
        self,
        keypoints: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        intrinsics: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        B, N, _ = keypoints.shape
        device = keypoints.device

        patch_size = 16
        kpts_pixel = keypoints * patch_size + patch_size / 2

        kpts_norm = kpts_pixel.clone()
        kpts_norm[:, :, 0] = 2.0 * kpts_pixel[:, :, 0] / (W - 1) - 1.0
        kpts_norm[:, :, 1] = 2.0 * kpts_pixel[:, :, 1] / (H - 1) - 1.0

        grid = kpts_norm.unsqueeze(1)
        depth_at_kpts = F.grid_sample(
            depth, grid, mode='nearest', align_corners=True
        ).squeeze(1).squeeze(1)

        K_inv = torch.inverse(intrinsics)

        x = kpts_pixel[:, :, 0]
        y = kpts_pixel[:, :, 1]
        pixels_hom = torch.stack([x, y, torch.ones_like(x)], dim=2)

        rays = torch.bmm(pixels_hom, K_inv.transpose(1, 2))
        points_3d = rays * depth_at_kpts.unsqueeze(2)

        points_3d_hom = torch.cat([points_3d, torch.ones(B, N, 1, device=device)], dim=2)
        points_3d_target = torch.bmm(points_3d_hom, pose.transpose(1, 2))[:, :, :3]

        pixels_target = torch.bmm(points_3d_target, intrinsics.transpose(1, 2))
        pixels_target = pixels_target[:, :, :2] / (pixels_target[:, :, 2:3] + 1e-7)

        pixels_target = (pixels_target - patch_size / 2) / patch_size

        return pixels_target


class DescriptorConsistencyLoss(nn.Module):
    """
    Contrastive loss for descriptors with proper numerical stability.

    Uses simple cosine distance loss + triplet-style margin loss.
    Much more stable than InfoNCE for this application.
    """

    def __init__(self, margin: float = 0.2, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute descriptor loss using cosine distance + triplet margin.

        This is MUCH more stable than InfoNCE and prevents negative loss.
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

            valid_mask = (idx1 < desc1.shape[1]) & (idx2 < desc2.shape[1])
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            if len(idx1) == 0:
                continue

            matched_desc1 = desc1[b, idx1]  # (M, D)
            matched_desc2 = desc2[b, idx2]  # (M, D)

            # POSITIVE loss: Minimize distance between matched pairs
            # Use cosine distance (1 - cosine_similarity)
            pos_sim = (matched_desc1 * matched_desc2).sum(dim=-1)  # Cosine similarity
            pos_loss = (1 - pos_sim).mean()  # Cosine distance

            # NEGATIVE loss: Maximize distance to non-matches (triplet-style)
            # Compute similarity to ALL descriptors
            sim_matrix = torch.mm(matched_desc1, desc2[b].t())  # (M, N)

            # Mask out the positive matches
            mask = torch.ones_like(sim_matrix, dtype=torch.bool)
            mask[torch.arange(len(idx1)), idx2] = False

            if mask.sum() > 0:
                # Get similarities to negatives
                neg_sims = sim_matrix[mask].reshape(len(idx1), -1)  # (M, N-1)

                # Find hardest negative (highest similarity)
                hard_neg_sim, _ = neg_sims.max(dim=1)  # (M,)

                # Triplet margin loss:
                # Want: pos_sim > hard_neg_sim + margin
                # Loss: max(0, hard_neg_sim - pos_sim + margin)
                triplet_loss = torch.relu(hard_neg_sim - pos_sim + self.margin).mean()
            else:
                triplet_loss = torch.tensor(0.0, device=device)

            # Combine both losses
            total_loss += pos_loss + triplet_loss
            num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device)


class UncertaintyCalibrationLoss(nn.Module):
    """Uncertainty calibration loss"""

    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predicted_confidence: torch.Tensor,
        actual_error: torch.Tensor
    ) -> torch.Tensor:
        # Normalize error to [0, 1]
        error_flat = actual_error.flatten()
        error_95th = torch.quantile(error_flat, 0.95)
        error_norm = torch.clamp(actual_error / (error_95th + 1e-6), 0, 1)

        target = 1 - error_norm.unsqueeze(-1)

        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_confidence, target)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_confidence, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class DescriptorDiversityLoss(nn.Module):
    """
    Explicit diversity loss to prevent descriptor collapse.

    Encourages descriptors to span the full descriptor space.
    """

    def __init__(self):
        super().__init__()

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss to prevent all descriptors becoming identical.

        Args:
            descriptors: (B, N, D) L2-normalized descriptors

        Returns:
            loss: Scalar diversity loss (lower = more diverse)
        """
        B, N, D = descriptors.shape

        # Flatten batch dimension
        desc_flat = descriptors.reshape(B * N, D)  # (B*N, D)

        # Compute pairwise cosine similarities
        # For L2-normalized vectors, cosine sim = dot product
        sim_matrix = torch.mm(desc_flat, desc_flat.t())  # (B*N, B*N)

        # We want LOW similarity between different descriptors
        # Remove diagonal (self-similarity = 1)
        mask = ~torch.eye(B * N, device=descriptors.device, dtype=torch.bool)
        off_diag_sims = sim_matrix[mask]

        # Loss: penalize HIGH similarities (we want them low)
        # Mean of all pairwise similarities should be close to 0
        diversity_loss = off_diag_sims.mean().abs()

        return diversity_loss