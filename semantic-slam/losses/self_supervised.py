"""
Self-Supervised Losses for Semantic SLAM
COMPLETE VERSION - All losses including selector training
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
    """Keypoint repeatability loss with proper scaling"""

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

        smooth_l1 = torch.where(
            distances < self.distance_threshold,
            0.5 * distances ** 2 / self.distance_threshold,
            distances - 0.5 * self.distance_threshold
        )

        loss = smooth_l1.mean()
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
    InfoNCE contrastive loss for descriptors with hard negative mining.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        num_negatives: int = 50,
        margin: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.margin = margin

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

            valid_mask = (idx1 < desc1.shape[1]) & (idx2 < desc2.shape[1])
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            if len(idx1) == 0:
                continue

            matched_desc1 = desc1[b, idx1]
            matched_desc2 = desc2[b, idx2]

            # InfoNCE Loss
            logits = torch.mm(matched_desc1, desc2[b].t()) / self.temperature
            target_positions = idx2
            infonce_loss = F.cross_entropy(logits, target_positions)

            # Hard negative mining
            sim_matrix = torch.mm(matched_desc1, desc2[b].t())
            neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
            neg_mask[torch.arange(len(idx1)), idx2] = False

            if neg_mask.sum() > 0:
                num_negs = min(self.num_negatives, neg_mask.sum() // len(idx1))

                if num_negs > 0:
                    neg_sims = sim_matrix.clone()
                    neg_sims[~neg_mask] = -float('inf')

                    hard_neg_sims, _ = torch.topk(neg_sims, k=num_negs, dim=1)
                    pos_sims = (matched_desc1 * matched_desc2).sum(dim=-1)

                    triplet_losses = torch.relu(
                        hard_neg_sims - pos_sims.unsqueeze(1) + self.margin
                    )

                    triplet_loss = triplet_losses.mean()
                else:
                    triplet_loss = torch.tensor(0.0, device=device)
            else:
                triplet_loss = torch.tensor(0.0, device=device)

            total_loss += infonce_loss + 0.5 * triplet_loss
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
    """Diversity loss to prevent descriptor collapse"""

    def __init__(self, target_similarity: float = 0.1):
        super().__init__()
        self.target_similarity = target_similarity

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        B, N, D = descriptors.shape
        device = descriptors.device

        total_loss = 0.0

        for b in range(B):
            desc_b = descriptors[b]
            sim_matrix = torch.mm(desc_b, desc_b.t())

            mask = ~torch.eye(N, device=device, dtype=torch.bool)
            off_diag_sims = sim_matrix[mask]

            squared_loss = ((off_diag_sims - self.target_similarity) ** 2).mean()

            high_sim_mask = off_diag_sims > 0.3
            if high_sim_mask.sum() > 0:
                repulsion_loss = off_diag_sims[high_sim_mask].mean()
            else:
                repulsion_loss = 0.0

            total_loss += squared_loss + 0.5 * repulsion_loss

        return total_loss / B


class PeakinessLoss(nn.Module):
    """
    NEW: Encourages saliency map to be PEAKED, not messy.

    FIXED: Adjusted for sigmoid (allows multiple peaks).
    Now just encourages clear peaks without forcing extreme sparsity.
    """

    def __init__(self, target_sparsity: float = 0.2):
        """
        Args:
            target_sparsity: Target fraction of active locations (20% instead of 10%)
        """
        super().__init__()
        self.target_sparsity = target_sparsity

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency_flat = saliency_map.squeeze(-1).reshape(B, H * W)

        # COMPONENT 1: Contrast Loss
        # Encourage high values to be HIGH and low values to be LOW
        # This creates clear peaks without forcing single-point behavior
        mean_val = saliency_flat.mean(dim=1, keepdim=True)
        above_mean = saliency_flat > mean_val
        below_mean = ~above_mean

        if above_mean.sum() > 0:
            high_vals = saliency_flat[above_mean]
            # Want high values close to 1.0
            high_loss = (1.0 - high_vals).mean()
        else:
            high_loss = torch.tensor(0.0, device=device)

        if below_mean.sum() > 0:
            low_vals = saliency_flat[below_mean]
            # Want low values close to 0.0
            low_loss = low_vals.mean()
        else:
            low_loss = torch.tensor(0.0, device=device)

        contrast_loss = high_loss + low_loss

        # COMPONENT 2: Soft Sparsity
        # Gentle encouragement toward target sparsity (not strict)
        l1_norm = saliency_flat.mean(dim=1)
        target_l1 = self.target_sparsity
        sparsity_loss = 0.5 * F.mse_loss(l1_norm, torch.full_like(l1_norm, target_l1))

        total_loss = contrast_loss + sparsity_loss

        return total_loss


class FeatureVarianceLoss(nn.Module):
    """
    NEW: Selector should choose regions with HIGH DINOv3 variance.

    High-variance regions (edges, corners, textures) are good for tracking.
    """

    def __init__(self, neighborhood_size: int = 3):
        super().__init__()
        self.neighborhood_size = neighborhood_size

    def forward(
        self,
        saliency_map: torch.Tensor,
        dino_features: torch.Tensor
    ) -> torch.Tensor:
        B, H, W, C = dino_features.shape
        device = dino_features.device

        # Compute local feature variance
        features = dino_features.permute(0, 3, 1, 2)

        kernel_size = self.neighborhood_size
        avg_pool = F.avg_pool2d(features, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        squared_diff = (features - avg_pool) ** 2
        local_variance = F.avg_pool2d(squared_diff, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        variance_map = local_variance.mean(dim=1, keepdim=True)

        # Normalize to [0, 1]
        variance_map = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min() + 1e-8)

        variance_map = variance_map.permute(0, 2, 3, 1)

        # Saliency should match variance
        loss = F.mse_loss(saliency_map, variance_map)

        return loss