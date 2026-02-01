"""
Geometric Constraint Losses
Epipolar consistency and depth reprojection for 3D-aware training.

FIXES:
- Replaced F.huber_loss (doesn't exist) with F.smooth_l1_loss (with beta parameter)
- Added better error handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EpipolarConsistencyLoss(nn.Module):
    """
    Epipolar consistency loss.

    Enforces that matched keypoints satisfy the epipolar constraint:
    kpts2^T @ F @ kpts1 ≈ 0

    This is CRITICAL for SLAM - ensures geometric validity of matches.
    """

    def __init__(self, threshold: float = 3.0):
        super().__init__()
        self.threshold = threshold  # Used as beta in smooth_l1_loss

    def forward(
        self,
        kpts1: torch.Tensor,
        kpts2: torch.Tensor,
        matches: torch.Tensor,
        K: torch.Tensor,
        relative_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute epipolar consistency loss.

        Args:
            kpts1: (B, N, 2) keypoints in frame 1 (pixel coords)
            kpts2: (B, M, 2) keypoints in frame 2 (pixel coords)
            matches: (B, K, 2) matched indices [idx1, idx2]
            K: (B, 3, 3) camera intrinsics
            relative_pose: (B, 4, 4) T_2_from_1

        Returns:
            loss: Scalar epipolar consistency loss
        """
        B = kpts1.shape[0]
        device = kpts1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            # Extract matched keypoints
            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            # Filter valid indices
            valid_mask = (idx1 < kpts1.shape[1]) & (idx2 < kpts2.shape[1]) & (idx1 >= 0) & (idx2 >= 0)
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            matched_kpts1 = kpts1[b, idx1]  # (K, 2)
            matched_kpts2 = kpts2[b, idx2]  # (K, 2)

            # Compute fundamental matrix from relative pose
            # F = K^-T @ [t]_x @ R @ K^-1
            R = relative_pose[b, :3, :3]
            t = relative_pose[b, :3, 3]

            # Skew-symmetric matrix [t]_x
            t_x = torch.zeros(3, 3, device=device)
            t_x[0, 1] = -t[2]
            t_x[0, 2] = t[1]
            t_x[1, 0] = t[2]
            t_x[1, 2] = -t[0]
            t_x[2, 0] = -t[1]
            t_x[2, 1] = t[0]

            K_inv = torch.inverse(K[b])
            F_matrix = K_inv.t() @ t_x @ R @ K_inv

            # Convert to homogeneous coordinates
            kpts1_h = torch.cat([matched_kpts1, torch.ones(len(matched_kpts1), 1, device=device)], dim=1)
            kpts2_h = torch.cat([matched_kpts2, torch.ones(len(matched_kpts2), 1, device=device)], dim=1)

            # Compute epipolar lines: l = F_matrix @ kpts1
            epipolar_lines = (F_matrix @ kpts1_h.t()).t()  # (K, 3)

            # Distance from kpts2 to epipolar line
            # d = |kpts2^T @ l| / sqrt(l_x^2 + l_y^2)
            numerator = torch.abs((kpts2_h * epipolar_lines).sum(dim=1))
            denominator = torch.sqrt(epipolar_lines[:, 0]**2 + epipolar_lines[:, 1]**2 + 1e-8)

            distances = numerator / denominator

            # FIXED: Use smooth_l1_loss (Huber) with beta parameter
            # beta acts as the threshold for quadratic vs linear loss
            loss = F.smooth_l1_loss(
                distances,
                torch.zeros_like(distances),
                reduction='mean',
                beta=self.threshold
            )

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class DepthReprojectionLoss(nn.Module):
    """
    Depth reprojection loss using RGB-D ground truth.

    Enforces that keypoints triangulate to correct 3D positions.
    Uses ground truth depth from TUM RGB-D dataset.

    FIXES:
    - Reduced clipping from 50 to 10 pixels (more reasonable for indoor scenes)
    - Added better validation of camera parameters
    - Improved error handling
    """

    def __init__(self, max_depth=10.0, min_depth=0.1, max_reproj_error=10.0):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.max_reproj_error = max_reproj_error  # Reduced from 50

    def forward(
        self,
        kpts1: torch.Tensor,
        kpts2: torch.Tensor,
        matches: torch.Tensor,
        depth1: torch.Tensor,
        K: torch.Tensor,
        relative_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute depth reprojection loss.

        Args:
            kpts1: (B, N, 2) keypoints in frame 1 (pixel coords)
            kpts2: (B, M, 2) keypoints in frame 2 (pixel coords)
            matches: (B, K, 2) matched indices
            depth1: (B, 1, H, W) depth map for frame 1 (meters)
            K: (B, 3, 3) camera intrinsics
            relative_pose: (B, 4, 4) T_2_from_1

        Returns:
            loss: Scalar reprojection loss
        """
        B, _, H, W = depth1.shape
        device = kpts1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            # Extract matched keypoints
            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            valid_mask = (idx1 < kpts1.shape[1]) & (idx2 < kpts2.shape[1]) & (idx1 >= 0) & (idx2 >= 0)
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            matched_kpts1 = kpts1[b, idx1]  # (K, 2)
            matched_kpts2 = kpts2[b, idx2]  # (K, 2)

            # CRITICAL: Validate keypoint coordinates are within image bounds
            valid_coords = (
                (matched_kpts1[:, 0] >= 0) & (matched_kpts1[:, 0] < W) &
                (matched_kpts1[:, 1] >= 0) & (matched_kpts1[:, 1] < H)
            )

            if valid_coords.sum() == 0:
                continue

            matched_kpts1 = matched_kpts1[valid_coords]
            matched_kpts2 = matched_kpts2[valid_coords]

            # Sample depth at keypoint locations
            # Normalize coordinates to [-1, 1] for grid_sample
            norm_coords = matched_kpts1.clone()
            norm_coords[:, 0] = 2.0 * matched_kpts1[:, 0] / (W - 1) - 1.0
            norm_coords[:, 1] = 2.0 * matched_kpts1[:, 1] / (H - 1) - 1.0

            # Clamp to valid range to prevent sampling errors
            norm_coords = torch.clamp(norm_coords, -1.0, 1.0)

            grid = norm_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, K, 2)
            depth_at_kpts = F.grid_sample(
                depth1[b:b+1],
                grid,
                mode='bilinear',
                align_corners=True
            ).squeeze()  # (K,)

            # Filter out invalid depth
            valid_depth = (depth_at_kpts > self.min_depth) & (depth_at_kpts < self.max_depth)
            if valid_depth.sum() == 0:
                continue

            matched_kpts1 = matched_kpts1[valid_depth]
            matched_kpts2 = matched_kpts2[valid_depth]
            depth_at_kpts = depth_at_kpts[valid_depth]

            # VALIDATE camera intrinsics (detect if still using wrong camera)
            fx = K[b, 0, 0].item()
            fy = K[b, 1, 1].item()

            # Sanity check: focal lengths should be reasonable for TUM (500-550 range)
            if fx < 400 or fx > 600 or fy < 400 or fy > 600:
                print(f"⚠️  WARNING: Unusual camera intrinsics: fx={fx:.1f}, fy={fy:.1f}")

            # Back-project to 3D
            K_inv = torch.inverse(K[b])
            kpts1_h = torch.cat([matched_kpts1, torch.ones(len(matched_kpts1), 1, device=device)], dim=1)

            # 3D points in frame 1
            points_3d = depth_at_kpts.unsqueeze(1) * (K_inv @ kpts1_h.t()).t()  # (K, 3)

            # Transform to frame 2
            T = relative_pose[b]
            R = T[:3, :3]
            t = T[:3, 3]
            points_3d_frame2 = (R @ points_3d.t()).t() + t  # (K, 3)

            # Check if points are in front of camera
            valid_proj = points_3d_frame2[:, 2] > 0.1
            if valid_proj.sum() == 0:
                continue

            matched_kpts2 = matched_kpts2[valid_proj]
            points_3d_frame2 = points_3d_frame2[valid_proj]

            # Project to frame 2
            kpts2_proj = (K[b] @ points_3d_frame2.t()).t()  # (K, 3)
            kpts2_proj = kpts2_proj[:, :2] / (kpts2_proj[:, 2:3] + 1e-8)

            # Reprojection error
            error = torch.norm(kpts2_proj - matched_kpts2, dim=1)

            # FIXED: Reduced clipping from 50 to 10 pixels
            error = torch.clamp(error, max=self.max_reproj_error)
            loss = error.mean()

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

class PhotometricConsistencyLoss(nn.Module):
    """
    Photometric consistency loss.
    Warped image should match original image.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        depth1: torch.Tensor,
        K: torch.Tensor,
        relative_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute photometric consistency.

        Args:
            img1: (B, 3, H, W) frame 1
            img2: (B, 3, H, W) frame 2
            depth1: (B, 1, H, W) depth for frame 1
            K: (B, 3, 3) intrinsics
            relative_pose: (B, 4, 4) T_2_from_1

        Returns:
            loss: Photometric loss
        """
        B, _, H, W = img1.shape
        device = img1.device

        # Create pixel grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        pixel_coords = torch.stack([x_grid, y_grid], dim=-1).float()  # (H, W, 2)

        total_loss = 0.0

        for b in range(B):
            # Back-project to 3D
            depth_b = depth1[b, 0]

            # Valid depth mask
            valid_mask = (depth_b > 0.1) & (depth_b < 10.0)

            if valid_mask.sum() < 100:  # Need enough valid pixels
                continue

            # Homogeneous coordinates
            K_inv = torch.inverse(K[b])
            pixels_h = torch.cat([pixel_coords, torch.ones(H, W, 1, device=device)], dim=-1)

            # 3D points
            depth_flat = depth_b.reshape(-1, 1)  # (H*W, 1)
            points_3d = depth_flat * (K_inv @ pixels_h.reshape(-1, 3).t()).t()
            points_3d = points_3d.reshape(H, W, 3)

            # Transform to frame 2
            T = relative_pose[b]
            R = T[:3, :3]
            t = T[:3, 3]

            points_3d_flat = points_3d.reshape(-1, 3)
            points_3d_frame2 = (R @ points_3d_flat.t()).t() + t
            points_3d_frame2 = points_3d_frame2.reshape(H, W, 3)

            # Project to frame 2
            pixels_frame2 = (K[b] @ points_3d_frame2.reshape(-1, 3).t()).t()
            pixels_frame2 = pixels_frame2.reshape(H, W, 3)
            pixels_frame2 = pixels_frame2[..., :2] / (pixels_frame2[..., 2:3] + 1e-8)

            # Normalize for grid_sample
            norm_coords = pixels_frame2.clone()
            norm_coords[..., 0] = 2.0 * pixels_frame2[..., 0] / (W - 1) - 1.0
            norm_coords[..., 1] = 2.0 * pixels_frame2[..., 1] / (H - 1) - 1.0

            # Sample from img2
            warped_img = F.grid_sample(
                img2[b:b+1],
                norm_coords.unsqueeze(0),
                mode='bilinear',
                align_corners=True,
                padding_mode='zeros'
            ).squeeze(0)

            # Photometric error
            error = torch.abs(warped_img - img1[b])

            # Mask invalid regions
            error = error * valid_mask.unsqueeze(0)

            loss = error.sum() / (valid_mask.sum() * 3 + 1e-8)

            if not torch.isnan(loss):
                total_loss += loss

        return total_loss / B