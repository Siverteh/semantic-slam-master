"""
Self-Supervised Losses for Semantic SLAM
Photometric, stability, descriptor, and uncertainty losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PhotometricLoss(nn.Module):
    """
    Photometric consistency loss using depth warping.
    Ensures corresponding pixels have similar appearance.
    """
    
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
        """
        Args:
            rgb1: (B, 3, H, W) RGB image at time t
            rgb2: (B, 3, H, W) RGB image at time t+1
            depth1: (B, 1, H, W) depth at time t
            relative_pose: (B, 4, 4) relative camera pose T_t+1^t
            intrinsics: (B, 3, 3) camera intrinsics (optional, defaults to TUM)
            
        Returns:
            loss: Scalar photometric loss
        """
        B, _, H, W = rgb1.shape
        device = rgb1.device
        
        # Default TUM RGB-D intrinsics (approximation for 640x480 -> 518x518)
        if intrinsics is None:
            fx = fy = 525.0 * (H / 480.0)  # Scale to current resolution
            cx = cy = H / 2.0
            intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=device).unsqueeze(0).repeat(B, 1, 1)
        
        # Warp rgb1 to rgb2 using depth and pose
        rgb1_warped = self._warp_image(rgb1, depth1, relative_pose, intrinsics)
        
        # Compute photometric error
        error = torch.abs(rgb1_warped - rgb2)
        
        # Mask out invalid regions (warped outside image or invalid depth)
        valid_mask = (depth1 > 0) & (depth1 < 10.0)  # Valid depth range
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
        """
        Warp image using depth and camera pose.
        
        Args:
            image: (B, 3, H, W)
            depth: (B, 1, H, W)
            pose: (B, 4, 4) T_target^source
            intrinsics: (B, 3, 3)
            
        Returns:
            warped: (B, 3, H, W) warped image
        """
        B, _, H, W = image.shape
        device = image.device
        
        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Flatten and add batch dimension
        x = x.reshape(1, -1).expand(B, -1)  # (B, H*W)
        y = y.reshape(1, -1).expand(B, -1)
        z = depth.reshape(B, -1)  # (B, H*W)
        
        # Backproject to 3D
        K_inv = torch.inverse(intrinsics)  # (B, 3, 3)
        
        # Homogeneous pixel coordinates
        pixels = torch.stack([x, y, torch.ones_like(x)], dim=2)  # (B, H*W, 3)
        
        # 3D points in camera frame: P = Z * K^-1 @ pixels
        rays = torch.bmm(pixels, K_inv.transpose(1, 2))  # (B, H*W, 3)
        points_3d = rays * z.unsqueeze(2)  # (B, H*W, 3)
        
        # Transform to target frame
        points_3d_hom = torch.cat([points_3d, torch.ones(B, H*W, 1, device=device)], dim=2)  # (B, H*W, 4)
        points_3d_target = torch.bmm(points_3d_hom, pose.transpose(1, 2))[:, :, :3]  # (B, H*W, 3)
        
        # Project to target image
        pixels_target = torch.bmm(points_3d_target, intrinsics.transpose(1, 2))  # (B, H*W, 3)
        pixels_target = pixels_target[:, :, :2] / (pixels_target[:, :, 2:3] + 1e-7)  # (B, H*W, 2)
        
        # Normalize to [-1, 1] for grid_sample
        pixels_target[:, :, 0] = 2.0 * pixels_target[:, :, 0] / (W - 1) - 1.0  # x
        pixels_target[:, :, 1] = 2.0 * pixels_target[:, :, 1] / (H - 1) - 1.0  # y
        
        # Reshape to grid
        grid = pixels_target.reshape(B, H, W, 2)
        
        # Sample image
        warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return warped


class RepeatabilityLoss(nn.Module):
    """
    Keypoint repeatability loss.
    Ensures keypoints are stable across frames.
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
        """
        Compute repeatability of keypoints across frames.
        
        Args:
            keypoints1: (B, N, 2) keypoints in frame 1 (x, y) in patch coordinates
            keypoints2: (B, N, 2) keypoints in frame 2
            depth1: (B, 1, H, W) depth map
            relative_pose: (B, 4, 4) camera pose
            intrinsics: (B, 3, 3) camera intrinsics
            
        Returns:
            loss: Repeatability loss (lower = more repeatable)
        """
        B, N, _ = keypoints1.shape
        device = keypoints1.device
        H, W = depth1.shape[2:]
        
        # Default intrinsics
        if intrinsics is None:
            fx = fy = 525.0 * (H / 480.0)
            cx = cy = H / 2.0
            intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=device).unsqueeze(0).repeat(B, 1, 1)
        
        # Project keypoints1 to frame 2
        projected_kpts = self._project_keypoints(
            keypoints1, depth1, relative_pose, intrinsics, H, W
        )
        
        # Compute distance between projected and detected keypoints
        distances = torch.norm(projected_kpts - keypoints2, dim=2)  # (B, N)
        
        # Loss is the average distance for repeatable keypoints
        repeatable_mask = distances < self.distance_threshold
        
        if repeatable_mask.sum() > 0:
            loss = distances[repeatable_mask].mean()
        else:
            # If no keypoints are repeatable, penalize heavily
            loss = distances.mean()
        
        return loss
    
    def _project_keypoints(
        self,
        keypoints: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        H: int,
        W: int
    ) -> torch.Tensor:
        """Project keypoints from frame 1 to frame 2"""
        B, N, _ = keypoints.shape
        device = keypoints.device
        
        if intrinsics is None:
            fx = fy = 525.0 * (H / 480.0)
            cx = cy = H / 2.0
            intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=device).unsqueeze(0).repeat(B, 1, 1)
        
        # Sample depth at keypoint locations
        # Normalize keypoint coords to [-1, 1] for grid_sample
        kpts_norm = keypoints.clone()
        kpts_norm[:, :, 0] = 2.0 * keypoints[:, :, 0] / (W - 1) - 1.0
        kpts_norm[:, :, 1] = 2.0 * keypoints[:, :, 1] / (H - 1) - 1.0
        
        grid = kpts_norm.unsqueeze(1)  # (B, 1, N, 2)
        depth_at_kpts = F.grid_sample(
            depth, grid, mode='nearest', align_corners=True
        ).squeeze(1).squeeze(1)  # (B, N)
        
        # Backproject to 3D
        K_inv = torch.inverse(intrinsics)
        
        x = keypoints[:, :, 0]  # (B, N)
        y = keypoints[:, :, 1]
        pixels_hom = torch.stack([x, y, torch.ones_like(x)], dim=2)  # (B, N, 3)
        
        rays = torch.bmm(pixels_hom, K_inv.transpose(1, 2))
        points_3d = rays * depth_at_kpts.unsqueeze(2)
        
        # Transform to frame 2
        points_3d_hom = torch.cat([points_3d, torch.ones(B, N, 1, device=device)], dim=2)
        points_3d_target = torch.bmm(points_3d_hom, pose.transpose(1, 2))[:, :, :3]
        
        # Project to frame 2
        pixels_target = torch.bmm(points_3d_target, intrinsics.transpose(1, 2))
        pixels_target = pixels_target[:, :, :2] / (pixels_target[:, :, 2:3] + 1e-7)
        
        return pixels_target


class DescriptorConsistencyLoss(nn.Module):
    """
    Descriptor consistency loss.
    Matched keypoints should have similar descriptors.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            desc1: (B, N1, D) descriptors from frame 1
            desc2: (B, N2, D) descriptors from frame 2
            matches: (B, M, 2) matched indices (i, j)
            
        Returns:
            loss: Descriptor consistency loss
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
            
            matched_desc1 = desc1[b, idx1]
            matched_desc2 = desc2[b, idx2]
            
            # Positive pairs
            pos_dist = 1 - (matched_desc1 * matched_desc2).sum(dim=-1)
            pos_loss = pos_dist.mean()
            
            # Negative pairs (random sampling)
            M = matched_desc1.shape[0]
            N2 = desc2.shape[1]
            neg_idx = torch.randint(0, N2, (M,), device=device)
            
            neg_mask = neg_idx != idx2
            if neg_mask.sum() > 0:
                neg_desc2 = desc2[b, neg_idx[neg_mask]]
                neg_desc1 = matched_desc1[neg_mask]
                
                neg_sim = (neg_desc1 * neg_desc2).sum(dim=-1)
                neg_loss = torch.relu(neg_sim - self.margin).mean()
            else:
                neg_loss = 0.0
            
            total_loss += pos_loss + neg_loss
            num_valid += 1
        
        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device)


class UncertaintyCalibrationLoss(nn.Module):
    """
    Uncertainty calibration loss.
    Confidence should predict actual matching error.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        predicted_confidence: torch.Tensor,
        actual_error: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted_confidence: (B, N, 1) in [0, 1]
            actual_error: (B, N) actual reprojection/matching error
            
        Returns:
            loss: Calibration loss
        """
        # Normalize error to [0, 1]
        error_norm = actual_error / (actual_error.max() + 1e-6)
        
        # Target: high confidence = low error
        target = 1 - error_norm.unsqueeze(-1)
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_confidence, target)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_confidence, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss