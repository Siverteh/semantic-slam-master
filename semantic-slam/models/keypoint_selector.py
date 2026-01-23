"""
Keypoint Selector Head
Learns to identify stable, trackable keypoints from DINOv3 features
Inspired by R2D2's repeatability map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KeypointSelector(nn.Module):
    """
    Predicts saliency heatmap indicating which locations are good keypoints.
    Uses lightweight CNN architecture for efficiency.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        """
        Args:
            input_dim: DINOv3 feature dimension (384 for ViT-S)
            hidden_dim: Hidden layer dimension
            num_layers: Number of convolutional layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Build convolutional layers
        layers = []
        in_channels = input_dim
        
        for i in range(num_layers):
            out_channels = hidden_dim if i < num_layers - 1 else 1
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity()
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Final activation to produce probabilities
        self.activation = nn.Sigmoid()
        
    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dino_features: (B, H, W, C) DINOv3 patch features
            
        Returns:
            saliency_map: (B, H, W, 1) keypoint probability map in [0, 1]
        """
        # Permute to (B, C, H, W) for conv layers
        x = dino_features.permute(0, 3, 1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Apply sigmoid activation
        x = self.activation(x)
        
        # Permute back to (B, H, W, 1)
        saliency_map = x.permute(0, 2, 3, 1)
        
        return saliency_map
    
    def select_keypoints(
        self,
        saliency_map: torch.Tensor,
        num_keypoints: int = 500,
        nms_radius: int = 2,
        threshold: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top keypoints from saliency map with non-maximum suppression.
        
        Args:
            saliency_map: (B, H, W, 1) saliency heatmap
            num_keypoints: Maximum number of keypoints to select
            nms_radius: Radius for non-maximum suppression
            threshold: Minimum saliency threshold
            
        Returns:
            keypoints: (B, N, 2) keypoint coordinates in [0, H-1] x [0, W-1]
            scores: (B, N) saliency scores for each keypoint
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device
        
        # Squeeze last dimension
        saliency = saliency_map.squeeze(-1)  # (B, H, W)
        
        # Apply NMS if requested
        if nms_radius > 0:
            saliency = self._apply_nms(saliency, nms_radius)
        
        # Threshold
        saliency = torch.where(saliency > threshold, saliency, torch.zeros_like(saliency))
        
        # Select top-k keypoints per batch
        keypoints_list = []
        scores_list = []
        
        for b in range(B):
            sal_b = saliency[b]  # (H, W)
            
            # Get valid coordinates
            valid_mask = sal_b > 0
            valid_scores = sal_b[valid_mask]
            valid_coords = torch.nonzero(valid_mask)  # (N_valid, 2) as (y, x)
            
            if len(valid_scores) == 0:
                # No valid keypoints, return random points
                random_y = torch.randint(0, H, (num_keypoints,), device=device)
                random_x = torch.randint(0, W, (num_keypoints,), device=device)
                kpts = torch.stack([random_x, random_y], dim=1).float()
                scrs = torch.zeros(num_keypoints, device=device)
            else:
                # Select top-k
                k = min(num_keypoints, len(valid_scores))
                top_scores, top_indices = torch.topk(valid_scores, k)
                top_coords = valid_coords[top_indices]  # (k, 2) as (y, x)
                
                # Convert to (x, y) format
                kpts = torch.stack([top_coords[:, 1], top_coords[:, 0]], dim=1).float()
                scrs = top_scores
                
                # Pad if necessary
                if k < num_keypoints:
                    pad_size = num_keypoints - k
                    kpts = torch.cat([kpts, kpts[-1:].repeat(pad_size, 1)], dim=0)
                    scrs = torch.cat([scrs, torch.zeros(pad_size, device=device)], dim=0)
            
            keypoints_list.append(kpts)
            scores_list.append(scrs)
        
        keypoints = torch.stack(keypoints_list, dim=0)  # (B, N, 2)
        scores = torch.stack(scores_list, dim=0)  # (B, N)
        
        return keypoints, scores
    
    def _apply_nms(self, saliency: torch.Tensor, radius: int) -> torch.Tensor:
        """
        Apply non-maximum suppression to saliency map.
        
        Args:
            saliency: (B, H, W) saliency scores
            radius: NMS radius
            
        Returns:
            nms_saliency: (B, H, W) after NMS
        """
        # Max pooling to find local maxima
        kernel_size = 2 * radius + 1
        max_pooled = F.max_pool2d(
            saliency.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=radius
        ).squeeze(1)
        
        # Keep only local maxima
        nms_mask = (saliency == max_pooled)
        nms_saliency = saliency * nms_mask.float()
        
        return nms_saliency