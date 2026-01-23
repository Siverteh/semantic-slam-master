"""
Frozen DINOv3 Backbone for Semantic SLAM
Loads pretrained DINOv3 and extracts patch features for downstream heads
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple


class DinoBackbone(nn.Module):
    """
    Wrapper for frozen DINOv3 backbone.
    Extracts patch features at specified resolution.
    """
    
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        input_size: int = 448,
        freeze: bool = True
    ):
        """
        Args:
            model_name: DINOv3 model variant via timm
                       - vit_small_patch16_dinov3.lvd1689m (384-dim, 22M params)
                       - vit_base_patch16_dinov3.lvd1689m (768-dim, 86M params)
                       - vit_large_patch16_dinov3.lvd1689m (1024-dim, 304M params)
            input_size: Input image size (448 recommended for better resolution)
            freeze: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.input_size = input_size
        
        # Load pretrained DINOv3 from timm
        print(f"Loading {model_name} from timm...")
        self.dino = timm.create_model(
            model_name,
            pretrained=True,
            dynamic_img_size=True  # Allows flexible input sizes
        )
        
        # Get model info
        self.patch_size = 16  # DINOv3 uses 16x16 patches
        self.embed_dim = self.dino.embed_dim    # 384 for ViT-S
        self.num_patches = (input_size // self.patch_size) ** 2  # 28x28 = 784
        self.grid_size = input_size // self.patch_size  # 28
        self.n_storage_tokens = 4  # DINOv3 has 4 storage tokens after CLS
        
        # Freeze backbone if requested
        if freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
            print(f"✓ Frozen DINOv3 backbone: {self.embed_dim}D features, {self.grid_size}x{self.grid_size} patches")
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features from images.
        
        Args:
            images: (B, 3, H, W) RGB images
            
        Returns:
            patch_features: (B, grid_size, grid_size, embed_dim) spatial features
        """
        B = images.shape[0]
        
        # Extract features using forward_features (returns all tokens)
        with torch.set_grad_enabled(self.training and not self._is_frozen()):
            features = self.dino.forward_features(images)
        
        # features shape: (B, num_tokens, embed_dim)
        # num_tokens = 1 (CLS) + 4 (storage) + num_patches
        
        # Extract only patch tokens (skip CLS token at index 0 and 4 storage tokens)
        patch_tokens = features[:, 1 + self.n_storage_tokens:, :]  # (B, num_patches, embed_dim)
        
        # Reshape to spatial grid
        patch_features = patch_tokens.reshape(
            B, self.grid_size, self.grid_size, self.embed_dim
        )
        
        return patch_features
    
    def _is_frozen(self) -> bool:
        """Check if backbone is frozen"""
        return not next(self.dino.parameters()).requires_grad
    
    def extract_at_keypoints(
        self,
        patch_features: torch.Tensor,
        keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features at specific keypoint locations using bilinear sampling.
        
        Args:
            patch_features: (B, H, W, C) feature map
            keypoints: (B, N, 2) keypoint coordinates in [0, H-1] x [0, W-1]
            
        Returns:
            sampled_features: (B, N, C) features at keypoint locations
        """
        B, H, W, C = patch_features.shape
        N = keypoints.shape[1]
        
        # Normalize coordinates to [-1, 1] for grid_sample
        norm_coords = keypoints.clone()
        norm_coords[:, :, 0] = 2.0 * keypoints[:, :, 0] / (W - 1) - 1.0  # x
        norm_coords[:, :, 1] = 2.0 * keypoints[:, :, 1] / (H - 1) - 1.0  # y
        
        # Reshape for grid_sample: (B, N, 2) -> (B, 1, N, 2)
        grid = norm_coords.unsqueeze(1)
        
        # Permute features for grid_sample: (B, H, W, C) -> (B, C, H, W)
        features_bhwc = patch_features.permute(0, 3, 1, 2)
        
        # Sample features at keypoint locations
        sampled = torch.nn.functional.grid_sample(
            features_bhwc, grid, mode='bilinear', align_corners=True
        )
        
        # Reshape output: (B, C, 1, N) -> (B, N, C)
        sampled_features = sampled.squeeze(2).permute(0, 2, 1)
        
        return sampled_features