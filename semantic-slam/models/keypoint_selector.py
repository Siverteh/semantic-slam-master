"""
Semantic Edge-based Keypoint Detector
Grid-aligned detection following DINO-VO architecture.
Detects keypoints at semantic discontinuities (your idea!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class KeypointSelector(nn.Module):
    """
    Semantic edge-based keypoint detector.

    Key ideas:
    1. Detect semantic discontinuities in DINO features (edges between objects)
    2. Grid-aligned to 16×16 patches (ensures each keypoint queries one DINO patch)
    3. Learnable refinement on top of gradient-based detection

    This combines:
    - Your idea: semantic edges are good keypoints!
    - DINO-VO: grid-aligned detection for coarse features
    - Geometric constraints: edges correspond to 3D boundaries
    """

    def __init__(
        self,
        input_dim: int = 384,
        patch_size: int = 16
    ):
        super().__init__()

        self.input_dim = input_dim
        self.patch_size = patch_size

        # Learnable edge detector (refines gradient-based detection)
        self.edge_refiner = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_semantic_edges(
        self,
        dino_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute semantic edge strength from DINO features.

        This is your idea! Semantic discontinuities = object boundaries = good keypoints.

        Args:
            dino_features: (B, H, W, C) DINO patch features

        Returns:
            edge_strength: (B, H, W, 1) semantic edge magnitudes
        """
        # Convert to (B, C, H, W) for convolution
        features = dino_features.permute(0, 3, 1, 2)

        # Compute gradients in feature space
        grad_x = features[:, :, :, 1:] - features[:, :, :, :-1]  # (B, C, H, W-1)
        grad_y = features[:, :, 1:, :] - features[:, :, :-1, :]  # (B, C, H-1, W)

        # Pad to same size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))  # Pad right
        grad_y = F.pad(grad_y, (0, 0, 0, 1))  # Pad bottom

        # L2 norm across feature dimensions = edge strength
        edge_strength = torch.sqrt(
            (grad_x ** 2).sum(dim=1, keepdim=True) +
            (grad_y ** 2).sum(dim=1, keepdim=True) +
            1e-8
        )

        # Normalize to [0, 1]
        edge_strength = edge_strength / (edge_strength.max() + 1e-8)

        # Convert back to (B, H, W, 1)
        edge_strength = edge_strength.permute(0, 2, 3, 1)

        return edge_strength

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Predict keypoint saliency map.

        Combines:
        - Geometric edge detection (gradient-based)
        - Learned refinement (neural network)

        Args:
            dino_features: (B, H, W, C) in PATCH space (28×28 for 448×448 input)

        Returns:
            saliency_map: (B, H, W, 1) scores in [0, 1]
        """
        # Compute semantic edges (your idea!)
        semantic_edges = self.compute_semantic_edges(dino_features)

        # Learned refinement
        features_bchw = dino_features.permute(0, 3, 1, 2)
        learned_saliency = self.edge_refiner(features_bchw)
        learned_saliency = torch.sigmoid(learned_saliency)
        learned_saliency = learned_saliency.permute(0, 2, 3, 1)

        # Combine: geometric edges guide learned saliency
        saliency_map = 0.5 * semantic_edges + 0.5 * learned_saliency

        return saliency_map

    def select_keypoints(
        self,
        saliency_map: torch.Tensor,
        num_keypoints: int = 500,
        nms_radius: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select keypoints using grid-aligned detection.

        CRITICAL: Grid-aligned to ensure each keypoint queries exactly one DINO patch.

        Args:
            saliency_map: (B, H, W, 1) saliency scores
            num_keypoints: Number of keypoints to select
            nms_radius: NMS radius in patch space

        Returns:
            keypoints: (B, N, 2) in PATCH coordinates [0, H-1] x [0, W-1]
            scores: (B, N) saliency scores
        """
        B, H, W, _ = saliency_map.shape
        device = saliency_map.device

        saliency = saliency_map.squeeze(-1)  # (B, H, W)

        keypoints_list = []
        scores_list = []

        for b in range(B):
            sal_b = saliency[b]

            # Apply NMS
            sal_nms = self._apply_nms(sal_b.unsqueeze(0), nms_radius).squeeze(0)

            # Select top-k
            sal_flat = sal_nms.flatten()
            top_scores, top_indices = torch.topk(sal_flat, min(num_keypoints, len(sal_flat)))

            # Convert to 2D coordinates (patch space)
            y_coords = top_indices // W
            x_coords = top_indices % W

            kpts = torch.stack([x_coords, y_coords], dim=1).float()

            # Ensure exactly num_keypoints
            if len(kpts) < num_keypoints:
                # Pad with duplicates of best keypoint
                pad_size = num_keypoints - len(kpts)
                best_kpt = kpts[0:1].repeat(pad_size, 1)
                best_score = top_scores[0:1].repeat(pad_size)

                kpts = torch.cat([kpts, best_kpt], dim=0)
                top_scores = torch.cat([top_scores, best_score], dim=0)

            keypoints_list.append(kpts)
            scores_list.append(top_scores)

        keypoints = torch.stack(keypoints_list, dim=0)
        scores = torch.stack(scores_list, dim=0)

        return keypoints, scores

    def _apply_nms(
        self,
        saliency: torch.Tensor,
        radius: int
    ) -> torch.Tensor:
        """Non-maximum suppression"""
        if radius == 0:
            return saliency

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