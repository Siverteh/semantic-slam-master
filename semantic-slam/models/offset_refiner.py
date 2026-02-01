"""
Sub-pixel Offset Refiner
Predicts sub-pixel offsets to improve keypoint localization precision.
Following XFeat and ALIKE approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OffsetRefiner(nn.Module):
    """
    Predicts sub-pixel offsets for keypoint locations.

    Goal: Refine grid-aligned keypoints to sub-pixel precision.

    Input: Fused features at keypoint locations
    Output: Offsets in range [-0.5, 0.5] pixels
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.input_dim = input_dim

        # MLP for offset prediction
        self.offset_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),  # (dx, dy) offsets
            nn.Tanh()  # Range [-1, 1], will scale to [-0.5, 0.5]
        )

    def forward(
        self,
        features: torch.Tensor,
        keypoints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict sub-pixel offsets.

        Args:
            features: (B, N, C) features at keypoints
            keypoints: (B, N, 2) keypoint coordinates in PATCH space

        Returns:
            refined_keypoints: (B, N, 2) refined coordinates
            offsets: (B, N, 2) predicted offsets
        """
        # Predict offsets
        offsets = self.offset_mlp(features)  # (B, N, 2) in [-1, 1]

        # Scale to [-0.5, 0.5] pixels
        offsets = offsets * 0.5

        # Add to integer keypoints
        refined_keypoints = keypoints + offsets

        return refined_keypoints, offsets

    def compute_offset_loss(
        self,
        predicted_offsets: torch.Tensor,
        keypoints1: torch.Tensor,
        keypoints2_warped: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute offset consistency loss.

        Idea: Predicted offsets should align keypoints with warped correspondences.

        Args:
            predicted_offsets: (B, N, 2) predicted offsets
            keypoints1: (B, N, 2) original keypoints
            keypoints2_warped: (B, N, 2) corresponding keypoints from frame 2, warped to frame 1
            valid_mask: (B, N) mask for valid correspondences

        Returns:
            loss: Scalar offset loss
        """
        # Refined keypoints
        refined_kpts = keypoints1 + predicted_offsets

        # Distance to warped correspondences
        distance = torch.norm(refined_kpts - keypoints2_warped, dim=-1)

        # Mask invalid correspondences
        distance = distance * valid_mask

        # Loss: L1 distance
        loss = distance.sum() / (valid_mask.sum() + 1e-8)

        return loss