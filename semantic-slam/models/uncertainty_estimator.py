"""
Uncertainty Estimator Head
Predicts confidence scores for keypoints and descriptors
Used to weight features in bundle adjustment (inspired by R2D2's reliability map)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyEstimator(nn.Module):
    """
    Predicts per-keypoint confidence scores.
    Combines DINOv3 features + descriptor for uncertainty estimation.
    """
    
    def __init__(
        self,
        dino_dim: int = 384,
        descriptor_dim: int = 128,
        hidden_dim: int = 128
    ):
        """
        Args:
            dino_dim: DINOv3 feature dimension
            descriptor_dim: Refined descriptor dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.dino_dim = dino_dim
        self.descriptor_dim = descriptor_dim
        
        # Small MLP to predict confidence from concatenated features
        input_dim = dino_dim + descriptor_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(
        self,
        dino_features: torch.Tensor,
        descriptors: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict uncertainty scores for keypoints.
        
        Args:
            dino_features: (B, N, dino_dim) DINOv3 features at keypoints
            descriptors: (B, N, descriptor_dim) refined descriptors
            
        Returns:
            confidence: (B, N, 1) confidence scores in [0, 1]
        """
        # Concatenate features
        combined = torch.cat([dino_features, descriptors], dim=-1)  # (B, N, input_dim)
        
        # Predict confidence
        confidence = self.mlp(combined)  # (B, N, 1)
        
        return confidence
    
    def compute_calibration_loss(
        self,
        predicted_confidence: torch.Tensor,
        actual_error: torch.Tensor,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Calibration loss: confidence should correlate with actual error.
        Higher confidence -> lower error
        
        Args:
            predicted_confidence: (B, N, 1) predicted confidence in [0, 1]
            actual_error: (B, N) actual matching/reprojection error
            epsilon: Small constant for numerical stability
            
        Returns:
            loss: Scalar calibration loss
        """
        # Normalize error to [0, 1] range (lower is better)
        error_normalized = actual_error / (actual_error.max() + epsilon)
        
        # We want: high confidence <-> low error
        # So predicted_confidence should match (1 - error_normalized)
        target = 1 - error_normalized.unsqueeze(-1)  # (B, N, 1)
        
        # MSE loss between predicted confidence and target
        loss = F.mse_loss(predicted_confidence, target)
        
        return loss
    
    def compute_expected_error_loss(
        self,
        predicted_confidence: torch.Tensor,
        actual_error: torch.Tensor
    ) -> torch.Tensor:
        """
        Alternative loss: predict expected error directly.
        Confidence = 1 / (1 + error)
        
        Args:
            predicted_confidence: (B, N, 1) predicted confidence
            actual_error: (B, N) actual error
            
        Returns:
            loss: Scalar loss
        """
        # Convert confidence to predicted error
        predicted_error = (1 / (predicted_confidence.squeeze(-1) + 1e-6)) - 1
        
        # L1 loss between predicted and actual error
        loss = F.l1_loss(predicted_error, actual_error)
        
        return loss
    
    def filter_keypoints_by_confidence(
        self,
        keypoints: torch.Tensor,
        descriptors: torch.Tensor,
        confidence: torch.Tensor,
        threshold: float = 0.5
    ) -> tuple:
        """
        Filter out low-confidence keypoints.
        
        Args:
            keypoints: (B, N, 2) keypoint coordinates
            descriptors: (B, N, D) descriptors
            confidence: (B, N, 1) confidence scores
            threshold: Minimum confidence threshold
            
        Returns:
            filtered_keypoints: (B, M, 2)
            filtered_descriptors: (B, M, D)
            filtered_confidence: (B, M, 1)
        """
        B, N, _ = keypoints.shape
        
        filtered_kpts = []
        filtered_desc = []
        filtered_conf = []
        
        for b in range(B):
            mask = confidence[b, :, 0] >= threshold
            
            if mask.sum() == 0:
                # Keep at least one keypoint
                mask[confidence[b, :, 0].argmax()] = True
            
            filtered_kpts.append(keypoints[b, mask])
            filtered_desc.append(descriptors[b, mask])
            filtered_conf.append(confidence[b, mask])
        
        # Pad to same length
        max_len = max(kpts.shape[0] for kpts in filtered_kpts)
        
        padded_kpts = []
        padded_desc = []
        padded_conf = []
        
        for kpts, desc, conf in zip(filtered_kpts, filtered_desc, filtered_conf):
            if len(kpts) < max_len:
                pad_size = max_len - len(kpts)
                kpts = torch.cat([kpts, kpts[-1:].repeat(pad_size, 1)], dim=0)
                desc = torch.cat([desc, desc[-1:].repeat(pad_size, 1)], dim=0)
                conf = torch.cat([conf, torch.zeros(pad_size, 1, device=conf.device)], dim=0)
            
            padded_kpts.append(kpts)
            padded_desc.append(desc)
            padded_conf.append(conf)
        
        return (
            torch.stack(padded_kpts, dim=0),
            torch.stack(padded_desc, dim=0),
            torch.stack(padded_conf, dim=0)
        )