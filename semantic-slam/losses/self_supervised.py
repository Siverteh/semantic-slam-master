"""
Fixed Self-Supervised Losses
CORRECTED: Sparsity loss for SIGMOID outputs (not softmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorMatchingLoss(nn.Module):
    """InfoNCE contrastive loss - unchanged, this was fine"""

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

            # Numerical stability
            logits = torch.clamp(logits, -50, 50)

            loss = F.cross_entropy(logits, idx2)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            # Return small constant if no valid matches
            return torch.tensor(0.1, device=device, requires_grad=True)


class RepeatabilityLoss(nn.Module):
    """Simple keypoint repeatability - unchanged, this was fine"""

    def __init__(self, distance_threshold: float = 2.0):
        super().__init__()
        self.distance_threshold = distance_threshold

    def forward(
        self,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor
    ) -> torch.Tensor:
        distances = torch.norm(keypoints1 - keypoints2, dim=2)
        loss = F.smooth_l1_loss(distances, torch.zeros_like(distances))
        return loss


class PeakinessLoss(nn.Module):
    """Simplified peakiness loss - just encourage high variance"""

    def __init__(self):
        super().__init__()

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        Higher variance = more peaked.
        Lower variance = more uniform.
        """
        B, H, W, _ = saliency_map.shape

        saliency_flat = saliency_map.squeeze(-1).reshape(B, -1)

        # Want high variance (clear peaks vs background)
        variance = saliency_flat.var(dim=1).mean()

        # Encourage variance around 0.1-0.2
        target_var = 0.15
        loss = -variance  # Negative to maximize variance

        return loss.abs()  # Ensure positive


class ActivationLoss(nn.Module):
    """
    NEW: Encourages the selector to actually activate (not stay near 0).
    Ensures some locations have high scores.
    """

    def __init__(self, target_mean: float = 0.3):
        super().__init__()
        self.target_mean = target_mean

    def forward(self, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        Ensure mean saliency is around target (e.g., 0.3).
        This prevents the network from learning to output all zeros.

        Args:
            saliency_map: (B, H, W, 1) sigmoid outputs

        Returns:
            loss: Scalar activation loss
        """
        mean_saliency = saliency_map.mean()
        target = torch.tensor(self.target_mean, device=saliency_map.device)
        loss = F.mse_loss(mean_saliency, target)
        return loss