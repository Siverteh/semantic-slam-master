"""
Descriptor Refiner Head
Refines DINOv3 semantic features into compact, matchable SLAM descriptors
Adds geometric information to semantic features (inspired by RoMa)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorRefiner(nn.Module):
    """
    Refines DINOv3 384-dim features into compact 128-dim descriptors.
    Uses MLP with L2 normalization for descriptor matching.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3
    ):
        """
        Args:
            input_dim: DINOv3 feature dimension (384 for ViT-S)
            hidden_dim: Hidden layer dimension
            output_dim: Final descriptor dimension (128 for efficiency)
            num_layers: Number of MLP layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(dims[i + 1]))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Refine DINOv3 features into descriptors.
        
        Args:
            dino_features: (B, N, C) features at N keypoint locations
                          OR (B, H, W, C) dense feature map
            
        Returns:
            descriptors: (B, N, output_dim) L2-normalized descriptors
                        OR (B, H, W, output_dim) dense descriptors
        """
        input_shape = dino_features.shape
        is_dense = len(input_shape) == 4
        
        if is_dense:
            # Dense mode: (B, H, W, C)
            B, H, W, C = input_shape
            # Flatten spatial dimensions
            x = dino_features.reshape(B * H * W, C)
        else:
            # Sparse mode: (B, N, C)
            B, N, C = input_shape
            x = dino_features.reshape(B * N, C)
        
        # Apply MLP
        descriptors = self.mlp(x)
        
        # L2 normalize
        descriptors = F.normalize(descriptors, p=2, dim=-1)
        
        # Reshape back
        if is_dense:
            descriptors = descriptors.reshape(B, H, W, self.output_dim)
        else:
            descriptors = descriptors.reshape(B, N, self.output_dim)
        
        return descriptors
    
    def compute_descriptor_loss(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor,
        margin: float = 0.2
    ) -> torch.Tensor:
        """
        Compute descriptor matching loss using cosine similarity.
        
        Args:
            desc1: (B, N1, D) descriptors from frame 1
            desc2: (B, N2, D) descriptors from frame 2
            matches: (B, M, 2) matched keypoint indices (i, j)
            margin: Margin for contrastive loss
            
        Returns:
            loss: Scalar descriptor loss
        """
        B = desc1.shape[0]
        device = desc1.device
        
        total_loss = 0.0
        num_valid = 0
        
        for b in range(B):
            if matches[b].shape[0] == 0:
                continue
            
            # Get matched descriptors
            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()
            
            matched_desc1 = desc1[b, idx1]  # (M, D)
            matched_desc2 = desc2[b, idx2]  # (M, D)
            
            # Positive pairs: matched descriptors should be similar
            pos_sim = (matched_desc1 * matched_desc2).sum(dim=-1)  # (M,)
            pos_loss = (1 - pos_sim).clamp(min=0).mean()
            
            # Negative pairs: non-matched descriptors should be dissimilar
            # Sample random negatives
            M = matched_desc1.shape[0]
            N2 = desc2.shape[1]
            neg_idx = torch.randint(0, N2, (M,), device=device)
            
            # Ensure negatives are actually different
            neg_mask = neg_idx != idx2
            if neg_mask.sum() > 0:
                neg_desc2 = desc2[b, neg_idx[neg_mask]]
                neg_desc1 = matched_desc1[neg_mask]
                
                neg_sim = (neg_desc1 * neg_desc2).sum(dim=-1)  # (M_neg,)
                neg_loss = (neg_sim - margin).clamp(min=0).mean()
            else:
                neg_loss = 0.0
            
            total_loss += pos_loss + neg_loss
            num_valid += 1
        
        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device)