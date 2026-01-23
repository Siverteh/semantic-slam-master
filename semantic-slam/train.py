"""
Main Training Script for Semantic SLAM Heads
Trains keypoint selector, descriptor refiner, and uncertainty estimator
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict

# Import models
from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
from models.uncertainty_estimator import UncertaintyEstimator

# Import dataset and losses
from data.tum_dataset import TUMDataset
from losses.self_supervised import (
    PhotometricLoss,
    RepeatabilityLoss,
    DescriptorConsistencyLoss,
    UncertaintyCalibrationLoss
)


class SemanticSLAMTrainer:
    """End-to-end trainer for semantic SLAM heads"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        print("Initializing models...")
        self.backbone = DinoBackbone(
            model_name=config['model']['backbone'],
            input_size=config['model']['input_size'],
            freeze=True
        ).to(self.device)
        
        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            hidden_dim=config['model']['selector_hidden'],
            num_layers=config['model']['selector_layers']
        ).to(self.device)
        
        self.refiner = DescriptorRefiner(
            input_dim=self.backbone.embed_dim,
            hidden_dim=config['model']['refiner_hidden'],
            output_dim=config['model']['descriptor_dim'],
            num_layers=config['model']['refiner_layers']
        ).to(self.device)
        
        self.estimator = UncertaintyEstimator(
            dino_dim=self.backbone.embed_dim,
            descriptor_dim=config['model']['descriptor_dim'],
            hidden_dim=config['model']['estimator_hidden']
        ).to(self.device)
        
        # Count parameters
        total_params = (
            sum(p.numel() for p in self.selector.parameters() if p.requires_grad) +
            sum(p.numel() for p in self.refiner.parameters() if p.requires_grad) +
            sum(p.numel() for p in self.estimator.parameters() if p.requires_grad)
        )
        print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
        
        # Initialize losses
        self.photo_loss = PhotometricLoss()
        self.repeat_loss = RepeatabilityLoss(
            distance_threshold=config['loss']['repeat_threshold']
        )
        self.desc_loss = DescriptorConsistencyLoss(
            margin=config['loss']['desc_margin']
        )
        self.uncert_loss = UncertaintyCalibrationLoss(
            loss_type=config['loss']['uncert_type']
        )
        
        # Loss weights
        self.loss_weights = config['loss']['weights']
        
        # Optimizer
        self.optimizer = AdamW(
            list(self.selector.parameters()) +
            list(self.refiner.parameters()) +
            list(self.estimator.parameters()),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['lr_min']
        )
        
        # Dataset
        self.train_loader = self._create_dataloader(
            config['dataset']['train_sequences'],
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        
        self.val_loader = self._create_dataloader(
            config['dataset']['val_sequences'],
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        
        # Logging
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project'],
                name=config['logging']['run_name'],
                config=config
            )
    
    def _create_dataloader(
        self,
        sequences: list,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        """Create dataloader from multiple sequences"""
        datasets = []
        for seq in sequences:
            dataset = TUMDataset(
                dataset_root=self.config['dataset']['root'],
                sequence=seq,
                input_size=self.config['model']['input_size'],
                frame_spacing=self.config['dataset']['frame_spacing'],
                max_frames=self.config['dataset']['max_frames']
            )
            datasets.append(dataset)
        
        # Concatenate datasets
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset(datasets)
        
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.selector.train()
        self.refiner.train()
        self.estimator.train()
        
        total_loss = 0.0
        losses_dict = {
            'photo': 0.0,
            'repeat': 0.0,
            'desc': 0.0,
            'uncert': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb1 = batch['rgb1'].to(self.device)
            rgb2 = batch['rgb2'].to(self.device)
            depth1 = batch['depth1'].to(self.device)
            depth2 = batch['depth2'].to(self.device)
            
            if 'relative_pose' in batch:
                rel_pose = batch['relative_pose'].to(self.device)
            else:
                # Skip if no pose available
                continue
            
            # Forward pass
            loss, loss_components = self._forward_pass(
                rgb1, rgb2, depth1, depth2, rel_pose
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.selector.parameters()) +
                list(self.refiner.parameters()) +
                list(self.estimator.parameters()),
                max_norm=self.config['training']['grad_clip']
            )
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            for key in losses_dict:
                losses_dict[key] += loss_components[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'photo': f"{loss_components['photo']:.4f}",
                'repeat': f"{loss_components['repeat']:.4f}"
            })
        
        # Average losses
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        for key in losses_dict:
            losses_dict[key] /= n_batches
        
        return {'total': avg_loss, **losses_dict}
    
    def _forward_pass(
        self,
        rgb1: torch.Tensor,
        rgb2: torch.Tensor,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
        rel_pose: torch.Tensor
    ) -> tuple:
        """Complete forward pass through all heads"""
        
        # Extract DINOv2 features
        with torch.no_grad():
            feat1 = self.backbone(rgb1)  # (B, H, W, C)
            feat2 = self.backbone(rgb2)
        
        # Keypoint selection
        saliency1 = self.selector(feat1)  # (B, H, W, 1)
        saliency2 = self.selector(feat2)
        
        # Select keypoints
        kpts1, scores1 = self.selector.select_keypoints(
            saliency1,
            num_keypoints=self.config['model']['num_keypoints']
        )
        kpts2, scores2 = self.selector.select_keypoints(
            saliency2,
            num_keypoints=self.config['model']['num_keypoints']
        )
        
        # Extract features at keypoints
        feat_at_kpts1 = self.backbone.extract_at_keypoints(feat1, kpts1)
        feat_at_kpts2 = self.backbone.extract_at_keypoints(feat2, kpts2)
        
        # Refine descriptors
        desc1 = self.refiner(feat_at_kpts1)  # (B, N, D)
        desc2 = self.refiner(feat_at_kpts2)
        
        # Estimate uncertainty
        conf1 = self.estimator(feat_at_kpts1, desc1)  # (B, N, 1)
        conf2 = self.estimator(feat_at_kpts2, desc2)
        
        # Compute losses
        
        # 1. Photometric loss (dense)
        loss_photo = self.photo_loss(rgb1, rgb2, depth1, rel_pose)
        
        # 2. Repeatability loss (sparse keypoints)
        loss_repeat = self.repeat_loss(kpts1, kpts2, depth1, rel_pose)
        
        # 3. Descriptor loss (need to find matches first)
        matches = self._find_matches(desc1, desc2, kpts1, kpts2, depth1, rel_pose)
        loss_desc = self.desc_loss(desc1, desc2, matches)
        
        # 4. Uncertainty loss (predict reprojection error)
        reproj_error1 = self._compute_reprojection_error(kpts1, kpts2, depth1, rel_pose)
        loss_uncert = self.uncert_loss(conf1, reproj_error1)
        
        # Weighted combination
        w = self.loss_weights
        total_loss = (
            w['photo'] * loss_photo +
            w['repeat'] * loss_repeat +
            w['desc'] * loss_desc +
            w['uncert'] * loss_uncert
        )
        
        loss_components = {
            'photo': loss_photo.item(),
            'repeat': loss_repeat.item(),
            'desc': loss_desc.item(),
            'uncert': loss_uncert.item()
        }
        
        return total_loss, loss_components
    
    def _find_matches(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        kpts1: torch.Tensor,
        kpts2: torch.Tensor,
        depth1: torch.Tensor,
        rel_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Find matches between keypoints using descriptor similarity
        and geometric consistency.
        """
        B, N, D = desc1.shape
        device = desc1.device
        
        matches_list = []
        
        for b in range(B):
            # Compute descriptor similarity matrix
            sim_matrix = torch.mm(desc1[b], desc2[b].t())  # (N, N)
            
            # Find mutual nearest neighbors
            nn12 = sim_matrix.argmax(dim=1)  # (N,)
            nn21 = sim_matrix.argmax(dim=0)  # (N,)
            
            # Mutual nearest neighbors
            mutual_mask = nn21[nn12] == torch.arange(N, device=device)
            
            # Get matched pairs
            idx1 = torch.nonzero(mutual_mask).squeeze(1)
            idx2 = nn12[idx1]
            
            if len(idx1) > 0:
                matches_b = torch.stack([idx1, idx2], dim=1)
            else:
                # No matches found, create empty tensor
                matches_b = torch.zeros(0, 2, device=device, dtype=torch.long)
            
            matches_list.append(matches_b)
        
        # Pad to same length
        max_matches = max(m.shape[0] for m in matches_list)
        if max_matches == 0:
            return torch.zeros(B, 1, 2, device=device, dtype=torch.long)
        
        padded_matches = []
        for m in matches_list:
            if m.shape[0] < max_matches:
                pad = torch.zeros(max_matches - m.shape[0], 2, device=device, dtype=torch.long)
                m = torch.cat([m, pad], dim=0)
            padded_matches.append(m)
        
        return torch.stack(padded_matches, dim=0)
    
    def _compute_reprojection_error(
        self,
        kpts1: torch.Tensor,
        kpts2: torch.Tensor,
        depth1: torch.Tensor,
        rel_pose: torch.Tensor
    ) -> torch.Tensor:
        """Compute reprojection error for uncertainty calibration"""
        # Project kpts1 to frame 2 and measure distance to kpts2
        B, N, _ = kpts1.shape
        H, W = depth1.shape[2:]
        device = kpts1.device
        
        # Use RepeatabilityLoss's projection method
        projected_kpts = self.repeat_loss._project_keypoints(
            kpts1, depth1, rel_pose,
            intrinsics=None,  # Will use default
            H=H, W=W
        )
        
        # Compute L2 distance
        error = torch.norm(projected_kpts - kpts2, dim=2)  # (B, N)
        
        return error
    
    def validate(self) -> Dict[str, float]:
        """Validation pass"""
        self.selector.eval()
        self.refiner.eval()
        self.estimator.eval()
        
        total_loss = 0.0
        losses_dict = {'photo': 0.0, 'repeat': 0.0, 'desc': 0.0, 'uncert': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                rgb1 = batch['rgb1'].to(self.device)
                rgb2 = batch['rgb2'].to(self.device)
                depth1 = batch['depth1'].to(self.device)
                depth2 = batch['depth2'].to(self.device)
                
                if 'relative_pose' not in batch:
                    continue
                
                rel_pose = batch['relative_pose'].to(self.device)
                
                loss, loss_components = self._forward_pass(
                    rgb1, rgb2, depth1, depth2, rel_pose
                )
                
                total_loss += loss.item()
                for key in losses_dict:
                    losses_dict[key] += loss_components[key]
        
        n_batches = len(self.val_loader)
        avg_loss = total_loss / n_batches
        for key in losses_dict:
            losses_dict[key] /= n_batches
        
        return {'total': avg_loss, **losses_dict}
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config['training']['val_interval'] == 0:
                val_losses = self.validate()
                
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_losses['total']:.4f}")
                print(f"  Val Loss: {val_losses['total']:.4f}")
                
                # Log to wandb
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'train/total': train_losses['total'],
                        'train/photo': train_losses['photo'],
                        'train/repeat': train_losses['repeat'],
                        'train/desc': train_losses['desc'],
                        'train/uncert': train_losses['uncert'],
                        'val/total': val_losses['total'],
                        'val/photo': val_losses['photo'],
                        'val/repeat': val_losses['repeat'],
                        'val/desc': val_losses['desc'],
                        'val/uncert': val_losses['uncert'],
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                
                # Save best model
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pth', epoch, val_losses['total'])
                    print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint periodically
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch}.pth', epoch, train_losses['total'])
        
        print("\nTraining complete!")
    
    def save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save model checkpoint"""
        save_dir = Path(self.config['training']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'selector_state_dict': self.selector.state_dict(),
            'refiner_state_dict': self.refiner.state_dict(),
            'estimator_state_dict': self.estimator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, save_dir / filename)


def main():
    # Load config
    config_path = "configs/train_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and train
    trainer = SemanticSLAMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()