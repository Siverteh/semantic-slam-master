"""
Main Training Script for Semantic SLAM Heads
FIXED: Removed variance regularization, better metrics, cleaner code
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
import numpy as np

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
    UncertaintyCalibrationLoss,
    DescriptorDiversityLoss  # NEW!
)


class SemanticSLAMTrainer:
    """End-to-end trainer for semantic SLAM heads"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "="*70)
        print("INITIALIZING SEMANTIC SLAM TRAINING (FIXED VERSION)")
        print("="*70)

        # Initialize models
        print("\n📦 Loading models...")
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
        selector_params = sum(p.numel() for p in self.selector.parameters() if p.requires_grad)
        refiner_params = sum(p.numel() for p in self.refiner.parameters() if p.requires_grad)
        estimator_params = sum(p.numel() for p in self.estimator.parameters() if p.requires_grad)
        total_params = selector_params + refiner_params + estimator_params

        print(f"  ✓ Keypoint Selector:  {selector_params/1e6:.2f}M params")
        print(f"  ✓ Descriptor Refiner: {refiner_params/1e6:.2f}M params")
        print(f"  ✓ Uncertainty Estimator: {estimator_params/1e6:.2f}M params")
        print(f"  ✓ Total trainable:    {total_params/1e6:.2f}M params")
        print(f"  ✓ Device: {self.device}")

        # Initialize losses
        print("\n📊 Initializing losses...")
        self.photo_loss = PhotometricLoss()
        self.repeat_loss = RepeatabilityLoss(
            distance_threshold=config['loss']['repeat_threshold']
        )
        self.desc_loss = DescriptorConsistencyLoss(
            margin=config['loss']['desc_margin'],
            temperature=config['loss'].get('desc_temperature', 0.1)
        )
        self.uncert_loss = UncertaintyCalibrationLoss(
            loss_type=config['loss']['uncert_type']
        )
        self.diversity_loss = DescriptorDiversityLoss()  # NEW!

        # Loss weights
        self.loss_weights = config['loss']['weights']
        print(f"  Loss weights: {self.loss_weights}")

        # Optimizer
        self.optimizer = AdamW(
            list(self.selector.parameters()) +
            list(self.refiner.parameters()) +
            list(self.estimator.parameters()),
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay'])
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=float(config['training']['lr_min'])
        )

        # Warmup
        self.warmup_epochs = config['training'].get('warmup_epochs', 0)
        self.base_lr = float(config['training']['lr'])

        # Dataset
        print("\n📂 Loading datasets...")
        self.train_loader = self._create_dataloader(
            config['dataset']['train_sequences'],
            batch_size=config['training']['batch_size'],
            shuffle=True,
            is_train=True
        )

        self.val_loader = self._create_dataloader(
            config['dataset']['val_sequences'],
            batch_size=config['training']['batch_size'],
            shuffle=False,
            is_train=False
        )

        print(f"  ✓ Train batches: {len(self.train_loader)}")
        print(f"  ✓ Val batches: {len(self.val_loader)}")

        # Logging
        if config['logging']['use_wandb']:
            print("\n🔗 Initializing Weights & Biases...")
            wandb.init(
                project=config['logging']['project'],
                name=config['logging']['run_name'],
                config=config
            )
            print(f"  ✓ Run: {config['logging']['run_name']}")

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        print("\n" + "="*70)
        print("✓ INITIALIZATION COMPLETE - ARCHITECTURE FIXED")
        print("="*70 + "\n")

    def _create_dataloader(
        self,
        sequences: list,
        batch_size: int,
        shuffle: bool,
        is_train: bool = True
    ) -> DataLoader:
        """Create dataloader from multiple sequences"""
        datasets = []
        for seq in sequences:
            dataset = TUMDataset(
                dataset_root=self.config['dataset']['root'],
                sequence=seq,
                input_size=self.config['model']['input_size'],
                frame_spacing=self.config['dataset']['frame_spacing'],
                max_frames=self.config['dataset']['max_frames'],
                augmentation=self.config['dataset'].get('augmentation'),
                is_train=is_train
            )
            datasets.append(dataset)

        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset(datasets)

        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )

    def _get_lr(self, epoch: int) -> float:
        """Get learning rate with warmup"""
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            return self.optimizer.param_groups[0]['lr']

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.selector.train()
        self.refiner.train()
        self.estimator.train()

        # Apply warmup LR
        if epoch < self.warmup_epochs:
            lr = self._get_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        total_loss = 0.0
        losses_dict = {'photo': 0.0, 'repeat': 0.0, 'desc': 0.0, 'uncert': 0.0, 'diversity': 0.0}

        # Metrics tracking
        metrics = {
            'num_matches': [],
            'mean_confidence': [],
            'saliency_max': [],
            'saliency_mean': [],
            'desc_diversity': []  # RENAMED from desc_variance
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb1 = batch['rgb1'].to(self.device)
            rgb2 = batch['rgb2'].to(self.device)
            depth1 = batch['depth1'].to(self.device)
            depth2 = batch['depth2'].to(self.device)

            if 'relative_pose' not in batch:
                continue

            rel_pose = batch['relative_pose'].to(self.device)

            # Forward pass
            loss, loss_components, batch_metrics = self._forward_pass(
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

            # Accumulate losses
            total_loss += loss.item()
            for key in losses_dict:
                losses_dict[key] += loss_components[key]

            # Accumulate metrics
            for key in metrics:
                if key in batch_metrics:
                    metrics[key].append(batch_metrics[key])

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'L_desc': f"{loss_components['desc']:.3f}",
                'matches': f"{batch_metrics.get('num_matches', 0):.0f}",
                'desc_div': f"{batch_metrics.get('desc_diversity', 0):.3f}",  # RENAMED
                'lr': f"{current_lr:.1e}"
            })

            self.global_step += 1

            # Log to wandb
            if self.config['logging']['use_wandb'] and batch_idx % self.config['logging'].get('log_interval', 50) == 0:
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/photo': loss_components['photo'],
                    'batch/repeat': loss_components['repeat'],
                    'batch/desc': loss_components['desc'],
                    'batch/uncert': loss_components['uncert'],
                    'batch/num_matches': batch_metrics.get('num_matches', 0),
                    'batch/desc_variance': batch_metrics.get('desc_variance', 0),
                    'lr': current_lr,
                    'step': self.global_step
                })

        # Average losses and metrics
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        for key in losses_dict:
            losses_dict[key] /= n_batches

        avg_metrics = {}
        for key in metrics:
            if len(metrics[key]) > 0:
                avg_metrics[key] = np.mean(metrics[key])

        return {'total': avg_loss, **losses_dict, **avg_metrics}

    def _forward_pass(
        self,
        rgb1: torch.Tensor,
        rgb2: torch.Tensor,
        depth1: torch.Tensor,
        depth2: torch.Tensor,
        rel_pose: torch.Tensor
    ) -> tuple:
        """Complete forward pass through all heads"""

        # Extract DINOv3 features (now with proper normalization!)
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

        # Refine descriptors (L2 norm happens INSIDE refiner at the end)
        desc1 = self.refiner(feat_at_kpts1)  # (B, N, D)
        desc2 = self.refiner(feat_at_kpts2)

        # Estimate uncertainty
        conf1 = self.estimator(feat_at_kpts1, desc1)  # (B, N, 1)
        conf2 = self.estimator(feat_at_kpts2, desc2)

        # Compute losses

        # 1. Photometric loss
        loss_photo = self.photo_loss(rgb1, rgb2, depth1, rel_pose)

        # 2. Repeatability loss
        loss_repeat = self.repeat_loss(kpts1, kpts2, depth1, rel_pose)

        # 3. Descriptor loss
        matches = self._find_matches(desc1, desc2, kpts1, kpts2, depth1, rel_pose)
        loss_desc = self.desc_loss(desc1, desc2, matches)

        # 4. Uncertainty loss
        reproj_error1 = self._compute_reprojection_error(kpts1, kpts2, depth1, rel_pose)
        loss_uncert = self.uncert_loss(conf1, reproj_error1)

        # 5. Diversity loss (NEW - prevents descriptor collapse!)
        loss_diversity = self.diversity_loss(desc1)

        # Weighted combination
        w = self.loss_weights
        total_loss = (
            w['photo'] * loss_photo +
            w['repeat'] * loss_repeat +
            w['desc'] * loss_desc +
            w['uncert'] * loss_uncert +
            w['diversity'] * loss_diversity  # NEW!
        )

        loss_components = {
            'photo': loss_photo.item(),
            'repeat': loss_repeat.item(),
            'desc': loss_desc.item(),
            'uncert': loss_uncert.item(),
            'diversity': loss_diversity.item()  # NEW!
        }

        # Compute batch metrics
        # FIXED: For L2-normalized descriptors, use pairwise similarity instead of variance
        desc_flat = desc1.reshape(-1, desc1.shape[-1])  # (B*N, D)

        # Compute pairwise cosine similarities (for L2-normalized = dot product)
        # Sample subset for efficiency
        if desc_flat.shape[0] > 500:
            indices = torch.randperm(desc_flat.shape[0])[:500]
            desc_sample = desc_flat[indices]
        else:
            desc_sample = desc_flat

        sim_matrix = torch.mm(desc_sample, desc_sample.t())
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(sim_matrix.shape[0], device=sim_matrix.device, dtype=torch.bool)
        pairwise_sims = sim_matrix[mask]

        # Mean absolute similarity (0 = diverse, 1 = collapsed)
        desc_diversity = pairwise_sims.abs().mean().item()

        batch_metrics = {
            'num_matches': matches.shape[1],
            'mean_confidence': conf1.mean().item(),
            'saliency_max': saliency1.max().item(),
            'saliency_mean': saliency1.mean().item(),
            'desc_diversity': desc_diversity  # RENAMED from desc_variance
        }

        return total_loss, loss_components, batch_metrics

    def _find_matches(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        kpts1: torch.Tensor,
        kpts2: torch.Tensor,
        depth1: torch.Tensor,
        rel_pose: torch.Tensor
    ) -> torch.Tensor:
        """Find matches using mutual nearest neighbors"""
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
        B, N, _ = kpts1.shape
        H, W = depth1.shape[2:]
        device = kpts1.device

        # Create default intrinsics
        fx = fy = 525.0 * (H / 480.0)
        cx = cy = H / 2.0
        intrinsics = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], device=device).unsqueeze(0).repeat(B, 1, 1)

        # Project kpts1 to frame 2
        projected_kpts = self.repeat_loss._project_keypoints(
            kpts1, depth1, rel_pose,
            intrinsics=intrinsics,
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
        losses_dict = {'photo': 0.0, 'repeat': 0.0, 'desc': 0.0, 'uncert': 0.0, 'diversity': 0.0}
        metrics = {
            'num_matches': [],
            'mean_confidence': [],
            'desc_diversity': []  # RENAMED
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                rgb1 = batch['rgb1'].to(self.device)
                rgb2 = batch['rgb2'].to(self.device)
                depth1 = batch['depth1'].to(self.device)
                depth2 = batch['depth2'].to(self.device)

                if 'relative_pose' not in batch:
                    continue

                rel_pose = batch['relative_pose'].to(self.device)

                loss, loss_components, batch_metrics = self._forward_pass(
                    rgb1, rgb2, depth1, depth2, rel_pose
                )

                total_loss += loss.item()
                for key in losses_dict:
                    losses_dict[key] += loss_components[key]

                for key in metrics:
                    if key in batch_metrics:
                        metrics[key].append(batch_metrics[key])

        n_batches = len(self.val_loader)
        avg_loss = total_loss / n_batches
        for key in losses_dict:
            losses_dict[key] /= n_batches

        avg_metrics = {}
        for key in metrics:
            if len(metrics[key]) > 0:
                avg_metrics[key] = np.mean(metrics[key])

        return {'total': avg_loss, **losses_dict, **avg_metrics}

    def train(self):
        """Main training loop"""
        print("🚀 Starting training...\n")

        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            if epoch % self.config['training']['val_interval'] == 0:
                val_losses = self.validate()

                # Print epoch summary
                print(f"\n{'='*70}")
                print(f"EPOCH {epoch}/{self.config['training']['epochs']} SUMMARY")
                print(f"{'='*70}")
                print(f"{'Metric':<20} {'Train':>12} {'Val':>12}")
                print(f"{'-'*70}")
                print(f"{'Total Loss':<20} {train_losses['total']:>12.4f} {val_losses['total']:>12.4f}")
                print(f"{'  Photometric':<20} {train_losses['photo']:>12.4f} {val_losses['photo']:>12.4f}")
                print(f"{'  Repeatability':<20} {train_losses['repeat']:>12.4f} {val_losses['repeat']:>12.4f}")
                print(f"{'  Descriptor':<20} {train_losses['desc']:>12.4f} {val_losses['desc']:>12.4f}")
                print(f"{'  Uncertainty':<20} {train_losses['uncert']:>12.4f} {val_losses['uncert']:>12.4f}")
                print(f"{'  Diversity':<20} {train_losses['diversity']:>12.4f} {val_losses['diversity']:>12.4f}")
                print(f"{'-'*70}")
                print(f"{'Avg Matches':<20} {train_losses.get('num_matches', 0):>12.1f} {val_losses.get('num_matches', 0):>12.1f}")
                print(f"{'Mean Confidence':<20} {train_losses.get('mean_confidence', 0):>12.3f} {val_losses.get('mean_confidence', 0):>12.3f}")
                print(f"{'Desc Diversity':<20} {train_losses.get('desc_diversity', 0):>12.3f} {val_losses.get('desc_diversity', 0):>12.3f}")  # RENAMED
                print(f"{'Learning Rate':<20} {self.optimizer.param_groups[0]['lr']:>12.1e}")
                print(f"{'='*70}\n")

                # Check for descriptor health (UPDATED)
                desc_div = train_losses.get('desc_diversity', 0)
                if desc_div > 0.5:
                    print(f"⚠️  WARNING: Descriptor diversity low ({desc_div:.3f} > 0.5)")
                    print("   Descriptors may be too similar (collapse)")
                elif desc_div < 0.1:
                    print(f"✓ Descriptor diversity excellent ({desc_div:.3f} < 0.1)")
                    print("   Descriptors are well-separated!")

                # Check diversity loss (should decrease as descriptors learn to match)
                diversity_loss = train_losses.get('diversity', 0)
                if diversity_loss < 0.02:
                    print(f"✓ Diversity loss healthy ({diversity_loss:.3f})")
                    print("   Descriptors learning to match correctly!")

                # Check matches
                num_matches = train_losses.get('num_matches', 0)
                if num_matches < 100:
                    print(f"⚠️  WARNING: Few matches ({num_matches:.0f})")
                elif num_matches > 200:
                    print(f"✓ Good match count ({num_matches:.0f})")

                print()

                # Log to wandb
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'train/total': train_losses['total'],
                        'train/photo': train_losses['photo'],
                        'train/repeat': train_losses['repeat'],
                        'train/desc': train_losses['desc'],
                        'train/uncert': train_losses['uncert'],
                        'train/diversity': train_losses['diversity'],
                        'train/num_matches': train_losses.get('num_matches', 0),
                        'train/desc_diversity': train_losses.get('desc_diversity', 0),  # RENAMED
                        'val/total': val_losses['total'],
                        'val/photo': val_losses['photo'],
                        'val/repeat': val_losses['repeat'],
                        'val/desc': val_losses['desc'],
                        'val/uncert': val_losses['uncert'],
                        'val/diversity': val_losses['diversity'],
                        'val/num_matches': val_losses.get('num_matches', 0),
                        'val/desc_diversity': val_losses.get('desc_diversity', 0),  # RENAMED
                        'lr': self.optimizer.param_groups[0]['lr']
                    })

                # Save best model
                if val_losses['total'] < self.best_val_loss:
                    improvement = self.best_val_loss - val_losses['total']
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pth', epoch, val_losses['total'])
                    print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})\n")

            # Step scheduler
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Save checkpoint periodically
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch}.pth', epoch, train_losses['total'])

        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)

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
    config_path = "configs/train_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer = SemanticSLAMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()