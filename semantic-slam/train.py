"""
SOTA 2025 Training Script for Semantic Keypoint Detection
Based on DeDoDe v2, ALIKE, DINO-VO, XFeat, and Keypt2Subpx (2024-2025)

Single-stage joint training of:
- Keypoint Selector (saliency + sub-pixel offset heads)
- Descriptor Refiner (4-layer MLP with residuals)

Key Training Details:
- Optimizer: AdamW with lr=1e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR to lr_min=1e-6
- Gradient clipping: max_norm=1.0
- Epochs: 40 (DeDoDe v2 shows shorter training is better)
- DINOv3 backbone is FROZEN

EXACTLY 4 LOSSES:
- Descriptor Matching (weight=10.0)
- Homographic Consistency (weight=1.0)
- Dispersity Peakiness (weight=5.0)
- Offset Consistency (weight=0.15)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional

# Optional: Weights & Biases for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging to console only")

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
from data.tum_dataset import TUMDataset
from losses.self_supervised import (
    DescriptorMatchingLoss,
    HomographicConsistencyLoss,
    DispersityPeakinessLoss,
    OffsetConsistencyLoss
)


class SemanticSLAMTrainer:
    """
    SOTA 2025 Trainer for Semantic Keypoint Detection.

    Single-stage joint training of selector + descriptor heads with:
    - Frozen DINOv3 backbone
    - Sub-pixel offset prediction
    - Exactly 4 losses (no more!)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._print_header()
        self._init_models()
        self._init_losses()
        self._init_optimizer()
        self._init_dataloaders()
        self._init_logging()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        print("\n" + "="*70)
        print("✓ INITIALIZATION COMPLETE")
        print("="*70 + "\n")

    def _print_header(self):
        """Print training configuration header"""
        print("\n" + "="*70)
        print("SEMANTIC KEYPOINT DETECTION - SOTA 2025 TRAINING")
        print("="*70)
        print("Architecture:")
        print("  • DINOv3-ViT-S backbone (FROZEN)")
        print("  • Keypoint Selector with Sub-Pixel Offset Head")
        print("  • Descriptor Refiner (4-layer MLP with residuals)")
        print("\nLosses (exactly 4):")
        print("  • Descriptor Matching (InfoNCE, weight=10.0)")
        print("  • Homographic Consistency (entropy, weight=1.0)")
        print("  • Dispersity Peakiness (weight=5.0)")
        print("  • Offset Consistency (weight=0.15)")
        print("="*70)

    def _init_models(self):
        """Initialize all models"""
        print("\n📦 Loading models...")

        # DINOv3 Backbone (FROZEN)
        self.backbone = DinoBackbone(
            model_name=self.config['model']['backbone'],
            input_size=self.config['model']['input_size'],
            freeze=True
        ).to(self.device)

        # Keypoint Selector (with offset head)
        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['selector_hidden']
        ).to(self.device)

        # Descriptor Refiner
        self.refiner = DescriptorRefiner(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['refiner_hidden'],
            output_dim=self.config['model']['descriptor_dim'],
            num_layers=self.config['model']['refiner_layers']
        ).to(self.device)

        # Count parameters
        selector_params = sum(p.numel() for p in self.selector.parameters() if p.requires_grad)
        refiner_params = sum(p.numel() for p in self.refiner.parameters() if p.requires_grad)
        total_params = selector_params + refiner_params

        print(f"  ✓ DINOv3 Backbone: {self.backbone.embed_dim}-dim features (frozen)")
        print(f"  ✓ Keypoint Selector: {selector_params/1e6:.2f}M params (saliency + offset)")
        print(f"  ✓ Descriptor Refiner: {refiner_params/1e6:.2f}M params")
        print(f"  ✓ Total trainable: {total_params/1e6:.2f}M params")

    def _init_losses(self):
        """Initialize exactly 4 losses"""
        print("\n📊 Initializing losses (exactly 4)...")

        loss_config = self.config['loss']

        # Loss 1: Descriptor Matching (InfoNCE)
        self.desc_loss = DescriptorMatchingLoss(
            temperature=loss_config['temperature']
        )

        # Loss 2: Homographic Consistency (spatial entropy)
        self.homographic_loss = HomographicConsistencyLoss(
            grid_size=self.config['model']['input_size'] // 16  # 28 for 448
        )

        # Loss 3: Dispersity Peakiness
        self.peakiness_loss = DispersityPeakinessLoss(
            target_variance=loss_config['target_variance'],
            target_mean=loss_config['target_mean']
        )

        # Loss 4: Offset Consistency
        self.offset_loss = OffsetConsistencyLoss(
            max_offset_magnitude=loss_config['max_offset']
        )

        # Loss weights
        self.loss_weights = loss_config['weights']

        print(f"  ✓ Descriptor Matching: weight={self.loss_weights['descriptor']}")
        print(f"  ✓ Homographic Consistency: weight={self.loss_weights['homographic']}")
        print(f"  ✓ Dispersity Peakiness: weight={self.loss_weights['peakiness']}")
        print(f"  ✓ Offset Consistency: weight={self.loss_weights['offset']}")

    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        print("\n⚙️ Setting up optimizer...")

        # Combine parameters from both heads
        params = list(self.selector.parameters()) + list(self.refiner.parameters())

        # AdamW optimizer (as specified)
        self.optimizer = AdamW(
            params,
            lr=float(self.config['training']['lr']),
            weight_decay=float(self.config['training']['weight_decay'])
        )

        # CosineAnnealingLR scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=float(self.config['training']['lr_min'])
        )

        print(f"  ✓ AdamW: lr={self.config['training']['lr']}, weight_decay={self.config['training']['weight_decay']}")
        print(f"  ✓ CosineAnnealingLR: T_max={self.config['training']['epochs']}, eta_min={self.config['training']['lr_min']}")

    def _init_dataloaders(self):
        """Initialize train and validation dataloaders"""
        print("\n📂 Loading datasets...")

        self.train_loader = self._create_dataloader(
            self.config['dataset']['train_sequences'],
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            is_train=True
        )

        self.val_loader = self._create_dataloader(
            self.config['dataset']['val_sequences'],
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            is_train=False
        )

        print(f"  ✓ Train: {len(self.train_loader)} batches")
        print(f"  ✓ Val: {len(self.val_loader)} batches")

    def _create_dataloader(
        self,
        sequences: list,
        batch_size: int,
        shuffle: bool,
        is_train: bool
    ) -> DataLoader:
        """Create dataloader from sequences"""
        datasets = []
        for seq in sequences:
            dataset = TUMDataset(
                dataset_root=self.config['dataset']['root'],
                sequence=seq,
                input_size=self.config['model']['input_size'],
                frame_spacing=self.config['dataset']['frame_spacing'],
                max_frames=self.config['dataset'].get('max_frames'),
                augmentation=self.config['dataset'].get('augmentation'),
                is_train=is_train
            )
            datasets.append(dataset)

        combined_dataset = ConcatDataset(datasets)

        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            drop_last=True  # Ensure consistent batch sizes
        )

    def _init_logging(self):
        """Initialize logging (wandb if available)"""
        if self.config['logging']['use_wandb'] and WANDB_AVAILABLE:
            print("\n🔗 Initializing Weights & Biases...")
            wandb.init(
                project=self.config['logging']['project'],
                name=self.config['logging']['run_name'],
                config=self.config
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

    def train(self):
        """Main training loop"""
        print("\n🚀 Starting training...\n")

        epochs = self.config['training']['epochs']

        for epoch in range(1, epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validation
            if epoch % self.config['training']['val_interval'] == 0:
                val_metrics = self.validate()
                self._print_epoch_summary(epoch, train_metrics, val_metrics)
                self._log_metrics(epoch, train_metrics, val_metrics)

                # Save best model
                if val_metrics['total'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics['total'])
                    print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})\n")

            # Update learning rate
            self.scheduler.step()

            # Save periodic checkpoint
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, train_metrics['total'])

        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print(f"✓ Best validation loss: {self.best_val_loss:.4f}")
        print("="*70)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.selector.train()
        self.refiner.train()

        # Metrics accumulators
        total_losses = {
            'total': 0.0,
            'descriptor': 0.0,
            'homographic': 0.0,
            'peakiness': 0.0,
            'offset': 0.0
        }
        metrics = {
            'num_matches': [],
            'mean_saliency': [],
            'saliency_variance': [],
            'mean_offset_magnitude': []
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb1 = batch['rgb1'].to(self.device)
            rgb2 = batch['rgb2'].to(self.device)

            # Forward pass
            loss, loss_components, batch_metrics = self._forward_pass(rgb1, rgb2)

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf at batch {batch_idx}, skipping...")
                continue

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (max_norm=1.0 as specified)
            torch.nn.utils.clip_grad_norm_(
                list(self.selector.parameters()) + list(self.refiner.parameters()),
                max_norm=self.config['training']['grad_clip']
            )

            self.optimizer.step()

            # Accumulate metrics
            for key in total_losses:
                total_losses[key] += loss_components.get(key, 0.0)

            for key in metrics:
                if key in batch_metrics:
                    metrics[key].append(batch_metrics[key])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'desc': f"{loss_components['descriptor']:.3f}",
                'offset': f"{batch_metrics.get('mean_offset_magnitude', 0):.3f}",
                'matches': f"{batch_metrics.get('num_matches', 0):.0f}"
            })

            self.global_step += 1

            # Periodic wandb logging
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/descriptor': loss_components['descriptor'],
                    'batch/homographic': loss_components['homographic'],
                    'batch/peakiness': loss_components['peakiness'],
                    'batch/offset': loss_components['offset'],
                    'batch/matches': batch_metrics.get('num_matches', 0),
                    'batch/offset_magnitude': batch_metrics.get('mean_offset_magnitude', 0),
                    'step': self.global_step
                })

        # Average losses
        n = len(self.train_loader)
        for key in total_losses:
            total_losses[key] /= n

        # Average metrics
        for key in metrics:
            if len(metrics[key]) > 0:
                total_losses[key] = np.mean(metrics[key])

        return total_losses

    def _forward_pass(
        self,
        rgb1: torch.Tensor,
        rgb2: torch.Tensor
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Complete forward pass through all components.

        Returns:
            loss: Total weighted loss
            loss_components: Dictionary of individual losses
            metrics: Dictionary of batch metrics
        """
        # Step 1: Extract frozen DINOv3 features
        with torch.no_grad():
            feat1 = self.backbone(rgb1)  # (B, H, W, C)
            feat2 = self.backbone(rgb2)

        # Step 2: Keypoint selection (saliency + offsets)
        saliency1, offset1 = self.selector(feat1)
        saliency2, offset2 = self.selector(feat2)

        # Step 3: Select keypoints with sub-pixel refinement
        num_kpts = self.config['model']['num_keypoints']
        kpts1, scores1, kpt_offsets1 = self.selector.select_keypoints(
            saliency1, offset1, num_keypoints=num_kpts
        )
        kpts2, scores2, kpt_offsets2 = self.selector.select_keypoints(
            saliency2, offset2, num_keypoints=num_kpts
        )

        # Step 4: Extract DINOv3 features at refined keypoint locations
        feat_at_kpts1 = self.backbone.extract_at_keypoints(feat1, kpts1)
        feat_at_kpts2 = self.backbone.extract_at_keypoints(feat2, kpts2)

        # Step 5: Refine descriptors
        desc1 = self.refiner(feat_at_kpts1)  # (B, N, 128)
        desc2 = self.refiner(feat_at_kpts2)

        # Step 6: Find mutual nearest neighbor matches
        matches = self._find_mutual_nn_matches(desc1, desc2)

        # Step 7: Compute all 4 losses
        # Loss 1: Descriptor matching
        loss_desc = self.desc_loss(desc1, desc2, matches)

        # Loss 2: Homographic consistency (spatial entropy)
        loss_homo = (self.homographic_loss(saliency1) +
                    self.homographic_loss(saliency2)) / 2

        # Loss 3: Dispersity peakiness
        loss_peak = (self.peakiness_loss(saliency1, kpts1) +
                    self.peakiness_loss(saliency2, kpts2)) / 2

        # Loss 4: Offset consistency
        # Need integer keypoint coordinates for offset extraction (detached)
        kpts1_int = kpts1.detach().floor().long()
        kpts2_int = kpts2.detach().floor().long()
        loss_offset = self.offset_loss(offset1, offset2, matches, kpts1_int, kpts2_int)

        # Handle NaN losses
        loss_desc = self._safe_loss(loss_desc, 0.1)
        loss_homo = self._safe_loss(loss_homo, 0.0)
        loss_peak = self._safe_loss(loss_peak, 0.0)
        loss_offset = self._safe_loss(loss_offset, 0.0)

        # Weighted sum
        w = self.loss_weights
        total_loss = (
            w['descriptor'] * loss_desc +
            w['homographic'] * loss_homo +
            w['peakiness'] * loss_peak +
            w['offset'] * loss_offset
        )

        # Collect metrics
        loss_components = {
            'total': total_loss.item(),
            'descriptor': loss_desc.item(),
            'homographic': loss_homo.item(),
            'peakiness': loss_peak.item(),
            'offset': loss_offset.item()
        }

        # Compute additional metrics
        sal_np = saliency1.detach().cpu().numpy()
        offset_magnitude = torch.sqrt((kpt_offsets1 ** 2).sum(dim=-1)).mean().item()

        batch_metrics = {
            'num_matches': matches.shape[1] if len(matches.shape) > 1 else 0,
            'mean_saliency': float(np.mean(sal_np)),
            'saliency_variance': float(np.var(sal_np)),
            'mean_offset_magnitude': offset_magnitude
        }

        return total_loss, loss_components, batch_metrics

    def _find_mutual_nn_matches(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor
    ) -> torch.Tensor:
        """
        Find mutual nearest neighbor matches between descriptors.

        Args:
            desc1: (B, N, D) descriptors from frame 1
            desc2: (B, N, D) descriptors from frame 2

        Returns:
            matches: (B, M, 2) where matches[b, i] = [idx1, idx2]
        """
        B, N, D = desc1.shape
        device = desc1.device

        matches_list = []

        for b in range(B):
            # Compute similarity matrix
            sim_matrix = torch.mm(desc1[b], desc2[b].t())  # (N, N)

            # Forward nearest neighbors (1 → 2)
            nn12 = sim_matrix.argmax(dim=1)  # (N,)

            # Backward nearest neighbors (2 → 1)
            nn21 = sim_matrix.argmax(dim=0)  # (N,)

            # Mutual nearest neighbors
            indices = torch.arange(N, device=device)
            mutual_mask = nn21[nn12] == indices

            # Get matched indices
            idx1 = torch.nonzero(mutual_mask).squeeze(1)
            idx2 = nn12[idx1]

            if len(idx1) > 0:
                matches_b = torch.stack([idx1, idx2], dim=1)  # (M, 2)
            else:
                matches_b = torch.zeros(0, 2, device=device, dtype=torch.long)

            matches_list.append(matches_b)

        # Pad to same length
        max_matches = max(m.shape[0] for m in matches_list)
        if max_matches == 0:
            return torch.zeros(B, 1, 2, device=device, dtype=torch.long)

        padded = []
        for m in matches_list:
            if m.shape[0] < max_matches:
                pad = torch.zeros(max_matches - m.shape[0], 2, device=device, dtype=torch.long)
                m = torch.cat([m, pad], dim=0)
            padded.append(m)

        return torch.stack(padded, dim=0)

    def _safe_loss(self, loss: torch.Tensor, default: float) -> torch.Tensor:
        """Return default if loss is NaN/Inf"""
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(default, device=loss.device, requires_grad=True)
        return loss

    def validate(self) -> Dict[str, float]:
        """Validation pass"""
        self.selector.eval()
        self.refiner.eval()

        total_losses = {
            'total': 0.0,
            'descriptor': 0.0,
            'homographic': 0.0,
            'peakiness': 0.0,
            'offset': 0.0
        }
        metrics = {
            'num_matches': [],
            'mean_saliency': [],
            'saliency_variance': [],
            'mean_offset_magnitude': []
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                rgb1 = batch['rgb1'].to(self.device)
                rgb2 = batch['rgb2'].to(self.device)

                loss, loss_components, batch_metrics = self._forward_pass(rgb1, rgb2)

                for key in total_losses:
                    total_losses[key] += loss_components.get(key, 0.0)

                for key in metrics:
                    if key in batch_metrics:
                        metrics[key].append(batch_metrics[key])

        # Average
        n = len(self.val_loader)
        for key in total_losses:
            total_losses[key] /= n

        for key in metrics:
            if len(metrics[key]) > 0:
                total_losses[key] = np.mean(metrics[key])

        return total_losses

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Print epoch summary"""
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{self.config['training']['epochs']}")
        print(f"{'='*70}")
        print(f"{'Metric':<25} {'Train':>12} {'Val':>12}")
        print(f"{'-'*70}")
        print(f"{'Total Loss':<25} {train_metrics['total']:>12.4f} {val_metrics['total']:>12.4f}")
        print(f"{'  Descriptor (w=10.0)':<25} {train_metrics['descriptor']:>12.4f} {val_metrics['descriptor']:>12.4f}")
        print(f"{'  Homographic (w=1.0)':<25} {train_metrics['homographic']:>12.4f} {val_metrics['homographic']:>12.4f}")
        print(f"{'  Peakiness (w=5.0)':<25} {train_metrics['peakiness']:>12.4f} {val_metrics['peakiness']:>12.4f}")
        print(f"{'  Offset (w=0.15)':<25} {train_metrics['offset']:>12.4f} {val_metrics['offset']:>12.4f}")
        print(f"{'-'*70}")
        print(f"{'Matches':<25} {train_metrics.get('num_matches', 0):>12.1f} {val_metrics.get('num_matches', 0):>12.1f}")
        print(f"{'Mean Saliency':<25} {train_metrics.get('mean_saliency', 0):>12.3f} {val_metrics.get('mean_saliency', 0):>12.3f}")
        print(f"{'Saliency Variance':<25} {train_metrics.get('saliency_variance', 0):>12.4f} {val_metrics.get('saliency_variance', 0):>12.4f}")
        print(f"{'Mean Offset Mag.':<25} {train_metrics.get('mean_offset_magnitude', 0):>12.3f} {val_metrics.get('mean_offset_magnitude', 0):>12.3f}")
        print(f"{'='*70}")

        # Quality checks
        offset_mag = val_metrics.get('mean_offset_magnitude', 0)
        if offset_mag < 0.5:
            print("✅ Good offset magnitude (<0.5) - sub-pixel accuracy achieved!")
        else:
            print("⚠️ Offset magnitude high - may need more training")

        sal_var = val_metrics.get('saliency_variance', 0)
        if 0.15 <= sal_var <= 0.30:
            print("✅ Good saliency variance - learning peaked distribution!")
        else:
            print("ℹ️ Saliency variance outside target range [0.15, 0.30]")

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics to wandb"""
        if not self.use_wandb:
            return

        wandb.log({
            'epoch': epoch,
            'train/total': train_metrics['total'],
            'train/descriptor': train_metrics['descriptor'],
            'train/homographic': train_metrics['homographic'],
            'train/peakiness': train_metrics['peakiness'],
            'train/offset': train_metrics['offset'],
            'val/total': val_metrics['total'],
            'val/descriptor': val_metrics['descriptor'],
            'val/homographic': val_metrics['homographic'],
            'val/peakiness': val_metrics['peakiness'],
            'val/offset': val_metrics['offset'],
            'val/matches': val_metrics.get('num_matches', 0),
            'val/offset_magnitude': val_metrics.get('mean_offset_magnitude', 0),
            'val/saliency_variance': val_metrics.get('saliency_variance', 0),
            'lr': self.scheduler.get_last_lr()[0]
        })

    def save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save checkpoint"""
        save_dir = Path(self.config['training']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'selector_state_dict': self.selector.state_dict(),
            'refiner_state_dict': self.refiner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }

        torch.save(checkpoint, save_dir / filename)
        print(f"  ✓ Saved checkpoint: {save_dir / filename}")


def main():
    """Main entry point"""
    config_path = "configs/train_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer = SemanticSLAMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
