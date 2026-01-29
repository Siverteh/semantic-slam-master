"""
Two-Stage Training Script (R2D2-style)

Stage 1 (epochs 1-40): Train detector + descriptor
  - Learns to detect stable keypoints
  - Learns discriminative descriptors
  - No uncertainty estimation

Stage 2 (epochs 41-60): Train uncertainty (detector + descriptor FROZEN)
  - Learns to predict reliability
  - No gradient flow to detector/descriptor

Reference: R2D2 (CVPR 2019) Section 3.3
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

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
from models.uncertainty_estimator import UncertaintyEstimator

from data.tum_dataset import TUMDataset
from losses.self_supervised import (
    DescriptorMatchingLoss,
    HomographicConsistencyLoss,  # FIXED: SuperPoint-style
    DispersityPeakinessLoss,  # FIXED: ALIKE-inspired
    OffsetConsistencyLoss,
    UncertaintyCalibrationLoss
)


class SemanticSLAMTrainer:
    """
    Two-stage trainer following R2D2 methodology.

    Stage 1: Detector + Descriptor learning
    Stage 2: Uncertainty/Reliability learning (frozen detector + descriptor)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "="*70)
        print("SEMANTIC SLAM - TWO-STAGE TRAINING (R2D2-style)")
        print("="*70)
        print("Stage 1: Detector + Descriptor (self-supervised)")
        print("Stage 2: Uncertainty Estimation (frozen detector + descriptor)")
        print("="*70 + "\n")

        # Initialize models
        print("📦 Loading models...")
        self.backbone = DinoBackbone(
            model_name=config['model']['backbone'],
            input_size=config['model']['input_size'],
            freeze=True
        ).to(self.device)

        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            hidden_dim=config['model']['selector_hidden']
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
            hidden_dim=int(config['model'].get('estimator_hidden', 128))
        ).to(self.device)

        # Parameter counts
        selector_params = sum(p.numel() for p in self.selector.parameters() if p.requires_grad)
        refiner_params = sum(p.numel() for p in self.refiner.parameters() if p.requires_grad)
        estimator_params = sum(p.numel() for p in self.estimator.parameters() if p.requires_grad)

        print(f"  ✓ Keypoint Selector:  {selector_params/1e6:.2f}M params")
        print(f"  ✓ Descriptor Refiner: {refiner_params/1e6:.2f}M params")
        print(f"  ✓ Uncertainty Est.:   {estimator_params/1e6:.2f}M params")

        # Initialize losses (only essential ones)
        print("\n📊 Initializing losses (4 essential, SOTA-informed)...")
        self.desc_loss = DescriptorMatchingLoss(
            temperature=float(config['loss'].get('desc_temperature', 0.07))
        )
        self.homographic_loss = HomographicConsistencyLoss()  # FIXED: SuperPoint-style
        self.peakiness_loss = DispersityPeakinessLoss(  # FIXED: ALIKE-inspired
            target_variance=float(config['loss'].get('target_variance', 0.22)),
            sparsity_target=float(config['loss'].get('sparsity_target', 0.30))  # Lowered
        )
        self.offset_loss = OffsetConsistencyLoss()
        self.uncertainty_loss = UncertaintyCalibrationLoss()

        #  Loss weights
        self.loss_weights = config['loss']['weights']
        print(f"  ✓ Stage 1 losses: descriptor, homographic, peakiness, offset")
        print(f"  ✓ Weights: desc={self.loss_weights['desc']}, " +
              f"homographic={self.loss_weights['homographic']}, " +
              f"peakiness={self.loss_weights['peakiness']}, " +
              f"offset={self.loss_weights['offset']}")
        print(f"  ✓ Stage 2 loss: uncertainty (detector + descriptor frozen)")

        # Stage 1 optimizer (selector + refiner)
        self.optimizer_stage1 = AdamW(
            list(self.selector.parameters()) + list(self.refiner.parameters()),
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay'])
        )

        # Stage 2 optimizer (estimator only)
        self.optimizer_stage2 = AdamW(
            self.estimator.parameters(),
            lr=float(config['training'].get('lr_stage2', float(config['training']['lr']) * 0.1)),
            weight_decay=float(config['training']['weight_decay'])
        )

        # Schedulers
        stage1_epochs = int(config['training']['stage1_epochs'])
        stage2_epochs = int(config['training']['stage2_epochs'])

        self.scheduler_stage1 = CosineAnnealingLR(
            self.optimizer_stage1,
            T_max=stage1_epochs,
            eta_min=float(config['training']['lr_min'])
        )

        self.scheduler_stage2 = CosineAnnealingLR(
            self.optimizer_stage2,
            T_max=stage2_epochs,
            eta_min=float(config['training']['lr_min'])
        )

        # Datasets
        print("\n📂 Loading datasets...")
        self.train_loader = self._create_dataloader(
            config['dataset']['train_sequences'],
            batch_size=int(config['training']['batch_size']),
            shuffle=True,
            is_train=True
        )

        self.val_loader = self._create_dataloader(
            config['dataset']['val_sequences'],
            batch_size=int(config['training']['batch_size']),
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

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.current_stage = 1

        print("\n" + "="*70)
        print("✓ INITIALIZATION COMPLETE")
        print("="*70 + "\n")

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
            num_workers=int(self.config['training']['num_workers']),
            pin_memory=True
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch (stage-aware)"""

        if self.current_stage == 1:
            return self._train_epoch_stage1(epoch)
        else:
            return self._train_epoch_stage2(epoch)

    def _train_epoch_stage1(self, epoch: int) -> Dict[str, float]:
        """Stage 1: Train detector + descriptor"""
        self.selector.train()
        self.refiner.train()

        total_loss = 0.0
        losses_dict = {'desc': 0.0, 'homographic': 0.0, 'peakiness': 0.0, 'offset': 0.0}
        metrics = {'num_matches': [], 'mean_saliency': [], 'saliency_variance': []}

        pbar = tqdm(self.train_loader, desc=f"[Stage 1] Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            rgb1 = batch['rgb1'].to(self.device)
            rgb2 = batch['rgb2'].to(self.device)

            # Forward pass (stage 1)
            loss, loss_components, batch_metrics = self._forward_stage1(rgb1, rgb2)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf at batch {batch_idx}, skipping...")
                continue

            # Backward
            self.optimizer_stage1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.selector.parameters()) + list(self.refiner.parameters()),
                max_norm=float(self.config['training']['grad_clip'])
            )
            self.optimizer_stage1.step()

            # Accumulate
            total_loss += loss.item()
            for key in losses_dict:
                losses_dict[key] += loss_components[key]

            for key in metrics:
                if key in batch_metrics:
                    metrics[key].append(batch_metrics[key])

            # Update progress
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'desc': f"{loss_components['desc']:.3f}",
                'matches': f"{batch_metrics.get('num_matches', 0):.0f}"
            })

            self.global_step += 1

            # Log to wandb
            if self.config['logging']['use_wandb'] and batch_idx % 50 == 0:
                wandb.log({
                    'stage': 1,
                    'batch/loss': loss.item(),
                    'batch/desc': loss_components['desc'],
                    'batch/homographic': loss_components['homographic'],
                    'batch/peakiness': loss_components['peakiness'],
                    'batch/offset': loss_components['offset'],
                    'batch/matches': batch_metrics.get('num_matches', 0),
                    'step': self.global_step
                })

        # Average
        n = len(self.train_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        avg_metrics = {}
        for key in metrics:
            if len(metrics[key]) > 0:
                avg_metrics[key] = np.mean(metrics[key])

        return {'total': avg_loss, **losses_dict, **avg_metrics}

    def _train_epoch_stage2(self, epoch: int) -> Dict[str, float]:
        """Stage 2: Train uncertainty (detector + descriptor frozen)"""
        # Ensure frozen
        self.selector.eval()
        self.refiner.eval()
        self.estimator.train()

        total_loss = 0.0
        losses_dict = {'uncertainty': 0.0}
        metrics = {'mean_confidence': [], 'mean_error': []}

        pbar = tqdm(self.train_loader, desc=f"[Stage 2] Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            rgb1 = batch['rgb1'].to(self.device)
            rgb2 = batch['rgb2'].to(self.device)

            # Forward pass (stage 2)
            loss, loss_components, batch_metrics = self._forward_stage2(rgb1, rgb2)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf at batch {batch_idx}, skipping...")
                continue

            # Backward
            self.optimizer_stage2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.estimator.parameters(),
                max_norm=float(self.config['training']['grad_clip'])
            )
            self.optimizer_stage2.step()

            # Accumulate
            total_loss += loss.item()
            losses_dict['uncertainty'] += loss_components['uncertainty']

            for key in metrics:
                if key in batch_metrics:
                    metrics[key].append(batch_metrics[key])

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'conf': f"{batch_metrics.get('mean_confidence', 0):.3f}"
            })

            self.global_step += 1

            if self.config['logging']['use_wandb'] and batch_idx % 50 == 0:
                wandb.log({
                    'stage': 2,
                    'batch/loss': loss.item(),
                    'batch/uncertainty': loss_components['uncertainty'],
                    'batch/confidence': batch_metrics.get('mean_confidence', 0),
                    'step': self.global_step
                })

        n = len(self.train_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        avg_metrics = {}
        for key in metrics:
            if len(metrics[key]) > 0:
                avg_metrics[key] = np.mean(metrics[key])

        return {'total': avg_loss, **losses_dict, **avg_metrics}

    def _forward_stage1(self, rgb1: torch.Tensor, rgb2: torch.Tensor) -> tuple:
        """Stage 1 forward: detector + descriptor losses"""

        # Extract DINOv3 features (frozen)
        with torch.no_grad():
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)

        # Keypoint detection (NEW: with sub-pixel offsets)
        saliency1, offsets1 = self.selector(feat1)
        saliency2, offsets2 = self.selector(feat2)

        # Select keypoints with sub-pixel refinement
        kpts1, scores1 = self.selector.select_keypoints(
            saliency1, offsets1,
            num_keypoints=self.config['model']['num_keypoints']
        )
        kpts2, scores2 = self.selector.select_keypoints(
            saliency2, offsets2,
            num_keypoints=self.config['model']['num_keypoints']
        )

        # Extract features at keypoints
        feat_at_kpts1 = self.backbone.extract_at_keypoints(feat1, kpts1)
        feat_at_kpts2 = self.backbone.extract_at_keypoints(feat2, kpts2)

        # Refine descriptors
        desc1 = self.refiner(feat_at_kpts1)
        desc2 = self.refiner(feat_at_kpts2)

        # ============ COMPUTE STAGE 1 LOSSES ============

        # Find matches
        matches = self._find_matches(desc1, desc2)

        # 1. Descriptor matching (most important)
        loss_desc = self.desc_loss(desc1, desc2, matches)

        # 2. FIXED: Homographic consistency (SuperPoint-style)
        loss_homographic = self.homographic_loss(saliency1, kpts1)

        # 3. FIXED: Dispersity + Peakiness (ALIKE-inspired)
        loss_peakiness = self.peakiness_loss(saliency1, kpts1)

        # 4. Offset consistency
        loss_offset = self.offset_loss(offsets1, offsets2, kpts1, kpts2, matches)

        # Handle NaN
        if torch.isnan(loss_desc):
            loss_desc = torch.tensor(0.1, device=loss_desc.device, requires_grad=True)
        if torch.isnan(loss_homographic):
            loss_homographic = torch.tensor(0.0, device=loss_homographic.device, requires_grad=True)
        if torch.isnan(loss_peakiness):
            loss_peakiness = torch.tensor(0.0, device=loss_peakiness.device, requires_grad=True)
        if torch.isnan(loss_offset):
            loss_offset = torch.tensor(0.0, device=loss_offset.device, requires_grad=True)

        # Weighted combination
        w = self.loss_weights
        total_loss = (
            w['desc'] * loss_desc +
            w['homographic'] * loss_homographic +
            w['peakiness'] * loss_peakiness +
            w['offset'] * loss_offset
        )

        loss_components = {
            'desc': loss_desc.item(),
            'homographic': loss_homographic.item(),
            'peakiness': loss_peakiness.item(),
            'offset': loss_offset.item()
        }

        # Metrics
        sal_np = saliency1.detach().cpu().numpy()
        batch_metrics = {
            'num_matches': matches.shape[1],
            'mean_saliency': float(np.mean(sal_np)),
            'saliency_variance': float(np.var(sal_np))
        }

        return total_loss, loss_components, batch_metrics

    def _forward_stage2(self, rgb1: torch.Tensor, rgb2: torch.Tensor) -> tuple:
        """Stage 2 forward: uncertainty loss (everything else frozen)"""

        # All feature extraction is frozen
        with torch.no_grad():
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)

            saliency1, offsets1 = self.selector(feat1)
            saliency2, offsets2 = self.selector(feat2)

            kpts1, scores1 = self.selector.select_keypoints(
                saliency1, offsets1,
                num_keypoints=self.config['model']['num_keypoints']
            )
            kpts2, scores2 = self.selector.select_keypoints(
                saliency2, offsets2,
                num_keypoints=self.config['model']['num_keypoints']
            )

            feat_at_kpts1 = self.backbone.extract_at_keypoints(feat1, kpts1)
            feat_at_kpts2 = self.backbone.extract_at_keypoints(feat2, kpts2)

            desc1 = self.refiner(feat_at_kpts1)
            desc2 = self.refiner(feat_at_kpts2)

            matches = self._find_matches(desc1, desc2)

        # NOW train uncertainty estimator
        confidence1 = self.estimator(feat_at_kpts1, desc1)

        # Compute actual matching error (ground truth for uncertainty)
        actual_error = self._compute_match_error(desc1, desc2, matches)

        # Uncertainty calibration loss
        loss_uncertainty = self.uncertainty_loss(confidence1, actual_error)

        if torch.isnan(loss_uncertainty):
            loss_uncertainty = torch.tensor(0.01, device=loss_uncertainty.device, requires_grad=True)

        total_loss = loss_uncertainty

        loss_components = {
            'uncertainty': loss_uncertainty.item()
        }

        batch_metrics = {
            'mean_confidence': confidence1.mean().item(),
            'mean_error': actual_error.mean().item()
        }

        return total_loss, loss_components, batch_metrics

    def _find_matches(self, desc1: torch.Tensor, desc2: torch.Tensor) -> torch.Tensor:
        """Find mutual nearest neighbor matches"""
        B, N, D = desc1.shape
        device = desc1.device

        matches_list = []

        for b in range(B):
            sim_matrix = torch.mm(desc1[b], desc2[b].t())
            nn12 = sim_matrix.argmax(dim=1)
            nn21 = sim_matrix.argmax(dim=0)
            mutual_mask = nn21[nn12] == torch.arange(N, device=device)

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

        padded = []
        for m in matches_list:
            if m.shape[0] < max_matches:
                pad = torch.zeros(max_matches - m.shape[0], 2, device=device, dtype=torch.long)
                m = torch.cat([m, pad], dim=0)
            padded.append(m)

        return torch.stack(padded, dim=0)

    def _compute_match_error(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor,
        matches: torch.Tensor
    ) -> torch.Tensor:
        """Compute descriptor distance for matched pairs (ground truth for uncertainty)"""
        B, N, D = desc1.shape
        errors = []

        for b in range(B):
            if matches[b].shape[0] == 0:
                errors.append(torch.zeros(N, device=desc1.device))
                continue

            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            valid_mask = (idx1 < N) & (idx2 < desc2.shape[1])
            if valid_mask.sum() == 0:
                errors.append(torch.zeros(N, device=desc1.device))
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            # Descriptor distance (1 - cosine similarity)
            matched_desc1 = desc1[b, idx1]
            matched_desc2 = desc2[b, idx2]
            distances = 1.0 - (matched_desc1 * matched_desc2).sum(dim=-1)

            # Initialize error array
            error_b = torch.ones(N, device=desc1.device)  # Default: high error
            error_b[idx1] = distances  # Update with actual errors

            errors.append(error_b)

        return torch.stack(errors, dim=0)

    def validate(self) -> Dict[str, float]:
        """Validation (stage-aware)"""
        if self.current_stage == 1:
            return self._validate_stage1()
        else:
            return self._validate_stage2()

    def _validate_stage1(self) -> Dict[str, float]:
        """Stage 1 validation"""
        self.selector.eval()
        self.refiner.eval()

        total_loss = 0.0
        losses_dict = {'desc': 0.0, 'homographic': 0.0, 'peakiness': 0.0, 'offset': 0.0}
        metrics = {'num_matches': [], 'mean_saliency': []}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                rgb1 = batch['rgb1'].to(self.device)
                rgb2 = batch['rgb2'].to(self.device)

                loss, loss_components, batch_metrics = self._forward_stage1(rgb1, rgb2)

                total_loss += loss.item()
                for key in losses_dict:
                    losses_dict[key] += loss_components[key]

                for key in metrics:
                    if key in batch_metrics:
                        metrics[key].append(batch_metrics[key])

        n = len(self.val_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        avg_metrics = {}
        for key in metrics:
            if len(metrics[key]) > 0:
                avg_metrics[key] = np.mean(metrics[key])

        return {'total': avg_loss, **losses_dict, **avg_metrics}

    def _validate_stage2(self) -> Dict[str, float]:
        """Stage 2 validation"""
        self.selector.eval()
        self.refiner.eval()
        self.estimator.eval()

        total_loss = 0.0
        losses_dict = {'uncertainty': 0.0}
        metrics = {'mean_confidence': []}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                rgb1 = batch['rgb1'].to(self.device)
                rgb2 = batch['rgb2'].to(self.device)

                loss, loss_components, batch_metrics = self._forward_stage2(rgb1, rgb2)

                total_loss += loss.item()
                losses_dict['uncertainty'] += loss_components['uncertainty']
                metrics['mean_confidence'].append(batch_metrics['mean_confidence'])

        n = len(self.val_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        avg_metrics = {'mean_confidence': np.mean(metrics['mean_confidence'])}

        return {'total': avg_loss, **losses_dict, **avg_metrics}

    def train(self):
        """Main two-stage training loop"""
        print("🚀 Starting two-stage training...\n")

        stage1_epochs = self.config['training']['stage1_epochs']
        stage2_epochs = self.config['training']['stage2_epochs']
        total_epochs = stage1_epochs + stage2_epochs

        # ============ STAGE 1 ============
        print("\n" + "🔥"*35)
        print("STAGE 1: Training Detector + Descriptor")
        print("🔥"*35 + "\n")

        self.current_stage = 1

        for epoch in range(1, stage1_epochs + 1):
            train_losses = self.train_epoch(epoch)

            if epoch % int(self.config['training']['val_interval']) == 0:
                val_losses = self.validate()
                self._print_summary(epoch, stage1_epochs, train_losses, val_losses, stage=1)

                # Log
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'stage': 1,
                        'train/total': train_losses['total'],
                        'val/total': val_losses['total']
                    })

                # Save best
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best_stage1.pth', epoch, val_losses['total'])
                    print(f"✓ Saved best Stage 1 model (val_loss: {self.best_val_loss:.4f})\n")

            self.scheduler_stage1.step()

        # Save Stage 1 final
        self.save_checkpoint('stage1_final.pth', stage1_epochs, val_losses['total'])
        print(f"\n✓ Stage 1 complete. Saved final checkpoint.\n")

        # ============ STAGE 2 ============
        print("\n" + "🎯"*35)
        print("STAGE 2: Training Uncertainty (frozen detector + descriptor)")
        print("🎯"*35 + "\n")

        # Freeze selector and refiner
        for param in self.selector.parameters():
            param.requires_grad = False
        for param in self.refiner.parameters():
            param.requires_grad = False

        self.selector.eval()
        self.refiner.eval()

        print("✓ Detector and Descriptor are now FROZEN")
        print("✓ Training only Uncertainty Estimator\n")

        self.current_stage = 2
        self.best_val_loss = float('inf')  # Reset

        for epoch in range(stage1_epochs + 1, total_epochs + 1):
            train_losses = self.train_epoch(epoch)

            if epoch % int(self.config['training']['val_interval']) == 0:
                val_losses = self.validate()
                self._print_summary(
                    epoch - stage1_epochs,
                    stage2_epochs,
                    train_losses,
                    val_losses,
                    stage=2
                )

                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'stage': 2,
                        'train/uncertainty': train_losses['total'],
                        'val/uncertainty': val_losses['total']
                    })

                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pth', epoch, val_losses['total'])
                    print(f"✓ Saved best Stage 2 model (val_loss: {self.best_val_loss:.4f})\n")

            self.scheduler_stage2.step()

        print("\n" + "="*70)
        print("✓ TWO-STAGE TRAINING COMPLETE!")
        print("="*70)
        print(f"✓ Best Stage 1 checkpoint: checkpoints/best_stage1.pth")
        print(f"✓ Final model: checkpoints/best_model.pth")

    def _print_summary(
        self,
        epoch: int,
        max_epoch: int,
        train_losses: dict,
        val_losses: dict,
        stage: int
    ):
        """Print epoch summary"""
        print(f"\n{'='*70}")
        print(f"[STAGE {stage}] EPOCH {epoch}/{max_epoch}")
        print(f"{'='*70}")
        print(f"{'Metric':<25} {'Train':>12} {'Val':>12}")
        print(f"{'-'*70}")
        print(f"{'Total Loss':<25} {train_losses['total']:>12.4f} {val_losses['total']:>12.4f}")

        if stage == 1:
            print(f"{'  Descriptor':<25} {train_losses['desc']:>12.4f} {val_losses['desc']:>12.4f}")
            print(f"{'  Homographic':<25} {train_losses['homographic']:>12.4f} {val_losses['homographic']:>12.4f}")
            print(f"{'  Peakiness':<25} {train_losses['peakiness']:>12.4f} {val_losses['peakiness']:>12.4f}")
            print(f"{'  Offset':<25} {train_losses['offset']:>12.4f} {val_losses['offset']:>12.4f}")
        else:
            print(f"{'  Uncertainty':<25} {train_losses['uncertainty']:>12.4f} {val_losses['uncertainty']:>12.4f}")

        print(f"{'='*70}\n")

    def save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save checkpoint"""
        save_dir = Path(self.config['training']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'stage': self.current_stage,
            'selector_state_dict': self.selector.state_dict(),
            'refiner_state_dict': self.refiner.state_dict(),
            'estimator_state_dict': self.estimator.state_dict(),
            'config': self.config
        }

        if self.current_stage == 1:
            checkpoint['optimizer_state_dict'] = self.optimizer_stage1.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler_stage1.state_dict()
        else:
            checkpoint['optimizer_state_dict'] = self.optimizer_stage2.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler_stage2.state_dict()

        torch.save(checkpoint, save_dir / filename)


def main():
    config_path = "configs/train_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer = SemanticSLAMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()