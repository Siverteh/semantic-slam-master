"""
Training Script - DESCRIPTOR QUALITY FOCUSED
Key improvements:
1. Proper descriptor variance loss
2. All loss weights from config (including edge/sparsity!)
3. Better monitoring of descriptor quality
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

from data.tum_dataset import TUMDataset
from losses.self_supervised import (
    DescriptorMatchingLoss,
    DescriptorVarianceLoss,  # NEW
    RepeatabilityLoss,
    PeakinessLoss,
    ActivationLoss,
    EdgeAwarenessLoss,
    SpatialSparsityLoss
)


class SemanticSLAMTrainer:
    """Trainer with improved descriptor quality focus"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "="*70)
        print("SEMANTIC SLAM TRAINING - DESCRIPTOR QUALITY FOCUSED")
        print("="*70)
        print("Key improvements:")
        print("  ✓ Stronger descriptor loss (weight: 8.0)")
        print("  ✓ Descriptor variance regularization")
        print("  ✓ Proper edge/sparsity weights")
        print("  ✓ Better temperature for InfoNCE")
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

        # Count parameters
        selector_params = sum(p.numel() for p in self.selector.parameters() if p.requires_grad)
        refiner_params = sum(p.numel() for p in self.refiner.parameters() if p.requires_grad)
        total_params = selector_params + refiner_params

        print(f"  ✓ Keypoint Selector:  {selector_params/1e6:.2f}M params")
        print(f"  ✓ Descriptor Refiner: {refiner_params/1e6:.2f}M params (improved)")
        print(f"  ✓ Total trainable:    {total_params/1e6:.2f}M params")

        # Initialize losses
        print("\n📊 Initializing losses...")
        self.desc_loss = DescriptorMatchingLoss(
            temperature=config['loss']['desc_temperature'],
            num_negatives=config['loss'].get('desc_negatives', 40)
        )
        self.variance_loss = DescriptorVarianceLoss(
            min_variance=0.005  # FIXED: Correct target for 128-dim descriptors
        )
        self.repeat_loss = RepeatabilityLoss(
            distance_threshold=config['loss']['repeat_threshold']
        )
        self.peakiness_loss = PeakinessLoss(
            target_variance=config['loss']['target_variance']
        )
        self.activation_loss = ActivationLoss(
            target_mean=config['loss']['sparsity_target']
        )
        self.edge_loss = EdgeAwarenessLoss(
            edge_threshold=config['loss']['edge_threshold']
        ).to(self.device)
        self.sparsity_loss = SpatialSparsityLoss(
            sparsity_target=config['loss']['sparsity_target'],
            penalty_weight=config['loss']['sparsity_penalty']
        ).to(self.device)

        # Loss weights (NOW ALL FROM CONFIG!)
        self.loss_weights = config['loss']['weights']
        print(f"  ✓ Loss weights: {self.loss_weights}")
        print(f"  ✓ Descriptor loss weight: {self.loss_weights['desc']} (STRONG!)")
        print(f"  ✓ Variance loss weight: {self.loss_weights['variance']}")

        # Optimizer
        self.optimizer = AdamW(
            list(self.selector.parameters()) + list(self.refiner.parameters()),
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay'])
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=float(config['training']['lr_min'])
        )

        # Datasets
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

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

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
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.selector.train()
        self.refiner.train()

        total_loss = 0.0
        losses_dict = {
            'desc': 0.0,
            'variance': 0.0,
            'repeat': 0.0,
            'peakiness': 0.0,
            'activation': 0.0,
            'edge': 0.0,
            'sparsity': 0.0
        }

        # Metrics
        metrics = {
            'num_matches': [],
            'mean_saliency': [],
            'max_saliency': [],
            'saliency_variance': [],
            'descriptor_variance': []  # NEW: monitor descriptor quality
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            rgb1 = batch['rgb1'].to(self.device)
            rgb2 = batch['rgb2'].to(self.device)

            # Forward pass
            loss, loss_components, batch_metrics = self._forward_pass(rgb1, rgb2)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf detected at batch {batch_idx}, skipping...")
                continue

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.selector.parameters()) + list(self.refiner.parameters()),
                max_norm=self.config['training']['grad_clip']
            )
            self.optimizer.step()

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
                'var': f"{loss_components['variance']:.3f}",
                'matches': f"{batch_metrics.get('num_matches', 0):.0f}",
                'desc_var': f"{batch_metrics.get('descriptor_variance', 0):.3f}"
            })

            self.global_step += 1

            # Log to wandb
            if self.config['logging']['use_wandb'] and batch_idx % 50 == 0:
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/desc': loss_components['desc'],
                    'batch/variance': loss_components['variance'],
                    'batch/repeat': loss_components['repeat'],
                    'batch/edge': loss_components['edge'],
                    'batch/matches': batch_metrics.get('num_matches', 0),
                    'batch/descriptor_variance': batch_metrics.get('descriptor_variance', 0),
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

    def _forward_pass(
        self,
        rgb1: torch.Tensor,
        rgb2: torch.Tensor
    ) -> tuple:
        """Complete forward pass with ALL losses"""

        # Extract DINOv3 features
        with torch.no_grad():
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)

        # Keypoint selection
        saliency1 = self.selector(feat1)
        saliency2 = self.selector(feat2)

        # Safety checks
        if torch.isnan(saliency1).any():
            saliency1 = torch.sigmoid(torch.zeros_like(saliency1))
        if torch.isnan(saliency2).any():
            saliency2 = torch.sigmoid(torch.zeros_like(saliency2))

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
        desc1 = self.refiner(feat_at_kpts1)
        desc2 = self.refiner(feat_at_kpts2)

        # ============ COMPUTE LOSSES ============

        # 1. Descriptor matching loss (STRONG WEIGHT!)
        matches = self._find_matches(desc1, desc2)
        loss_desc = self.desc_loss(desc1, desc2, matches)

        # 2. NEW: Descriptor variance loss (prevent collapse!)
        loss_variance = self.variance_loss(desc1)

        # 3. Repeatability loss
        loss_repeat = self.repeat_loss(saliency1, saliency2)

        # 4. Peakiness loss
        loss_peakiness = self.peakiness_loss(saliency1)

        # 5. Activation loss
        loss_activation = self.activation_loss(saliency1)

        # 6. Edge awareness loss
        loss_edge = self.edge_loss(saliency1, rgb1)

        # 7. Spatial sparsity loss
        loss_sparsity = self.sparsity_loss(saliency1)

        # Replace NaN with fallback
        if torch.isnan(loss_desc):
            loss_desc = torch.tensor(0.1, device=loss_desc.device, requires_grad=True)
        if torch.isnan(loss_variance):
            loss_variance = torch.tensor(0.0, device=loss_variance.device, requires_grad=True)
        if torch.isnan(loss_repeat):
            loss_repeat = torch.tensor(0.0, device=loss_repeat.device, requires_grad=True)
        if torch.isnan(loss_peakiness):
            loss_peakiness = torch.tensor(0.0, device=loss_peakiness.device, requires_grad=True)
        if torch.isnan(loss_activation):
            loss_activation = torch.tensor(0.0, device=loss_activation.device, requires_grad=True)
        if torch.isnan(loss_edge):
            loss_edge = torch.tensor(0.0, device=loss_edge.device, requires_grad=True)
        if torch.isnan(loss_sparsity):
            loss_sparsity = torch.tensor(0.0, device=loss_sparsity.device, requires_grad=True)

        # Weighted combination (ALL FROM CONFIG!)
        w = self.loss_weights
        total_loss = (
            w['desc'] * loss_desc +
            w['variance'] * loss_variance +
            w['repeat'] * loss_repeat +
            w['peakiness'] * loss_peakiness +
            w['activation'] * loss_activation +
            w['edge'] * loss_edge +
            w['sparsity'] * loss_sparsity
        )

        loss_components = {
            'desc': loss_desc.item(),
            'variance': loss_variance.item(),
            'repeat': loss_repeat.item(),
            'peakiness': loss_peakiness.item(),
            'activation': loss_activation.item(),
            'edge': loss_edge.item(),
            'sparsity': loss_sparsity.item()
        }

        # Compute statistics
        sal_np = saliency1.detach().cpu().numpy()
        desc_np = desc1.detach().cpu().numpy()

        # Descriptor variance (higher = better)
        desc_variance = np.var(desc_np)

        batch_metrics = {
            'num_matches': matches.shape[1],
            'mean_saliency': float(np.mean(sal_np)),
            'max_saliency': float(np.max(sal_np)),
            'saliency_variance': float(np.var(sal_np)),
            'descriptor_variance': float(desc_variance)  # NEW
        }

        return total_loss, loss_components, batch_metrics

    def _find_matches(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor
    ) -> torch.Tensor:
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

    def validate(self) -> Dict[str, float]:
        """Validation pass"""
        self.selector.eval()
        self.refiner.eval()

        total_loss = 0.0
        losses_dict = {
            'desc': 0.0,
            'variance': 0.0,
            'repeat': 0.0,
            'peakiness': 0.0,
            'activation': 0.0,
            'edge': 0.0,
            'sparsity': 0.0
        }
        metrics = {
            'num_matches': [],
            'mean_saliency': [],
            'max_saliency': [],
            'saliency_variance': [],
            'descriptor_variance': []
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                rgb1 = batch['rgb1'].to(self.device)
                rgb2 = batch['rgb2'].to(self.device)

                loss, loss_components, batch_metrics = self._forward_pass(rgb1, rgb2)

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

    def train(self):
        """Main training loop"""
        print("🚀 Starting training...\n")

        for epoch in range(1, self.config['training']['epochs'] + 1):
            train_losses = self.train_epoch(epoch)

            if epoch % self.config['training']['val_interval'] == 0:
                val_losses = self.validate()

                # Print summary
                print(f"\n{'='*70}")
                print(f"EPOCH {epoch}/{self.config['training']['epochs']}")
                print(f"{'='*70}")
                print(f"{'Metric':<25} {'Train':>12} {'Val':>12}")
                print(f"{'-'*70}")
                print(f"{'Total Loss':<25} {train_losses['total']:>12.4f} {val_losses['total']:>12.4f}")
                print(f"{'  Descriptor':<25} {train_losses['desc']:>12.4f} {val_losses['desc']:>12.4f}")
                print(f"{'  Variance':<25} {train_losses['variance']:>12.4f} {val_losses['variance']:>12.4f}")
                print(f"{'  Repeatability':<25} {train_losses['repeat']:>12.4f} {val_losses['repeat']:>12.4f}")
                print(f"{'  Peakiness':<25} {train_losses['peakiness']:>12.4f} {val_losses['peakiness']:>12.4f}")
                print(f"{'  Activation':<25} {train_losses['activation']:>12.4f} {val_losses['activation']:>12.4f}")
                print(f"{'  Edge Awareness':<25} {train_losses['edge']:>12.4f} {val_losses['edge']:>12.4f}")
                print(f"{'  Spatial Sparsity':<25} {train_losses['sparsity']:>12.4f} {val_losses['sparsity']:>12.4f}")
                print(f"{'-'*70}")
                print(f"{'Matches':<25} {train_losses.get('num_matches', 0):>12.1f} {val_losses.get('num_matches', 0):>12.1f}")
                print(f"{'Mean Saliency':<25} {train_losses.get('mean_saliency', 0):>12.3f} {val_losses.get('mean_saliency', 0):>12.3f}")
                print(f"{'Max Saliency':<25} {train_losses.get('max_saliency', 0):>12.3f} {val_losses.get('max_saliency', 0):>12.3f}")
                print(f"{'Saliency Variance':<25} {train_losses.get('saliency_variance', 0):>12.3f} {val_losses.get('saliency_variance', 0):>12.3f}")
                print(f"{'Descriptor Variance':<25} {train_losses.get('descriptor_variance', 0):>12.3f} {val_losses.get('descriptor_variance', 0):>12.3f}")
                print(f"{'='*70}\n")

                # Check descriptor quality
                desc_var = val_losses.get('descriptor_variance', 0)
                if desc_var > 0.006:  # FIXED: Correct threshold for 128-dim descriptors
                    print("✅ Good descriptor variance! Descriptors are diverse.")
                elif desc_var < 0.003:  # FIXED: Collapse threshold
                    print("⚠️ Low descriptor variance - possible collapse! Increase variance loss weight.")
                else:
                    print("ℹ️  Descriptor variance is acceptable but could be higher.")

                # Log to wandb
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'train/total': train_losses['total'],
                        'train/desc': train_losses['desc'],
                        'train/variance': train_losses['variance'],
                        'train/repeat': train_losses['repeat'],
                        'train/peakiness': train_losses['peakiness'],
                        'train/activation': train_losses['activation'],
                        'train/edge': train_losses['edge'],
                        'train/sparsity': train_losses['sparsity'],
                        'val/total': val_losses['total'],
                        'val/desc': val_losses['desc'],
                        'val/variance': val_losses['variance'],
                        'val/repeat': val_losses['repeat'],
                        'val/peakiness': val_losses['peakiness'],
                        'val/activation': val_losses['activation'],
                        'val/edge': val_losses['edge'],
                        'val/sparsity': val_losses['sparsity'],
                        'val/descriptor_variance': val_losses.get('descriptor_variance', 0)
                    })

                # Save best
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pth', epoch, val_losses['total'])
                    print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})\n")

            self.scheduler.step()

        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)

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


def main():
    config_path = "configs/train_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer = SemanticSLAMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()