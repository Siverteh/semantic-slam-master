"""
Training Script - FIXED
Key changes:
1. InfoNCE loss instead of AP loss
2. Geometric consistency regularization in Stage 2
3. Better matching with adaptive threshold
4. Detailed logging of keypoint quality
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
from models.geometric_cnn import GeometricCNN, FeatureFusion
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
from models.offset_refiner import OffsetRefiner

from data.tum_dataset import TUMDataset
from losses.self_supervised import (
    InfoNCELoss,  # CHANGED from APLoss
    DescriptorVarianceLoss,
    RepeatabilityLoss,
    SemanticEdgeLoss,
    GeometricConsistencyLoss  # NEW!
)
from losses.geometric_losses import (
    EpipolarConsistencyLoss,
    DepthReprojectionLoss,
    PhotometricConsistencyLoss
)


class HybridSLAMTrainer:
    """3-Stage hybrid semantic-geometric SLAM training"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "="*70)
        print("HYBRID SEMANTIC-GEOMETRIC SLAM TRAINING - FIXED")
        print("="*70)
        print("Architecture: DINOv3 (frozen) + Lightweight CNN (trained)")
        print("Losses: InfoNCE + Epipolar + Depth + Geometric Consistency")
        print("="*70 + "\n")

        self._init_models()
        self._init_losses()
        self._init_datasets()

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.current_stage = 1

        print("✓ Initialization complete\n")

    def _init_models(self):
        """Initialize all model components"""
        print("📦 Loading models...")

        self.backbone = DinoBackbone(
            model_name=self.config['model']['backbone'],
            input_size=self.config['model']['input_size'],
            freeze=True
        ).to(self.device)

        self.geometric_cnn = GeometricCNN(
            input_channels=3,
            output_channels=self.config['model']['geometric_channels']
        ).to(self.device)

        self.fusion = FeatureFusion(
            semantic_dim=self.backbone.embed_dim,
            geometric_dim=self.config['model']['geometric_channels'],
            output_dim=self.config['model']['fusion_dim']
        ).to(self.device)

        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            patch_size=self.backbone.patch_size
        ).to(self.device)

        self.refiner = DescriptorRefiner(
            input_dim=self.config['model']['fusion_dim'],
            hidden_dim=self.config['model']['refiner_hidden'],
            output_dim=self.config['model']['descriptor_dim'],
            num_layers=self.config['model']['refiner_layers']
        ).to(self.device)

        if self.config['model']['enable_offset']:
            self.offset_refiner = OffsetRefiner(
                input_dim=self.config['model']['fusion_dim'],
                hidden_dim=self.config['model']['offset_hidden']
            ).to(self.device)
        else:
            self.offset_refiner = None

        geo_params = sum(p.numel() for p in self.geometric_cnn.parameters())
        sel_params = sum(p.numel() for p in self.selector.parameters())
        ref_params = sum(p.numel() for p in self.refiner.parameters())
        fus_params = sum(p.numel() for p in self.fusion.parameters())
        total = geo_params + sel_params + ref_params + fus_params

        print(f"  ✓ Geometric CNN:     {geo_params/1e6:.2f}M params")
        print(f"  ✓ Keypoint Selector: {sel_params/1e6:.2f}M params")
        print(f"  ✓ Feature Fusion:    {fus_params/1e6:.2f}M params")
        print(f"  ✓ Descriptor Refiner:{ref_params/1e6:.2f}M params")
        print(f"  ✓ Total trainable:   {total/1e6:.2f}M params")

    def _init_losses(self):
        """Initialize loss functions"""
        print("\n📊 Initializing losses...")

        # CHANGED: InfoNCE instead of AP
        self.infonce_loss = InfoNCELoss(temperature=0.07)
        self.variance_loss = DescriptorVarianceLoss(
            min_variance=self.config['loss']['min_variance']
        )

        # Geometric losses
        self.epipolar_loss = EpipolarConsistencyLoss(
            threshold=self.config['loss']['epipolar_threshold']
        )
        self.depth_reproj_loss = DepthReprojectionLoss()
        self.photometric_loss = PhotometricConsistencyLoss()

        # Detector losses
        self.repeat_loss = RepeatabilityLoss()
        self.semantic_edge_loss = SemanticEdgeLoss()

        # NEW: Geometric consistency loss for Stage 2
        self.geometric_consistency_loss = GeometricConsistencyLoss()

        self.loss_weights = self.config['loss']['weights']
        print(f"  ✓ Using InfoNCE loss (better than AP for semantic structure)")
        print(f"  ✓ Loss weights: {self.loss_weights}")

    def _init_datasets(self):
        """Initialize train and validation datasets"""
        print("\n📂 Loading datasets...")

        train_datasets = []
        for seq in self.config['dataset']['train_sequences']:
            dataset = TUMDataset(
                dataset_root=self.config['dataset']['root'],
                sequence=seq,
                input_size=self.config['model']['input_size'],
                frame_spacing=self.config['dataset']['frame_spacing'],
                max_frames=self.config['dataset']['max_frames'],
                augmentation=self.config['dataset'].get('augmentation'),
                is_train=True
            )
            train_datasets.append(dataset)

        from torch.utils.data import ConcatDataset
        combined_train = ConcatDataset(train_datasets)

        self.train_loader = DataLoader(
            combined_train,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )

        val_datasets = []
        for seq in self.config['dataset']['val_sequences']:
            dataset = TUMDataset(
                dataset_root=self.config['dataset']['root'],
                sequence=seq,
                input_size=self.config['model']['input_size'],
                frame_spacing=self.config['dataset']['frame_spacing'],
                max_frames=200,
                augmentation=None,
                is_train=False
            )
            val_datasets.append(dataset)

        combined_val = ConcatDataset(val_datasets)

        self.val_loader = DataLoader(
            combined_val,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )

        print(f"  ✓ Train batches: {len(self.train_loader)}")
        print(f"  ✓ Val batches: {len(self.val_loader)}")

    def train(self, start_stage=1, checkpoint_path=None):
        """Main training loop with 3 stages"""

        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project'],
                name=self.config['logging']['run_name'],
                config=self.config
            )

        if checkpoint_path:
            print(f"\n📥 Loading checkpoint: {checkpoint_path}")
            self._load_checkpoint(checkpoint_path)

        if start_stage <= 1:
            print("\n" + "="*70)
            print("STAGE 1: GEOMETRIC CNN PRE-TRAINING")
            print("="*70)
            self._train_stage1()

        if start_stage <= 2:
            print("\n" + "="*70)
            print("STAGE 2: FULL JOINT TRAINING (with Geometric Regularization)")
            print("="*70)
            self._train_stage2()

        if self.config['model']['enable_offset']:
            print("\n" + "="*70)
            print("STAGE 3: OFFSET REFINEMENT")
            print("="*70)
            self._train_stage3()

        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)

        if self.config['logging']['use_wandb']:
            wandb.finish()

    def _train_stage1(self):
        """Stage 1: Pre-train geometric CNN"""
        self.current_stage = 1

        optimizer = AdamW(
            self.geometric_cnn.parameters(),
            lr=self.config['training']['stage1_lr'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['stage1_epochs'],
            eta_min=self.config['training']['stage1_lr'] / 10
        )

        for epoch in range(1, self.config['training']['stage1_epochs'] + 1):
            train_losses = self._train_epoch_stage1(optimizer, epoch)
            val_losses = self._validate_stage1()

            self._print_epoch_summary(epoch, train_losses, val_losses, "Stage1")

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('stage1_best.pth', epoch, val_losses['total'])

            scheduler.step()

        self.save_checkpoint('stage1_final.pth', epoch, val_losses['total'])

    def _train_stage2(self):
        """Stage 2: Train all components with geometric regularization"""
        self.current_stage = 2

        params = (
            list(self.geometric_cnn.parameters()) +
            list(self.selector.parameters()) +
            list(self.fusion.parameters()) +
            list(self.refiner.parameters())
        )

        optimizer = AdamW(
            params,
            lr=self.config['training']['stage2_lr'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['stage2_epochs'],
            eta_min=self.config['training']['stage2_lr_min']
        )

        # Load stage 1 weights
        try:
            checkpoint = torch.load('checkpoints/stage1_best.pth')
            self.geometric_cnn.load_state_dict(checkpoint['geometric_cnn_state_dict'])
            print("✓ Loaded Stage 1 geometric CNN weights")
        except:
            print("⚠️ Could not load Stage 1 weights, starting from scratch")

        for epoch in range(1, self.config['training']['stage2_epochs'] + 1):
            train_losses = self._train_epoch_stage2(optimizer, epoch)
            val_losses = self._validate_stage2()

            self._print_epoch_summary(epoch, train_losses, val_losses, "Stage2")

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('stage2_best.pth', epoch, val_losses['total'])

            scheduler.step()

        self.save_checkpoint('stage2_final.pth', epoch, val_losses['total'])

    def _train_epoch_stage1(self, optimizer, epoch):
        """Training epoch for stage 1"""
        self.geometric_cnn.train()

        total_loss = 0.0
        losses_dict = {
            'photometric': 0.0,
            'edge_response': 0.0,
            'smooth': 0.0,
            'collapse': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Stage1 Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            rgb1 = batch['rgb1'].to(self.device)
            rgb2 = batch['rgb2'].to(self.device)
            depth1 = batch['depth1'].to(self.device)

            geo_feat1 = self.geometric_cnn(rgb1)
            geo_feat2 = self.geometric_cnn(rgb2)

            K, relative_pose = self._get_camera_params(batch)

            # Photometric loss
            loss_photo = self.photometric_loss(rgb1, rgb2, depth1, K, relative_pose)

            # Edge response loss
            loss_edge = self.geometric_consistency_loss(geo_feat1, rgb1)

            # Feature regularization
            geo_norm1 = torch.norm(geo_feat1, dim=1, keepdim=True)
            geo_norm2 = torch.norm(geo_feat2, dim=1, keepdim=True)

            grad_x = torch.abs(geo_norm1[:, :, :, 1:] - geo_norm1[:, :, :, :-1])
            grad_y = torch.abs(geo_norm1[:, :, 1:, :] - geo_norm1[:, :, :-1, :])
            loss_smooth = (grad_x.mean() + grad_y.mean())

            loss_collapse = torch.abs(1.0 - geo_norm1.mean()) + torch.abs(1.0 - geo_norm2.mean())

            # Total loss (prioritize edge response)
            loss = (
                0.3 * loss_photo +
                0.5 * loss_edge +
                0.1 * loss_smooth +
                0.1 * loss_collapse
            )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.geometric_cnn.parameters(),
                self.config['training']['grad_clip']
            )
            optimizer.step()

            total_loss += loss.item()
            losses_dict['photometric'] += loss_photo.item()
            losses_dict['edge_response'] += loss_edge.item()
            losses_dict['smooth'] += loss_smooth.item()
            losses_dict['collapse'] += loss_collapse.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'edge': f"{loss_edge.item():.3f}"
            })

            self.global_step += 1

        n = len(self.train_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        return {'total': avg_loss, **losses_dict}

    def _train_epoch_stage2(self, optimizer, epoch):
        """
        Training epoch for stage 2

        CRITICAL: Now includes geometric consistency loss to prevent degradation!
        """
        self.geometric_cnn.train()
        self.selector.train()
        self.fusion.train()
        self.refiner.train()

        total_loss = 0.0
        losses_dict = {
            'infonce': 0.0,
            'variance': 0.0,
            'epipolar': 0.0,
            'depth_reproj': 0.0,
            'photometric': 0.0,
            'repeatability': 0.0,
            'semantic_edge': 0.0,
            'geo_consistency': 0.0  # NEW!
        }

        # Track keypoint quality
        avg_keypoint_scores = []
        avg_num_good_keypoints = []

        pbar = tqdm(self.train_loader, desc=f"Stage2 Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            loss, loss_components, kpt_stats = self._forward_pass_stage2(batch)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()

            params = (
                list(self.geometric_cnn.parameters()) +
                list(self.selector.parameters()) +
                list(self.fusion.parameters()) +
                list(self.refiner.parameters())
            )
            torch.nn.utils.clip_grad_norm_(params, self.config['training']['grad_clip'])

            optimizer.step()

            total_loss += loss.item()
            for key in losses_dict:
                losses_dict[key] += loss_components.get(key, 0.0)

            # Track keypoint quality
            avg_keypoint_scores.append(kpt_stats['avg_score'])
            avg_num_good_keypoints.append(kpt_stats['num_good'])

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'infonce': f"{loss_components.get('infonce', 0):.3f}",
                'kpts': f"{kpt_stats['num_good']:.0f}/{self.config['model']['num_keypoints']}"
            })

            self.global_step += 1

            # Detailed logging every 100 batches
            if batch_idx % 100 == 0:
                print(f"\n  [Keypoint Quality] Avg score: {kpt_stats['avg_score']:.3f}, "
                      f"Good keypoints: {kpt_stats['num_good']}/{self.config['model']['num_keypoints']}")

            if self.config['logging']['use_wandb'] and batch_idx % 50 == 0:
                wandb.log({
                    'stage2/batch_loss': loss.item(),
                    'stage2/infonce_loss': loss_components.get('infonce', 0),
                    'stage2/geo_consistency': loss_components.get('geo_consistency', 0),
                    'stage2/avg_keypoint_score': kpt_stats['avg_score'],
                    'stage2/num_good_keypoints': kpt_stats['num_good'],
                    'step': self.global_step
                })

        n = len(self.train_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        # Print epoch keypoint statistics
        print(f"\n  [Epoch {epoch} Keypoint Stats]")
        print(f"    Avg keypoint score: {np.mean(avg_keypoint_scores):.3f}")
        print(f"    Avg good keypoints: {np.mean(avg_num_good_keypoints):.1f}/{self.config['model']['num_keypoints']}")

        return {'total': avg_loss, **losses_dict}

    def _forward_pass_stage2(self, batch):
        """
        Complete forward pass for stage 2

        FIXED: Now returns keypoint quality statistics
        """
        rgb1 = batch['rgb1'].to(self.device)
        rgb2 = batch['rgb2'].to(self.device)
        depth1 = batch['depth1'].to(self.device)

        # Extract features
        with torch.no_grad():
            dino_feat1 = self.backbone(rgb1)
            dino_feat2 = self.backbone(rgb2)

        geo_feat1 = self.geometric_cnn(rgb1)
        geo_feat2 = self.geometric_cnn(rgb2)

        # Detect keypoints (FIXED: now adaptive!)
        saliency1 = self.selector(dino_feat1)
        saliency2 = self.selector(dino_feat2)

        kpts1, scores1 = self.selector.select_keypoints(
            saliency1,
            num_keypoints=self.config['model']['num_keypoints'],
            nms_radius=4,  # Pixel-level NMS
            score_threshold=0.1  # Adaptive threshold
        )
        kpts2, scores2 = self.selector.select_keypoints(
            saliency2,
            num_keypoints=self.config['model']['num_keypoints'],
            nms_radius=4,
            score_threshold=0.1
        )

        # Keypoint quality stats
        kpt_stats = {
            'avg_score': scores1.mean().item(),
            'num_good': (scores1 > 0.1).sum().item()
        }

        # Convert to pixel coords
        kpts1_pixel = self.backbone.patch_to_pixel(kpts1)
        kpts2_pixel = self.backbone.patch_to_pixel(kpts2)

        # Extract features at keypoints
        dino_at_kpts1 = self.backbone.extract_at_keypoints(dino_feat1, kpts1)
        dino_at_kpts2 = self.backbone.extract_at_keypoints(dino_feat2, kpts2)

        geo_at_kpts1 = self.geometric_cnn.extract_at_keypoints(geo_feat1, kpts1_pixel)
        geo_at_kpts2 = self.geometric_cnn.extract_at_keypoints(geo_feat2, kpts2_pixel)

        # Fuse
        fused1 = self.fusion(dino_at_kpts1, geo_at_kpts1)
        fused2 = self.fusion(dino_at_kpts2, geo_at_kpts2)

        # Refine descriptors
        desc1 = self.refiner(fused1)
        desc2 = self.refiner(fused2)

        # Find matches (IMPROVED)
        matches = self._find_matches(desc1, desc2, threshold=0.7)

        # Camera params
        K, relative_pose = self._get_camera_params(batch)

        # Compute losses
        w = self.loss_weights

        # CHANGED: InfoNCE instead of AP
        loss_infonce = self.infonce_loss(desc1, desc2, matches)
        loss_var = self.variance_loss(desc1)
        loss_epi = self.epipolar_loss(kpts1_pixel, kpts2_pixel, matches, K, relative_pose)
        loss_depth = self.depth_reproj_loss(kpts1_pixel, kpts2_pixel, matches, depth1, K, relative_pose)
        loss_photo = self.photometric_loss(rgb1, rgb2, depth1, K, relative_pose)
        loss_repeat = self.repeat_loss(saliency1, saliency2)

        # Semantic edge loss
        semantic_edges1 = self.selector.compute_semantic_edges(dino_feat1)
        loss_semantic_edge = self.semantic_edge_loss(saliency1, semantic_edges1)

        # NEW: Geometric consistency loss (prevent degradation!)
        loss_geo_consistency = self.geometric_consistency_loss(geo_feat1, rgb1)

        # Total loss (UPDATED weights)
        total_loss = (
            2.0 * loss_infonce +          # Primary descriptor loss
            1.0 * loss_var +
            1.5 * loss_epi +
            1.0 * loss_depth +
            0.3 * loss_photo +
            0.5 * loss_repeat +
            0.5 * loss_semantic_edge +
            0.5 * loss_geo_consistency    # NEW: Keep geometric features edge-responsive!
        )

        loss_components = {
            'infonce': loss_infonce.item(),
            'variance': loss_var.item(),
            'epipolar': loss_epi.item(),
            'depth_reproj': loss_depth.item(),
            'photometric': loss_photo.item(),
            'repeatability': loss_repeat.item(),
            'semantic_edge': loss_semantic_edge.item(),
            'geo_consistency': loss_geo_consistency.item()
        }

        return total_loss, loss_components, kpt_stats

    def _find_matches(self, desc1, desc2, threshold=0.7):
        """
        IMPROVED: Better matching with adaptive threshold
        """
        B, N, D = desc1.shape
        M = desc2.shape[1]
        device = desc1.device

        matches_list = []

        for b in range(B):
            # Similarity matrix
            sim_matrix = torch.mm(desc1[b], desc2[b].t())  # (N, M)

            # Find mutual nearest neighbors
            nn12_sim, nn12 = sim_matrix.max(dim=1)
            nn21_sim, nn21 = sim_matrix.max(dim=0)

            mutual_mask = nn21[nn12] == torch.arange(N, device=device)
            conf_mask = nn12_sim > threshold

            final_mask = mutual_mask & conf_mask

            idx1 = torch.nonzero(final_mask).squeeze(1)
            idx2 = nn12[idx1]

            if len(idx1) > 0:
                matches_b = torch.stack([idx1, idx2], dim=1)
            else:
                # Dummy match
                matches_b = torch.zeros(1, 2, device=device, dtype=torch.long)

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

    def _get_camera_params(self, batch):
        """Extract camera intrinsics"""
        B = batch['rgb1'].shape[0]
        device = self.device

        K_list = []
        for b in range(B):
            fx = batch['fx'][b].item()
            fy = batch['fy'][b].item()
            cx = batch['cx'][b].item()
            cy = batch['cy'][b].item()

            K_b = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=device, dtype=torch.float32)
            K_list.append(K_b)

        K = torch.stack(K_list, dim=0)

        if 'relative_pose' in batch:
            relative_pose = batch['relative_pose'].to(device).float()
        else:
            relative_pose = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1).float()

        return K, relative_pose

    def _validate_stage1(self):
        """Validation for stage 1"""
        self.geometric_cnn.eval()

        total_loss = 0.0
        losses_dict = {
            'photometric': 0.0,
            'edge_response': 0.0,
            'smooth': 0.0,
            'collapse': 0.0
        }

        with torch.no_grad():
            for batch in self.val_loader:
                rgb1 = batch['rgb1'].to(self.device)
                rgb2 = batch['rgb2'].to(self.device)
                depth1 = batch['depth1'].to(self.device)

                geo_feat1 = self.geometric_cnn(rgb1)
                geo_feat2 = self.geometric_cnn(rgb2)

                K, relative_pose = self._get_camera_params(batch)

                loss_photo = self.photometric_loss(rgb1, rgb2, depth1, K, relative_pose)
                loss_edge = self.geometric_consistency_loss(geo_feat1, rgb1)

                geo_norm1 = torch.norm(geo_feat1, dim=1, keepdim=True)
                geo_norm2 = torch.norm(geo_feat2, dim=1, keepdim=True)

                grad_x = torch.abs(geo_norm1[:, :, :, 1:] - geo_norm1[:, :, :, :-1])
                grad_y = torch.abs(geo_norm1[:, :, 1:, :] - geo_norm1[:, :, :-1, :])
                loss_smooth = (grad_x.mean() + grad_y.mean())

                loss_collapse = torch.abs(1.0 - geo_norm1.mean()) + torch.abs(1.0 - geo_norm2.mean())

                loss = (
                    0.3 * loss_photo +
                    0.5 * loss_edge +
                    0.1 * loss_smooth +
                    0.1 * loss_collapse
                )

                total_loss += loss.item()
                losses_dict['photometric'] += loss_photo.item()
                losses_dict['edge_response'] += loss_edge.item()
                losses_dict['smooth'] += loss_smooth.item()
                losses_dict['collapse'] += loss_collapse.item()

        n = len(self.val_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        return {'total': avg_loss, **losses_dict}

    def _validate_stage2(self):
        """Validation for stage 2"""
        self.geometric_cnn.eval()
        self.selector.eval()
        self.fusion.eval()
        self.refiner.eval()
        self.backbone.eval()

        total_loss = 0.0
        losses_dict = {
            'infonce': 0.0,
            'variance': 0.0,
            'epipolar': 0.0,
            'depth_reproj': 0.0,
            'photometric': 0.0,
            'repeatability': 0.0,
            'semantic_edge': 0.0,
            'geo_consistency': 0.0
        }
        num_valid = 0

        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    loss, loss_components, _ = self._forward_pass_stage2(batch)

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    total_loss += loss.item()
                    for key in losses_dict:
                        losses_dict[key] += loss_components.get(key, 0.0)
                    num_valid += 1
                except Exception as e:
                    continue

        if num_valid == 0:
            return {'total': 0.0, **losses_dict}

        avg_loss = total_loss / num_valid
        for key in losses_dict:
            losses_dict[key] /= num_valid

        return {'total': avg_loss, **losses_dict}

    def _train_stage3(self):
        """Stage 3: Offset refinement"""
        if self.offset_refiner is None:
            print("⚠️ Offset refiner not enabled. Skipping Stage 3.")
            return

        self.current_stage = 3

        # Freeze all other modules
        self.backbone.eval()
        self.geometric_cnn.eval()
        self.selector.eval()
        self.fusion.eval()
        self.refiner.eval()
        self.offset_refiner.train()

        optimizer = AdamW(
            self.offset_refiner.parameters(),
            lr=self.config['training']['stage3_lr'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['stage3_epochs'],
            eta_min=self.config['training']['stage3_lr'] / 10
        )

        for epoch in range(1, self.config['training']['stage3_epochs'] + 1):
            train_losses = self._train_epoch_stage3(optimizer, epoch)
            val_losses = self._validate_stage3()

            self._print_epoch_summary(epoch, train_losses, val_losses, "Stage3")

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('stage3_best.pth', epoch, val_losses['total'])

            scheduler.step()

        self.save_checkpoint('stage3_final.pth', epoch, val_losses['total'])

    def _train_epoch_stage3(self, optimizer, epoch):
        """Training epoch for stage 3 (offset refinement)"""
        self.offset_refiner.train()

        total_loss = 0.0
        losses_dict = {
            'offset': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Stage3 Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            loss, loss_components = self._forward_pass_stage3(batch)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.offset_refiner.parameters(),
                self.config['training']['grad_clip']
            )
            optimizer.step()

            total_loss += loss.item()
            losses_dict['offset'] += loss_components.get('offset', 0.0)

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'offset': f"{loss_components.get('offset', 0):.3f}"
            })

            self.global_step += 1

        n = len(self.train_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        return {'total': avg_loss, **losses_dict}

    def _validate_stage3(self):
        """Validation for stage 3 (offset refinement)"""
        self.offset_refiner.eval()

        total_loss = 0.0
        losses_dict = {
            'offset': 0.0
        }
        num_valid = 0

        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    loss, loss_components = self._forward_pass_stage3(batch)

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    total_loss += loss.item()
                    losses_dict['offset'] += loss_components.get('offset', 0.0)
                    num_valid += 1
                except Exception:
                    continue

        if num_valid == 0:
            return {'total': 0.0, **losses_dict}

        avg_loss = total_loss / num_valid
        for key in losses_dict:
            losses_dict[key] /= num_valid

        return {'total': avg_loss, **losses_dict}

    def _forward_pass_stage3(self, batch):
        """Forward pass for stage 3 (offset refinement)"""
        rgb1 = batch['rgb1'].to(self.device)
        rgb2 = batch['rgb2'].to(self.device)
        depth2 = batch['depth2'].to(self.device)

        with torch.no_grad():
            dino_feat1 = self.backbone(rgb1)
            dino_feat2 = self.backbone(rgb2)

            geo_feat1 = self.geometric_cnn(rgb1)
            geo_feat2 = self.geometric_cnn(rgb2)

            saliency1 = self.selector(dino_feat1)
            saliency2 = self.selector(dino_feat2)

            kpts1, _ = self.selector.select_keypoints(
                saliency1,
                num_keypoints=self.config['model']['num_keypoints'],
                nms_radius=self.config['keypoints']['nms_radius'],
                score_threshold=self.config['keypoints']['score_threshold']
            )
            kpts2, _ = self.selector.select_keypoints(
                saliency2,
                num_keypoints=self.config['model']['num_keypoints'],
                nms_radius=self.config['keypoints']['nms_radius'],
                score_threshold=self.config['keypoints']['score_threshold']
            )

            kpts1_pixel = self.backbone.patch_to_pixel(kpts1)
            kpts2_pixel = self.backbone.patch_to_pixel(kpts2)

            dino_at_kpts1 = self.backbone.extract_at_keypoints(dino_feat1, kpts1)
            dino_at_kpts2 = self.backbone.extract_at_keypoints(dino_feat2, kpts2)

            geo_at_kpts1 = self.geometric_cnn.extract_at_keypoints(geo_feat1, kpts1_pixel)
            geo_at_kpts2 = self.geometric_cnn.extract_at_keypoints(geo_feat2, kpts2_pixel)

            fused1 = self.fusion(dino_at_kpts1, geo_at_kpts1)
            fused2 = self.fusion(dino_at_kpts2, geo_at_kpts2)

            desc1 = self.refiner(fused1)
            desc2 = self.refiner(fused2)

        matches = self._find_matches(desc1, desc2, threshold=0.7)
        K, relative_pose = self._get_camera_params(batch)

        loss_offset = self._compute_offset_loss(
            kpts1,
            kpts2,
            matches,
            fused1,
            depth2,
            K,
            relative_pose
        )

        w = self.loss_weights
        total_loss = w['offset'] * loss_offset

        loss_components = {
            'offset': loss_offset.item()
        }

        return total_loss, loss_components

    def _compute_offset_loss(
        self,
        kpts1: torch.Tensor,
        kpts2: torch.Tensor,
        matches: torch.Tensor,
        fused1: torch.Tensor,
        depth2: torch.Tensor,
        K: torch.Tensor,
        relative_pose: torch.Tensor
    ) -> torch.Tensor:
        """Compute offset loss using depth-based warping from frame 2 to frame 1."""
        B, _, H, W = depth2.shape
        device = kpts1.device

        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            if matches[b].shape[0] == 0:
                continue

            idx1 = matches[b, :, 0].long()
            idx2 = matches[b, :, 1].long()

            valid_mask = (idx1 < kpts1.shape[1]) & (idx2 < kpts2.shape[1]) & (idx1 >= 0) & (idx2 >= 0)
            if valid_mask.sum() == 0:
                continue

            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]

            kpts1_b = kpts1[b, idx1]  # (K, 2) patch coords
            kpts2_b = kpts2[b, idx2]  # (K, 2) patch coords
            fused1_b = fused1[b, idx1]

            # Convert kpts2 to pixel coords for depth lookup
            kpts2_pixel = self.backbone.patch_to_pixel(kpts2_b.unsqueeze(0)).squeeze(0)

            # Sample depth at kpts2 locations
            norm_coords = kpts2_pixel.clone()
            norm_coords[:, 0] = 2.0 * kpts2_pixel[:, 0] / (W - 1) - 1.0
            norm_coords[:, 1] = 2.0 * kpts2_pixel[:, 1] / (H - 1) - 1.0
            norm_coords = torch.clamp(norm_coords, -1.0, 1.0)

            grid = norm_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, K, 2)
            depth_at_kpts = torch.nn.functional.grid_sample(
                depth2[b:b+1],
                grid,
                mode='bilinear',
                align_corners=True
            ).squeeze()

            valid_depth = (depth_at_kpts > self.depth_reproj_loss.min_depth) & (
                depth_at_kpts < self.depth_reproj_loss.max_depth
            )
            if valid_depth.sum() == 0:
                continue

            kpts1_b = kpts1_b[valid_depth]
            kpts2_pixel = kpts2_pixel[valid_depth]
            fused1_b = fused1_b[valid_depth]
            depth_at_kpts = depth_at_kpts[valid_depth]

            # Back-project to 3D in frame 2
            K_inv = torch.inverse(K[b])
            kpts2_h = torch.cat([
                kpts2_pixel,
                torch.ones(len(kpts2_pixel), 1, device=device)
            ], dim=1)
            points_3d_2 = depth_at_kpts.unsqueeze(1) * (K_inv @ kpts2_h.t()).t()

            # Transform to frame 1
            T_2_from_1 = relative_pose[b]
            T_1_from_2 = torch.inverse(T_2_from_1)
            R = T_1_from_2[:3, :3]
            t = T_1_from_2[:3, 3]
            points_3d_1 = (R @ points_3d_2.t()).t() + t

            valid_proj = points_3d_1[:, 2] > 0.1
            if valid_proj.sum() == 0:
                continue

            points_3d_1 = points_3d_1[valid_proj]
            kpts1_b = kpts1_b[valid_proj]
            fused1_b = fused1_b[valid_proj]

            # Project to frame 1
            kpts1_proj = (K[b] @ points_3d_1.t()).t()
            kpts1_proj = kpts1_proj[:, :2] / (kpts1_proj[:, 2:3] + 1e-8)

            # Convert to patch coords
            kpts1_proj_patch = self.backbone.pixel_to_patch(kpts1_proj.unsqueeze(0)).squeeze(0)

            # Predict offsets for matched keypoints
            _, offsets = self.offset_refiner(fused1_b.unsqueeze(0), kpts1_b.unsqueeze(0))
            offsets = offsets.squeeze(0)

            valid_mask = torch.ones(offsets.shape[0], device=device)
            loss = self.offset_refiner.compute_offset_loss(offsets, kpts1_b, kpts1_proj_patch, valid_mask)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def _print_epoch_summary(self, epoch, train_losses, val_losses, stage):
        """Print epoch summary"""
        print(f"\n{'='*70}")
        print(f"{stage} - EPOCH {epoch}")
        print(f"{'='*70}")
        print(f"Train Loss: {train_losses['total']:.4f}")

        train_components = {k: v for k, v in train_losses.items() if k != 'total'}
        if train_components:
            loss_str = "  " + " | ".join([f"{k}: {v:.4f}" for k, v in train_components.items()])
            print(loss_str)

        print(f"Val Loss:   {val_losses['total']:.4f}")

        val_components = {k: v for k, v in val_losses.items() if k != 'total'}
        if val_components:
            loss_str = "  " + " | ".join([f"{k}: {v:.4f}" for k, v in val_components.items()])
            print(loss_str)

        print(f"{'='*70}\n")

    def save_checkpoint(self, filename, epoch, loss):
        """Save checkpoint"""
        save_dir = Path(self.config['training']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'stage': self.current_stage,
            'geometric_cnn_state_dict': self.geometric_cnn.state_dict(),
            'selector_state_dict': self.selector.state_dict(),
            'fusion_state_dict': self.fusion.state_dict(),
            'refiner_state_dict': self.refiner.state_dict(),
            'config': self.config
        }

        if self.offset_refiner:
            checkpoint['offset_refiner_state_dict'] = self.offset_refiner.state_dict()

        torch.save(checkpoint, save_dir / filename)
        print(f"✓ Saved checkpoint: {filename}")

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'geometric_cnn_state_dict' in checkpoint:
            self.geometric_cnn.load_state_dict(checkpoint['geometric_cnn_state_dict'])
            print("  ✓ Loaded geometric_cnn")

        if 'selector_state_dict' in checkpoint:
            self.selector.load_state_dict(checkpoint['selector_state_dict'])
            print("  ✓ Loaded selector")

        if 'fusion_state_dict' in checkpoint:
            self.fusion.load_state_dict(checkpoint['fusion_state_dict'])
            print("  ✓ Loaded fusion")

        if 'refiner_state_dict' in checkpoint:
            self.refiner.load_state_dict(checkpoint['refiner_state_dict'])
            print("  ✓ Loaded refiner")

        print(f"  ✓ Checkpoint from stage {checkpoint.get('stage', '?')}, epoch {checkpoint.get('epoch', '?')}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Hybrid Semantic-Geometric SLAM')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--start-stage', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['training']['stage1_lr'] = float(config['training']['stage1_lr'])
    config['training']['stage2_lr'] = float(config['training']['stage2_lr'])
    config['training']['stage2_lr_min'] = float(config['training']['stage2_lr_min'])
    config['training']['stage3_lr'] = float(config['training']['stage3_lr'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])

    trainer = HybridSLAMTrainer(config)

    checkpoint = args.checkpoint
    if args.start_stage == 2 and checkpoint is None:
        checkpoint = 'checkpoints/stage1_best.pth'
    elif args.start_stage == 3 and checkpoint is None:
        checkpoint = 'checkpoints/stage2_best.pth'

    trainer.train(start_stage=args.start_stage, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()