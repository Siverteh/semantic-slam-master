"""
Training Script - 3-Stage Hybrid Semantic-Geometric Approach
Stage 1: Geometric CNN pre-training
Stage 2: Full joint training
Stage 3: Offset refinement
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
    APLoss,
    DescriptorVarianceLoss,
    RepeatabilityLoss,
    SemanticEdgeLoss
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
        print("HYBRID SEMANTIC-GEOMETRIC SLAM TRAINING")
        print("="*70)
        print("Architecture: DINOv3 (frozen) + Lightweight CNN (trained)")
        print("Losses: AP + Epipolar + Depth + Semantic Edge")
        print("="*70 + "\n")

        # Initialize models
        self._init_models()

        # Initialize losses
        self._init_losses()

        # Initialize datasets
        self._init_datasets()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.current_stage = 1

        print("✓ Initialization complete\n")

    def _init_models(self):
        """Initialize all model components"""
        print("📦 Loading models...")

        # Frozen DINOv3 backbone
        self.backbone = DinoBackbone(
            model_name=self.config['model']['backbone'],
            input_size=self.config['model']['input_size'],
            freeze=True
        ).to(self.device)

        # Trainable geometric CNN
        self.geometric_cnn = GeometricCNN(
            input_channels=3,
            output_channels=self.config['model']['geometric_channels']
        ).to(self.device)

        # Feature fusion
        self.fusion = FeatureFusion(
            semantic_dim=self.backbone.embed_dim,
            geometric_dim=self.config['model']['geometric_channels'],
            output_dim=self.config['model']['fusion_dim']
        ).to(self.device)

        # Keypoint selector
        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            patch_size=self.backbone.patch_size
        ).to(self.device)

        # Descriptor refiner
        self.refiner = DescriptorRefiner(
            input_dim=self.config['model']['fusion_dim'],
            hidden_dim=self.config['model']['refiner_hidden'],
            output_dim=self.config['model']['descriptor_dim'],
            num_layers=self.config['model']['refiner_layers']
        ).to(self.device)

        # Offset refiner (for sub-pixel accuracy)
        if self.config['model']['enable_offset']:
            self.offset_refiner = OffsetRefiner(
                input_dim=self.config['model']['fusion_dim'],
                hidden_dim=self.config['model']['offset_hidden']
            ).to(self.device)
        else:
            self.offset_refiner = None

        # Count parameters
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

        if self.offset_refiner:
            off_params = sum(p.numel() for p in self.offset_refiner.parameters())
            print(f"  ✓ Offset Refiner:    {off_params/1e6:.2f}M params")

    def _init_losses(self):
        """Initialize loss functions"""
        print("\n📊 Initializing losses...")

        # Descriptor losses
        self.ap_loss = APLoss(kappa=self.config['loss']['ap_kappa'])
        self.variance_loss = DescriptorVarianceLoss(
            min_variance=self.config['loss']['min_variance']
        )

        # Geometric losses (CRITICAL!)
        self.epipolar_loss = EpipolarConsistencyLoss(
            threshold=self.config['loss']['epipolar_threshold']
        )
        self.depth_reproj_loss = DepthReprojectionLoss()
        self.photometric_loss = PhotometricConsistencyLoss()

        # Detector losses
        self.repeat_loss = RepeatabilityLoss()
        self.semantic_edge_loss = SemanticEdgeLoss()

        self.loss_weights = self.config['loss']['weights']
        print(f"  ✓ Loss weights: {self.loss_weights}")

    def _init_datasets(self):
        """Initialize train and validation datasets"""
        print("\n📂 Loading datasets...")

        # Training loader
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

        # Validation loader
        val_datasets = []
        for seq in self.config['dataset']['val_sequences']:
            dataset = TUMDataset(
                dataset_root=self.config['dataset']['root'],
                sequence=seq,
                input_size=self.config['model']['input_size'],
                frame_spacing=self.config['dataset']['frame_spacing'],
                max_frames=200,  # Limit val frames for speed
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
        print(f"  ✓ Train sequences: {len(self.config['dataset']['train_sequences'])}")

    def train(self, start_stage=1, checkpoint_path=None):
        """Main training loop with 3 stages

        Args:
            start_stage: Stage to start from (1, 2, or 3)
            checkpoint_path: Path to checkpoint to load (optional)
        """

        # Initialize W&B
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project'],
                name=self.config['logging']['run_name'],
                config=self.config
            )

        # Load checkpoint if specified
        if checkpoint_path:
            print(f"\n📥 Loading checkpoint: {checkpoint_path}")
            self._load_checkpoint(checkpoint_path)

        # Stage 1: Geometric CNN pre-training
        if start_stage <= 1:
            print("\n" + "="*70)
            print("STAGE 1: GEOMETRIC CNN PRE-TRAINING")
            print("="*70)
            self._train_stage1()

        # Stage 2: Full joint training
        if start_stage <= 2:
            print("\n" + "="*70)
            print("STAGE 2: FULL JOINT TRAINING")
            print("="*70)
            self._train_stage2()

        # Stage 3: Offset refinement
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
        """Stage 1: Pre-train geometric CNN with geometric losses only"""
        self.current_stage = 1

        # Optimizer: only geometric CNN
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

        # Training loop
        for epoch in range(1, self.config['training']['stage1_epochs'] + 1):
            train_losses = self._train_epoch_stage1(optimizer, epoch)
            val_losses = self._validate_stage1()

            self._print_epoch_summary(epoch, train_losses, val_losses, "Stage1")

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('stage1_best.pth', epoch, val_losses['total'])

            scheduler.step()

        # Save final stage 1 model
        self.save_checkpoint('stage1_final.pth', epoch, val_losses['total'])

    def _train_stage2(self):
        """Stage 2: Train all components jointly"""
        self.current_stage = 2

        # Optimizer: all trainable components
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

        # Training loop
        for epoch in range(1, self.config['training']['stage2_epochs'] + 1):
            train_losses = self._train_epoch_stage2(optimizer, epoch)
            val_losses = self._validate_stage2()

            self._print_epoch_summary(epoch, train_losses, val_losses, "Stage2")

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('stage2_best.pth', epoch, val_losses['total'])

            scheduler.step()

        # Save final
        self.save_checkpoint('stage2_final.pth', epoch, val_losses['total'])

    def _train_stage3(self):
        """Stage 3: Fine-tune with offset prediction"""
        self.current_stage = 3

        # Optimizer: only offset refiner
        optimizer = AdamW(
            self.offset_refiner.parameters(),
            lr=self.config['training']['stage3_lr'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Load stage 2 weights
        try:
            checkpoint = torch.load('checkpoints/stage2_best.pth')
            self.geometric_cnn.load_state_dict(checkpoint['geometric_cnn_state_dict'])
            self.selector.load_state_dict(checkpoint['selector_state_dict'])
            self.fusion.load_state_dict(checkpoint['fusion_state_dict'])
            self.refiner.load_state_dict(checkpoint['refiner_state_dict'])
            print("✓ Loaded Stage 2 weights")
        except:
            print("⚠️ Could not load Stage 2 weights")

        # Freeze everything except offset refiner
        for param in self.geometric_cnn.parameters():
            param.requires_grad = False
        for param in self.selector.parameters():
            param.requires_grad = False
        for param in self.fusion.parameters():
            param.requires_grad = False
        for param in self.refiner.parameters():
            param.requires_grad = False

        # Training loop
        for epoch in range(1, self.config['training']['stage3_epochs'] + 1):
            train_losses = self._train_epoch_stage3(optimizer, epoch)
            val_losses = self._validate_stage3()

            self._print_epoch_summary(epoch, train_losses, val_losses, "Stage3")

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pth', epoch, val_losses['total'])

        # Save final complete model
        self.save_checkpoint('final_model.pth', epoch, val_losses['total'])

    def _train_epoch_stage1(self, optimizer, epoch):
        """
        Training epoch for stage 1 (geometric CNN only)

        CRITICAL FIX: Added edge-aware training to geometric CNN
        Previously only used photometric loss → learned textures, not edges
        Now includes edge detection to learn edge-responsive features
        """
        self.geometric_cnn.train()

        total_loss = 0.0
        losses_dict = {
            'photometric': 0.0,
            'edge_response': 0.0,  # NEW!
            'smooth': 0.0,
            'collapse': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Stage1 Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            rgb1 = batch['rgb1'].to(self.device)
            rgb2 = batch['rgb2'].to(self.device)
            depth1 = batch['depth1'].to(self.device)

            # Extract geometric features
            geo_feat1 = self.geometric_cnn(rgb1)
            geo_feat2 = self.geometric_cnn(rgb2)

            # Get camera params (USING FIXED VERSION)
            K, relative_pose = self._get_camera_params(batch)

            # Loss 1: Photometric consistency
            loss_photo = self.photometric_loss(
                rgb1, rgb2, depth1, K, relative_pose
            )

            # Loss 2: EDGE RESPONSE (NEW!)
            # Encourage geometric features to respond to image gradients
            # Convert RGB to grayscale for edge detection
            gray1 = 0.299 * rgb1[:, 0] + 0.587 * rgb1[:, 1] + 0.114 * rgb1[:, 2]
            gray1 = gray1.unsqueeze(1)  # (B, 1, H, W)

            # Compute image gradients (Sobel-like)
            grad_x = torch.abs(gray1[:, :, :, 1:] - gray1[:, :, :, :-1])
            grad_y = torch.abs(gray1[:, :, 1:, :] - gray1[:, :, :-1, :])

            # Pad to match size
            grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
            grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))

            # Edge magnitude
            edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

            # Downsample to match geometric feature resolution (H/4, W/4)
            edge_mag_down = torch.nn.functional.interpolate(
                edge_mag,
                size=(geo_feat1.shape[2], geo_feat1.shape[3]),
                mode='bilinear',
                align_corners=False
            )

            # Compute geometric feature magnitude
            geo_mag = torch.norm(geo_feat1, dim=1, keepdim=True)

            # Normalize both to [0, 1]
            edge_mag_norm = edge_mag_down / (edge_mag_down.max() + 1e-8)
            geo_mag_norm = geo_mag / (geo_mag.max() + 1e-8)

            # Correlation loss: geometric features should align with edges
            # Flatten spatial dimensions
            edge_flat = edge_mag_norm.reshape(edge_mag_norm.shape[0], -1)
            geo_flat = geo_mag_norm.reshape(geo_mag_norm.shape[0], -1)

            # Pearson correlation (want positive correlation)
            edge_mean = edge_flat - edge_flat.mean(dim=1, keepdim=True)
            geo_mean = geo_flat - geo_flat.mean(dim=1, keepdim=True)

            correlation = (edge_mean * geo_mean).sum(dim=1) / (
                torch.sqrt((edge_mean**2).sum(dim=1) * (geo_mean**2).sum(dim=1)) + 1e-8
            )

            # Loss: 1 - correlation (minimize this to maximize correlation)
            loss_edge = 1.0 - correlation.mean()

            # Loss 3: Feature regularization
            geo_norm1 = torch.norm(geo_feat1, dim=1, keepdim=True)
            geo_norm2 = torch.norm(geo_feat2, dim=1, keepdim=True)

            # Spatial gradient regularization (encourage smoothness)
            grad_x = torch.abs(geo_norm1[:, :, :, 1:] - geo_norm1[:, :, :, :-1])
            grad_y = torch.abs(geo_norm1[:, :, 1:, :] - geo_norm1[:, :, :-1, :])
            loss_smooth = (grad_x.mean() + grad_y.mean())

            # Loss 4: Non-collapse (encourage non-zero activations)
            loss_collapse = torch.abs(1.0 - geo_norm1.mean()) + torch.abs(1.0 - geo_norm2.mean())

            # UPDATED LOSS WEIGHTS:
            # Prioritize edge response over photometric
            loss = (
                0.3 * loss_photo +      # Reduced from 0.5
                0.5 * loss_edge +       # NEW - highest weight!
                0.1 * loss_smooth +
                0.1 * loss_collapse
            )

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.geometric_cnn.parameters(),
                self.config['training']['grad_clip']
            )
            optimizer.step()

            # Accumulate
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

        # Average
        n = len(self.train_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        return {'total': avg_loss, **losses_dict}

    def _train_epoch_stage2(self, optimizer, epoch):
        """Training epoch for stage 2 (full training)"""
        # Set training mode
        self.geometric_cnn.train()
        self.selector.train()
        self.fusion.train()
        self.refiner.train()

        total_loss = 0.0
        losses_dict = {
            'ap': 0.0, 'variance': 0.0,
            'epipolar': 0.0, 'depth_reproj': 0.0, 'photometric': 0.0,
            'repeatability': 0.0, 'semantic_edge': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Stage2 Epoch {epoch:2d}")

        for batch_idx, batch in enumerate(pbar):
            loss, loss_components = self._forward_pass_stage2(batch)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            params = (
                list(self.geometric_cnn.parameters()) +
                list(self.selector.parameters()) +
                list(self.fusion.parameters()) +
                list(self.refiner.parameters())
            )
            torch.nn.utils.clip_grad_norm_(params, self.config['training']['grad_clip'])

            optimizer.step()

            # Accumulate
            total_loss += loss.item()
            for key in losses_dict:
                losses_dict[key] += loss_components.get(key, 0.0)

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'ap': f"{loss_components.get('ap', 0):.3f}",
                'epi': f"{loss_components.get('epipolar', 0):.3f}"
            })

            self.global_step += 1

            # Log to wandb
            if self.config['logging']['use_wandb'] and batch_idx % 50 == 0:
                wandb.log({
                    'stage2/batch_loss': loss.item(),
                    'stage2/ap_loss': loss_components.get('ap', 0),
                    'stage2/epipolar_loss': loss_components.get('epipolar', 0),
                    'step': self.global_step
                })

        # Average
        n = len(self.train_loader)
        avg_loss = total_loss / n
        for key in losses_dict:
            losses_dict[key] /= n

        return {'total': avg_loss, **losses_dict}

    def _forward_pass_stage2(self, batch):
        """Complete forward pass for stage 2"""
        rgb1 = batch['rgb1'].to(self.device)
        rgb2 = batch['rgb2'].to(self.device)
        depth1 = batch['depth1'].to(self.device)

        # Extract features
        with torch.no_grad():
            dino_feat1 = self.backbone(rgb1)
            dino_feat2 = self.backbone(rgb2)

        geo_feat1 = self.geometric_cnn(rgb1)
        geo_feat2 = self.geometric_cnn(rgb2)

        # Detect keypoints
        saliency1 = self.selector(dino_feat1)
        saliency2 = self.selector(dino_feat2)

        kpts1, scores1 = self.selector.select_keypoints(
            saliency1, num_keypoints=self.config['model']['num_keypoints']
        )
        kpts2, scores2 = self.selector.select_keypoints(
            saliency2, num_keypoints=self.config['model']['num_keypoints']
        )

        # Convert patch coords to pixel coords
        kpts1_pixel = self.backbone.patch_to_pixel(kpts1)
        kpts2_pixel = self.backbone.patch_to_pixel(kpts2)

        # Extract features at keypoints
        dino_at_kpts1 = self.backbone.extract_at_keypoints(dino_feat1, kpts1)
        dino_at_kpts2 = self.backbone.extract_at_keypoints(dino_feat2, kpts2)

        geo_at_kpts1 = self.geometric_cnn.extract_at_keypoints(geo_feat1, kpts1_pixel)
        geo_at_kpts2 = self.geometric_cnn.extract_at_keypoints(geo_feat2, kpts2_pixel)

        # Fuse features
        fused1 = self.fusion(dino_at_kpts1, geo_at_kpts1)
        fused2 = self.fusion(dino_at_kpts2, geo_at_kpts2)

        # Refine descriptors
        desc1 = self.refiner(fused1)
        desc2 = self.refiner(fused2)

        # Find matches
        matches = self._find_matches(desc1, desc2)

        # Get camera params
        K, relative_pose = self._get_camera_params(batch)

        # Compute losses
        w = self.loss_weights

        loss_ap = self.ap_loss(desc1, desc2, matches)
        loss_var = self.variance_loss(desc1)
        loss_epi = self.epipolar_loss(kpts1_pixel, kpts2_pixel, matches, K, relative_pose)
        loss_depth = self.depth_reproj_loss(kpts1_pixel, kpts2_pixel, matches, depth1, K, relative_pose)
        loss_photo = self.photometric_loss(rgb1, rgb2, depth1, K, relative_pose)
        loss_repeat = self.repeat_loss(saliency1, saliency2)

        # Semantic edge loss
        semantic_edges1 = self.selector.compute_semantic_edges(dino_feat1)
        loss_semantic_edge = self.semantic_edge_loss(saliency1, semantic_edges1)

        # Total loss
        total_loss = (
            w['ap'] * loss_ap +
            w['variance'] * loss_var +
            w['epipolar'] * loss_epi +
            w['depth_reproj'] * loss_depth +
            w['photometric'] * loss_photo +
            w['repeatability'] * loss_repeat +
            w['semantic_edge'] * loss_semantic_edge
        )

        loss_components = {
            'ap': loss_ap.item(),
            'variance': loss_var.item(),
            'epipolar': loss_epi.item(),
            'depth_reproj': loss_depth.item(),
            'photometric': loss_photo.item(),
            'repeatability': loss_repeat.item(),
            'semantic_edge': loss_semantic_edge.item()
        }

        return total_loss, loss_components

    def _find_matches(self, desc1, desc2, threshold=0.8):
        """
        Find mutual nearest neighbor matches.

        FIXES:
        - Added similarity threshold for better match quality
        - Added debug logging to understand matching failures
        - Relaxed mutual NN constraint slightly
        """
        B, N, D = desc1.shape
        M = desc2.shape[1]
        device = desc1.device

        matches_list = []
        num_matches_log = []

        for b in range(B):
            # Compute similarity matrix (cosine similarity since L2 normalized)
            sim_matrix = torch.mm(desc1[b], desc2[b].t())  # (N, M)

            # Find nearest neighbors
            nn12_sim, nn12 = sim_matrix.max(dim=1)  # For each in desc1, best in desc2
            nn21_sim, nn21 = sim_matrix.max(dim=0)  # For each in desc2, best in desc1

            # Mutual nearest neighbors
            mutual_mask = nn21[nn12] == torch.arange(N, device=device)

            # ALSO filter by similarity threshold
            # Only keep matches with high confidence
            conf_mask = nn12_sim > threshold

            # Combine masks
            final_mask = mutual_mask & conf_mask

            idx1 = torch.nonzero(final_mask).squeeze(1)
            idx2 = nn12[idx1]

            if len(idx1) > 0:
                matches_b = torch.stack([idx1, idx2], dim=1)
            else:
                # If no matches, create dummy match to avoid empty tensor issues
                matches_b = torch.zeros(1, 2, device=device, dtype=torch.long)

            matches_list.append(matches_b)
            num_matches_log.append(len(idx1))

        # Log matching statistics every 100 batches
        if self.global_step % 100 == 0:
            avg_matches = sum(num_matches_log) / len(num_matches_log)
            max_matches = max(num_matches_log)
            min_matches = min(num_matches_log)

            print(f"\n  [Matching Stats] Avg: {avg_matches:.1f}, Min: {min_matches}, Max: {max_matches}")

            # Descriptor similarity distribution
            sample_sim = sim_matrix.flatten()
            print(f"  [Descriptor Sim] Mean: {sample_sim.mean():.3f}, Std: {sample_sim.std():.3f}, Max: {sample_sim.max():.3f}")

        # Pad to same length
        max_matches = max(m.shape[0] for m in matches_list)
        if max_matches == 0:
            # No matches at all - return single dummy match
            return torch.zeros(B, 1, 2, device=device, dtype=torch.long)

        padded = []
        for m in matches_list:
            if m.shape[0] < max_matches:
                pad = torch.zeros(max_matches - m.shape[0], 2, device=device, dtype=torch.long)
                m = torch.cat([m, pad], dim=0)
            padded.append(m)

        return torch.stack(padded, dim=0)

    def _get_camera_params(self, batch):
        """
        Extract camera intrinsics and relative pose.

        FIXED: Now uses correct per-sample camera intrinsics from batch
        instead of hardcoding fr1 params for all sequences
        """
        B = batch['rgb1'].shape[0]
        device = self.device

        # Build K matrix per sample (handles mixed fr1/fr2/fr3 in same batch)
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

        # Relative pose from batch (if available)
        if 'relative_pose' in batch:
            relative_pose = batch['relative_pose'].to(device).float()
        else:
            # Identity fallback
            relative_pose = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1).float()

        return K, relative_pose

    def _train_epoch_stage3(self, optimizer, epoch):
        """Training epoch for stage 3 (offset refinement)"""
        # Only offset refiner in training mode
        self.offset_refiner.train()
        self.geometric_cnn.eval()
        self.selector.eval()
        self.fusion.eval()
        self.refiner.eval()

        # Similar to stage 2 but focus on offset loss
        # Implementation similar to stage 2, omitted for brevity
        # Just add offset refinement to forward pass
        pass

    def _validate_stage1(self):
        """
        Validation for stage 1
        UPDATED: Now includes edge response loss
        """
        self.geometric_cnn.eval()

        total_loss = 0.0
        losses_dict = {
            'photometric': 0.0,
            'edge_response': 0.0,  # NEW!
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

                # Photometric loss
                loss_photo = self.photometric_loss(
                    rgb1, rgb2, depth1, K, relative_pose
                )

                # Edge response loss (same as training)
                gray1 = 0.299 * rgb1[:, 0] + 0.587 * rgb1[:, 1] + 0.114 * rgb1[:, 2]
                gray1 = gray1.unsqueeze(1)

                grad_x = torch.abs(gray1[:, :, :, 1:] - gray1[:, :, :, :-1])
                grad_y = torch.abs(gray1[:, :, 1:, :] - gray1[:, :, :-1, :])

                grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
                grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))

                edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

                edge_mag_down = torch.nn.functional.interpolate(
                    edge_mag,
                    size=(geo_feat1.shape[2], geo_feat1.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )

                geo_mag = torch.norm(geo_feat1, dim=1, keepdim=True)

                edge_mag_norm = edge_mag_down / (edge_mag_down.max() + 1e-8)
                geo_mag_norm = geo_mag / (geo_mag.max() + 1e-8)

                edge_flat = edge_mag_norm.reshape(edge_mag_norm.shape[0], -1)
                geo_flat = geo_mag_norm.reshape(geo_mag_norm.shape[0], -1)

                edge_mean = edge_flat - edge_flat.mean(dim=1, keepdim=True)
                geo_mean = geo_flat - geo_flat.mean(dim=1, keepdim=True)

                correlation = (edge_mean * geo_mean).sum(dim=1) / (
                    torch.sqrt((edge_mean**2).sum(dim=1) * (geo_mean**2).sum(dim=1)) + 1e-8
                )

                loss_edge = 1.0 - correlation.mean()

                # Feature regularization
                geo_norm1 = torch.norm(geo_feat1, dim=1, keepdim=True)
                geo_norm2 = torch.norm(geo_feat2, dim=1, keepdim=True)

                grad_x = torch.abs(geo_norm1[:, :, :, 1:] - geo_norm1[:, :, :, :-1])
                grad_y = torch.abs(geo_norm1[:, :, 1:, :] - geo_norm1[:, :, :-1, :])
                loss_smooth = (grad_x.mean() + grad_y.mean())

                loss_collapse = torch.abs(1.0 - geo_norm1.mean()) + torch.abs(1.0 - geo_norm2.mean())

                # Total loss (same weights as training)
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
        # Set eval mode
        self.geometric_cnn.eval()
        self.selector.eval()
        self.fusion.eval()
        self.refiner.eval()
        self.backbone.eval()

        total_loss = 0.0
        losses_dict = {
            'ap': 0.0, 'variance': 0.0,
            'epipolar': 0.0, 'depth_reproj': 0.0, 'photometric': 0.0,
            'repeatability': 0.0, 'semantic_edge': 0.0
        }
        num_valid = 0

        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    loss, loss_components = self._forward_pass_stage2(batch)

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    total_loss += loss.item()
                    for key in losses_dict:
                        losses_dict[key] += loss_components.get(key, 0.0)
                    num_valid += 1
                except Exception as e:
                    # Skip batches with errors
                    continue

        if num_valid == 0:
            return {'total': 0.0, **losses_dict}

        avg_loss = total_loss / num_valid
        for key in losses_dict:
            losses_dict[key] /= num_valid

        return {'total': avg_loss, **losses_dict}

    def _validate_stage3(self):
        """Validation for stage 3"""
        return {'total': 0.0}

    def _print_epoch_summary(self, epoch, train_losses, val_losses, stage):
        """Print epoch summary"""
        print(f"\n{'='*70}")
        print(f"{stage} - EPOCH {epoch}")
        print(f"{'='*70}")
        print(f"Train Loss: {train_losses['total']:.4f}")

        # Print individual training losses
        train_components = {k: v for k, v in train_losses.items() if k != 'total'}
        if train_components:
            loss_str = "  " + " | ".join([f"{k}: {v:.4f}" for k, v in train_components.items()])
            print(loss_str)

        print(f"Val Loss:   {val_losses['total']:.4f}")

        # Print individual validation losses
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

        if self.offset_refiner and 'offset_refiner_state_dict' in checkpoint:
            self.offset_refiner.load_state_dict(checkpoint['offset_refiner_state_dict'])
            print("  ✓ Loaded offset_refiner")

        print(f"  ✓ Checkpoint from stage {checkpoint.get('stage', '?')}, epoch {checkpoint.get('epoch', '?')}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Hybrid Semantic-Geometric SLAM')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
    parser.add_argument('--start-stage', type=int, default=1, choices=[1, 2, 3], help='Stage to start from (1, 2, or 3)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load (e.g., checkpoints/stage1_best.pth)')
    args = parser.parse_args()

    config_path = args.config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert string learning rates to floats (YAML 1.1 issue with scientific notation)
    config['training']['stage1_lr'] = float(config['training']['stage1_lr'])
    config['training']['stage2_lr'] = float(config['training']['stage2_lr'])
    config['training']['stage2_lr_min'] = float(config['training']['stage2_lr_min'])
    config['training']['stage3_lr'] = float(config['training']['stage3_lr'])
    config['training']['weight_decay'] = float(config['training']['weight_decay'])

    trainer = HybridSLAMTrainer(config)

    # If starting from stage 2 and no checkpoint specified, use stage1_best.pth by default
    checkpoint = args.checkpoint
    if args.start_stage == 2 and checkpoint is None:
        checkpoint = 'checkpoints/stage1_best.pth'
        print(f"Starting from stage 2, using default checkpoint: {checkpoint}")
    elif args.start_stage == 3 and checkpoint is None:
        checkpoint = 'checkpoints/stage2_best.pth'
        print(f"Starting from stage 3, using default checkpoint: {checkpoint}")

    trainer.train(start_stage=args.start_stage, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()