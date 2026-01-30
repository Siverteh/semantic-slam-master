"""
SOTA 2025 Visualization Script for Semantic Keypoint Detection
Supports sub-pixel keypoint visualization with offset arrows.

Features:
- Saliency map visualization
- Sub-pixel keypoint locations (shows offset refinement)
- Offset field visualization
- Match visualization between frame pairs
- Quality metrics display
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import yaml
import cv2
from typing import Dict, Optional, Tuple

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
import torchvision.transforms as transforms


class SubPixelVisualizer:
    """
    Visualizer for SOTA 2025 keypoint detection with sub-pixel refinement.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print("="*60)
        print("SOTA 2025 Keypoint Visualizer")
        print("="*60)

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._load_models(checkpoint_path)
        self._setup_transforms()

        print("✓ Ready for visualization")
        print("="*60 + "\n")

    def _load_models(self, checkpoint_path: str):
        """Load all models from checkpoint"""
        print("Loading models...")

        # Backbone
        self.backbone = DinoBackbone(
            model_name=self.config['model']['backbone'],
            input_size=self.config['model']['input_size'],
            freeze=True
        ).to(self.device)

        # Selector (with offset head)
        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['selector_hidden']
        ).to(self.device)

        # Refiner
        self.refiner = DescriptorRefiner(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['refiner_hidden'],
            output_dim=self.config['model']['descriptor_dim']
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.selector.load_state_dict(checkpoint['selector_state_dict'])
        self.refiner.load_state_dict(checkpoint['refiner_state_dict'])

        self.selector.eval()
        self.refiner.eval()

        print(f"  ✓ Loaded from epoch {checkpoint['epoch']}")
        print(f"  ✓ Checkpoint loss: {checkpoint['loss']:.4f}")

    def _setup_transforms(self):
        """Setup image transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((self.config['model']['input_size'],
                             self.config['model']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract_features(self, image_path: str) -> Dict:
        """
        Extract all features from a single image.

        Returns dict with:
        - image: Original image (numpy)
        - saliency_map: (H, W) saliency scores
        - offset_map: (H, W, 2) sub-pixel offsets
        - keypoints_patch: (N, 2) keypoints in patch coords (with sub-pixel)
        - keypoints_pixel: (N, 2) keypoints in pixel coords
        - keypoints_int: (N, 2) integer keypoints (before offset)
        - offsets: (N, 2) applied offsets
        - scores: (N,) saliency scores
        - descriptors: (N, D) descriptors
        """
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image.resize(
            (self.config['model']['input_size'], self.config['model']['input_size'])
        ))
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features
        dino_features = self.backbone(image_tensor)

        # Get saliency and offsets
        saliency_map, offset_map = self.selector(dino_features)

        # Select keypoints with sub-pixel refinement
        keypoints, scores, offsets = self.selector.select_keypoints(
            saliency_map, offset_map,
            num_keypoints=self.config['model']['num_keypoints']
        )

        # Convert to pixel coordinates
        keypoints_pixel = self.backbone.patch_to_pixel(keypoints)

        # Get integer keypoints (before offset)
        keypoints_int = keypoints - 0.5 * offsets
        keypoints_int_pixel = self.backbone.patch_to_pixel(keypoints_int)

        # Extract descriptors
        feat_at_kpts = self.backbone.extract_at_keypoints(dino_features, keypoints)
        descriptors = self.refiner(feat_at_kpts)

        return {
            'image': image_np,
            'saliency_map': saliency_map[0, :, :, 0].cpu().numpy(),
            'offset_map': offset_map[0].cpu().numpy(),
            'keypoints_patch': keypoints[0].cpu().numpy(),
            'keypoints_pixel': keypoints_pixel[0].cpu().numpy(),
            'keypoints_int_pixel': keypoints_int_pixel[0].cpu().numpy(),
            'offsets': offsets[0].cpu().numpy(),
            'scores': scores[0].cpu().numpy(),
            'descriptors': descriptors[0].cpu().numpy()
        }

    def visualize_single(
        self,
        features: Dict,
        output_dir: str = "visualization_output",
        name: str = "viz"
    ):
        """
        Create comprehensive visualization for a single image.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        image = features['image']
        saliency = features['saliency_map']
        offset_map = features['offset_map']
        kpts_pixel = features['keypoints_pixel']
        kpts_int_pixel = features['keypoints_int_pixel']
        offsets = features['offsets']
        scores = features['scores']

        # Compute statistics
        offset_magnitudes = np.sqrt((offsets ** 2).sum(axis=1))
        mean_offset = offset_magnitudes.mean()
        sal_variance = saliency.var()
        sal_mean = saliency.mean()

        print(f"\n📊 Statistics for {name}:")
        print(f"  Saliency - Mean: {sal_mean:.3f}, Variance: {sal_variance:.4f}")
        print(f"  Offsets - Mean magnitude: {mean_offset:.3f} patches")
        print(f"  Offset range: [{offsets.min():.2f}, {offsets.max():.2f}]")

        # Create visualizations
        self._save_keypoints_plot(image, kpts_pixel, scores, output_dir, name)
        self._save_subpixel_plot(image, kpts_pixel, kpts_int_pixel, offsets, output_dir, name)
        self._save_saliency_plot(image, saliency, output_dir, name)
        self._save_offset_field_plot(offset_map, output_dir, name)
        self._save_stats_plot(saliency, offsets, scores, output_dir, name)
        self._save_combined_plot(image, saliency, kpts_pixel, kpts_int_pixel,
                                offsets, scores, output_dir, name)

        print(f"✓ Saved visualizations to {output_dir}/")

    def _save_keypoints_plot(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        output_dir: Path,
        name: str
    ):
        """Save keypoints on image"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        scatter = ax.scatter(
            keypoints[:, 0], keypoints[:, 1],
            c=scores, cmap='hot', s=25, alpha=0.8,
            edgecolors='white', linewidths=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Saliency Score')

        ax.set_title('Keypoints (colored by saliency)', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.savefig(output_dir / f"{name}_1_keypoints.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _save_subpixel_plot(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        keypoints_int: np.ndarray,
        offsets: np.ndarray,
        output_dir: Path,
        name: str
    ):
        """Save sub-pixel refinement visualization with offset arrows"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        # Plot integer grid points
        ax.scatter(
            keypoints_int[:100, 0], keypoints_int[:100, 1],
            c='blue', s=30, alpha=0.6, marker='s', label='Grid location'
        )

        # Plot refined keypoints
        ax.scatter(
            keypoints[:100, 0], keypoints[:100, 1],
            c='red', s=30, alpha=0.8, marker='o', label='Sub-pixel refined'
        )

        # Draw arrows showing offset
        patch_size = self.config['model']['input_size'] / 28  # ~16
        for i in range(min(100, len(keypoints))):
            dx = (keypoints[i, 0] - keypoints_int[i, 0])
            dy = (keypoints[i, 1] - keypoints_int[i, 1])
            if abs(dx) > 0.1 or abs(dy) > 0.1:  # Only show significant offsets
                ax.arrow(
                    keypoints_int[i, 0], keypoints_int[i, 1],
                    dx, dy,
                    head_width=2, head_length=1, fc='yellow', ec='yellow', alpha=0.7
                )

        ax.legend(loc='upper right')
        ax.set_title('Sub-Pixel Refinement (arrows show offset)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.savefig(output_dir / f"{name}_2_subpixel.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _save_saliency_plot(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
        output_dir: Path,
        name: str
    ):
        """Save saliency map visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        # Saliency map
        im = axes[1].imshow(saliency, cmap='hot', interpolation='nearest')
        axes[1].set_title(f'Saliency Map\nMean: {saliency.mean():.3f}, Var: {saliency.var():.4f}',
                         fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        axes[2].imshow(image)
        saliency_up = cv2.resize(saliency, (image.shape[1], image.shape[0]))
        axes[2].imshow(saliency_up, cmap='hot', alpha=0.5)
        axes[2].set_title('Saliency Overlay', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_3_saliency.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _save_offset_field_plot(
        self,
        offset_map: np.ndarray,
        output_dir: Path,
        name: str
    ):
        """Save offset field visualization"""
        H, W, _ = offset_map.shape

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # X offsets
        im0 = axes[0].imshow(offset_map[:, :, 0], cmap='RdBu', vmin=-1, vmax=1)
        axes[0].set_title('X Offset (dx)', fontsize=12)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        # Y offsets
        im1 = axes[1].imshow(offset_map[:, :, 1], cmap='RdBu', vmin=-1, vmax=1)
        axes[1].set_title('Y Offset (dy)', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Offset magnitude
        magnitude = np.sqrt(offset_map[:, :, 0]**2 + offset_map[:, :, 1]**2)
        im2 = axes[2].imshow(magnitude, cmap='viridis', vmin=0, vmax=1.5)
        axes[2].set_title(f'Offset Magnitude\nMean: {magnitude.mean():.3f}', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.suptitle('Offset Field Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_4_offset_field.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _save_stats_plot(
        self,
        saliency: np.ndarray,
        offsets: np.ndarray,
        scores: np.ndarray,
        output_dir: Path,
        name: str
    ):
        """Save statistics plots"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Saliency histogram
        axes[0].hist(saliency.flatten(), bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[0].axvline(saliency.mean(), color='blue', linestyle='--', label=f'Mean: {saliency.mean():.3f}')
        axes[0].set_xlabel('Saliency Value')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Saliency Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Offset magnitude histogram
        offset_mag = np.sqrt((offsets ** 2).sum(axis=1))
        axes[1].hist(offset_mag, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[1].axvline(offset_mag.mean(), color='blue', linestyle='--',
                       label=f'Mean: {offset_mag.mean():.3f}')
        axes[1].axvline(0.5, color='red', linestyle=':', label='Target: 0.5')
        axes[1].set_xlabel('Offset Magnitude (patches)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Offset Magnitude Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Score distribution
        axes[2].hist(scores, bins=30, color='purple', alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Keypoint Score')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Keypoint Score Distribution')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_5_statistics.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _save_combined_plot(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
        kpts_pixel: np.ndarray,
        kpts_int_pixel: np.ndarray,
        offsets: np.ndarray,
        scores: np.ndarray,
        output_dir: Path,
        name: str
    ):
        """Save combined overview plot"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)

        # Row 1: Main views
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        scatter = ax1.scatter(kpts_pixel[:, 0], kpts_pixel[:, 1],
                             c=scores, cmap='hot', s=20, alpha=0.8)
        ax1.set_title('Keypoints (colored by score)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(scatter, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(saliency, cmap='hot', interpolation='nearest')
        ax2.set_title(f'Saliency Map\nVar: {saliency.var():.4f}', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image)
        ax3.scatter(kpts_int_pixel[:50, 0], kpts_int_pixel[:50, 1],
                   c='blue', s=25, marker='s', alpha=0.6, label='Grid')
        ax3.scatter(kpts_pixel[:50, 0], kpts_pixel[:50, 1],
                   c='red', s=25, marker='o', alpha=0.8, label='Refined')
        ax3.legend(loc='upper right')
        ax3.set_title('Sub-Pixel Refinement', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # Row 2: Statistics and metrics
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(saliency.flatten(), bins=50, color='red', alpha=0.7)
        ax4.axvline(saliency.mean(), color='blue', linestyle='--', linewidth=2)
        ax4.set_xlabel('Saliency')
        ax4.set_ylabel('Count')
        ax4.set_title('Saliency Distribution')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        offset_mag = np.sqrt((offsets ** 2).sum(axis=1))
        ax5.hist(offset_mag, bins=30, color='green', alpha=0.7)
        ax5.axvline(offset_mag.mean(), color='blue', linestyle='--', linewidth=2)
        ax5.axvline(0.5, color='red', linestyle=':', linewidth=2)
        ax5.set_xlabel('Offset Magnitude')
        ax5.set_ylabel('Count')
        ax5.set_title('Offset Distribution')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        # Quality metrics text
        target_var = '✅' if 0.15 <= saliency.var() <= 0.30 else '❌'
        target_offset = '✅' if offset_mag.mean() < 0.5 else '❌'

        stats_text = f"""
QUALITY METRICS
{'='*40}

Saliency Statistics:
  Mean:      {saliency.mean():.4f}
  Variance:  {saliency.var():.4f}  {target_var} (target: 0.15-0.30)
  Max:       {saliency.max():.4f}

Offset Statistics:
  Mean Mag:  {offset_mag.mean():.4f}  {target_offset} (target: <0.5)
  Max Mag:   {offset_mag.max():.4f}
  Range X:   [{offsets[:, 0].min():.2f}, {offsets[:, 0].max():.2f}]
  Range Y:   [{offsets[:, 1].min():.2f}, {offsets[:, 1].max():.2f}]

Keypoints:
  Count:     {len(scores)}
  Min Score: {scores.min():.4f}
  Max Score: {scores.max():.4f}

{'='*40}
SUB-PIXEL REFINEMENT: {'WORKING' if offset_mag.mean() > 0.01 else 'MINIMAL'}
        """
        ax6.text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax6.transAxes)

        plt.suptitle('SOTA 2025 Keypoint Detection - Overview',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(output_dir / f"{name}_combined.png", dpi=150, bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def visualize_matches(
        self,
        image_path1: str,
        image_path2: str,
        output_dir: str = "visualization_output",
        name: str = "matches"
    ):
        """Visualize matches between two images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Extract features
        feat1 = self.extract_features(image_path1)
        feat2 = self.extract_features(image_path2)

        # Find matches
        desc1 = torch.from_numpy(feat1['descriptors']).unsqueeze(0)
        desc2 = torch.from_numpy(feat2['descriptors']).unsqueeze(0)

        matches = self._find_matches(desc1, desc2)

        print(f"\n📊 Found {len(matches)} mutual nearest neighbor matches")

        # Create match visualization
        self._save_match_plot(
            feat1['image'], feat2['image'],
            feat1['keypoints_pixel'], feat2['keypoints_pixel'],
            matches, output_dir, name
        )

    def _find_matches(
        self,
        desc1: torch.Tensor,
        desc2: torch.Tensor
    ) -> np.ndarray:
        """Find mutual nearest neighbor matches"""
        desc1 = desc1[0]  # (N, D)
        desc2 = desc2[0]  # (N, D)

        # Similarity matrix
        sim = torch.mm(desc1, desc2.t())

        # Forward and backward NN
        nn12 = sim.argmax(dim=1)
        nn21 = sim.argmax(dim=0)

        # Mutual matches
        N = desc1.shape[0]
        indices = torch.arange(N)
        mutual = nn21[nn12] == indices

        idx1 = torch.where(mutual)[0].numpy()
        idx2 = nn12[mutual].numpy()

        return np.stack([idx1, idx2], axis=1)

    def _save_match_plot(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        matches: np.ndarray,
        output_dir: Path,
        name: str
    ):
        """Save match visualization"""
        # Create side-by-side image
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        combined[:h1, :w1] = image1
        combined[:h2, w1:] = image2

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(combined)

        # Draw matches
        colors = plt.cm.hsv(np.linspace(0, 1, len(matches)))

        for i, (idx1, idx2) in enumerate(matches[:100]):  # Limit to 100
            pt1 = kpts1[idx1]
            pt2 = kpts2[idx2] + np.array([w1, 0])

            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                   c=colors[i], linewidth=0.5, alpha=0.6)
            ax.scatter([pt1[0]], [pt1[1]], c=[colors[i]], s=20)
            ax.scatter([pt2[0]], [pt2[1]], c=[colors[i]], s=20)

        ax.set_title(f'Matches: {len(matches)} (showing up to 100)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.savefig(output_dir / f"{name}_matches.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='SOTA 2025 Keypoint Visualization with Sub-Pixel Refinement'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--image2', type=str, default=None,
                       help='Second image for match visualization')
    parser.add_argument('--output', type=str, default='visualization_output',
                       help='Output directory')
    parser.add_argument('--name', type=str, default='viz',
                       help='Output file prefix')

    args = parser.parse_args()

    visualizer = SubPixelVisualizer(args.checkpoint, args.config)

    if args.image2:
        # Match visualization
        visualizer.visualize_matches(args.image, args.image2, args.output, args.name)
    else:
        # Single image visualization
        features = visualizer.extract_features(args.image)
        visualizer.visualize_single(features, args.output, args.name)

    print(f"\n✓ Visualizations saved to {args.output}/")


if __name__ == "__main__":
    main()
