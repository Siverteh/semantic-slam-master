"""
Fixed Visualization Script
CRITICAL: Properly converts PATCH coordinates → PIXEL coordinates for display
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import yaml

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
import torchvision.transforms as transforms


class FixedVisualizer:
    """Visualizer with proper coordinate handling"""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print("Loading models...")

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load models
        self.backbone = DinoBackbone(
            model_name=self.config['model']['backbone'],
            input_size=self.config['model']['input_size'],
            freeze=True
        ).to(self.device)

        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['selector_hidden']
        ).to(self.device)

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

        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")

        # Transform
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
    def extract_features(self, image_path: str):
        """Extract features with proper coordinate tracking"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract DINOv3 features (PATCH space)
        dino_features = self.backbone(image_tensor)

        # Keypoint selection (PATCH space)
        saliency_map = self.selector(dino_features)
        keypoints_patch, scores = self.selector.select_keypoints(
            saliency_map,
            num_keypoints=self.config['model']['num_keypoints']
        )

        # Convert PATCH → PIXEL for visualization
        keypoints_pixel = self.backbone.patch_to_pixel(keypoints_patch)

        # Extract features at keypoints
        features_at_kpts = self.backbone.extract_at_keypoints(dino_features, keypoints_patch)

        # Refine descriptors
        descriptors = self.refiner(features_at_kpts)

        return {
            'image': image,
            'saliency_map': saliency_map[0, :, :, 0].cpu().numpy(),
            'keypoints_patch': keypoints_patch[0].cpu().numpy(),
            'keypoints_pixel': keypoints_pixel[0].cpu().numpy(),
            'scores': scores[0].cpu().numpy(),
            'descriptors': descriptors[0].cpu().numpy()
        }

    def visualize_frame(self, features: dict, output_path: str = None):
        """Visualize with correct coordinate display"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        image = features['image']
        image_np = np.array(image.resize((self.config['model']['input_size'],
                                         self.config['model']['input_size'])))

        # 1. Image with keypoints (PIXEL coordinates for display!)
        axes[0, 0].imshow(image_np)
        kpts_pixel = features['keypoints_pixel']
        scores = features['scores']

        # Color by score
        scatter = axes[0, 0].scatter(
            kpts_pixel[:, 0], kpts_pixel[:, 1],
            c=scores, cmap='hot', s=30, alpha=0.8,
            edgecolors='white', linewidths=0.5
        )
        axes[0, 0].set_title(f'Keypoints (N={len(kpts_pixel)}) in PIXEL space',
                            fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(scatter, ax=axes[0, 0], label='Score')

        # 2. Saliency map (PATCH space - 28x28)
        saliency = features['saliency_map']
        im1 = axes[0, 1].imshow(saliency, cmap='hot', interpolation='nearest')
        axes[0, 1].set_title(f'Saliency Map (PATCH space: {saliency.shape[0]}x{saliency.shape[1]})',
                            fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Saliency')

        # 3. Grid overlay showing patch boundaries
        axes[1, 0].imshow(image_np)

        # Draw patch grid
        patch_size = self.backbone.patch_size
        for i in range(0, self.config['model']['input_size'], patch_size):
            axes[1, 0].axhline(i, color='cyan', alpha=0.3, linewidth=0.5)
            axes[1, 0].axvline(i, color='cyan', alpha=0.3, linewidth=0.5)

        # Plot keypoints
        axes[1, 0].scatter(
            kpts_pixel[:, 0], kpts_pixel[:, 1],
            c='red', s=20, alpha=0.8, edgecolors='white', linewidths=0.5
        )
        axes[1, 0].set_title(f'Grid Alignment (16x16 patches)',
                            fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # 4. Coordinate info
        axes[1, 1].axis('off')
        info_text = f"""
COORDINATE SYSTEM INFO:

DINOv3 Configuration:
- Patch size: {self.backbone.patch_size}x{self.backbone.patch_size}
- Grid size: {self.backbone.grid_h}x{self.backbone.grid_w}
- Input size: {self.config['model']['input_size']}x{self.config['model']['input_size']}

Keypoint Statistics:
- Total keypoints: {len(kpts_pixel)}
- PATCH coords: [0, {self.backbone.grid_h-1}] x [0, {self.backbone.grid_w-1}]
- PIXEL coords: [0, {self.config['model']['input_size']-1}] x [0, {self.config['model']['input_size']-1}]

Conversion:
- Patch → Pixel: pixel = patch * {self.backbone.patch_size} + {self.backbone.patch_size/2}
- Pixel → Patch: patch = (pixel - {self.backbone.patch_size/2}) / {self.backbone.patch_size}

Saliency Map:
- Mean: {saliency.mean():.4f}
- Max: {saliency.max():.4f}
- Min: {saliency.min():.4f}

This is GRID-ALIGNED following DINO-VO!
        """

        axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                       verticalalignment='center')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to {output_path}")
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize with proper coordinates')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='visualization_fixed.png')

    args = parser.parse_args()

    visualizer = FixedVisualizer(args.checkpoint, args.config)
    features = visualizer.extract_features(args.image)
    visualizer.visualize_frame(features, args.output)

    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print("="*70)
    print("\nCheck that:")
    print("  1. Keypoints align to patch grid boundaries")
    print("  2. Saliency map is 28x28 (not messy)")
    print("  3. Keypoints are distributed (not all in corner)")
    print("="*70)


if __name__ == "__main__":
    main()