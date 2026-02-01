"""
Stage 2 Model Visualization Script
Visualizes keypoints, saliency maps, and descriptor matches from trained Stage 2 model
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
from pathlib import Path
import yaml
import cv2
from tqdm import tqdm

from models.dino_backbone import DinoBackbone
from models.geometric_cnn import GeometricCNN, FeatureFusion
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
import torchvision.transforms as transforms


class Stage2Visualizer:
    """Visualizer for Stage 2 trained model"""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load all models
        print("\n📦 Loading models...")

        # Frozen DINOv3 backbone
        self.backbone = DinoBackbone(
            model_name=self.config['model']['backbone'],
            input_size=self.config['model']['input_size'],
            freeze=True
        ).to(self.device)

        # Geometric CNN
        self.geometric_cnn = GeometricCNN(
            input_channels=3,
            output_channels=self.config['model']['geometric_channels']
        ).to(self.device)

        # Feature Fusion
        self.fusion = FeatureFusion(
            semantic_dim=self.backbone.embed_dim,
            geometric_dim=self.config['model']['geometric_channels'],
            output_dim=self.config['model']['fusion_dim']
        ).to(self.device)

        # Keypoint Selector
        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            patch_size=self.backbone.patch_size
        ).to(self.device)

        # Descriptor Refiner
        self.refiner = DescriptorRefiner(
            input_dim=self.config['model']['fusion_dim'],
            hidden_dim=self.config['model']['refiner_hidden'],
            output_dim=self.config['model']['descriptor_dim'],
            num_layers=self.config['model']['refiner_layers']
        ).to(self.device)

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.geometric_cnn.load_state_dict(checkpoint['geometric_cnn_state_dict'])
        self.selector.load_state_dict(checkpoint['selector_state_dict'])
        self.fusion.load_state_dict(checkpoint['fusion_state_dict'])
        self.refiner.load_state_dict(checkpoint['refiner_state_dict'])

        print(f"✓ Loaded checkpoint from stage {checkpoint['stage']}, epoch {checkpoint['epoch']}")

        # Set to eval mode
        self.backbone.eval()
        self.geometric_cnn.eval()
        self.fusion.eval()
        self.selector.eval()
        self.refiner.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize(self.config['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def visualize_single_image(self, image_path: str, output_dir: str = 'visualization_output'):
        """Visualize keypoints and saliency for a single image"""
        Path(output_dir).mkdir(exist_ok=True)

        print(f"\n📸 Processing: {image_path}")

        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        h_orig, w_orig = image_np.shape[:2]

        # Resize for model
        image_resized = image.resize((self.config['model']['input_size'],
                                     self.config['model']['input_size']))
        image_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            # Semantic features (frozen)
            dino_feat = self.backbone(image_tensor)

            # Geometric features
            geo_feat = self.geometric_cnn(image_tensor)

            # Keypoint selection
            saliency = self.selector(dino_feat)
            kpts, scores = self.selector.select_keypoints(
                saliency,
                num_keypoints=self.config['model']['num_keypoints']
            )

            # Convert to pixel coordinates
            kpts_pixel = self.backbone.patch_to_pixel(kpts)

            # Extract features at keypoints
            dino_at_kpts = self.backbone.extract_at_keypoints(dino_feat, kpts)
            geo_at_kpts = self.geometric_cnn.extract_at_keypoints(geo_feat, kpts_pixel)

            # Fuse features
            fused = self.fusion(dino_at_kpts, geo_at_kpts)

            # Refine descriptors
            descriptors = self.refiner(fused)

        # Convert to numpy
        saliency_np = saliency[0, :, :, 0].cpu().numpy()
        kpts_np = kpts_pixel[0].cpu().numpy()
        scores_np = scores[0].cpu().numpy()
        desc_np = descriptors[0].cpu().numpy()

        # Get geometric features for visualization
        geo_feat_np = geo_feat[0].cpu().numpy()
        geo_norm = np.sqrt(np.sum(geo_feat_np**2, axis=0))

        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Stage 2 Visualization: {Path(image_path).name}', fontsize=16)

        # 1. Original image with keypoints
        ax = axes[0, 0]
        ax.imshow(image_resized)
        ax.scatter(kpts_np[:, 0], kpts_np[:, 1], c='red', s=20, marker='x', linewidth=2)
        ax.set_title('Detected Keypoints')
        ax.axis('off')

        # 2. Saliency map
        ax = axes[0, 1]
        im = ax.imshow(saliency_np, cmap='hot')
        ax.set_title('Saliency Map')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        # 3. Geometric features norm
        ax = axes[0, 2]
        im = ax.imshow(geo_norm, cmap='viridis')
        ax.set_title('Geometric Feature Norm')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        # 4. Keypoint scores
        ax = axes[1, 0]
        ax.imshow(image_resized)
        scatter = ax.scatter(kpts_np[:, 0], kpts_np[:, 1], c=scores_np,
                            s=30, cmap='jet', alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_title('Keypoint Scores')
        ax.axis('off')
        plt.colorbar(scatter, ax=ax)

        # 5. Edge map for comparison
        ax = axes[1, 1]
        edges = cv2.Canny(cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2GRAY), 50, 150)
        ax.imshow(edges, cmap='gray')
        ax.scatter(kpts_np[:, 0], kpts_np[:, 1], c='red', s=20, marker='x', linewidth=2)
        ax.set_title('Edge Detection + Keypoints')
        ax.axis('off')

        # 6. Descriptor space info
        ax = axes[1, 2]
        ax.text(0.5, 0.8, 'Descriptor Statistics', ha='center', fontsize=12, weight='bold',
               transform=ax.transAxes)
        desc_stats = f"""
        Num Keypoints: {len(kpts_np)}
        Descriptor Dim: {desc_np.shape[1]}

        Descriptor L2 norm:
          Mean: {np.mean(np.linalg.norm(desc_np, axis=1)):.4f}
          Std: {np.std(np.linalg.norm(desc_np, axis=1)):.4f}

        Saliency Stats:
          Min: {saliency_np.min():.4f}
          Max: {saliency_np.max():.4f}
          Mean: {saliency_np.mean():.4f}

        Keypoint Score Stats:
          Min: {scores_np.min():.4f}
          Max: {scores_np.max():.4f}
          Mean: {scores_np.mean():.4f}
        """
        ax.text(0.1, 0.5, desc_stats, ha='left', va='center', fontsize=10,
               family='monospace', transform=ax.transAxes)
        ax.axis('off')

        # Save figure
        output_path = Path(output_dir) / f'{Path(image_path).stem}_viz.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        plt.close()

        return {
            'keypoints': kpts_np,
            'scores': scores_np,
            'saliency': saliency_np,
            'descriptors': desc_np,
            'geo_norm': geo_norm
        }

    @torch.no_grad()
    def visualize_matches(self, image1_path: str, image2_path: str,
                         output_dir: str = 'visualization_output'):
        """Visualize matches between two images"""
        Path(output_dir).mkdir(exist_ok=True)

        print(f"\n🔗 Matching: {image1_path} ↔ {image2_path}")

        # Process both images
        results1 = self.visualize_single_image(image1_path, output_dir)
        results2 = self.visualize_single_image(image2_path, output_dir)

        # Find matches using mutual nearest neighbors
        desc1 = torch.tensor(results1['descriptors'], dtype=torch.float32).to(self.device)
        desc2 = torch.tensor(results2['descriptors'], dtype=torch.float32).to(self.device)

        # Similarity matrix
        sim = torch.mm(desc1, desc2.t())  # (N, M)

        # Mutual nearest neighbors
        nn12 = sim.argmax(dim=1)
        nn21 = sim.argmax(dim=0)
        mutual_mask = nn21[nn12] == torch.arange(len(desc1)).to(self.device)

        matched_idx1 = torch.nonzero(mutual_mask).squeeze(1).cpu().numpy()
        matched_idx2 = nn12[mutual_mask].cpu().numpy()

        print(f"Found {len(matched_idx1)} mutual matches")

        # Visualize matches
        img1 = cv2.imread(image1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (self.config['model']['input_size'],
                                 self.config['model']['input_size']))

        img2 = cv2.imread(image2_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, (self.config['model']['input_size'],
                                 self.config['model']['input_size']))

        # Horizontal concatenation
        h, w = img1.shape[:2]
        canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
        canvas[:, :w] = img1
        canvas[:, w:] = img2

        # Draw matches
        for i in range(min(len(matched_idx1), 50)):  # Show top 50 matches
            idx1 = matched_idx1[i]
            idx2 = matched_idx2[i]

            pt1 = tuple(results1['keypoints'][idx1].astype(int))
            pt2 = tuple((results2['keypoints'][idx2] + w).astype(int))

            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2, 3, color, -1)
            cv2.line(canvas, pt1, pt2, color, 1)

        # Save
        output_path = Path(output_dir) / 'matches.png'
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), canvas_rgb)
        print(f"✓ Saved matches visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Stage 2 Model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/stage2_best.pth',
                       help='Path to stage 2 checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Single image to visualize')
    parser.add_argument('--image1', type=str, default=None,
                       help='First image for matching')
    parser.add_argument('--image2', type=str, default=None,
                       help='Second image for matching')
    parser.add_argument('--dataset-dir', type=str, default=None,
                       help='Directory with dataset to visualize')
    parser.add_argument('--output-dir', type=str, default='visualization_output',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num-images', type=int, default=5,
                       help='Number of images to visualize from dataset')

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = Stage2Visualizer(args.checkpoint, args.config, device=args.device)

    # Single image
    if args.image:
        visualizer.visualize_single_image(args.image, args.output_dir)

    # Match two images
    elif args.image1 and args.image2:
        visualizer.visualize_matches(args.image1, args.image2, args.output_dir)

    # Dataset visualization
    elif args.dataset_dir:
        dataset_path = Path(args.dataset_dir)
        image_files = list(dataset_path.glob('rgb/*.png')) or list(dataset_path.glob('*.png'))
        image_files = sorted(image_files)[:args.num_images]

        if not image_files:
            print("❌ No images found in dataset directory")
            return

        print(f"📁 Visualizing {len(image_files)} images from {args.dataset_dir}")
        for img_path in image_files:
            visualizer.visualize_single_image(str(img_path), args.output_dir)

    else:
        # Default: visualize from TUM dataset
        data_root = Path('data/tum_rgbd')
        sequences = ['rgbd_dataset_freiburg1_desk', 'rgbd_dataset_freiburg1_xyz']

        for seq in sequences:
            seq_dir = data_root / seq
            if seq_dir.exists():
                print(f"\n🎬 Visualizing sequence: {seq}")
                rgb_dir = seq_dir / 'rgb'
                if rgb_dir.exists():
                    images = sorted(list(rgb_dir.glob('*.png')))[:args.num_images]
                    for img_path in images:
                        visualizer.visualize_single_image(str(img_path), args.output_dir)


if __name__ == '__main__':
    main()
