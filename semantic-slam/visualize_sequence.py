"""
Sequence Visualization Script
Visualize what your trained semantic heads are actually doing on a TUM sequence!

Shows:
1. Keypoint detections (from selector head)
2. Saliency maps (what the selector sees)
3. Descriptor similarities (what the refiner produces)
4. Confidence/uncertainty (what the estimator predicts)
5. Frame-to-frame matches
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import yaml
import cv2
from tqdm import tqdm

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
from models.uncertainty_estimator import UncertaintyEstimator
import torchvision.transforms as transforms


class SequenceVisualizer:
    """Visualize semantic SLAM heads on image sequences"""

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
            hidden_dim=self.config['model']['selector_hidden'],
            num_layers=self.config['model']['selector_layers']
        ).to(self.device)

        self.refiner = DescriptorRefiner(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['refiner_hidden'],
            output_dim=self.config['model']['descriptor_dim'],
            num_layers=self.config['model']['refiner_layers']
        ).to(self.device)

        self.estimator = UncertaintyEstimator(
            dino_dim=self.backbone.embed_dim,
            descriptor_dim=self.config['model']['descriptor_dim'],
            hidden_dim=self.config['model']['estimator_hidden']
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.selector.load_state_dict(checkpoint['selector_state_dict'])
        self.refiner.load_state_dict(checkpoint['refiner_state_dict'])
        self.estimator.load_state_dict(checkpoint['estimator_state_dict'])

        # Set to eval mode
        self.selector.eval()
        self.refiner.eval()
        self.estimator.eval()

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
        """Extract all features from an image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract DINOv3 features
        dino_features = self.backbone(image_tensor)  # (1, H, W, C)

        # Keypoint selection
        saliency_map = self.selector(dino_features)  # (1, H, W, 1)
        keypoints, scores = self.selector.select_keypoints(
            saliency_map,
            num_keypoints=self.config['model']['num_keypoints']
        )

        # Extract features at keypoints
        features_at_kpts = self.backbone.extract_at_keypoints(dino_features, keypoints)

        # Refine descriptors
        descriptors = self.refiner(features_at_kpts)

        # Estimate confidence
        confidence = self.estimator(features_at_kpts, descriptors)

        return {
            'image': image,
            'image_tensor': image_tensor,
            'dino_features': dino_features,
            'saliency_map': saliency_map[0, :, :, 0].cpu().numpy(),
            'keypoints': keypoints[0].cpu().numpy(),
            'scores': scores[0].cpu().numpy(),
            'descriptors': descriptors[0].cpu().numpy(),
            'confidence': confidence[0, :, 0].cpu().numpy()
        }

    def visualize_single_frame(self, features: dict, output_path: str = None):
        """
        Visualize all three heads on a single frame.

        Shows:
        1. Original image with keypoints (colored by confidence)
        2. Saliency map (what selector sees)
        3. Confidence map (what estimator predicts)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        image = features['image']
        image_np = np.array(image.resize((self.config['model']['input_size'],
                                         self.config['model']['input_size'])))

        # 1. Original with keypoints colored by confidence
        axes[0, 0].imshow(image_np)
        kpts = features['keypoints']
        conf = features['confidence']

        # Sort by confidence for better visualization
        sort_idx = np.argsort(conf)
        kpts_sorted = kpts[sort_idx]
        conf_sorted = conf[sort_idx]

        scatter = axes[0, 0].scatter(
            kpts_sorted[:, 0], kpts_sorted[:, 1],
            c=conf_sorted, cmap='hot', s=30, alpha=0.8, edgecolors='white', linewidths=0.5
        )
        axes[0, 0].set_title(f'Keypoints (N={len(kpts)}) colored by Confidence', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(scatter, ax=axes[0, 0], label='Confidence')

        # 2. Saliency map (what selector sees)
        saliency = features['saliency_map']
        im1 = axes[0, 1].imshow(saliency, cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title('Saliency Map (Selector Head)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Saliency')

        # 3. Confidence distribution
        axes[1, 0].hist(features['confidence'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        axes[1, 0].axvline(features['confidence'].mean(), color='red', linestyle='--',
                          label=f'Mean: {features["confidence"].mean():.3f}')
        axes[1, 0].set_xlabel('Confidence Score', fontsize=12)
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Confidence Distribution (Estimator Head)', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Keypoint score distribution
        axes[1, 1].hist(features['scores'], bins=50, alpha=0.7, edgecolor='black', color='coral')
        axes[1, 1].axvline(features['scores'].mean(), color='red', linestyle='--',
                          label=f'Mean: {features["scores"].mean():.3f}')
        axes[1, 1].set_xlabel('Keypoint Score', fontsize=12)
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Keypoint Score Distribution (Selector Head)', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_matches(self, features1: dict, features2: dict, output_path: str = None, top_k: int = 50):
        """
        Visualize descriptor matches between two frames.

        Shows what the descriptor refiner is doing!
        """
        # Get descriptors
        desc1 = features1['descriptors']
        desc2 = features2['descriptors']
        kpts1 = features1['keypoints']
        kpts2 = features2['keypoints']
        conf1 = features1['confidence']
        conf2 = features2['confidence']

        # Compute similarity matrix
        sim_matrix = np.dot(desc1, desc2.T)  # (N1, N2)

        # Find mutual nearest neighbors
        nn12 = sim_matrix.argmax(axis=1)
        nn21 = sim_matrix.argmax(axis=0)

        mutual_mask = nn21[nn12] == np.arange(len(desc1))

        # Get matches
        matches = []
        for i in np.where(mutual_mask)[0]:
            j = nn12[i]
            matches.append((i, j, sim_matrix[i, j]))

        matches.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
        matches = matches[:top_k]  # Top matches

        # Create visualization
        img1 = np.array(features1['image'].resize((self.config['model']['input_size'],
                                                   self.config['model']['input_size'])))
        img2 = np.array(features2['image'].resize((self.config['model']['input_size'],
                                                   self.config['model']['input_size'])))

        # Concatenate images side by side
        h, w = img1.shape[:2]
        canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
        canvas[:, :w] = img1
        canvas[:, w:] = img2

        # Draw matches
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR) if len(canvas.shape) == 3 else canvas

        for idx, (i, j, sim) in enumerate(matches):
            # Color based on similarity (green = high, red = low)
            color_val = int(255 * sim)
            color = (0, color_val, 255 - color_val)  # BGR

            pt1 = (int(kpts1[i, 0]), int(kpts1[i, 1]))
            pt2 = (int(kpts2[j, 0] + w), int(kpts2[j, 1]))

            # Draw line
            cv2.line(canvas_rgb, pt1, pt2, color, 1, cv2.LINE_AA)

            # Draw circles
            cv2.circle(canvas_rgb, pt1, 3, color, -1)
            cv2.circle(canvas_rgb, pt2, 3, color, -1)

        canvas_rgb = cv2.cvtColor(canvas_rgb, cv2.COLOR_BGR2RGB)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.imshow(canvas_rgb)
        ax.set_title(f'Descriptor Matches (Top {len(matches)} of {len(desc1)} mutual NN)\n'
                    f'Green = High Similarity, Red = Low Similarity',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add statistics
        stats_text = f'Total mutual matches: {mutual_mask.sum()}\n'
        stats_text += f'Mean similarity: {np.mean([m[2] for m in matches]):.3f}\n'
        stats_text += f'Frame 1: {len(kpts1)} keypoints\n'
        stats_text += f'Frame 2: {len(kpts2)} keypoints'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved matches to {output_path}")
        else:
            plt.show()

        plt.close()

        return mutual_mask.sum()

    def visualize_sequence(
        self,
        sequence_dir: str,
        output_dir: str,
        num_frames: int = 10,
        frame_spacing: int = 10
    ):
        """
        Visualize semantic features across an entire sequence.

        Args:
            sequence_dir: Path to TUM RGB-D sequence
            output_dir: Where to save visualizations
            num_frames: How many frames to process
            frame_spacing: Spacing between frames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get image files
        rgb_dir = Path(sequence_dir) / 'rgb'
        image_files = sorted(list(rgb_dir.glob('*.png')))

        if len(image_files) == 0:
            print(f"No images found in {rgb_dir}")
            return

        # Select frames
        indices = np.linspace(0, len(image_files)-1, num_frames, dtype=int)
        selected_files = [image_files[i] for i in indices]

        print(f"\n{'='*70}")
        print(f"VISUALIZING SEQUENCE: {Path(sequence_dir).name}")
        print(f"{'='*70}")
        print(f"Total frames: {len(image_files)}")
        print(f"Selected: {num_frames} frames")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")

        # Process each frame
        all_features = []
        print("Extracting features...")
        for i, img_file in enumerate(tqdm(selected_files)):
            features = self.extract_features(str(img_file))
            features['filename'] = img_file.name
            all_features.append(features)

            # Visualize single frame
            frame_output = output_dir / f"frame_{i:03d}_analysis.png"
            self.visualize_single_frame(features, str(frame_output))

        # Visualize frame-to-frame matches
        print("\nVisualizing frame-to-frame matches...")
        match_counts = []
        for i in range(len(all_features) - 1):
            match_output = output_dir / f"matches_{i:03d}_to_{i+1:03d}.png"
            num_matches = self.visualize_matches(
                all_features[i],
                all_features[i+1],
                str(match_output),
                top_k=100
            )
            match_counts.append(num_matches)

        # Create summary visualization
        self._create_summary(all_features, match_counts, output_dir)

        print(f"\n{'='*70}")
        print("✓ VISUALIZATION COMPLETE!")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}")
        print(f"Files created:")
        print(f"  - {num_frames} frame analyses")
        print(f"  - {num_frames-1} match visualizations")
        print(f"  - 1 summary plot")
        print(f"{'='*70}\n")

    def _create_summary(self, all_features: list, match_counts: list, output_dir: Path):
        """Create summary statistics plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        frames = np.arange(len(all_features))

        # 1. Number of keypoints per frame
        num_kpts = [len(f['keypoints']) for f in all_features]
        axes[0, 0].plot(frames, num_kpts, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Frame', fontsize=12)
        axes[0, 0].set_ylabel('Number of Keypoints', fontsize=12)
        axes[0, 0].set_title('Keypoints per Frame', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(np.mean(num_kpts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(num_kpts):.1f}')
        axes[0, 0].legend()

        # 2. Mean confidence per frame
        mean_conf = [f['confidence'].mean() for f in all_features]
        axes[0, 1].plot(frames, mean_conf, 'o-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel('Frame', fontsize=12)
        axes[0, 1].set_ylabel('Mean Confidence', fontsize=12)
        axes[0, 1].set_title('Mean Confidence per Frame', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(np.mean(mean_conf), color='red', linestyle='--',
                          label=f'Mean: {np.mean(mean_conf):.3f}')
        axes[0, 1].legend()

        # 3. Frame-to-frame matches
        if match_counts:
            axes[1, 0].plot(frames[:-1], match_counts, 'o-', linewidth=2, markersize=8, color='green')
            axes[1, 0].set_xlabel('Frame Pair', fontsize=12)
            axes[1, 0].set_ylabel('Number of Matches', fontsize=12)
            axes[1, 0].set_title('Frame-to-Frame Matches', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(np.mean(match_counts), color='red', linestyle='--',
                              label=f'Mean: {np.mean(match_counts):.1f}')
            axes[1, 0].legend()

        # 4. Saliency statistics
        mean_saliency = [f['saliency_map'].mean() for f in all_features]
        max_saliency = [f['saliency_map'].max() for f in all_features]

        axes[1, 1].plot(frames, mean_saliency, 'o-', linewidth=2, markersize=8,
                       label='Mean Saliency', color='purple')
        axes[1, 1].plot(frames, max_saliency, 's-', linewidth=2, markersize=8,
                       label='Max Saliency', color='magenta')
        axes[1, 1].set_xlabel('Frame', fontsize=12)
        axes[1, 1].set_ylabel('Saliency', fontsize=12)
        axes[1, 1].set_title('Saliency Statistics', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved summary to {output_dir / 'summary.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize semantic SLAM features on image sequences'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--sequence', type=str, required=True,
                       help='Path to TUM RGB-D sequence (e.g., data/tum_rgbd/rgbd_dataset_freiburg1_desk)')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_frames', type=int, default=10,
                       help='Number of frames to visualize')
    parser.add_argument('--frame_spacing', type=int, default=10,
                       help='Spacing between frames')

    args = parser.parse_args()

    # Create visualizer
    visualizer = SequenceVisualizer(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )

    # Visualize sequence
    visualizer.visualize_sequence(
        sequence_dir=args.sequence,
        output_dir=args.output,
        num_frames=args.num_frames,
        frame_spacing=args.frame_spacing
    )


if __name__ == "__main__":
    main()