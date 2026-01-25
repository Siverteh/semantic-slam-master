"""
Visualize Matches Between Two Frames
Shows frame-to-frame matching quality
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import yaml
import cv2

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
import torchvision.transforms as transforms


class MatchVisualizer:
    """Visualize matches between frame pairs"""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

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
        """Extract features from single image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract DINOv3 features
        dino_features = self.backbone(image_tensor)

        # Keypoint selection
        saliency_map = self.selector(dino_features)
        keypoints_patch, scores = self.selector.select_keypoints(
            saliency_map,
            num_keypoints=self.config['model']['num_keypoints']
        )

        # Convert to pixel coordinates
        keypoints_pixel = self.backbone.patch_to_pixel(keypoints_patch)

        # Extract features at keypoints
        features_at_kpts = self.backbone.extract_at_keypoints(dino_features, keypoints_patch)

        # Refine descriptors
        descriptors = self.refiner(features_at_kpts)

        return {
            'image': image,
            'keypoints_pixel': keypoints_pixel[0].cpu().numpy(),
            'scores': scores[0].cpu().numpy(),
            'descriptors': descriptors[0].cpu().numpy()
        }

    def find_matches(self, desc1: np.ndarray, desc2: np.ndarray, ratio_thresh: float = 0.8):
        """Find matches using mutual nearest neighbors + ratio test"""
        # Compute similarity matrix
        sim_matrix = np.dot(desc1, desc2.T)

        # Mutual nearest neighbors
        nn12 = sim_matrix.argmax(axis=1)
        nn21 = sim_matrix.argmax(axis=0)

        matches = []
        for i in range(len(desc1)):
            j = nn12[i]
            if nn21[j] == i:  # Mutual match
                sim = sim_matrix[i, j]
                # Simple ratio test: check if this is significantly better than second best
                sims = sim_matrix[i].copy()
                sims[j] = -1  # Exclude best match
                second_best = sims.max()

                if sim > second_best * ratio_thresh:
                    matches.append((i, j, sim))

        return matches

    def visualize_matches(
        self,
        image1_path: str,
        image2_path: str,
        output_path: str = None,
        max_matches: int = 100
    ):
        """Visualize matches between two frames"""
        print(f"\nProcessing frame pair...")

        # Extract features
        feat1 = self.extract_features(image1_path)
        feat2 = self.extract_features(image2_path)

        # Find matches
        matches = self.find_matches(
            feat1['descriptors'],
            feat2['descriptors'],
            ratio_thresh=0.8
        )

        print(f"  Found {len(matches)} matches")

        # Sort by similarity
        matches.sort(key=lambda x: x[2], reverse=True)
        matches = matches[:max_matches]

        # Create visualization
        img1 = np.array(feat1['image'].resize((self.config['model']['input_size'],
                                               self.config['model']['input_size'])))
        img2 = np.array(feat2['image'].resize((self.config['model']['input_size'],
                                               self.config['model']['input_size'])))

        h, w = img1.shape[:2]
        canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
        canvas[:, :w] = img1
        canvas[:, w:] = img2

        # Draw matches
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        for idx, (i, j, sim) in enumerate(matches):
            # Color by similarity (green = high, red = low)
            color_val = int(255 * sim)
            color = (0, color_val, 255 - color_val)  # BGR

            pt1 = (int(feat1['keypoints_pixel'][i, 0]),
                   int(feat1['keypoints_pixel'][i, 1]))
            pt2 = (int(feat2['keypoints_pixel'][j, 0] + w),
                   int(feat2['keypoints_pixel'][j, 1]))

            # Draw line
            cv2.line(canvas_bgr, pt1, pt2, color, 1, cv2.LINE_AA)

            # Draw circles
            cv2.circle(canvas_bgr, pt1, 3, color, -1)
            cv2.circle(canvas_bgr, pt2, 3, color, -1)

        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

        # Create plot (matches only)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(canvas_rgb)
        ax.set_title(
            f'Matches (Top {len(matches)} of {len(feat1["descriptors"])} kpts)\n'
            f'Green = High Similarity, Red = Low Similarity',
            fontsize=12,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved to {output_path}")
        else:
            plt.show()

        plt.close()

        return len(matches)


def main():
    parser = argparse.ArgumentParser(description='Visualize frame-to-frame matches')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--frame1', type=str, required=True,
                       help='Path to first frame')
    parser.add_argument('--frame2', type=str, required=True,
                       help='Path to second frame')
    parser.add_argument('--output', type=str, default='matches.png',
                       help='Output path')
    parser.add_argument('--max_matches', type=int, default=100,
                       help='Max matches to display')

    args = parser.parse_args()

    visualizer = MatchVisualizer(args.checkpoint, args.config)

    num_matches = visualizer.visualize_matches(
        args.frame1,
        args.frame2,
        args.output,
        args.max_matches
    )

    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print("="*70)
    print(f"\nMatches found: {num_matches}")

    if num_matches > 100:
        print("✅ Excellent! Many good matches")
    elif num_matches > 50:
        print("✅ Good match quality")
    elif num_matches > 20:
        print("⚠️  Fair - consider retraining with better contrast")
    else:
        print("❌ Poor - check if frames are from same sequence")

    print("="*70)


if __name__ == "__main__":
    main()