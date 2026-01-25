"""
Enhanced Visualization Script
Shows saliency alignment with edges and corners
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


class EdgeAwareVisualizer:
    """Visualizer that shows edge/corner alignment"""

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
        """Extract features and compute edge maps"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image.resize((self.config['model']['input_size'],
                                         self.config['model']['input_size'])))
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

        # Compute edge map for comparison
        edge_map = self._compute_edge_map(image_np)

        return {
            'image': image_np,
            'saliency_map': saliency_map[0, :, :, 0].cpu().numpy(),
            'keypoints_patch': keypoints_patch[0].cpu().numpy(),
            'keypoints_pixel': keypoints_pixel[0].cpu().numpy(),
            'scores': scores[0].cpu().numpy(),
            'edge_map': edge_map
        }

    def _compute_edge_map(self, image_np: np.ndarray) -> np.ndarray:
        """Compute edge map using Sobel"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)

        # Downsample to saliency resolution (28x28)
        edge_map_28 = cv2.resize(edge_magnitude, (28, 28), interpolation=cv2.INTER_AREA)

        return edge_map_28

    def visualize_frame(self, features: dict, output_path: str = None):
        """Visualize with edge alignment analysis"""
        # Create output directory
        if output_path:
            output_dir = Path("visualization_output")
            output_dir.mkdir(exist_ok=True)
            base_name = Path(output_path).stem
        else:
            output_dir = Path("visualization_output")
            output_dir.mkdir(exist_ok=True)
            base_name = "visualization"

        image_np = features['image']
        saliency = features['saliency_map']
        edge_map = features['edge_map']
        kpts_pixel = features['keypoints_pixel']
        scores = features['scores']

        # Compute statistics needed for multiple plots
        correlation = np.corrcoef(edge_map.flatten(), saliency.flatten())[0, 1]
        variance = saliency.var()
        std = saliency.std()
        high_sal_pct = (saliency > 0.6).sum() / saliency.size * 100
        top_k = 100
        top_sal_indices = np.argsort(saliency.flatten())[-top_k:]
        top_sal_edge_strength = edge_map.flatten()[top_sal_indices].mean()
        sal_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        edge_norm = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
        diff = np.abs(sal_norm - edge_norm)

        # Save individual plots
        print("Saving individual visualizations...")

        # 1. Original image with keypoints
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(image_np)
        scatter = ax1.scatter(
            kpts_pixel[:, 0], kpts_pixel[:, 1],
            c=scores, cmap='hot', s=30, alpha=0.8,
            edgecolors='white', linewidths=0.5
        )
        ax1.set_title('Keypoints (colored by score)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(scatter, ax=ax1, fraction=0.046)
        plt.savefig(output_dir / f"{base_name}_1_keypoints.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Saliency map
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        im2 = ax2.imshow(saliency, cmap='hot', interpolation='nearest')
        ax2.set_title(f'Saliency Map\nMean: {saliency.mean():.3f}, Max: {saliency.max():.3f}',
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        plt.savefig(output_dir / f"{base_name}_2_saliency_map.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Edge map
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        im3 = ax3.imshow(edge_map, cmap='gray', interpolation='nearest')
        ax3.set_title('Edge Map (Ground Truth)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        plt.savefig(output_dir / f"{base_name}_3_edge_map.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 4. Saliency overlay
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        ax4.imshow(image_np)
        saliency_upsampled = cv2.resize(saliency, (448, 448), interpolation=cv2.INTER_LINEAR)
        ax4.imshow(saliency_upsampled, cmap='hot', alpha=0.5)
        ax4.set_title('Saliency Overlay', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.savefig(output_dir / f"{base_name}_4_saliency_overlay.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 5. Edge overlay
        fig5, ax5 = plt.subplots(figsize=(8, 8))
        ax5.imshow(image_np)
        edge_upsampled = cv2.resize(edge_map, (448, 448), interpolation=cv2.INTER_LINEAR)
        ax5.imshow(edge_upsampled, cmap='gray', alpha=0.5)
        ax5.set_title('Edge Overlay', fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.savefig(output_dir / f"{base_name}_5_edge_overlay.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 6. Difference map
        fig6, ax6 = plt.subplots(figsize=(8, 8))
        im6 = ax6.imshow(diff, cmap='RdYlGn_r', interpolation='nearest')
        ax6.set_title(f'Alignment Error\nMean diff: {diff.mean():.3f}',
                     fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        plt.savefig(output_dir / f"{base_name}_6_alignment_error.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 7. Saliency histogram
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        ax7.hist(saliency.flatten(), bins=50, alpha=0.7, edgecolor='black', color='red')
        ax7.axvline(saliency.mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
        ax7.axvline(saliency.max(), color='green', linestyle='--', linewidth=2, label='Max')
        ax7.set_xlabel('Saliency Value')
        ax7.set_ylabel('Count')
        ax7.set_title('Saliency Distribution', fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"{base_name}_7_saliency_histogram.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 8. Correlation plot
        fig8, ax8 = plt.subplots(figsize=(8, 6))
        ax8.scatter(edge_map.flatten(), saliency.flatten(), alpha=0.3, s=10)
        z = np.polyfit(edge_map.flatten(), saliency.flatten(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(edge_map.min(), edge_map.max(), 100)
        ax8.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Corr: {correlation:.3f}')
        ax8.set_xlabel('Edge Strength')
        ax8.set_ylabel('Saliency')
        ax8.set_title('Edge-Saliency Correlation', fontsize=12, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"{base_name}_8_correlation_plot.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 9. Statistics text
        fig9, ax9 = plt.subplots(figsize=(8, 8))
        ax9.axis('off')
        stats_text = f"""
SALIENCY STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Mean:      {saliency.mean():.4f}
Median:    {np.median(saliency):.4f}
Max:       {saliency.max():.4f}
Min:       {saliency.min():.4f}
Std Dev:   {std:.4f}
Variance:  {variance:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━

QUALITY METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Edge Correlation:    {correlation:.3f}
High Sal. Regions:   {high_sal_pct:.1f}%
Top-{top_k} Edge Str: {top_sal_edge_strength:.3f}
━━━━━━━━━━━━━━━━━━━━━━━━━━

TARGET RANGES:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Mean:      0.40-0.50 {'✅' if 0.40 <= saliency.mean() <= 0.50 else '❌'}
Max:       0.70-0.90 {'✅' if 0.70 <= saliency.max() <= 0.90 else '❌'}
Variance:  0.18-0.28 {'✅' if 0.18 <= variance <= 0.28 else '❌'}
Correlation: >0.40   {'✅' if correlation > 0.40 else '❌'}
━━━━━━━━━━━━━━━━━━━━━━━━━━

{'✅ GOOD: Learning edges/corners!' if correlation > 0.40 and variance > 0.15 else '❌ NEEDS WORK: Low correlation/variance'}
        """
        ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        plt.savefig(output_dir / f"{base_name}_9_statistics.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved 9 individual visualizations to {output_dir}/")

        # Now create the combined figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: Original views
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_np)
        scatter = ax1.scatter(
            kpts_pixel[:, 0], kpts_pixel[:, 1],
            c=scores, cmap='hot', s=30, alpha=0.8,
            edgecolors='white', linewidths=0.5
        )
        ax1.set_title('Keypoints (colored by score)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(scatter, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(saliency, cmap='hot', interpolation='nearest')
        ax2.set_title(f'Saliency Map\nMean: {saliency.mean():.3f}, Max: {saliency.max():.3f}',
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(edge_map, cmap='gray', interpolation='nearest')
        ax3.set_title('Edge Map (Ground Truth)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # Row 2: Overlays
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(image_np)
        ax4.imshow(saliency_upsampled, cmap='hot', alpha=0.5)
        ax4.set_title('Saliency Overlay', fontsize=12, fontweight='bold')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(image_np)
        ax5.imshow(edge_upsampled, cmap='gray', alpha=0.5)
        ax5.set_title('Edge Overlay', fontsize=12, fontweight='bold')
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(diff, cmap='RdYlGn_r', interpolation='nearest')
        ax6.set_title(f'Alignment Error\nMean diff: {diff.mean():.3f}',
                     fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)

        # Row 3: Analysis
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(saliency.flatten(), bins=50, alpha=0.7, edgecolor='black', color='red')
        ax7.axvline(saliency.mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
        ax7.axvline(saliency.max(), color='green', linestyle='--', linewidth=2, label='Max')
        ax7.set_xlabel('Saliency Value')
        ax7.set_ylabel('Count')
        ax7.set_title('Saliency Distribution', fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[2, 1])
        ax8.scatter(edge_map.flatten(), saliency.flatten(), alpha=0.3, s=10)
        ax8.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Corr: {correlation:.3f}')
        ax8.set_xlabel('Edge Strength')
        ax8.set_ylabel('Saliency')
        ax8.set_title('Edge-Saliency Correlation', fontsize=12, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.suptitle('Edge-Aware Saliency Analysis', fontsize=16, fontweight='bold', y=0.98)

        # Save combined figure
        combined_path = output_dir / f"{base_name}_combined.png"
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved combined visualization to {combined_path}")
        plt.close()

        # Print summary
        print("\n" + "="*70)
        print("SALIENCY ANALYSIS SUMMARY")
        print("="*70)
        print(f"Edge Correlation: {correlation:.3f} {'✅ GOOD' if correlation > 0.40 else '❌ LOW'}")
        print(f"Variance:         {variance:.3f} {'✅ GOOD' if 0.18 <= variance <= 0.28 else '❌ OUT OF RANGE'}")
        print(f"Max Saliency:     {saliency.max():.3f} {'✅ GOOD' if 0.70 <= saliency.max() <= 0.90 else '❌ OUT OF RANGE'}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize edge-aware saliency')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--image', type=str, default='test_image.png')
    parser.add_argument('--output', type=str, default='edge_aware_viz.png')

    args = parser.parse_args()

    visualizer = EdgeAwareVisualizer(args.checkpoint, args.config)
    features = visualizer.extract_features(args.image)
    visualizer.visualize_frame(features, args.output)

    print("\n✓ Check the visualization_output folder for all visualizations!")


if __name__ == "__main__":
    main()