"""
Compare Multiple Checkpoints
Helps you find the best epoch (not always the last one!)
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


def load_model(checkpoint_path: str, config: dict, device):
    """Load a single checkpoint"""
    backbone = DinoBackbone(
        model_name=config['model']['backbone'],
        input_size=config['model']['input_size'],
        freeze=True
    ).to(device)

    selector = KeypointSelector(
        input_dim=backbone.embed_dim,
        hidden_dim=config['model']['selector_hidden']
    ).to(device)

    refiner = DescriptorRefiner(
        input_dim=backbone.embed_dim,
        hidden_dim=config['model']['refiner_hidden'],
        output_dim=config['model']['descriptor_dim']
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    selector.load_state_dict(checkpoint['selector_state_dict'])
    refiner.load_state_dict(checkpoint['refiner_state_dict'])

    selector.eval()
    refiner.eval()

    return backbone, selector, refiner, checkpoint['epoch']


def extract_features(image_path: str, backbone, selector, refiner, transform, device):
    """Extract features from image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        dino_features = backbone(image_tensor)
        saliency_map = selector(dino_features)
        keypoints_patch, scores = selector.select_keypoints(saliency_map, num_keypoints=500)
        features_at_kpts = backbone.extract_at_keypoints(dino_features, keypoints_patch)
        descriptors = refiner(features_at_kpts)

    return {
        'saliency': saliency_map[0, :, :, 0].cpu().numpy(),
        'keypoints': keypoints_patch[0].cpu().numpy(),
        'scores': scores[0].cpu().numpy(),
        'descriptors': descriptors[0].cpu().numpy()
    }


def compare_checkpoints(checkpoint_paths: list, image_path: str, config_path: str, output_path: str):
    """Compare multiple checkpoints on same image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((config['model']['input_size'], config['model']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load original image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((config['model']['input_size'], config['model']['input_size']))

    # Compare each checkpoint
    results = []
    print(f"\nComparing {len(checkpoint_paths)} checkpoints...")

    for ckpt_path in checkpoint_paths:
        print(f"  Loading {Path(ckpt_path).name}...")
        backbone, selector, refiner, epoch = load_model(ckpt_path, config, device)
        features = extract_features(image_path, backbone, selector, refiner, transform, device)

        results.append({
            'epoch': epoch,
            'path': ckpt_path,
            'features': features
        })

    # Create comparison plot
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))

    if n_models == 1:
        axes = axes.reshape(2, 1)

    for idx, result in enumerate(results):
        features = result['features']
        saliency = features['saliency']

        # Row 1: Saliency maps
        im = axes[0, idx].imshow(saliency, cmap='hot', vmin=0, vmax=1)
        axes[0, idx].set_title(
            f"Epoch {result['epoch']}\n"
            f"Mean: {saliency.mean():.3f}, Max: {saliency.max():.3f}\n"
            f"Var: {saliency.var():.4f}",
            fontsize=10, fontweight='bold'
        )
        axes[0, idx].axis('off')
        plt.colorbar(im, ax=axes[0, idx])

        # Row 2: Original image with keypoints
        axes[1, idx].imshow(image)

        # Convert keypoints to pixel space
        kpts_pixel = features['keypoints'] * 16 + 8
        scores = features['scores']

        # Color by score
        scatter = axes[1, idx].scatter(
            kpts_pixel[:, 0], kpts_pixel[:, 1],
            c=scores, cmap='hot', s=20, alpha=0.7,
            vmin=0, vmax=1, edgecolors='white', linewidths=0.5
        )
        axes[1, idx].set_title(f"Keypoints (colored by score)", fontsize=10)
        axes[1, idx].axis('off')
        plt.colorbar(scatter, ax=axes[1, idx])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to {output_path}")

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Epoch':<10} {'Mean Sal':<12} {'Max Sal':<12} {'Variance':<12} {'Recommendation'}")
    print("-"*70)

    for result in results:
        sal = result['features']['saliency']
        mean_sal = sal.mean()
        max_sal = sal.max()
        var_sal = sal.var()

        # Simple recommendation
        if 0.10 < var_sal < 0.20 and 0.6 < max_sal < 0.95:
            rec = "✅ Excellent"
        elif 0.05 < var_sal < 0.25 and 0.4 < max_sal < 0.95:
            rec = "✓ Good"
        else:
            rec = "⚠️  Check"

        print(f"{result['epoch']:<10} {mean_sal:<12.4f} {max_sal:<12.4f} {var_sal:<12.4f} {rec}")

    print("="*70)
    print("\nGuidelines:")
    print("  - Variance: 0.10-0.20 is ideal (clear peaks vs background)")
    print("  - Max saliency: 0.60-0.95 is good (not saturated)")
    print("  - Mean saliency: 0.25-0.35 is typical")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare multiple checkpoints')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='Paths to checkpoint files (e.g., checkpoint_epoch10.pth checkpoint_epoch20.pth)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='checkpoint_comparison.png',
                       help='Output path')

    args = parser.parse_args()

    compare_checkpoints(args.checkpoints, args.image, args.config, args.output)


if __name__ == "__main__":
    main()