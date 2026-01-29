"""
Clean Match Visualization - ACTUALLY USES QUALITY FILTERING
Shows only high-quality matches using descriptor sim + saliency + uncertainty
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
from models.uncertainty_estimator import UncertaintyEstimator


class SmartMatcher:
    """Matcher that uses descriptor + saliency + uncertainty for quality"""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load models
        self.backbone = DinoBackbone(
            model_name=self.config["model"]["backbone"],
            input_size=self.config["model"]["input_size"],
            freeze=True
        ).to(self.device)

        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config["model"]["selector_hidden"]
        ).to(self.device)

        self.refiner = DescriptorRefiner(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config["model"]["refiner_hidden"],
            output_dim=self.config["model"]["descriptor_dim"]
        ).to(self.device)

        self.uncertainty = UncertaintyEstimator(
            dino_dim=self.backbone.embed_dim,
            descriptor_dim=self.config["model"]["descriptor_dim"]
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.selector.load_state_dict(checkpoint["selector_state_dict"])
        self.refiner.load_state_dict(checkpoint["refiner_state_dict"])

        # Uncertainty estimator might not be in old checkpoints
        if "uncertainty_state_dict" in checkpoint:
            self.uncertainty.load_state_dict(checkpoint["uncertainty_state_dict"])
            print("✓ Loaded uncertainty estimator")
        else:
            print("⚠️  No uncertainty estimator in checkpoint (using random init)")

        self.selector.eval()
        self.refiner.eval()
        self.uncertainty.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.config["model"]["input_size"],
                             self.config["model"]["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract_features(self, image_path: str):
        """Extract keypoints, descriptors, saliency, and uncertainty - WITH DIAGNOSTICS"""

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # DINOv3 features
        patch_features = self.backbone(image_tensor)

        # Saliency and keypoint selection
        saliency_map = self.selector(patch_features)
        keypoints_patch, saliency_scores = self.selector.select_keypoints(
            saliency_map,
            num_keypoints=self.config["model"]["num_keypoints"]
        )

        # 🔍 DEBUG: Print what's actually happening
        sal_map_np = saliency_map[0, :, :, 0].cpu().numpy()
        sal_scores_np = saliency_scores[0].cpu().numpy()
        kpts_patch_np = keypoints_patch[0].cpu().numpy()

        print(f"\n  🔍 Saliency Diagnostics for {Path(image_path).name}:")
        print(f"    Saliency Map: mean={sal_map_np.mean():.3f}, max={sal_map_np.max():.3f}, min={sal_map_np.min():.3f}")
        print(f"    Selected keypoint scores: mean={sal_scores_np.mean():.3f}, min={sal_scores_np.min():.3f}, max={sal_scores_np.max():.3f}")
        print(f"    Keypoints with score < 0.3: {(sal_scores_np < 0.3).sum()}/{len(sal_scores_np)}")
        print(f"    Keypoints with score < 0.5: {(sal_scores_np < 0.5).sum()}/{len(sal_scores_np)}")
        print(f"    Keypoints with score > 0.7: {(sal_scores_np > 0.7).sum()}/{len(sal_scores_np)}")

        # 🔍 Check if keypoints match saliency map
        # For first 5 keypoints, check their saliency map values
        print(f"    First 5 keypoints patch coords and their saliency map values:")
        for i in range(min(5, len(kpts_patch_np))):
            x, y = kpts_patch_np[i]
            x_int, y_int = int(round(x)), int(round(y))
            x_int = np.clip(x_int, 0, 27)
            y_int = np.clip(y_int, 0, 27)
            map_val = sal_map_np[y_int, x_int]
            score_val = sal_scores_np[i]
            print(f"      KPt {i}: patch ({x:.1f}, {y:.1f}) -> map[{y_int},{x_int}]={map_val:.3f}, score={score_val:.3f}")

        # Extract DINO features at keypoints
        dino_at_kpts = self.backbone.extract_at_keypoints(patch_features, keypoints_patch)

        # Descriptors
        descriptors = self.refiner(dino_at_kpts)

        # Uncertainty (confidence)
        confidence = self.uncertainty(dino_at_kpts, descriptors)

        # Convert to pixel coordinates (model input space) then scale to original image size
        keypoints_pixel = self.backbone.patch_to_pixel(keypoints_patch)
        input_size = self.config["model"]["input_size"]
        scale = np.array([orig_w / input_size, orig_h / input_size], dtype=np.float32)
        keypoints_pixel = keypoints_pixel * torch.tensor(scale, device=keypoints_pixel.device)

        return {
            "image": image,
            "keypoints": keypoints_pixel[0].cpu().numpy(),  # (N, 2) scaled to original image
            "descriptors": descriptors[0].cpu().numpy(),    # (N, 128)
            "saliency": saliency_scores[0].cpu().numpy(),   # (N,)
            "confidence": confidence[0, :, 0].cpu().numpy(),  # (N,)
            "saliency_map": saliency_map[0, :, :, 0].cpu().numpy()  # (28, 28)
        }

    @staticmethod
    def save_saliency_debug_viz(feat1, feat2, output_path):
        """Save a diagnostic image showing saliency map and selected keypoints"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # Frame 1 - Image with keypoints colored by saliency
        ax = axes[0, 0]
        ax.imshow(np.array(feat1["image"]))
        scatter = ax.scatter(feat1["keypoints"][:, 0], feat1["keypoints"][:, 1],
                            c=feat1["saliency"], cmap='hot', s=50,
                            vmin=0, vmax=1, edgecolors='white', linewidths=1)
        ax.set_title(f'Frame 1 Keypoints (colored by saliency)\nMean sal: {feat1["saliency"].mean():.3f}')
        ax.axis('off')
        plt.colorbar(scatter, ax=ax)

        # Frame 1 - Saliency map
        ax = axes[0, 1]
        im = ax.imshow(feat1["saliency_map"], cmap='hot', interpolation='nearest')
        ax.set_title(f'Frame 1 Saliency Map\nMean: {feat1["saliency_map"].mean():.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        # Frame 2 - Image with keypoints
        ax = axes[1, 0]
        ax.imshow(np.array(feat2["image"]))
        scatter = ax.scatter(feat2["keypoints"][:, 0], feat2["keypoints"][:, 1],
                            c=feat2["saliency"], cmap='hot', s=50,
                            vmin=0, vmax=1, edgecolors='white', linewidths=1)
        ax.set_title(f'Frame 2 Keypoints (colored by saliency)\nMean sal: {feat2["saliency"].mean():.3f}')
        ax.axis('off')
        plt.colorbar(scatter, ax=ax)

        # Frame 2 - Saliency map
        ax = axes[1, 1]
        im = ax.imshow(feat2["saliency_map"], cmap='hot', interpolation='nearest')
        ax.set_title(f'Frame 2 Saliency Map\nMean: {feat2["saliency_map"].mean():.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        debug_path = output_path.replace('.png', '_SALIENCY_DEBUG.png')
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved saliency debug: {debug_path}")
        plt.close()

    @staticmethod
    def find_quality_matches(
        feat1: dict,
        feat2: dict,
        top_k: int = 50,
        min_desc_sim: float = 0.7,
        min_saliency: float = 0.3,
        min_confidence: float = 0.3,
        desc_weight: float = 0.5,
        sal_weight: float = 0.3,
        conf_weight: float = 0.2
    ):
        """
        Find top-k matches by COMBINED quality score.

        Quality = desc_weight * descriptor_sim
                + sal_weight * avg_saliency
                + conf_weight * avg_confidence

        Returns ONLY matches that pass ALL thresholds, sorted by quality.
        """

        desc1 = torch.from_numpy(feat1["descriptors"])  # (N, D)
        desc2 = torch.from_numpy(feat2["descriptors"])  # (M, D)

        # Cosine similarity matrix
        sim_matrix = torch.mm(desc1, desc2.t())  # (N, M)

        # Mutual nearest neighbors
        nn12_sim, nn12_idx = sim_matrix.max(dim=1)  # Best match in frame2 for each in frame1
        nn21_sim, nn21_idx = sim_matrix.max(dim=0)  # Best match in frame1 for each in frame2

        # Mutual: frame1[i] matches frame2[j] AND frame2[j] matches frame1[i]
        mutual_mask = (nn21_idx[nn12_idx] == torch.arange(desc1.shape[0]))

        idx1 = torch.nonzero(mutual_mask).squeeze(-1)
        idx2 = nn12_idx[idx1]

        if len(idx1) == 0:
            print("  ❌ No mutual matches found!")
            return np.zeros((0, 2), dtype=int), np.zeros(0)

        # Get quality scores for each match
        desc_similarities = sim_matrix[idx1, idx2].numpy()

        saliency1 = feat1["saliency"][idx1.numpy()]
        saliency2 = feat2["saliency"][idx2.numpy()]
        avg_saliency = (saliency1 + saliency2) / 2

        confidence1 = feat1["confidence"][idx1.numpy()]
        confidence2 = feat2["confidence"][idx2.numpy()]
        avg_confidence = (confidence1 + confidence2) / 2

        # FILTER: Apply minimum thresholds (THIS IS KEY!)
        valid_mask = (
            (desc_similarities >= min_desc_sim) &
            (avg_saliency >= min_saliency) &
            (avg_confidence >= min_confidence)
        )

        if valid_mask.sum() == 0:
            print(f"  ❌ No matches pass thresholds:")
            print(f"     desc_sim>={min_desc_sim}, saliency>={min_saliency}, conf>={min_confidence}")
            return np.zeros((0, 2), dtype=int), np.zeros(0)

        # Apply filter
        idx1 = idx1[valid_mask].numpy()
        idx2 = idx2[valid_mask].numpy()
        desc_similarities = desc_similarities[valid_mask]
        avg_saliency = avg_saliency[valid_mask]
        avg_confidence = avg_confidence[valid_mask]

        # Compute combined quality score
        quality_scores = (
            desc_weight * desc_similarities +
            sal_weight * avg_saliency +
            conf_weight * avg_confidence
        )

        # Sort by quality (best first) and take top-k
        sort_indices = np.argsort(-quality_scores)
        if len(sort_indices) > top_k:
            sort_indices = sort_indices[:top_k]

        matches = np.stack([idx1[sort_indices], idx2[sort_indices]], axis=1)
        quality_scores = quality_scores[sort_indices]

        print(f"  ✓ Found {len(matches)} high-quality matches (from {valid_mask.sum()} valid)")
        print(f"     Quality range: [{quality_scores.min():.3f}, {quality_scores.max():.3f}]")
        print(f"     Mean: desc={desc_similarities[sort_indices].mean():.3f}, "
              f"sal={avg_saliency[sort_indices].mean():.3f}, "
              f"conf={avg_confidence[sort_indices].mean():.3f}")

        return matches, quality_scores


def visualize_matches(
    feat1: dict,
    feat2: dict,
    matches: np.ndarray,
    quality: np.ndarray,
    output_path: str,
    show_all_keypoints: bool = False
):
    """
    Visualize matches with quality color-coding.

    Args:
        feat1, feat2: Feature dicts from extract_features()
        matches: (K, 2) match indices
        quality: (K,) quality scores [0-1]
        output_path: Where to save
        show_all_keypoints: If True, show ALL keypoints (not just matched ones)
    """

    img1 = np.array(feat1["image"])
    img2 = np.array(feat2["image"])
    kpts1 = feat1["keypoints"]
    kpts2 = feat2["keypoints"]

    if len(matches) == 0:
        print("  ⚠️  No matches to visualize!")
        return

    # Create side-by-side canvas
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    gap = 20
    canvas_h = max(h1, h2)
    canvas_w = w1 + gap + w2
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    canvas[:h1, :w1] = img1
    canvas[:h2, w1+gap:w1+gap+w2] = img2

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas)
    ax.axis("off")

    # Optional: Show ALL keypoints as small gray dots (for context)
    if show_all_keypoints:
        ax.scatter(kpts1[:, 0], kpts1[:, 1],
                  c='gray', s=5, alpha=0.3, marker='x')
        ax.scatter(kpts2[:, 0] + w1 + gap, kpts2[:, 1],
                  c='gray', s=5, alpha=0.3, marker='x')

    # Normalize quality for color mapping
    if quality.max() > quality.min():
        quality_norm = (quality - quality.min()) / (quality.max() - quality.min())
    else:
        quality_norm = np.ones_like(quality)

    # Draw matches (sorted by quality, so best ones drawn last/on top)
    for (i1, i2), q_norm, q_raw in zip(matches, quality_norm, quality):
        x1, y1 = kpts1[i1]
        x2, y2 = kpts2[i2]
        x2_shift = x2 + w1 + gap

        # Color: green=high quality, red=low quality
        color = plt.cm.RdYlGn(q_norm)

        # Line width by quality
        lw = 0.5 + 2.0 * q_norm

        # Draw line
        ax.plot([x1, x2_shift], [y1, y2],
               color=color, linewidth=lw, alpha=0.7, zorder=10)

        # Draw keypoints
        ax.scatter([x1, x2_shift], [y1, y2],
                  c=[color], s=30, alpha=0.9,
                  edgecolors='white', linewidths=1.0, zorder=20)

    # Title with statistics
    ax.set_title(
        f'Top {len(matches)} Quality Matches | '
        f'Quality: {quality.mean():.3f} ± {quality.std():.3f} | '
        f'Range: [{quality.min():.3f}, {quality.max():.3f}]',
        fontsize=14, fontweight='bold', pad=20
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                               norm=plt.Normalize(vmin=quality.min(), vmax=quality.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Match Quality', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize HIGH-QUALITY matches only')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--image_dir', type=str,
                       default='data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb')
    parser.add_argument('--spacing', type=int, default=5,
                       help='Frame spacing (1=consecutive, 5=skip 4 frames)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start from this frame')
    parser.add_argument('--num_pairs', type=int, default=5,
                       help='Number of frame pairs to visualize')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Show top K matches by quality')
    parser.add_argument('--output_dir', type=str, default='match_viz_clean')

    # Quality thresholds (STRICT - only show good matches!)
    parser.add_argument('--min_desc_sim', type=float, default=0.75,
                       help='Minimum descriptor similarity (0.75 = 75%)')
    parser.add_argument('--min_saliency', type=float, default=0.3,
                       help='Minimum saliency score (filters black monitor!)')
    parser.add_argument('--min_confidence', type=float, default=0.3,
                       help='Minimum uncertainty confidence')

    # Quality weights (must sum to 1.0)
    parser.add_argument('--desc_weight', type=float, default=0.5)
    parser.add_argument('--sal_weight', type=float, default=0.3)
    parser.add_argument('--conf_weight', type=float, default=0.2)

    parser.add_argument('--show_all_kpts', action='store_true',
                       help='Show all keypoints (not just matched ones)')

    args = parser.parse_args()

    # Setup
    image_dir = Path(args.image_dir)
    images = sorted(image_dir.glob('*.png'))
    if len(images) < args.spacing + 1:
        raise ValueError(f"Need at least {args.spacing + 1} images")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("HIGH-QUALITY MATCH VISUALIZATION")
    print("="*70)
    print(f"Images: {len(images)} in {image_dir.name}")
    print(f"Frame spacing: {args.spacing}")
    print(f"Top-k matches: {args.top_k}")
    print(f"\nQuality thresholds (strict!):")
    print(f"  Descriptor similarity ≥ {args.min_desc_sim}")
    print(f"  Saliency score ≥ {args.min_saliency}")
    print(f"  Confidence ≥ {args.min_confidence}")
    print(f"\nQuality weights:")
    print(f"  Descriptor: {args.desc_weight:.1%}")
    print(f"  Saliency: {args.sal_weight:.1%}")
    print(f"  Confidence: {args.conf_weight:.1%}")
    print("="*70 + "\n")

    # Load matcher
    matcher = SmartMatcher(args.checkpoint, args.config)

    # Process pairs
    pair_count = 0
    all_qualities = []

    for i in range(args.start_idx, len(images) - args.spacing, args.spacing):
        if pair_count >= args.num_pairs:
            break

        img1_path = images[i]
        img2_path = images[i + args.spacing]

        print(f"\nPair {pair_count + 1}/{args.num_pairs}:")
        print(f"  {img1_path.name} → {img2_path.name}")

        # Extract features
        feat1 = matcher.extract_features(str(img1_path))
        feat2 = matcher.extract_features(str(img2_path))

        out_name = f"pair_{pair_count:03d}_{img1_path.stem}_to_{img2_path.stem}.png"
        # matcher.save_saliency_debug_viz(feat1, feat2, str(output_dir / out_name))

        # Find quality matches
        matches, quality = matcher.find_quality_matches(
            feat1, feat2,
            top_k=args.top_k,
            min_desc_sim=args.min_desc_sim,
            min_saliency=args.min_saliency,
            min_confidence=args.min_confidence,
            desc_weight=args.desc_weight,
            sal_weight=args.sal_weight,
            conf_weight=args.conf_weight
        )

        if len(matches) > 0:
            all_qualities.extend(quality.tolist())

            # Visualize
            visualize_matches(
                feat1, feat2, matches, quality,
                output_path=str(output_dir / out_name),
                show_all_keypoints=args.show_all_kpts
            )

        pair_count += 1

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Pairs processed: {pair_count}")
    print(f"Total matches: {len(all_qualities)}")
    if all_qualities:
        print(f"Average quality: {np.mean(all_qualities):.3f}")
        print(f"Quality std: {np.std(all_qualities):.3f}")
        print(f"Range: [{np.min(all_qualities):.3f}, {np.max(all_qualities):.3f}]")
        print(f"High quality (>0.8): {sum(q > 0.8 for q in all_qualities)} matches")
    print(f"\n✓ Results saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()