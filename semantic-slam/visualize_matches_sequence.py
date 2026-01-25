"""
FIXED: Visualize keypoint matches with saliency-weighted quality.

Key improvements:
- Filters matches by BOTH descriptor similarity AND saliency
- Supports multiple frame spacings in one run
- Shows match quality distribution
- Color-codes by combined quality score
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner


class SequenceMatcher:
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

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

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.selector.load_state_dict(checkpoint["selector_state_dict"])
        self.refiner.load_state_dict(checkpoint["refiner_state_dict"])

        self.selector.eval()
        self.refiner.eval()

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
    def extract(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        patch_features = self.backbone(image_tensor)
        saliency = self.selector(patch_features)
        keypoints_patch, scores = self.selector.select_keypoints(
            saliency,
            num_keypoints=self.config["model"]["num_keypoints"]
        )

        descriptors = self.refiner(
            self.backbone.extract_at_keypoints(patch_features, keypoints_patch)
        )

        keypoints_pixel = self.backbone.patch_to_pixel(keypoints_patch)

        # Compute per-keypoint intensity on resized image (filters dark regions)
        resized = image.resize(
            (self.config["model"]["input_size"], self.config["model"]["input_size"])
        )
        gray = np.array(resized.convert("L"), dtype=np.float32) / 255.0
        kpts = keypoints_pixel[0].cpu().numpy()
        xs = np.clip(kpts[:, 0].round().astype(int), 0, gray.shape[1] - 1)
        ys = np.clip(kpts[:, 1].round().astype(int), 0, gray.shape[0] - 1)
        intensities = gray[ys, xs]

        return {
            "image": image,
            "saliency": saliency[0, :, :, 0].cpu().numpy(),
            "keypoints_pixel": kpts,
            "scores": scores[0].cpu().numpy(),
            "intensity": intensities,
            "descriptors": descriptors[0].cpu().numpy()
        }

    @staticmethod
    def match_with_quality(
        desc1: np.ndarray,
        desc2: np.ndarray,
        scores1: np.ndarray,
        scores2: np.ndarray,
        saliency_weight: float = 0.3,
        min_saliency: float = 0.2,
        min_descriptor_sim: float = 0.7,
        intensity1: Optional[np.ndarray] = None,
        intensity2: Optional[np.ndarray] = None,
        min_intensity: float = 0.1
    ):
        """
        FIXED: Match with combined descriptor similarity + saliency quality.

        Args:
            desc1: (N, D) descriptors from frame 1
            desc2: (M, D) descriptors from frame 2
            scores1: (N,) saliency scores for frame 1 keypoints
            scores2: (M,) saliency scores for frame 2 keypoints
            saliency_weight: How much to weight saliency (0.3 = 30%)
            min_saliency: Minimum saliency to consider (filter low-saliency matches)
            min_descriptor_sim: Minimum descriptor similarity
            intensity1: (N,) grayscale intensity for frame 1 keypoints
            intensity2: (M,) grayscale intensity for frame 2 keypoints
            min_intensity: Minimum brightness to consider (filters black regions)

        Returns:
            matches: (K, 2) match indices
            quality_scores: (K,) combined quality [0-1]
        """
        desc1_t = torch.from_numpy(desc1)
        desc2_t = torch.from_numpy(desc2)
        scores1_t = torch.from_numpy(scores1)
        scores2_t = torch.from_numpy(scores2)

        # Descriptor similarity matrix
        desc_sim = torch.mm(desc1_t, desc2_t.t())  # (N, M)

        # Mutual nearest neighbors
        nn12 = desc_sim.argmax(dim=1)
        nn21 = desc_sim.argmax(dim=0)
        mutual = nn21[nn12] == torch.arange(desc1_t.shape[0])

        idx1 = torch.nonzero(mutual).squeeze(1)
        idx2 = nn12[idx1]

        if idx1.numel() == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        # Get descriptor similarities for matches
        desc_similarities = desc_sim[idx1, idx2]

        # Get saliency scores for matches
        sal1 = scores1_t[idx1]
        sal2 = scores2_t[idx2]
        avg_saliency = (sal1 + sal2) / 2

        # FILTER: Remove low-saliency matches (this is the key fix!)
        saliency_mask = avg_saliency >= min_saliency
        desc_mask = desc_similarities >= min_descriptor_sim
        valid_mask = saliency_mask & desc_mask

        # Optional intensity filter (reject dark monitor matches)
        if intensity1 is not None and intensity2 is not None:
            int1 = torch.from_numpy(intensity1)[idx1]
            int2 = torch.from_numpy(intensity2)[idx2]
            avg_intensity = (int1 + int2) / 2
            intensity_mask = avg_intensity >= min_intensity
            valid_mask = valid_mask & intensity_mask

        if valid_mask.sum() == 0:
            print(f"⚠️  No matches above thresholds (min_sal={min_saliency}, min_desc={min_descriptor_sim})")
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        # Apply filter
        idx1 = idx1[valid_mask]
        idx2 = idx2[valid_mask]
        desc_similarities = desc_similarities[valid_mask]
        avg_saliency = avg_saliency[valid_mask]

        # Combined quality score: weighted sum of descriptor similarity + saliency
        quality_scores = (
            (1 - saliency_weight) * desc_similarities +
            saliency_weight * avg_saliency
        )

        matches = torch.stack([idx1, idx2], dim=1).cpu().numpy()
        quality_scores = quality_scores.cpu().numpy()

        return matches, quality_scores


def visualize_matches(
    image1: Image.Image,
    image2: Image.Image,
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    scores1: np.ndarray,
    scores2: np.ndarray,
    matches: np.ndarray,
    match_quality: np.ndarray,
    output_path: str = None,
    max_matches: int = 100,
    gap: int = 20,
    spacing: int = 1
):
    """Visualize matches with quality indicators"""

    img1 = np.array(image1)
    img2 = np.array(image2)

    if len(matches) == 0:
        print(f"⚠️  No matches found for spacing={spacing}")
        return

    # Sort by quality and take top-k
    if len(matches) > max_matches:
        order = np.argsort(-match_quality)[:max_matches]
        matches = matches[order]
        match_quality = match_quality[order]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + gap + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 + gap:w1 + gap + w2] = img2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(f'Matches (spacing={spacing}, n={len(matches)})',
                 fontsize=14, fontweight='bold')

    # Color by quality (green=high, red=low)
    quality_norm = (match_quality - match_quality.min()) / (np.ptp(match_quality) + 1e-8)

    for (i1, i2), q in zip(matches, quality_norm):
        x1, y1 = kpts1[i1]
        x2, y2 = kpts2[i2]
        x2_shifted = x2 + w1 + gap

        # Color: green=high quality, red=low quality
        color = plt.cm.RdYlGn(q)

        # Line thickness by quality
        linewidth = 0.5 + 1.5 * q

        ax.plot([x1, x2_shifted], [y1, y2], color=color,
                linewidth=linewidth, alpha=0.7)
        ax.scatter([x1, x2_shifted], [y1, y2], color=color,
                   s=15, alpha=0.9, edgecolors='white', linewidths=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")
        print(f"  Matches: {len(matches)}, Mean quality: {match_quality.mean():.3f}")
    else:
        plt.show()

    plt.close()


def process_spacing(
    matcher: SequenceMatcher,
    images: list,
    spacing: int,
    output_dir: Path,
    max_pairs: int,
    max_matches: int,
    gap: int,
    saliency_weight: float,
    min_saliency: float,
    min_descriptor_sim: float,
    min_intensity: float
):
    """Process all pairs for a given spacing"""

    print(f"\n{'='*70}")
    print(f"Processing spacing={spacing} frames")
    print(f"{'='*70}")

    spacing_dir = output_dir / f"spacing_{spacing}"
    spacing_dir.mkdir(exist_ok=True)

    pair_count = 0
    all_quality_scores = []

    for i in range(0, len(images) - spacing, spacing):
        if pair_count >= max_pairs:
            break

        img_path1 = images[i]
        img_path2 = images[i + spacing]

        print(f"\nPair {pair_count + 1}/{max_pairs}: {img_path1.name} → {img_path2.name}")

        f1 = matcher.extract(str(img_path1))
        f2 = matcher.extract(str(img_path2))

        matches, match_quality = matcher.match_with_quality(
            f1["descriptors"],
            f2["descriptors"],
            f1["scores"],
            f2["scores"],
            saliency_weight=saliency_weight,
            min_saliency=min_saliency,
            min_descriptor_sim=min_descriptor_sim,
            intensity1=f1["intensity"],
            intensity2=f2["intensity"],
            min_intensity=min_intensity
        )

        if len(matches) > 0:
            all_quality_scores.extend(match_quality.tolist())

        out_name = f"matches_{img_path1.stem}_to_{img_path2.stem}.png"
        out_path = spacing_dir / out_name

        visualize_matches(
            f1["image"],
            f2["image"],
            f1["keypoints_pixel"],
            f2["keypoints_pixel"],
            f1["scores"],
            f2["scores"],
            matches,
            match_quality,
            output_path=str(out_path),
            max_matches=max_matches,
            gap=gap,
            spacing=spacing
        )

        pair_count += 1

    # Summary for this spacing
    if all_quality_scores:
        print(f"\n{'='*70}")
        print(f"Summary for spacing={spacing}")
        print(f"{'='*70}")
        print(f"Total pairs processed: {pair_count}")
        print(f"Total matches: {len(all_quality_scores)}")
        print(f"Average quality: {np.mean(all_quality_scores):.3f}")
        print(f"Quality range: [{np.min(all_quality_scores):.3f}, {np.max(all_quality_scores):.3f}]")
        print(f"High quality matches (>0.8): {sum(q > 0.8 for q in all_quality_scores)}")
        print(f"Output: {spacing_dir}")

    return all_quality_scores


def main():
    parser = argparse.ArgumentParser(
        description="Visualize matches with quality filtering (fixed version)"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--image_dir", type=str,
                       default="data/tum_rgbd/rgbd_dataset_freiburg1_desk/rgb")
    parser.add_argument("--pattern", type=str, default="*.png")
    parser.add_argument("--spacings", nargs='+', type=int, default=[1, 5, 10, 15, 20],
                       help="Frame spacings to test (e.g., 1 5 10 20)")
    parser.add_argument("--max_pairs", type=int, default=1,
                       help="Max pairs per spacing")
    parser.add_argument("--max_matches", type=int, default=50,
                       help="Max matches to visualize per pair")
    parser.add_argument("--gap", type=int, default=20,
                       help="Gap in pixels between images")
    parser.add_argument("--output_dir", type=str, default="match_viz")
    parser.add_argument("--device", type=str, default="cuda")

    # Quality filtering parameters
    parser.add_argument("--saliency_weight", type=float, default=0.3,
                       help="Weight for saliency in quality score (0.3 = 30%)")
    parser.add_argument("--min_saliency", type=float, default=0.5,
                       help="Minimum saliency threshold (filters black monitor!)")
    parser.add_argument("--min_descriptor_sim", type=float, default=0.7,
                       help="Minimum descriptor similarity threshold")
    parser.add_argument("--min_intensity", type=float, default=0.15,
                       help="Minimum grayscale intensity (filters dark regions)")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    images = sorted(image_dir.glob(args.pattern))
    if len(images) < 2:
        raise ValueError(f"Need at least 2 images in {image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MATCH VISUALIZATION (FIXED - Saliency-Aware)")
    print("="*70)
    print(f"Images: {len(images)} in {image_dir}")
    print(f"Spacings to test: {args.spacings}")
    print(f"Pairs per spacing: {args.max_pairs}")
    print(f"Quality thresholds:")
    print(f"  Min saliency: {args.min_saliency}")
    print(f"  Min descriptor sim: {args.min_descriptor_sim}")
    print(f"  Saliency weight: {args.saliency_weight}")
    print(f"  Min intensity: {args.min_intensity}")
    print("="*70)

    matcher = SequenceMatcher(args.checkpoint, args.config, device=args.device)

    # Process each spacing
    spacing_results = {}
    for spacing in args.spacings:
        quality_scores = process_spacing(
            matcher,
            images,
            spacing,
            output_dir,
            args.max_pairs,
            args.max_matches,
            args.gap,
            args.saliency_weight,
            args.min_saliency,
            args.min_descriptor_sim,
            args.min_intensity
        )
        spacing_results[spacing] = quality_scores

    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    for spacing, scores in spacing_results.items():
        if scores:
            print(f"Spacing {spacing:2d}: {len(scores):4d} matches, "
                  f"avg quality: {np.mean(scores):.3f}, "
                  f"high quality: {sum(q > 0.8 for q in scores):3d}")
        else:
            print(f"Spacing {spacing:2d}: No matches found")

    print(f"\n✓ Done. Results in {output_dir}/")
    print(f"  - Each spacing has its own subfolder")
    print(f"  - Visualizations show match quality with color coding")
    print(f"  - Green = high quality, Red = low quality")


if __name__ == "__main__":
    main()