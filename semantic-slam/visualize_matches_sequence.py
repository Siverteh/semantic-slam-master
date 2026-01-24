"""
Visualize keypoint matches across an image sequence.

This script:
- extracts DINOv3 patch features
- predicts saliency via the KeypointSelector head
- samples descriptors via DescriptorRefiner
- matches consecutive frames with mutual nearest neighbors
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
import matplotlib.pyplot as plt

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

        return {
            "image": image,
            "image_tensor": image_tensor,
            "saliency": saliency[0, :, :, 0].cpu().numpy(),
            "keypoints_patch": keypoints_patch[0].cpu().numpy(),
            "keypoints_pixel": keypoints_pixel[0].cpu().numpy(),
            "scores": scores[0].cpu().numpy(),
            "descriptors": descriptors[0].cpu().numpy()
        }

    @staticmethod
    def match_mutual_nn(desc1: np.ndarray, desc2: np.ndarray):
        desc1_t = torch.from_numpy(desc1)
        desc2_t = torch.from_numpy(desc2)
        sim = torch.mm(desc1_t, desc2_t.t())
        nn12 = sim.argmax(dim=1)
        nn21 = sim.argmax(dim=0)
        mutual = nn21[nn12] == torch.arange(desc1_t.shape[0])
        idx1 = torch.nonzero(mutual).squeeze(1)
        idx2 = nn12[idx1]
        if idx1.numel() == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        scores = sim[idx1, idx2].cpu().numpy()
        matches = torch.stack([idx1, idx2], dim=1).cpu().numpy()
        return matches, scores


def visualize_matches(
    image1: Image.Image,
    image2: Image.Image,
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    matches: np.ndarray,
    match_scores: np.ndarray,
    output_path: str = None,
    max_matches: int = 200,
    gap: int = 20
):
    img1 = np.array(image1)
    img2 = np.array(image2)

    if len(matches) == 0:
        print("No matches found.")
        return

    if len(matches) > max_matches:
        order = np.argsort(-match_scores)[:max_matches]
        matches = matches[order]
        match_scores = match_scores[order]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + gap + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 + gap:w1 + gap + w2] = img2

    plt.figure(figsize=(14, 8))
    plt.imshow(canvas)
    plt.axis("off")

    score_norm = (match_scores - match_scores.min()) / (np.ptp(match_scores) + 1e-8)

    for (i1, i2), s in zip(matches, score_norm):
        x1, y1 = kpts1[i1]
        x2, y2 = kpts2[i2]
        x2 += w1 + gap
        color = plt.cm.viridis(s)
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=1.0, alpha=0.8)
        plt.scatter([x1, x2], [y1, y2], color=color, s=8, alpha=0.9)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize matches across a sequence")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.png")
    parser.add_argument("--step", type=int, default=1, help="Frame step for pairs")
    parser.add_argument("--max_pairs", type=int, default=10)
    parser.add_argument("--max_matches", type=int, default=200)
    parser.add_argument("--gap", type=int, default=20, help="Gap in pixels between images")
    parser.add_argument("--output_dir", type=str, default="match_viz")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    images = sorted(image_dir.glob(args.pattern))
    if len(images) < 2:
        raise ValueError(f"Need at least 2 images in {image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matcher = SequenceMatcher(args.checkpoint, args.config, device=args.device)

    pair_count = 0
    for i in range(0, len(images) - args.step, args.step):
        if pair_count >= args.max_pairs:
            break

        img_path1 = images[i]
        img_path2 = images[i + args.step]

        f1 = matcher.extract(str(img_path1))
        f2 = matcher.extract(str(img_path2))

        matches, match_scores = matcher.match_mutual_nn(
            f1["descriptors"], f2["descriptors"]
        )

        out_name = f"matches_{img_path1.stem}_to_{img_path2.stem}.png"
        out_path = output_dir / out_name

        visualize_matches(
            f1["image"],
            f2["image"],
            f1["keypoints_pixel"],
            f2["keypoints_pixel"],
            matches,
            match_scores,
            output_path=str(out_path),
            max_matches=args.max_matches,
            gap=args.gap
        )

        pair_count += 1

    print(f"✓ Done. Wrote {pair_count} match visualizations to {output_dir}")


if __name__ == "__main__":
    main()
