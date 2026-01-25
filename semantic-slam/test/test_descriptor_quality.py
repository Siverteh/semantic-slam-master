"""
Test Descriptor Quality
Measures matching precision, recall, and inlier ratio.

Target: >80% inlier ratio, >70% precision@recall=0.5
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tum_dataset import TUMDataset
from models.descriptor_refiner import DescriptorRefiner
from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector


class DescriptorQualityTester:
    """Test descriptor matching quality"""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load models
        print("Loading models...")
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

        self.backbone.eval()
        self.selector.eval()
        self.refiner.eval()

        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor):
        """Extract keypoints and descriptors"""
        # DINOv3 features
        dino_features = self.backbone(image)

        # Keypoint selection
        saliency_map = self.selector(dino_features)
        keypoints_patch, scores = self.selector.select_keypoints(
            saliency_map,
            num_keypoints=self.config['model']['num_keypoints']
        )

        # Extract features at keypoints
        feat_at_kpts = self.backbone.extract_at_keypoints(dino_features, keypoints_patch)

        # Refine descriptors
        descriptors = self.refiner(feat_at_kpts)

        # Convert to pixel coordinates
        keypoints_pixel = self.backbone.patch_to_pixel(keypoints_patch)

        return (
            keypoints_pixel[0].cpu().numpy(),
            descriptors[0].cpu().numpy(),
            scores[0].cpu().numpy()
        )

    def find_mutual_nearest_neighbors(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.9
    ):
        """
        Find mutual nearest neighbor matches.

        Args:
            desc1: (N, D) descriptors from frame 1
            desc2: (M, D) descriptors from frame 2
            ratio_threshold: Lowe's ratio test threshold

        Returns:
            matches: (K, 2) array of indices [idx1, idx2]
            distances: (K,) array of cosine distances
        """
        # Compute similarity matrix (cosine similarity)
        sim_matrix = desc1 @ desc2.T  # (N, M)

        # Find nearest neighbors in both directions
        nn12_sim = sim_matrix.max(axis=1)
        nn12_idx = sim_matrix.argmax(axis=1)

        nn21_sim = sim_matrix.max(axis=0)
        nn21_idx = sim_matrix.argmax(axis=0)

        # Mutual nearest neighbors
        mutual_mask = nn21_idx[nn12_idx] == np.arange(len(desc1))

        # Apply ratio test (second best / best)
        sim_sorted = np.sort(sim_matrix, axis=1)[:, ::-1]
        ratio = sim_sorted[:, 1] / (sim_sorted[:, 0] + 1e-8)
        ratio_mask = ratio < ratio_threshold

        # Combine conditions
        valid_mask = mutual_mask & ratio_mask

        idx1 = np.where(valid_mask)[0]
        idx2 = nn12_idx[idx1]

        matches = np.stack([idx1, idx2], axis=1)
        distances = 1.0 - nn12_sim[idx1]  # Convert similarity to distance

        return matches, distances

    def compute_ground_truth_matches(
        self,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        H: np.ndarray,
        threshold: float = 3.0
    ):
        """
        Compute ground truth matches using pose.

        Args:
            kpts1: (N, 2) keypoints in frame 1
            kpts2: (M, 2) keypoints in frame 2
            H: (3, 3) homography from frame1 to frame2
            threshold: Distance threshold for match

        Returns:
            gt_matches: (K, 2) array of ground truth match indices
        """
        # Warp kpts1 to frame2
        kpts1_homo = np.concatenate([kpts1, np.ones((len(kpts1), 1))], axis=1)
        kpts1_warped = (H @ kpts1_homo.T).T
        kpts1_warped = kpts1_warped[:, :2] / kpts1_warped[:, 2:3]

        # Compute pairwise distances
        dists = np.linalg.norm(
            kpts1_warped[:, np.newaxis, :] - kpts2[np.newaxis, :, :],
            axis=2
        )

        # Find matches within threshold
        min_dists = dists.min(axis=1)
        min_idx = dists.argmin(axis=1)

        valid_mask = min_dists < threshold

        idx1 = np.where(valid_mask)[0]
        idx2 = min_idx[idx1]

        gt_matches = np.stack([idx1, idx2], axis=1)

        return gt_matches

    def evaluate_matches(
        self,
        pred_matches: np.ndarray,
        gt_matches: np.ndarray,
        num_kpts1: int,
        num_kpts2: int
    ):
        """
        Evaluate predicted matches against ground truth.

        Returns:
            dict with precision, recall, f1, inlier_ratio
        """
        # Convert to set for comparison
        pred_set = set(map(tuple, pred_matches))
        gt_set = set(map(tuple, gt_matches))

        # True positives: predicted matches that are correct
        tp = len(pred_set & gt_set)

        # False positives: predicted matches that are wrong
        fp = len(pred_set - gt_set)

        # False negatives: ground truth matches we missed
        fn = len(gt_set - pred_set)

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Inlier ratio: fraction of predicted matches that are correct
        inlier_ratio = tp / len(pred_matches) if len(pred_matches) > 0 else 0.0

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'inlier_ratio': inlier_ratio,
            'num_pred_matches': len(pred_matches),
            'num_gt_matches': len(gt_matches)
        }

    def test_sequence(
        self,
        sequence: str,
        num_pairs: int = 50,
        frame_spacing: int = 1
    ):
        """Test descriptor quality on a sequence"""
        print(f"\nTesting sequence: {sequence}")

        # Load dataset
        dataset = TUMDataset(
            dataset_root=self.config['dataset']['root'],
            sequence=sequence,
            input_size=self.config['model']['input_size'],
            frame_spacing=frame_spacing,
            max_frames=num_pairs + frame_spacing,
            augmentation=None,
            is_train=False
        )

        results = []

        for i in tqdm(range(min(num_pairs, len(dataset))), desc="Testing pairs"):
            batch = dataset[i]

            rgb1 = batch['rgb1'].unsqueeze(0).to(self.device)
            rgb2 = batch['rgb2'].unsqueeze(0).to(self.device)

            # Extract features
            kpts1, desc1, scores1 = self.extract_features(rgb1)
            kpts2, desc2, scores2 = self.extract_features(rgb2)

            # Find predicted matches
            pred_matches, match_dists = self.find_mutual_nearest_neighbors(desc1, desc2)

            # Compute ground truth matches
            if 'relative_pose' in batch:
                K = np.array([
                    [525.0, 0, 319.5],
                    [0, 525.0, 239.5],
                    [0, 0, 1]
                ])

                T_rel = batch['relative_pose'].numpy()
                R = T_rel[:3, :3]
                H = K @ R @ np.linalg.inv(K)

                gt_matches = self.compute_ground_truth_matches(kpts1, kpts2, H)

                # Evaluate
                metrics = self.evaluate_matches(pred_matches, gt_matches, len(kpts1), len(kpts2))
                metrics['mean_match_distance'] = match_dists.mean() if len(match_dists) > 0 else 0.0

                results.append(metrics)

        # Aggregate
        summary = {
            'sequence': sequence,
            'num_pairs': len(results),
            'mean_precision': np.mean([r['precision'] for r in results]),
            'std_precision': np.std([r['precision'] for r in results]),
            'mean_recall': np.mean([r['recall'] for r in results]),
            'std_recall': np.std([r['recall'] for r in results]),
            'mean_f1': np.mean([r['f1'] for r in results]),
            'std_f1': np.std([r['f1'] for r in results]),
            'mean_inlier_ratio': np.mean([r['inlier_ratio'] for r in results]),
            'std_inlier_ratio': np.std([r['inlier_ratio'] for r in results]),
            'mean_num_matches': np.mean([r['num_pred_matches'] for r in results]),
            'mean_match_distance': np.mean([r['mean_match_distance'] for r in results]),
            'all_results': results
        }

        return summary

    def visualize_results(self, results: list, output_path: str = 'test/results/descriptor_quality.png'):
        """Visualize descriptor quality results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sequences = [r['sequence'] for r in results]
        x = np.arange(len(sequences))

        # Plot 1: Precision and Recall
        ax1 = axes[0, 0]
        precision_means = [r['mean_precision'] * 100 for r in results]
        precision_stds = [r['std_precision'] * 100 for r in results]
        recall_means = [r['mean_recall'] * 100 for r in results]
        recall_stds = [r['std_recall'] * 100 for r in results]

        width = 0.35
        ax1.bar(x - width/2, precision_means, width, yerr=precision_stds,
                label='Precision', alpha=0.7, capsize=5, edgecolor='black')
        ax1.bar(x + width/2, recall_means, width, yerr=recall_stds,
                label='Recall', alpha=0.7, capsize=5, edgecolor='black')
        ax1.axhline(70, color='r', linestyle='--', label='Target: 70%')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.split('_')[-1] for s in sequences], rotation=45)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Precision and Recall')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Inlier Ratio
        ax2 = axes[0, 1]
        inlier_means = [r['mean_inlier_ratio'] * 100 for r in results]
        inlier_stds = [r['std_inlier_ratio'] * 100 for r in results]

        ax2.bar(x, inlier_means, yerr=inlier_stds, capsize=5, alpha=0.7, edgecolor='black')
        ax2.axhline(80, color='r', linestyle='--', label='Target: 80%')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.split('_')[-1] for s in sequences], rotation=45)
        ax2.set_ylabel('Inlier Ratio (%)')
        ax2.set_title('Match Inlier Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: F1 Score distribution
        ax3 = axes[1, 0]
        all_f1 = []
        for r in results:
            all_f1.extend([x['f1'] * 100 for x in r['all_results']])

        ax3.hist(all_f1, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(all_f1), color='r', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(all_f1):.1f}%')
        ax3.set_xlabel('F1 Score (%)')
        ax3.set_ylabel('Count')
        ax3.set_title('F1 Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')

        overall_inlier = np.mean([r['mean_inlier_ratio'] * 100 for r in results])
        overall_precision = np.mean([r['mean_precision'] * 100 for r in results])
        overall_recall = np.mean([r['mean_recall'] * 100 for r in results])
        overall_f1 = np.mean([r['mean_f1'] * 100 for r in results])

        summary_text = f"""
DESCRIPTOR QUALITY SUMMARY
{'='*45}

Overall Performance:
  Inlier Ratio:  {overall_inlier:.1f}%
  Precision:     {overall_precision:.1f}%
  Recall:        {overall_recall:.1f}%
  F1 Score:      {overall_f1:.1f}%

Targets:
  Inlier Ratio:  ≥80%  {'✅' if overall_inlier >= 80 else '❌'}
  Precision:     ≥70%  {'✅' if overall_precision >= 70 else '❌'}

Status: {'✅ PASS' if overall_inlier >= 80 and overall_precision >= 70 else '❌ FAIL'}
{'='*45}

Per-Sequence Inlier Ratio:
"""

        for r in results:
            seq_name = r['sequence'].split('_')[-1]
            inlier = r['mean_inlier_ratio'] * 100
            status = '✅' if inlier >= 80 else '❌'
            summary_text += f"\n  {seq_name:10s} {inlier:5.1f}% {status}"

        ax4.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center')

        plt.tight_layout()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {output_path}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test descriptor quality')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='../configs/train_config.yaml')
    parser.add_argument('--sequences', nargs='+',
                       default=['rgbd_dataset_freiburg1_plant',
                               'rgbd_dataset_freiburg1_desk',
                               'rgbd_dataset_freiburg1_room'])
    parser.add_argument('--num_pairs', type=int, default=50)
    parser.add_argument('--frame_spacing', type=int, default=1)
    parser.add_argument('--output', type=str, default='test/results/descriptor_quality.png')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("DESCRIPTOR QUALITY TEST")
    print("="*70)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Sequences:      {len(args.sequences)}")
    print(f"Pairs/sequence: {args.num_pairs}")
    print("="*70 + "\n")

    tester = DescriptorQualityTester(args.checkpoint, args.config)

    all_results = []
    for seq in args.sequences:
        result = tester.test_sequence(seq, args.num_pairs, args.frame_spacing)
        all_results.append(result)

        print(f"\n{'='*70}")
        print(f"Results for {seq}")
        print(f"{'='*70}")
        print(f"Precision:      {result['mean_precision']*100:.1f}% ± {result['std_precision']*100:.1f}%")
        print(f"Recall:         {result['mean_recall']*100:.1f}% ± {result['std_recall']*100:.1f}%")
        print(f"F1 Score:       {result['mean_f1']*100:.1f}% ± {result['std_f1']*100:.1f}%")
        print(f"Inlier Ratio:   {result['mean_inlier_ratio']*100:.1f}% ± {result['std_inlier_ratio']*100:.1f}%")
        print(f"Avg Matches:    {result['mean_num_matches']:.0f}")
        print(f"Match Distance: {result['mean_match_distance']:.3f}")

        status = "✅ PASS" if result['mean_inlier_ratio'] >= 0.80 else "❌ FAIL"
        print(f"Status:         {status}")
        print(f"{'='*70}")

    overall_inlier = np.mean([r['mean_inlier_ratio'] for r in all_results]) * 100
    print(f"\n{'='*70}")
    print(f"OVERALL INLIER RATIO: {overall_inlier:.1f}%")
    print(f"{'='*70}")

    if overall_inlier >= 80:
        print("✅ PASS: Descriptor quality is excellent!")
    else:
        print("❌ FAIL: Descriptor quality below target. Consider:")
        print("  - Increase descriptor loss weight")
        print("  - Add variance regularization")
        print("  - Check descriptor normalization")

    tester.visualize_results(all_results, args.output)

    # Save results
    import json
    results_json = {
        'overall_inlier_ratio': overall_inlier / 100,
        'overall_precision': np.mean([r['mean_precision'] for r in all_results]),
        'overall_recall': np.mean([r['mean_recall'] for r in all_results]),
        'sequences': []
    }
    for r in all_results:
        results_json['sequences'].append({
            'name': r['sequence'],
            'precision': r['mean_precision'],
            'recall': r['mean_recall'],
            'inlier_ratio': r['mean_inlier_ratio']
        })

    output_path = Path(args.output)
    json_path = output_path.with_name(f"{output_path.stem}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✓ Saved results to {json_path}")


if __name__ == "__main__":
    main()