"""
Test Keypoint Repeatability
Measures how consistently keypoints are detected across consecutive frames.

Target: >60% repeatability (ratio of keypoints within 2-3 pixels after pose correction)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from data.tum_dataset import TUMDataset


class RepeatabilityTester:
    """Test keypoint repeatability across frame pairs"""

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

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.selector.load_state_dict(checkpoint['selector_state_dict'])

        self.backbone.eval()
        self.selector.eval()

        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")

    @torch.no_grad()
    def detect_keypoints(self, image: torch.Tensor):
        """Detect keypoints in an image"""
        # Extract features
        dino_features = self.backbone(image)

        # Predict saliency
        saliency_map = self.selector(dino_features)

        # Select keypoints (in PATCH coordinates)
        keypoints_patch, scores = self.selector.select_keypoints(
            saliency_map,
            num_keypoints=self.config['model']['num_keypoints']
        )

        # Convert to pixel coordinates
        keypoints_pixel = self.backbone.patch_to_pixel(keypoints_patch)

        return keypoints_pixel[0].cpu().numpy(), scores[0].cpu().numpy()

    def compute_repeatability(
        self,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        H: np.ndarray = None,
        threshold: float = 3.0
    ) -> dict:
        """
        Compute repeatability between two sets of keypoints.

        Args:
            kpts1: (N, 2) keypoints in frame 1
            kpts2: (M, 2) keypoints in frame 2
            H: (3, 3) homography from frame1 to frame2 (if available)
            threshold: Distance threshold in pixels

        Returns:
            dict with repeatability metrics
        """
        if H is not None:
            # Warp kpts1 to frame2 coordinate system
            kpts1_homo = np.concatenate([kpts1, np.ones((len(kpts1), 1))], axis=1)
            kpts1_warped = (H @ kpts1_homo.T).T
            kpts1_warped = kpts1_warped[:, :2] / kpts1_warped[:, 2:3]
        else:
            # No pose correction - just use raw coordinates
            kpts1_warped = kpts1

        # Compute pairwise distances
        dists = np.linalg.norm(
            kpts1_warped[:, np.newaxis, :] - kpts2[np.newaxis, :, :],
            axis=2
        )

        # Find nearest neighbors
        min_dists = dists.min(axis=1)

        # Count matches within threshold
        repeatable = (min_dists < threshold).sum()

        # Repeatability ratio
        repeatability = repeatable / len(kpts1)

        return {
            'repeatability': repeatability,
            'repeatable_count': repeatable,
            'total_keypoints': len(kpts1),
            'mean_nn_distance': min_dists.mean(),
            'median_nn_distance': np.median(min_dists)
        }

    def test_sequence(
        self,
        sequence: str,
        num_pairs: int = 50,
        frame_spacing: int = 1,
        use_pose: bool = True
    ) -> dict:
        """
        Test repeatability on a sequence.

        Args:
            sequence: TUM sequence name
            num_pairs: Number of frame pairs to test
            frame_spacing: Spacing between frames
            use_pose: Use ground truth pose for correction

        Returns:
            dict with aggregated results
        """
        print(f"\nTesting sequence: {sequence}")
        print(f"  Pairs: {num_pairs}, spacing: {frame_spacing}, pose correction: {use_pose}")

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

            # Detect keypoints
            kpts1, scores1 = self.detect_keypoints(rgb1)
            kpts2, scores2 = self.detect_keypoints(rgb2)

            # Compute homography from pose if available
            H = None
            if use_pose and 'relative_pose' in batch:
                # Simplified: assume camera intrinsics (TUM fr1: ~520 focal length)
                K = np.array([
                    [525.0, 0, 319.5],
                    [0, 525.0, 239.5],
                    [0, 0, 1]
                ])

                # Relative pose (4x4)
                T_rel = batch['relative_pose'].numpy()
                R = T_rel[:3, :3]
                t = T_rel[:3, 3]

                # For small motion, approximate homography: H = K * R * K^-1
                # (ignoring translation, which is small for consecutive frames)
                H = K @ R @ np.linalg.inv(K)

            # Compute repeatability
            metrics = self.compute_repeatability(kpts1, kpts2, H, threshold=3.0)
            results.append(metrics)

        # Aggregate results
        repeatabilities = [r['repeatability'] for r in results]
        mean_dists = [r['mean_nn_distance'] for r in results]

        summary = {
            'sequence': sequence,
            'num_pairs': len(results),
            'mean_repeatability': np.mean(repeatabilities),
            'std_repeatability': np.std(repeatabilities),
            'median_repeatability': np.median(repeatabilities),
            'min_repeatability': np.min(repeatabilities),
            'max_repeatability': np.max(repeatabilities),
            'mean_distance': np.mean(mean_dists),
            'median_distance': np.median(mean_dists),
            'all_results': results
        }

        return summary

    def visualize_results(self, results: list, output_path: str = 'test/results/repeatability.png'):
        """Visualize repeatability results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Repeatability by sequence
        ax1 = axes[0, 0]
        sequences = [r['sequence'] for r in results]
        means = [r['mean_repeatability'] * 100 for r in results]
        stds = [r['std_repeatability'] * 100 for r in results]

        x = np.arange(len(sequences))
        ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
        ax1.axhline(60, color='r', linestyle='--', label='Target: 60%')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.split('_')[-1] for s in sequences], rotation=45)
        ax1.set_ylabel('Repeatability (%)')
        ax1.set_title('Mean Repeatability by Sequence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Repeatability distribution
        ax2 = axes[0, 1]
        all_reps = []
        for r in results:
            all_reps.extend([x['repeatability'] * 100 for x in r['all_results']])

        ax2.hist(all_reps, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(all_reps), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_reps):.1f}%')
        ax2.axvline(60, color='g', linestyle='--', linewidth=2, label='Target: 60%')
        ax2.set_xlabel('Repeatability (%)')
        ax2.set_ylabel('Count')
        ax2.set_title('Repeatability Distribution (All Pairs)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Mean distance by sequence
        ax3 = axes[1, 0]
        mean_dists = [r['mean_distance'] for r in results]
        median_dists = [r['median_distance'] for r in results]

        x = np.arange(len(sequences))
        width = 0.35
        ax3.bar(x - width/2, mean_dists, width, label='Mean', alpha=0.7, edgecolor='black')
        ax3.bar(x + width/2, median_dists, width, label='Median', alpha=0.7, edgecolor='black')
        ax3.axhline(3.0, color='r', linestyle='--', label='Threshold: 3px')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.split('_')[-1] for s in sequences], rotation=45)
        ax3.set_ylabel('Distance (pixels)')
        ax3.set_title('Nearest Neighbor Distance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create summary text
        overall_mean = np.mean([r['mean_repeatability'] * 100 for r in results])
        overall_std = np.std([r['mean_repeatability'] * 100 for r in results])
        pass_rate = sum(r['mean_repeatability'] >= 0.60 for r in results) / len(results) * 100

        summary_text = f"""
REPEATABILITY TEST SUMMARY
{'='*40}

Overall Performance:
  Mean:        {overall_mean:.1f}% ± {overall_std:.1f}%
  Pass Rate:   {pass_rate:.0f}% (≥60% repeatability)

Target:        60% repeatability
Status:        {'✅ PASS' if overall_mean >= 60 else '❌ FAIL'}

{'='*40}

Per-Sequence Results:
"""

        for r in results:
            seq_name = r['sequence'].split('_')[-1]
            rep_pct = r['mean_repeatability'] * 100
            status = '✅' if rep_pct >= 60 else '❌'
            summary_text += f"\n  {seq_name:10s} {rep_pct:5.1f}% {status}"

        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {output_path}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test keypoint repeatability')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='../configs/train_config.yaml')
    parser.add_argument('--sequences', nargs='+',
                       default=['rgbd_dataset_freiburg1_plant',
                               'rgbd_dataset_freiburg1_desk',
                               'rgbd_dataset_freiburg1_room'])
    parser.add_argument('--num_pairs', type=int, default=50,
                       help='Number of frame pairs to test per sequence')
    parser.add_argument('--frame_spacing', type=int, default=1,
                       help='Spacing between consecutive frames')
    parser.add_argument('--no_pose', action='store_true',
                       help='Disable pose correction (test raw repeatability)')
    parser.add_argument('--output', type=str, default='test/results/repeatability.png')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("REPEATABILITY TEST")
    print("="*70)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Sequences:      {len(args.sequences)}")
    print(f"Pairs/sequence: {args.num_pairs}")
    print(f"Frame spacing:  {args.frame_spacing}")
    print(f"Pose correction: {not args.no_pose}")
    print("="*70 + "\n")

    # Create tester
    tester = RepeatabilityTester(args.checkpoint, args.config)

    # Test each sequence
    all_results = []
    for seq in args.sequences:
        result = tester.test_sequence(
            seq,
            num_pairs=args.num_pairs,
            frame_spacing=args.frame_spacing,
            use_pose=not args.no_pose
        )
        all_results.append(result)

        # Print results
        print(f"\n{'='*70}")
        print(f"Results for {seq}")
        print(f"{'='*70}")
        print(f"Mean Repeatability:   {result['mean_repeatability']*100:.1f}% ± {result['std_repeatability']*100:.1f}%")
        print(f"Median Repeatability: {result['median_repeatability']*100:.1f}%")
        print(f"Range:                [{result['min_repeatability']*100:.1f}%, {result['max_repeatability']*100:.1f}%]")
        print(f"Mean NN Distance:     {result['mean_distance']:.2f} pixels")
        print(f"Median NN Distance:   {result['median_distance']:.2f} pixels")

        if result['mean_repeatability'] >= 0.60:
            print(f"Status:               ✅ PASS (≥60%)")
        else:
            print(f"Status:               ❌ FAIL (<60%)")
        print(f"{'='*70}")

    # Overall summary
    overall_mean = np.mean([r['mean_repeatability'] for r in all_results]) * 100
    print(f"\n{'='*70}")
    print(f"OVERALL REPEATABILITY: {overall_mean:.1f}%")
    print(f"{'='*70}")

    if overall_mean >= 60:
        print("✅ PASS: Keypoints are sufficiently repeatable for SLAM!")
    else:
        print("❌ FAIL: Repeatability below target. Consider:")
        print("  - Increase repeatability loss weight")
        print("  - Add data augmentation")
        print("  - Check if pose correction is working")

    # Visualize
    tester.visualize_results(all_results, args.output)

    # Save results
    import json
    results_json = {
        'overall_mean_repeatability': overall_mean / 100,
        'sequences': []
    }
    for r in all_results:
        results_json['sequences'].append({
            'name': r['sequence'],
            'mean_repeatability': r['mean_repeatability'],
            'std_repeatability': r['std_repeatability'],
            'mean_distance': r['mean_distance']
        })

    output_path = Path(args.output)
    json_path = output_path.with_name(f"{output_path.stem}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✓ Saved results to {json_path}")


if __name__ == "__main__":
    main()