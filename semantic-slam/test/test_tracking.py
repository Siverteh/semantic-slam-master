"""
Test Frame-to-Frame Tracking
Simulates SLAM tracking: can we track keypoints across consecutive frames?

Target: >90% tracking success rate
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

from data.tum_dataset import TUMDataset
from models.descriptor_refiner import DescriptorRefiner
from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector


class TrackingTester:
    """Test frame-to-frame tracking capability"""

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
        dino_features = self.backbone(image)
        saliency_map = self.selector(dino_features)
        keypoints_patch, scores = self.selector.select_keypoints(
            saliency_map,
            num_keypoints=self.config['model']['num_keypoints']
        )
        feat_at_kpts = self.backbone.extract_at_keypoints(dino_features, keypoints_patch)
        descriptors = self.refiner(feat_at_kpts)
        keypoints_pixel = self.backbone.patch_to_pixel(keypoints_patch)

        return (
            keypoints_pixel[0].cpu().numpy(),
            descriptors[0].cpu().numpy(),
            scores[0].cpu().numpy()
        )

    def track_frame_sequence(
        self,
        sequence: str,
        max_frames: int = 100,
        min_matches: int = 50,
        match_threshold: float = 0.8,
        frame_spacing: int = 1  # Actual gap between compared frames
    ):
        """
        Track through a sequence of frames.

        FIXED: Now actually tests with the specified frame spacing!
        Before: Was comparing consecutive frames in a spaced dataset
        Now: Compares frames with actual spacing in original sequence

        Args:
            sequence: TUM sequence name
            max_frames: Maximum number of comparisons to make
            min_matches: Minimum matches required for successful tracking
            match_threshold: Cosine similarity threshold for matches
            frame_spacing: ACTUAL gap in frames (1=consecutive, 10=skip 9 frames)

        Returns:
            dict with tracking statistics
        """
        print(f"\nTracking sequence: {sequence}")
        print(f"  Frame spacing: {frame_spacing} (comparing frame N to frame N+{frame_spacing})")
        print(f"  Min matches required: {min_matches}")

        # FIXED: Load full sequence with spacing=1
        full_dataset = TUMDataset(
            dataset_root=self.config['dataset']['root'],
            sequence=sequence,
            input_size=self.config['model']['input_size'],
            frame_spacing=1,  # Load consecutive frames
            max_frames=max_frames * frame_spacing,  # Need more frames for larger spacing
            augmentation=None,
            is_train=False
        )

        # Initialize tracking
        tracking_lost = False
        num_tracked_frames = 0
        num_lost_frames = 0
        match_counts = []
        match_ratios = []

        # FIXED: Load dataset WITHOUT spacing (get all frames)
        full_dataset = TUMDataset(
            dataset_root=self.config['dataset']['root'],
            sequence=sequence,
            input_size=self.config['model']['input_size'],
            frame_spacing=1,  # Always 1 for full sequence
            max_frames=max_frames * frame_spacing,  # Load enough frames
            augmentation=None,
            is_train=False
        )

        # Extract features from first frame
        first_batch = full_dataset[0]
        rgb_prev = first_batch['rgb1'].unsqueeze(0).to(self.device)
        kpts_prev, desc_prev, _ = self.extract_features(rgb_prev)

        # FIXED: Step through with actual spacing in original sequence
        for i in tqdm(range(frame_spacing, min(max_frames * frame_spacing, len(full_dataset)), frame_spacing), desc="Tracking"):
            batch = full_dataset[i]
            rgb_curr = batch['rgb1'].unsqueeze(0).to(self.device)  # Use rgb1, not rgb2

            # Extract features from current frame
            kpts_curr, desc_curr, _ = self.extract_features(rgb_curr)

            # Find matches
            sim_matrix = desc_prev @ desc_curr.T
            max_sim = sim_matrix.max(axis=1)
            matches = (max_sim > match_threshold).sum()

            match_counts.append(matches)
            match_ratio = matches / len(desc_prev)
            match_ratios.append(match_ratio)

            # Check if tracking succeeded
            if matches >= min_matches:
                num_tracked_frames += 1
                tracking_lost = False
            else:
                num_lost_frames += 1
                tracking_lost = True

            # Update previous frame (even if tracking lost - try to recover)
            kpts_prev = kpts_curr
            desc_prev = desc_curr

        # Compute statistics
        total_frames = len(match_counts)
        tracking_success_rate = num_tracked_frames / total_frames if total_frames > 0 else 0.0

        results = {
            'sequence': sequence,
            'frame_spacing': frame_spacing,  # NEW
            'total_frames': total_frames,
            'tracked_frames': num_tracked_frames,
            'lost_frames': num_lost_frames,
            'tracking_success_rate': tracking_success_rate,
            'mean_matches': np.mean(match_counts),
            'std_matches': np.std(match_counts),
            'min_matches': np.min(match_counts),
            'max_matches': np.max(match_counts),
            'mean_match_ratio': np.mean(match_ratios),
            'match_counts': match_counts,
            'match_ratios': match_ratios
        }

        return results

    def visualize_tracking(self, results: list, output_path: str = 'test/results/tracking.png'):
        """Visualize tracking results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Tracking success rate by sequence
        ax1 = axes[0, 0]
        sequences = [r['sequence'] for r in results]
        success_rates = [r['tracking_success_rate'] * 100 for r in results]

        x = np.arange(len(sequences))
        bars = ax1.bar(x, success_rates, alpha=0.7, edgecolor='black')

        # Color bars: green if >90%, yellow if 70-90%, red if <70%
        for i, rate in enumerate(success_rates):
            if rate >= 90:
                bars[i].set_color('green')
            elif rate >= 70:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('red')

        ax1.axhline(90, color='darkgreen', linestyle='--', linewidth=2, label='Target: 90%')
        ax1.axhline(70, color='orange', linestyle='--', linewidth=1, label='Acceptable: 70%')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.split('_')[-1] for s in sequences], rotation=45)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Tracking Success Rate')
        ax1.set_ylim([0, 105])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Match counts over time (first sequence)
        ax2 = axes[0, 1]
        first_result = results[0]
        frames = np.arange(len(first_result['match_counts']))
        match_counts = first_result['match_counts']

        ax2.plot(frames, match_counts, linewidth=1.5, alpha=0.7)
        ax2.axhline(first_result['mean_matches'], color='r', linestyle='--',
                   label=f"Mean: {first_result['mean_matches']:.0f}")
        ax2.axhline(50, color='orange', linestyle='--', label='Min threshold: 50')
        ax2.fill_between(frames, 0, match_counts, alpha=0.2)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Number of Matches')
        ax2.set_title(f"Matches Over Time ({first_result['sequence'].split('_')[-1]})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Match ratio distribution
        ax3 = axes[1, 0]
        all_ratios = []
        for r in results:
            all_ratios.extend([x * 100 for x in r['match_ratios']])

        ax3.hist(all_ratios, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(all_ratios), color='r', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(all_ratios):.1f}%')
        ax3.axvline(10, color='orange', linestyle='--', label='Min: 10%')
        ax3.set_xlabel('Match Ratio (%)')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Match Ratios (All Sequences)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')

        overall_success = np.mean([r['tracking_success_rate'] * 100 for r in results])
        overall_matches = np.mean([r['mean_matches'] for r in results])

        summary_text = f"""
TRACKING TEST SUMMARY
{'='*45}

Overall Performance:
  Success Rate:  {overall_success:.1f}%
  Avg Matches:   {overall_matches:.0f}

Target:          ≥90% success rate
Status:          {'✅ PASS' if overall_success >= 90 else '❌ FAIL'}

{'='*45}

Per-Sequence Success Rate:
"""

        for r in results:
            seq_name = r['sequence'].split('_')[-1]
            rate = r['tracking_success_rate'] * 100
            status = '✅' if rate >= 90 else ('⚠️' if rate >= 70 else '❌')
            summary_text += f"\n  {seq_name:10s} {rate:5.1f}% {status}"

        summary_text += f"\n\n{'='*45}\n"

        if overall_success >= 90:
            summary_text += "\n✅ Excellent tracking capability!"
        elif overall_success >= 70:
            summary_text += "\n⚠️  Acceptable but needs improvement"
        else:
            summary_text += "\n❌ Tracking too unstable for SLAM"

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
    parser = argparse.ArgumentParser(description='Test frame-to-frame tracking')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--sequences', nargs='+',
                       default=['rgbd_dataset_freiburg1_plant',
                               'rgbd_dataset_freiburg1_desk'])
    parser.add_argument('--max_frames', type=int, default=100,
                       help='Maximum frames to track per sequence')
    parser.add_argument('--min_matches', type=int, default=50,
                       help='Minimum matches required for successful tracking')
    parser.add_argument('--frame_spacing', type=int, default=1,
                       help='Spacing between frames (1=consecutive, 5=skip 4 frames)')
    parser.add_argument('--output', type=str, default='test/results/tracking.png')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("FRAME-TO-FRAME TRACKING TEST")
    print("="*70)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Sequences:      {len(args.sequences)}")
    print(f"Frames/seq:     {args.max_frames}")
    print(f"Min matches:    {args.min_matches}")
    print(f"Frame spacing:  {args.frame_spacing}")
    print("="*70 + "\n")

    tester = TrackingTester(args.checkpoint, args.config)

    all_results = []
    for seq in args.sequences:
        result = tester.track_frame_sequence(
            seq,
            max_frames=args.max_frames,
            min_matches=args.min_matches,
            frame_spacing=args.frame_spacing  # NEW
        )
        all_results.append(result)

        print(f"\n{'='*70}")
        print(f"Results for {seq}")
        print(f"{'='*70}")
        print(f"Total Frames:      {result['total_frames']}")
        print(f"Tracked:           {result['tracked_frames']} ({result['tracking_success_rate']*100:.1f}%)")
        print(f"Lost:              {result['lost_frames']}")
        print(f"Avg Matches:       {result['mean_matches']:.0f} ± {result['std_matches']:.0f}")
        print(f"Match Range:       [{result['min_matches']}, {result['max_matches']}]")
        print(f"Avg Match Ratio:   {result['mean_match_ratio']*100:.1f}%")

        if result['tracking_success_rate'] >= 0.90:
            print(f"Status:            ✅ EXCELLENT")
        elif result['tracking_success_rate'] >= 0.70:
            print(f"Status:            ⚠️  ACCEPTABLE")
        else:
            print(f"Status:            ❌ UNSTABLE")
        print(f"{'='*70}")

    # Overall summary
    overall_success = np.mean([r['tracking_success_rate'] for r in all_results]) * 100
    print(f"\n{'='*70}")
    print(f"OVERALL TRACKING SUCCESS: {overall_success:.1f}%")
    print(f"{'='*70}")

    if overall_success >= 90:
        print("✅ PASS: Tracking is robust enough for SLAM!")
    elif overall_success >= 70:
        print("⚠️  CAUTION: Tracking works but may be unreliable")
        print("   Consider increasing descriptor loss weight")
    else:
        print("❌ FAIL: Tracking too unstable for SLAM")
        print("   Possible issues:")
        print("   - Descriptor quality too low")
        print("   - Keypoints not repeatable enough")
        print("   - Need stronger feature learning")

    tester.visualize_tracking(all_results, args.output)

    # Save results
    import json
    results_json = {
        'overall_success_rate': overall_success / 100,
        'sequences': []
    }
    for r in all_results:
        results_json['sequences'].append({
            'name': r['sequence'],
            'success_rate': r['tracking_success_rate'],
            'mean_matches': r['mean_matches'],
            'mean_match_ratio': r['mean_match_ratio']
        })

    output_path = Path(args.output)
    json_path = output_path.with_name(f"{output_path.stem}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✓ Saved results to {json_path}")


if __name__ == "__main__":
    main()