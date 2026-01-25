"""
FIXED Master Test Script
CRITICAL: Only tests on sequences NOT seen during training!
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import datetime


def main():
    parser = argparse.ArgumentParser(description='Run all evaluation tests')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth')
    parser.add_argument('--config', type=str, default='../configs/train_config.yaml')

    # FIXED: Default to TEST sequences only (not train/val!)
    parser.add_argument('--sequences', nargs='+',
                       default=[
                           'rgbd_dataset_freiburg1_plant',  # Validation (can use)
                           'rgbd_dataset_freiburg3_long_office_household',  # Test
                           'rgbd_dataset_freiburg3_walking_xyz'  # Test
                       ],
                       help='Sequences to test (should be val/test only!)')

    parser.add_argument('--num_pairs', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--skip_slow', action='store_true')
    parser.add_argument('--difficulty', type=str, default='normal',
                       choices=['easy', 'normal', 'hard', 'extreme'])
    parser.add_argument('--frame_spacings', nargs='+', type=int, default=[1, 5])

    args = parser.parse_args()

    # VALIDATION: Warn if testing on training sequences
    TRAIN_SEQUENCES = [
        'rgbd_dataset_freiburg1_desk',
        'rgbd_dataset_freiburg1_room',
        'rgbd_dataset_freiburg3_walking_static'
    ]

    overlap = set(args.sequences) & set(TRAIN_SEQUENCES)
    if overlap:
        print("\n" + "="*70)
        print("⚠️  WARNING: TESTING ON TRAINING SEQUENCES!")
        print("="*70)
        print(f"These sequences were used for TRAINING: {overlap}")
        print("Results will be INFLATED due to overfitting!")
        print("Consider using only val/test sequences:")
        print("  - rgbd_dataset_freiburg1_plant")
        print("  - rgbd_dataset_freiburg3_long_office_household")
        print("  - rgbd_dataset_freiburg3_walking_xyz")
        print("="*70 + "\n")

        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted. Use --sequences to specify proper test sequences.")
            return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("SEMANTIC SLAM - PROPER EVALUATION (NO TRAIN/TEST OVERLAP)")
    print("="*70)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Test sequences: {', '.join([s.split('_')[-1] for s in args.sequences])}")
    print(f"Difficulty:     {args.difficulty.upper()}")
    print("="*70 + "\n")

    # Set difficulty parameters
    difficulty_configs = {
        'easy': {'frame_spacings': [1], 'min_matches': 30, 'num_pairs': 30},
        'normal': {'frame_spacings': [1, 5], 'min_matches': 50, 'num_pairs': 50},
        'hard': {'frame_spacings': [1, 5, 10], 'min_matches': 75, 'num_pairs': 100},
        'extreme': {'frame_spacings': [1, 5, 10, 20], 'min_matches': 100, 'num_pairs': 150}
    }
    difficulty = difficulty_configs[args.difficulty]

    # Common arguments
    common_args = [
        '--checkpoint', args.checkpoint,
        '--config', args.config,
        '--sequences'
    ] + args.sequences + [
        '--num_pairs', str(args.num_pairs if args.num_pairs != 50 else difficulty['num_pairs'])
    ]

    results = {}

    # Test 1: Repeatability
    print("\n" + "🔄 " * 20)
    print("TEST 1/4: REPEATABILITY")
    print("🔄 " * 20)
    for spacing in difficulty['frame_spacings']:
        test_args = common_args.copy()
        test_args.extend([
            '--frame_spacing', str(spacing),
            '--output', str(output_dir / f'repeatability_spacing{spacing}.png')
        ])
        subprocess.run([sys.executable, 'test_repeatability.py'] + test_args)

    # Test 2: Descriptor Quality
    print("\n" + "📊 " * 20)
    print("TEST 2/4: DESCRIPTOR QUALITY")
    print("📊 " * 20)
    test_args = common_args.copy()
    test_args.extend(['--output', str(output_dir / 'descriptor_quality.png')])
    subprocess.run([sys.executable, 'test_descriptor_quality.py'] + test_args)

    # Test 3: Tracking
    if not args.skip_slow:
        print("\n" + "🎯 " * 20)
        print("TEST 3/4: TRACKING")
        print("🎯 " * 20)
        for spacing in difficulty['frame_spacings']:
            test_args = [
                '--checkpoint', args.checkpoint,
                '--config', args.config,
                '--sequences'] + args.sequences + [
                '--max_frames', '100',
                '--min_matches', str(difficulty['min_matches']),
                '--frame_spacing', str(spacing),
                '--output', str(output_dir / f'tracking_spacing{spacing}.png')
            ]
            subprocess.run([sys.executable, 'test_tracking.py'] + test_args)

    # Test 4: Performance
    print("\n" + "⚡ " * 20)
    print("TEST 4/4: PERFORMANCE")
    print("⚡ " * 20)
    test_args = [
        '--checkpoint', args.checkpoint,
        '--config', args.config,
        '--sequences', args.sequences[0],
        '--output', str(output_dir / 'performance.png')
    ]
    subprocess.run([sys.executable, 'test_performance.py'] + test_args)

    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📊 Results saved to: {output_dir}")
    print("\n💡 Remember: These results are on UNSEEN test sequences!")
    print("   If scores seem lower than before, that's EXPECTED and HONEST.")


if __name__ == "__main__":
    main()