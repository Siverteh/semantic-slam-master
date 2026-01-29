"""
FIXED Master Test Script
CRITICAL: Only tests on sequences NOT seen during training!
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def load_json_results(path: Path) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_summary_report(
    output_dir: Path,
    checkpoint: str,
    config: str,
    sequences: list[str],
    difficulty: str,
    frame_spacings: list[int]
) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("SEMANTIC SLAM EVALUATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated:   {timestamp}")
    lines.append(f"Checkpoint:  {checkpoint}")
    lines.append(f"Config:      {config}")
    lines.append(f"Difficulty:  {difficulty.upper()}")
    lines.append(f"Spacings:    {', '.join(str(s) for s in frame_spacings)}")
    lines.append(f"Sequences:   {', '.join(sequences)}")
    lines.append(f"Output dir:  {output_dir}")
    lines.append("")

    # Repeatability by spacing
    lines.append("-" * 80)
    lines.append("REPEATABILITY (BY SPACING)")
    lines.append("-" * 80)
    for spacing in frame_spacings:
        result = load_json_results(output_dir / f"repeatability_spacing{spacing}_results.json")
        if not result:
            lines.append(f"Spacing {spacing}: results not found")
            continue
        lines.append(f"\nSpacing {spacing}:")
        lines.append(f"  Overall mean: {format_pct(result['overall_mean_repeatability'])}")
        for seq in result["sequences"]:
            name = seq["name"].split("_")[-1]
            mean = format_pct(seq["mean_repeatability"])
            std = format_pct(seq["std_repeatability"])
            dist = f"{seq['mean_distance']:.2f}px"
            lines.append(f"  - {name:20s} mean {mean} ± {std}, nn dist {dist}")

    lines.append("")

    # Descriptor quality (per spacing)
    lines.append("-" * 80)
    lines.append("DESCRIPTOR QUALITY (BY SPACING)")
    lines.append("-" * 80)
    any_desc = False
    for spacing in frame_spacings:
        desc_results = load_json_results(
            output_dir / f"descriptor_quality_spacing{spacing}_results.json"
        )
        if not desc_results:
            continue
        any_desc = True
        lines.append(f"\nSpacing {spacing}:")
        lines.append(f"  Overall precision: {format_pct(desc_results['overall_precision'])}")
        lines.append(f"  Overall recall:    {format_pct(desc_results['overall_recall'])}")
        lines.append(f"  Overall inlier:    {format_pct(desc_results['overall_inlier_ratio'])}")
        for seq in desc_results["sequences"]:
            name = seq["name"].split("_")[-1]
            precision = format_pct(seq["precision"])
            recall = format_pct(seq["recall"])
            inlier = format_pct(seq["inlier_ratio"])
            lines.append(f"  - {name:20s} inlier {inlier}, precision {precision}, recall {recall}")
    if not any_desc:
        fallback = load_json_results(output_dir / "descriptor_quality_results.json")
        if fallback:
            lines.append("No per-spacing descriptor results found; using single run:")
            lines.append(f"  Overall precision: {format_pct(fallback['overall_precision'])}")
            lines.append(f"  Overall recall:    {format_pct(fallback['overall_recall'])}")
            lines.append(f"  Overall inlier:    {format_pct(fallback['overall_inlier_ratio'])}")
            for seq in fallback["sequences"]:
                name = seq["name"].split("_")[-1]
                precision = format_pct(seq["precision"])
                recall = format_pct(seq["recall"])
                inlier = format_pct(seq["inlier_ratio"])
                lines.append(f"  - {name:20s} inlier {inlier}, precision {precision}, recall {recall}")
        else:
            lines.append("Descriptor quality results not found")

    lines.append("")

    # Tracking by spacing
    lines.append("-" * 80)
    lines.append("TRACKING (BY SPACING)")
    lines.append("-" * 80)
    for spacing in frame_spacings:
        result = load_json_results(output_dir / f"tracking_spacing{spacing}_results.json")
        if not result:
            lines.append(f"Spacing {spacing}: results not found")
            continue
        lines.append(f"\nSpacing {spacing}:")
        lines.append(f"  Overall success: {format_pct(result['overall_success_rate'])}")
        for seq in result["sequences"]:
            name = seq["name"].split("_")[-1]
            success = format_pct(seq["success_rate"])
            matches = f"{seq['mean_matches']:.1f}"
            ratio = format_pct(seq["mean_match_ratio"])
            lines.append(f"  - {name:20s} success {success}, mean matches {matches}, ratio {ratio}")

    lines.append("")

    # Performance (single run)
    perf_results = load_json_results(output_dir / "performance_results.json")
    lines.append("-" * 80)
    lines.append("PERFORMANCE")
    lines.append("-" * 80)
    if perf_results:
        lines.append(f"Average FPS: {perf_results['average_fps']:.1f}")
        for seq in perf_results["sequences"]:
            name = seq["name"].split("_")[-1]
            fps = f"{seq['fps']:.1f}"
            time_ms = f"{seq['total_time_ms']:.1f}ms"
            mem = f"{seq['memory_gb']:.3f}GB"
            lines.append(f"  - {name:20s} {fps} FPS, {time_ms}, {mem}")
    else:
        lines.append("Performance results not found")

    lines.append("\n" + "=" * 80)

    report_path = output_dir / "summary_report.txt"
    report_path.write_text("\n".join(lines))
    return report_path


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
    if args.frame_spacings:
        difficulty["frame_spacings"] = args.frame_spacings

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

    # Test 2: Descriptor Quality (per spacing)
    print("\n" + "📊 " * 20)
    print("TEST 2/4: DESCRIPTOR QUALITY")
    print("📊 " * 20)
    for spacing in difficulty['frame_spacings']:
        test_args = common_args.copy()
        test_args.extend([
            '--frame_spacing', str(spacing),
            '--output', str(output_dir / f'descriptor_quality_spacing{spacing}.png')
        ])
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

    # Summary report
    summary_path = write_summary_report(
        output_dir=output_dir,
        checkpoint=args.checkpoint,
        config=args.config,
        sequences=args.sequences,
        difficulty=args.difficulty,
        frame_spacings=difficulty["frame_spacings"]
    )

    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"📝 Summary report: {summary_path}")
    print("\n💡 Remember: These results are on UNSEEN test sequences!")
    print("   If scores seem lower than before, that's EXPECTED and HONEST.")


if __name__ == "__main__":
    main()