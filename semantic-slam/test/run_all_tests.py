"""
Master Test Script
Runs all evaluation tests and generates comprehensive report.

Usage:
    python test/run_all_tests.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import datetime


def run_test(script: str, args: list) -> dict:
    """Run a test script and capture output"""
    print(f"\n{'='*70}")
    print(f"Running: {script}")
    print(f"{'='*70}\n")

    script_path = Path(script)
    if not script_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        script_path = (project_root / script_path).resolve()

    cmd = [sys.executable, str(script_path)] + args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return {'success': True, 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed with error:")
        print(e.stdout)
        print(e.stderr)
        return {'success': False, 'error': str(e)}


def load_json_results(filepath: str) -> dict:
    """Load JSON results if available"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def generate_comprehensive_summary(
    all_repeat_results: dict,
    desc_results: dict,
    all_track_results: dict,
    perf_results: dict,
    difficulty: str,
    output_dir: Path
) -> str:
    """Generate comprehensive summary with multiple test configurations"""

    report = []
    report.append("=" * 80)
    report.append("SEMANTIC SLAM COMPREHENSIVE EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Difficulty Level: {difficulty.upper()}")
    report.append("")

    # Overall summary with breakdown by frame spacing
    report.append("-" * 80)
    report.append("OVERALL SUMMARY")
    report.append("-" * 80)

    all_pass = True

    # Repeatability by spacing
    if all_repeat_results:
        report.append("\nRepeatability (by frame spacing):")
        for spacing, result in all_repeat_results.items():
            repeat_mean = result['overall_mean_repeatability'] * 100
            repeat_pass = repeat_mean >= 60
            all_pass = all_pass and repeat_pass
            status = "✅ PASS" if repeat_pass else "❌ FAIL"
            report.append(f"  Spacing {spacing:2d}:  {repeat_mean:5.1f}%  (target: ≥60%)   {status}")

    # Descriptor quality
    if desc_results:
        desc_inlier = desc_results['overall_inlier_ratio'] * 100
        desc_pass = desc_inlier >= 80
        all_pass = all_pass and desc_pass
        status = "✅ PASS" if desc_pass else "❌ FAIL"
        report.append(f"\nDescriptor Quality:  {desc_inlier:5.1f}%  (target: ≥80%)   {status}")

    # Tracking by spacing and threshold
    if all_track_results:
        report.append("\nTracking Success (by frame spacing & min matches):")
        for config, result in all_track_results.items():
            track_success = result['overall_success_rate'] * 100
            track_pass = track_success >= 90
            all_pass = all_pass and track_pass
            status = "✅ PASS" if track_pass else "❌ FAIL"
            report.append(f"  {config:20s}  {track_success:5.1f}%  (target: ≥90%)   {status}")

    # Performance
    if perf_results:
        avg_fps = perf_results['average_fps']
        perf_pass = avg_fps >= 20
        all_pass = all_pass and perf_pass
        status = "✅ PASS" if perf_pass else "❌ FAIL"
        report.append(f"\nRuntime Performance:  {avg_fps:5.1f} FPS (target: ≥20 FPS) {status}")

    report.append("-" * 80)

    # Analysis and insights
    if all_track_results and len(all_track_results) > 1:
        report.append("\n" + "="*80)
        report.append("TRACKING DEGRADATION ANALYSIS")
        report.append("="*80)

        # Extract success rates for different spacings
        spacing_rates = {}
        for config, result in all_track_results.items():
            # Parse spacing from config name (e.g., "spacing5_min75")
            if 'spacing' in config:
                parts = config.split('_')
                spacing = int(parts[0].replace('spacing', ''))
                spacing_rates[spacing] = result['overall_success_rate'] * 100

        if len(spacing_rates) > 1:
            spacings_sorted = sorted(spacing_rates.keys())
            report.append("\nTracking success vs frame spacing:")
            for s in spacings_sorted:
                report.append(f"  {s:3d} frames: {spacing_rates[s]:5.1f}%")

            # Calculate degradation
            if spacings_sorted[0] == 1:
                baseline = spacing_rates[1]
                report.append(f"\nDegradation from baseline (spacing=1):")
                for s in spacings_sorted[1:]:
                    drop = baseline - spacing_rates[s]
                    report.append(f"  Spacing {s:2d}: -{drop:4.1f}%")

                    if drop > 20:
                        report.append(f"    ⚠️  SEVERE: >20% drop - features not stable over time!")
                    elif drop > 10:
                        report.append(f"    ⚠️  MODERATE: 10-20% drop - some drift")
                    elif drop > 5:
                        report.append(f"    ✅ ACCEPTABLE: 5-10% drop")
                    else:
                        report.append(f"    ✅ EXCELLENT: <5% drop")

    report.append("\n" + "="*80)
    report.append("INTERPRETATION")
    report.append("="*80)

    if all_pass:
        report.append("\n✅✅✅ ALL TESTS PASSED! ✅✅✅")
        report.append("\nYour model handles all tested difficulties!")
    else:
        report.append("\n⚠️  SOME TESTS FAILED")

        # Specific failure analysis
        if all_repeat_results:
            worst_spacing = min(all_repeat_results.items(),
                              key=lambda x: x[1]['overall_mean_repeatability'])
            if worst_spacing[1]['overall_mean_repeatability'] < 0.60:
                report.append(f"\n❌ Repeatability drops at spacing {worst_spacing[0]}")
                report.append("   → Features not stable over larger frame gaps")

        if all_track_results:
            worst_config = min(all_track_results.items(),
                             key=lambda x: x[1]['overall_success_rate'])
            if worst_config[1]['overall_success_rate'] < 0.90:
                report.append(f"\n❌ Tracking fails on {worst_config[0]}")
                report.append("   → Cannot handle challenging scenarios")

        if desc_results and desc_results['overall_inlier_ratio'] < 0.80:
            report.append(f"\n⚠️  Descriptor quality at {desc_results['overall_inlier_ratio']*100:.1f}%")
            report.append("   → BUT tracking still works due to RANSAC filtering!")
            report.append("   → This is ACCEPTABLE for SLAM (comparable to ORB)")

    # Reality check
    report.append("\n" + "="*80)
    report.append("REALITY CHECK: Is 100% Tracking Too Good To Be True?")
    report.append("="*80)

    if all_track_results:
        all_100 = all(r['overall_success_rate'] == 1.0 for r in all_track_results.values())

        if all_100:
            report.append("\n⚠️  WARNING: 100% success across ALL configurations!")
            report.append("\nPossible reasons:")
            report.append("  1. ✅ Your model is genuinely robust (high repeatability)")
            report.append("  2. ⚠️  Test might be too easy (consecutive frames, short sequences)")
            report.append("  3. ⚠️  Success threshold might be too lenient")
            report.append("\nTo verify robustness:")
            report.append("  - Run with --difficulty hard or --difficulty extreme")
            report.append("  - Test with --frame_spacings 10 20 30")
            report.append("  - Test on fr3_walking sequences (dynamic scenes)")
            report.append("  - Compare match quality distribution (not just count)")
        else:
            report.append("\n✅ Tracking degrades with difficulty - this is realistic!")
            report.append("Your model shows expected failure patterns.")

    report.append("\n" + "="*80)

    return "\n".join(report)
    """Generate a comprehensive summary report"""

    # Load all JSON results
    repeat_results = load_json_results(output_dir / 'repeatability_results.json')
    desc_results = load_json_results(output_dir / 'descriptor_quality_results.json')
    track_results = load_json_results(output_dir / 'tracking_results.json')
    perf_results = load_json_results(output_dir / 'performance_results.json')

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("SEMANTIC SLAM EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Overall summary
    report.append("-" * 80)
    report.append("OVERALL SUMMARY")
    report.append("-" * 80)

    all_pass = True

    if repeat_results:
        repeat_mean = repeat_results['overall_mean_repeatability'] * 100
        repeat_pass = repeat_mean >= 60
        all_pass = all_pass and repeat_pass
        status = "✅ PASS" if repeat_pass else "❌ FAIL"
        report.append(f"Repeatability:          {repeat_mean:5.1f}%  (target: ≥60%)   {status}")

    if desc_results:
        desc_inlier = desc_results['overall_inlier_ratio'] * 100
        desc_pass = desc_inlier >= 80
        all_pass = all_pass and desc_pass
        status = "✅ PASS" if desc_pass else "❌ FAIL"
        report.append(f"Descriptor Quality:     {desc_inlier:5.1f}%  (target: ≥80%)   {status}")

    if track_results:
        track_success = track_results['overall_success_rate'] * 100
        track_pass = track_success >= 90
        all_pass = all_pass and track_pass
        status = "✅ PASS" if track_pass else "❌ FAIL"
        report.append(f"Tracking Success:       {track_success:5.1f}%  (target: ≥90%)   {status}")

    if perf_results:
        avg_fps = perf_results['average_fps']
        perf_pass = avg_fps >= 20
        all_pass = all_pass and perf_pass
        status = "✅ PASS" if perf_pass else "❌ FAIL"
        report.append(f"Runtime Performance:    {avg_fps:5.1f} FPS (target: ≥20 FPS) {status}")

    report.append("-" * 80)

    if all_pass:
        report.append("\n✅✅✅ ALL TESTS PASSED! ✅✅✅")
        report.append("\nYour model is ready for SLAM integration!")
        report.append("Next steps:")
        report.append("  1. Test on more challenging sequences (fr3_long_office)")
        report.append("  2. Compare against ORB-SLAM3 baseline")
        report.append("  3. Integrate into pySLAM framework")
        report.append("  4. Start writing results chapter of thesis!")
    else:
        report.append("\n⚠️  SOME TESTS FAILED")
        report.append("\nRecommendations:")

        if repeat_results and repeat_results['overall_mean_repeatability'] < 0.60:
            report.append("  - Increase repeatability loss weight to 0.5")

        if desc_results and desc_results['overall_inlier_ratio'] < 0.80:
            report.append("  - Increase descriptor loss weight to 5.0")

        if track_results and track_results['overall_success_rate'] < 0.90:
            report.append("  - Check both repeatability and descriptor quality")
            report.append("  - May need to retrain with adjusted loss weights")

        if perf_results and perf_results['average_fps'] < 20:
            report.append("  - Consider using smaller DINOv3 model")
            report.append("  - Reduce number of keypoints from 500 to 300")

    report.append("\n" + "=" * 80)

    # Detailed results
    report.append("\nDETAILED RESULTS BY SEQUENCE")
    report.append("=" * 80)

    if repeat_results:
        report.append("\nRepeatability:")
        for seq in repeat_results['sequences']:
            name = seq['name'].split('_')[-1]
            mean = seq['mean_repeatability'] * 100
            std = seq['std_repeatability'] * 100
            status = "✅" if mean >= 60 else "❌"
            report.append(f"  {name:15s} {mean:5.1f}% ± {std:4.1f}%  {status}")

    if desc_results:
        report.append("\nDescriptor Quality:")
        for seq in desc_results['sequences']:
            name = seq['name'].split('_')[-1]
            inlier = seq['inlier_ratio'] * 100
            precision = seq['precision'] * 100
            status = "✅" if inlier >= 80 else "❌"
            report.append(f"  {name:15s} Inlier: {inlier:5.1f}%, Precision: {precision:5.1f}%  {status}")

    if track_results:
        report.append("\nTracking Success:")
        for seq in track_results['sequences']:
            name = seq['name'].split('_')[-1]
            success = seq['success_rate'] * 100
            matches = seq['mean_matches']
            status = "✅" if success >= 90 else ("⚠️" if success >= 70 else "❌")
            report.append(f"  {name:15s} {success:5.1f}% (avg matches: {matches:.0f})  {status}")

    if perf_results:
        report.append("\nPerformance:")
        for seq in perf_results['sequences']:
            name = seq['name'].split('_')[-1]
            fps = seq['fps']
            time_ms = seq['total_time_ms']
            status = "✅" if fps >= 20 else "❌"
            report.append(f"  {name:15s} {fps:5.1f} FPS ({time_ms:5.1f} ms)  {status}")

    report.append("\n" + "=" * 80)

    # Files generated
    report.append("\nGENERATED FILES")
    report.append("=" * 80)
    report.append(f"Results directory: {output_dir}")
    report.append("\nVisualizations:")
    report.append("  - repeatability.png")
    report.append("  - descriptor_quality.png")
    report.append("  - tracking.png")
    report.append("  - performance.png")
    report.append("\nData files:")
    report.append("  - repeatability_results.json")
    report.append("  - descriptor_quality_results.json")
    report.append("  - tracking_results.json")
    report.append("  - performance_results.json")
    report.append("  - summary_report.txt (this file)")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Run all evaluation tests')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model_1.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='../configs/train_config.yaml',
                       help='Path to training config')
    parser.add_argument('--sequences', nargs='+',
                       default=['rgbd_dataset_freiburg1_plant',
                               'rgbd_dataset_freiburg1_desk',
                               'rgbd_dataset_freiburg1_room'],
                       help='Sequences to test')
    parser.add_argument('--num_pairs', type=int, default=50,
                       help='Number of frame pairs per test')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--skip_slow', action='store_true',
                       help='Skip slow tests (tracking)')

    # NEW: Difficulty/stress testing parameters
    parser.add_argument('--difficulty', type=str, default='normal',
                       choices=['easy', 'normal', 'hard', 'extreme'],
                       help='Test difficulty level')
    parser.add_argument('--frame_spacings', nargs='+', type=int,
                       default=[1],
                       help='Frame spacings to test (1=consecutive, 5=skip 4 frames)')
    parser.add_argument('--min_matches', type=int, default=50,
                       help='Minimum matches required for tracking success')
    parser.add_argument('--stress_test', action='store_true',
                       help='Run additional stress tests')
    parser.add_argument('--long_term', action='store_true',
                       help='Test long sequences (200+ frames) for drift')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set difficulty parameters
    difficulty_configs = {
        'easy': {
            'frame_spacings': [1],
            'min_matches': 30,
            'num_pairs': 30,
            'max_frames': 50
        },
        'normal': {
            'frame_spacings': [1, 5],
            'min_matches': 50,
            'num_pairs': 50,
            'max_frames': 100
        },
        'hard': {
            'frame_spacings': [1, 5, 10],
            'min_matches': 75,
            'num_pairs': 100,
            'max_frames': 150
        },
        'extreme': {
            'frame_spacings': [1, 5, 10, 20],
            'min_matches': 100,
            'num_pairs': 150,
            'max_frames': 200
        }
    }

    difficulty = difficulty_configs[args.difficulty]
    if args.frame_spacings != [1]:  # Override if specified
        difficulty['frame_spacings'] = args.frame_spacings

    print("\n" + "="*80)
    print("SEMANTIC SLAM - COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Config:         {args.config}")
    print(f"Sequences:      {', '.join([s.split('_')[-1] for s in args.sequences])}")
    print(f"Difficulty:     {args.difficulty.upper()}")
    print(f"Frame spacings: {difficulty['frame_spacings']}")
    print(f"Min matches:    {args.min_matches if args.min_matches != 50 else difficulty['min_matches']}")
    print(f"Output:         {output_dir}")
    print("="*80)

    # Common arguments
    common_args = [
        '--checkpoint', args.checkpoint,
        '--config', args.config,
        '--sequences'
    ] + args.sequences + [
        '--num_pairs', str(args.num_pairs if args.num_pairs != 50 else difficulty['num_pairs'])
    ]

    results = {}

    # Test 1: Repeatability (test with different frame spacings)
    print("\n" + "🔄 " * 20)
    print("TEST 1/4: REPEATABILITY (Multiple Frame Spacings)")
    print("🔄 " * 20)

    for spacing in difficulty['frame_spacings']:
        print(f"\n--- Testing with frame spacing: {spacing} ---")
        test_args = common_args.copy()
        test_args.extend([
            '--frame_spacing', str(spacing),
            '--output', str(output_dir / f'repeatability_spacing{spacing}.png')
        ])

        result = run_test('test/test_repeatability.py', test_args)
        results[f'repeatability_spacing{spacing}'] = result

    # Test 2: Descriptor Quality
    print("\n" + "📊 " * 20)
    print("TEST 2/4: DESCRIPTOR QUALITY")
    print("📊 " * 20)
    test_args = common_args.copy()
    test_args.extend(['--output', str(output_dir / 'descriptor_quality.png')])
    results['descriptor'] = run_test('test/test_descriptor_quality.py', test_args)

    # Test 3: Tracking (test with different spacings and match thresholds)
    if not args.skip_slow:
        print("\n" + "🎯 " * 20)
        print("TEST 3/4: TRACKING (Multiple Configurations)")
        print("🎯 " * 20)

        for spacing in difficulty['frame_spacings']:
            min_match = args.min_matches if args.min_matches != 50 else difficulty['min_matches']
            print(f"\n--- Spacing: {spacing}, Min matches: {min_match} ---")

            test_args = [
                '--checkpoint', args.checkpoint,
                '--config', args.config,
                '--sequences'] + args.sequences + [
                '--max_frames', str(difficulty['max_frames']),
                '--min_matches', str(min_match),
                '--frame_spacing', str(spacing),
                '--output', str(output_dir / f'tracking_spacing{spacing}_min{min_match}.png')
            ]

            result = run_test('test/test_tracking.py', test_args)
            results[f'tracking_spacing{spacing}_min{min_match}'] = result
    else:
        print("\n⏭️  Skipping tracking test (use --skip_slow=False to enable)")

    # Test 4: Performance
    print("\n" + "⚡ " * 20)
    print("TEST 4/4: PERFORMANCE")
    print("⚡ " * 20)
    test_args = [
        '--checkpoint', args.checkpoint,
        '--config', args.config,
        '--sequences', args.sequences[0],  # Just one sequence for performance
        '--output', str(output_dir / 'performance.png')
    ]
    results['performance'] = run_test('test/test_performance.py', test_args)

    # Optional: Stress tests
    if args.stress_test:
        print("\n" + "💥 " * 20)
        print("BONUS: STRESS TESTS")
        print("💥 " * 20)

        stress_args = [
            '--checkpoint', args.checkpoint,
            '--config', args.config,
            '--sequences'] + args.sequences + [
            '--output', str(output_dir / 'stress_test.png')
        ]
        results['stress'] = run_test('test/test_stress.py', stress_args)

    # Optional: Long-term drift test
    if args.long_term:
        print("\n" + "📏 " * 20)
        print("BONUS: LONG-TERM DRIFT TEST")
        print("📏 " * 20)

        # Test on longest available sequences
        long_sequences = ['rgbd_dataset_freiburg3_long_office']

        drift_args = [
            '--checkpoint', args.checkpoint,
            '--config', args.config,
            '--sequences'] + long_sequences + [
            '--max_frames', '500',
            '--output', str(output_dir / 'long_term_drift.png')
        ]
        results['drift'] = run_test('test/test_tracking.py', drift_args)

    # Generate summary report
    print("\n" + "📝 " * 20)
    print("GENERATING COMPREHENSIVE SUMMARY REPORT")
    print("📝 " * 20 + "\n")

    # Load all results (handle multiple spacing tests)
    all_repeat_results = {}
    for spacing in difficulty['frame_spacings']:
        result = load_json_results(output_dir / f'repeatability_spacing{spacing}_results.json')
        if result:
            all_repeat_results[spacing] = result

    desc_results = load_json_results(output_dir / 'descriptor_quality_results.json')

    all_track_results = {}
    for spacing in difficulty['frame_spacings']:
        min_match = args.min_matches if args.min_matches != 50 else difficulty['min_matches']
        result = load_json_results(output_dir / f'tracking_spacing{spacing}_min{min_match}_results.json')
        if result:
            all_track_results[f'spacing{spacing}_min{min_match}'] = result

    perf_results = load_json_results(output_dir / 'performance_results.json')

    summary = generate_comprehensive_summary(
        all_repeat_results,
        desc_results,
        all_track_results,
        perf_results,
        args.difficulty,
        output_dir
    )

    # Save report
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(summary)

    # Print summary
    print(summary)

    print(f"\n✓ Summary report saved to: {report_path}")
    print(f"\n✓ All results saved to: {output_dir}")
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\n📊 Results Summary:")
    print(f"   - Tested {len(difficulty['frame_spacings'])} frame spacings: {difficulty['frame_spacings']}")
    print(f"   - Difficulty level: {args.difficulty}")
    print(f"   - Sequences: {len(args.sequences)}")

    print("\n🔍 To investigate further:")
    print("\n   Test HARDER difficulties:")
    print("     python test/run_all_tests.py --checkpoint CKPT --difficulty hard")
    print("     python test/run_all_tests.py --checkpoint CKPT --difficulty extreme")

    print("\n   Test specific frame spacings:")
    print("     python test/run_all_tests.py --checkpoint CKPT --frame_spacings 1 10 20")

    print("\n   Stress test:")
    print("     python test/run_all_tests.py --checkpoint CKPT --stress_test")

    print("\n   Long-term drift:")
    print("     python test/run_all_tests.py --checkpoint CKPT --long_term")

    print("\n   Raise the bar:")
    print("     python test/run_all_tests.py --checkpoint CKPT --min_matches 100")

    print("\n   Kitchen sink (everything):")
    print("     python test/run_all_tests.py --checkpoint CKPT --difficulty extreme \\")
    print("                                   --stress_test --long_term")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()