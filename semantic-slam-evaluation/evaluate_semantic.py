"""
Evaluate Semantic SLAM Results
Computes ATE and compares with ORB-SLAM3 and vanilla pySLAM baselines
"""

import numpy as np
from pathlib import Path
import subprocess
import sys
from typing import Dict, Tuple


def compute_ate_with_evo(traj_file: str, gt_file: str) -> Tuple[float, dict]:
    """
    Compute ATE using evo toolkit.

    Returns:
        ate_rmse: RMSE of absolute trajectory error
        metrics: Dictionary of all ATE metrics
    """
    try:
        # Run evo_ape
        result = subprocess.run(
            ['evo_ape', 'tum', gt_file, traj_file, '--align', '--silent'],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            print(f"  ⚠ evo_ape failed for {traj_file}")
            return float('inf'), {}

        # Parse output
        lines = result.stdout.strip().split('\n')
        metrics = {}

        for line in lines:
            if 'rmse' in line.lower():
                metrics['rmse'] = float(line.split()[-1])
            elif 'mean' in line.lower():
                metrics['mean'] = float(line.split()[-1])
            elif 'median' in line.lower():
                metrics['median'] = float(line.split()[-1])
            elif 'std' in line.lower():
                metrics['std'] = float(line.split()[-1])
            elif 'min' in line.lower():
                metrics['min'] = float(line.split()[-1])
            elif 'max' in line.lower():
                metrics['max'] = float(line.split()[-1])

        ate_rmse = metrics.get('rmse', float('inf'))

        return ate_rmse, metrics

    except Exception as e:
        print(f"  ✗ Error computing ATE: {e}")
        return float('inf'), {}


def load_baseline_results(baseline_dir: str, sequence: str) -> Dict[str, float]:
    """Load baseline ATE results"""
    results = {}

    # Try ORB-SLAM3
    orb_traj = Path(baseline_dir) / 'orb_slam3' / 'trajectories' / f'{sequence}_trajectory.txt'
    if orb_traj.exists():
        gt_file = f'/workspace/data/tum_rgbd/{sequence}/groundtruth.txt'
        ate, _ = compute_ate_with_evo(str(orb_traj), gt_file)
        results['ORB-SLAM3'] = ate

    # Try vanilla pySLAM
    pyslam_traj = Path(baseline_dir) / 'pyslam' / 'trajectories' / f'{sequence}_trajectory.txt'
    if pyslam_traj.exists():
        gt_file = f'/workspace/data/tum_rgbd/{sequence}/groundtruth.txt'
        ate, _ = compute_ate_with_evo(str(pyslam_traj), gt_file)
        results['pySLAM'] = ate

    return results


def evaluate_all_sequences(results_dir: str, baseline_dir: str):
    """Evaluate all sequences and compare with baselines"""
    results_path = Path(results_dir)
    trajectory_dir = results_path / 'trajectories'

    if not trajectory_dir.exists():
        print(f"✗ Trajectory directory not found: {trajectory_dir}")
        return

    # Find all semantic trajectories
    semantic_trajs = list(trajectory_dir.glob('*_semantic.txt'))

    if not semantic_trajs:
        print(f"✗ No semantic trajectories found in {trajectory_dir}")
        return

    print("\n" + "="*90)
    print("SEMANTIC SLAM EVALUATION RESULTS")
    print("="*90)

    # Results table
    all_results = []

    for traj_file in sorted(semantic_trajs):
        # Extract sequence name
        sequence = traj_file.stem.replace('_semantic', '')

        print(f"\n📊 Sequence: {sequence}")
        print("-" * 90)

        # Ground truth file
        gt_file = f'/workspace/data/tum_rgbd/{sequence}/groundtruth.txt'

        if not Path(gt_file).exists():
            print(f"  ⚠ Ground truth not found: {gt_file}")
            continue

        # Compute ATE for semantic model
        print(f"  Computing ATE for semantic model...")
        ate_semantic, metrics_semantic = compute_ate_with_evo(str(traj_file), gt_file)

        # Load baseline results
        baseline_results = load_baseline_results(baseline_dir, sequence)

        # Print results
        print(f"\n  {'Method':<20} {'ATE RMSE (m)':<15} {'Improvement':<15}")
        print(f"  {'-'*50}")

        # Semantic model
        print(f"  {'Semantic (Ours)':<20} {ate_semantic:<15.4f} {'baseline':<15}")

        # Baselines
        for method, ate_baseline in baseline_results.items():
            if ate_baseline != float('inf') and ate_semantic != float('inf'):
                improvement = (ate_baseline - ate_semantic) / ate_baseline * 100
                improvement_str = f"{improvement:+.1f}%"
                print(f"  {method:<20} {ate_baseline:<15.4f} {improvement_str:<15}")
            else:
                print(f"  {method:<20} {ate_baseline:<15.4f} {'N/A':<15}")

        # Store for summary
        result = {
            'sequence': sequence,
            'semantic': ate_semantic,
            'baselines': baseline_results
        }
        all_results.append(result)

        # Print detailed metrics for semantic
        if metrics_semantic:
            print(f"\n  Semantic Model Detailed Metrics:")
            print(f"    RMSE:   {metrics_semantic.get('rmse', 'N/A'):.4f} m")
            print(f"    Mean:   {metrics_semantic.get('mean', 'N/A'):.4f} m")
            print(f"    Median: {metrics_semantic.get('median', 'N/A'):.4f} m")
            print(f"    Std:    {metrics_semantic.get('std', 'N/A'):.4f} m")
            print(f"    Min:    {metrics_semantic.get('min', 'N/A'):.4f} m")
            print(f"    Max:    {metrics_semantic.get('max', 'N/A'):.4f} m")

    # Summary statistics
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)

    # Compute average improvements
    if all_results:
        semantic_ates = [r['semantic'] for r in all_results if r['semantic'] != float('inf')]

        print(f"\n📈 Average ATE (Semantic): {np.mean(semantic_ates):.4f} ± {np.std(semantic_ates):.4f} m")

        for method in ['ORB-SLAM3', 'pySLAM']:
            baseline_ates = []
            improvements = []

            for result in all_results:
                if method in result['baselines']:
                    ate_baseline = result['baselines'][method]
                    ate_semantic = result['semantic']

                    if ate_baseline != float('inf') and ate_semantic != float('inf'):
                        baseline_ates.append(ate_baseline)
                        improvement = (ate_baseline - ate_semantic) / ate_baseline * 100
                        improvements.append(improvement)

            if baseline_ates:
                avg_baseline = np.mean(baseline_ates)
                avg_improvement = np.mean(improvements)

                print(f"\n📈 vs {method}:")
                print(f"     Baseline ATE:    {avg_baseline:.4f} m")
                print(f"     Improvement:     {avg_improvement:+.1f}%")

                # Check hypothesis
                if avg_improvement >= 15.0:
                    print(f"     ✅ Hypothesis met! (≥15% improvement)")
                elif avg_improvement >= 10.0:
                    print(f"     ⚠️  Close to hypothesis (10-15% improvement)")
                else:
                    print(f"     ❌ Below hypothesis target (<10% improvement)")

    print("\n" + "="*90)

    # Save results to file
    results_file = results_path / 'evaluation_summary.txt'
    with open(results_file, 'w') as f:
        f.write("Semantic SLAM Evaluation Results\n")
        f.write("="*90 + "\n\n")

        for result in all_results:
            f.write(f"Sequence: {result['sequence']}\n")
            f.write(f"  Semantic ATE: {result['semantic']:.4f} m\n")

            for method, ate in result['baselines'].items():
                if ate != float('inf') and result['semantic'] != float('inf'):
                    improvement = (ate - result['semantic']) / ate * 100
                    f.write(f"  {method} ATE: {ate:.4f} m (improvement: {improvement:+.1f}%)\n")
            f.write("\n")

    print(f"\n✓ Results saved to: {results_file}")


def main():
    """Main evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate semantic SLAM results')
    parser.add_argument('--results_dir', type=str,
                       default='/workspace/semantic-slam-evaluation/results',
                       help='Directory containing semantic trajectories')
    parser.add_argument('--baseline_dir', type=str,
                       default='/workspace/experiments/baselines',
                       help='Directory containing baseline results')

    args = parser.parse_args()

    # Check if evo is installed
    try:
        subprocess.run(['evo_ape', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ evo toolkit not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'evo', '--upgrade'],
                      check=True)

    evaluate_all_sequences(args.results_dir, args.baseline_dir)


if __name__ == "__main__":
    main()