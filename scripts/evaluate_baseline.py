#!/usr/bin/env python3
"""
Evaluate ORB-SLAM3 baseline trajectories using evo toolkit.
Computes ATE, RPE and generates visualizations.
"""

import json
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from evo.core import sync, metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D
from evo.tools import file_interface


# Paths
PROJECT_ROOT = Path("/workspace")
DATA_PATH = PROJECT_ROOT / "data/tum_rgbd"
BASELINE_PATH = PROJECT_ROOT / "experiments/baselines/orb_slam3"
TRAJ_PATH = BASELINE_PATH / "trajectories"
PLOT_PATH = BASELINE_PATH / "plots"

# Create output directories
PLOT_PATH.mkdir(parents=True, exist_ok=True)


def load_trajectory(traj_file: Path, gt_file: Path) -> Tuple[PosePath3D, PosePath3D, float]:
    """
    Load estimated and ground truth trajectories.

    Args:
        traj_file: Path to ORB-SLAM3 trajectory (TUM format)
        gt_file: Path to ground truth trajectory

    Returns:
        (estimated_trajectory, ground_truth_trajectory, max_diff)
    """
    # Load trajectories
    traj_est = file_interface.read_tum_trajectory_file(str(traj_file))
    traj_gt = file_interface.read_tum_trajectory_file(str(gt_file))

    # Synchronize timestamps (max 0.01s difference)
    max_diff = 0.01
    traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est, max_diff)

    return traj_est, traj_gt, max_diff


def compute_ate(traj_est: PosePath3D, traj_gt: PosePath3D) -> Dict:
    """
    Compute Absolute Trajectory Error.

    Returns:
        Dictionary with ATE statistics
    """
    # Align trajectories using deepcopy and the align method
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_gt, correct_scale=False, correct_only_scale=False)

    # Compute ATE
    ate_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ate_metric.process_data((traj_gt, traj_est_aligned))

    ate_stats = ate_metric.get_all_statistics()

    return {
        "rmse": float(ate_stats["rmse"]),
        "mean": float(ate_stats["mean"]),
        "median": float(ate_stats["median"]),
        "std": float(ate_stats["std"]),
        "min": float(ate_stats["min"]),
        "max": float(ate_stats["max"]),
    }


def compute_rpe(traj_est: PosePath3D, traj_gt: PosePath3D,
                delta: float = 10) -> Dict:
    """
    Compute Relative Pose Error.

    Args:
        delta: Frame interval for RPE computation (number of frames)

    Returns:
        Dictionary with RPE statistics for translation and rotation, or None if computation fails
    """
    try:
        # RPE for translation - use time-based delta instead of distance
        rpe_trans = metrics.RPE(
            pose_relation=PoseRelation.translation_part,
            delta=delta,
            delta_unit=Unit.frames,  # Use frames instead of meters
            all_pairs=False
        )
        rpe_trans.process_data((traj_gt, traj_est))
        rpe_trans_stats = rpe_trans.get_all_statistics()

        # RPE for rotation
        rpe_rot = metrics.RPE(
            pose_relation=PoseRelation.rotation_angle_deg,
            delta=delta,
            delta_unit=Unit.frames,  # Use frames instead of meters
            all_pairs=False
        )
        rpe_rot.process_data((traj_gt, traj_est))
        rpe_rot_stats = rpe_rot.get_all_statistics()

        return {
            "translation": {
                "rmse": float(rpe_trans_stats["rmse"]),
                "mean": float(rpe_trans_stats["mean"]),
                "median": float(rpe_trans_stats["median"]),
                "std": float(rpe_trans_stats["std"]),
            },
            "rotation": {
                "rmse": float(rpe_rot_stats["rmse"]),
                "mean": float(rpe_rot_stats["mean"]),
                "median": float(rpe_rot_stats["median"]),
                "std": float(rpe_rot_stats["std"]),
            }
        }
    except Exception as e:
        # If RPE fails (trajectory too short, etc), return None
        print(f"  ⚠ RPE computation failed: {e}")
        return None


def plot_trajectory_comparison(traj_est: PosePath3D, traj_gt: PosePath3D,
                               output_file: Path, title: str):
    """Generate trajectory comparison plot."""
    # Align for visualization using deepcopy
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_gt, correct_scale=False, correct_only_scale=False)

    # Create plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories manually (avoid evo_plot issues)
    # Ground truth
    gt_xyz = traj_gt.positions_xyz
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
            '--', color='black', label='Ground Truth', alpha=0.5, linewidth=2)

    # Estimated (aligned)
    est_xyz = traj_est_aligned.positions_xyz
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2],
            '-', color='blue', label='ORB-SLAM3', linewidth=2)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_sequence(sequence_name: str) -> Dict:
    """
    Evaluate single sequence.

    Returns:
        Dictionary with all metrics
    """
    print(f"\nEvaluating: {sequence_name}")
    print("-" * 50)

    # File paths
    traj_file = TRAJ_PATH / f"{sequence_name}_trajectory.txt"
    gt_file = DATA_PATH / sequence_name / "groundtruth.txt"
    plot_file = PLOT_PATH / f"{sequence_name}_trajectory.png"

    # Check files exist
    if not traj_file.exists():
        print(f"⚠ Trajectory file not found: {traj_file}")
        return {"status": "missing_trajectory"}

    if not gt_file.exists():
        print(f"⚠ Ground truth not found: {gt_file}")
        return {"status": "missing_groundtruth"}

    try:
        # Load trajectories
        traj_est, traj_gt, max_diff = load_trajectory(traj_file, gt_file)

        # Compute metrics
        ate_results = compute_ate(traj_est, traj_gt)
        rpe_results = compute_rpe(traj_est, traj_gt, delta=10)  # 10 frames

        # Generate plots
        plot_trajectory_comparison(
            traj_est, traj_gt, plot_file,
            f"{sequence_name} - ORB-SLAM3 Baseline"
        )

        # Summary
        print(f"✓ ATE RMSE: {ate_results['rmse']:.4f} m")
        if rpe_results:
            print(f"✓ RPE Trans RMSE: {rpe_results['translation']['rmse']:.4f} m")
            print(f"✓ RPE Rot RMSE: {rpe_results['rotation']['rmse']:.4f} deg")
        print(f"✓ Plot saved: {plot_file}")

        result = {
            "status": "success",
            "ate": ate_results,
            "num_poses": len(traj_est.timestamps),
            "trajectory_length": float(traj_gt.path_length),
        }

        if rpe_results:
            result["rpe"] = rpe_results

        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        return {"status": "error", "error": str(e)}


def main():
    """Run evaluation on all sequences."""

    sequences = [
        "rgbd_dataset_freiburg1_desk",
        "rgbd_dataset_freiburg1_plant",
        "rgbd_dataset_freiburg1_room",
        "rgbd_dataset_freiburg3_long_office_household",
        "rgbd_dataset_freiburg3_walking_static",
        "rgbd_dataset_freiburg3_walking_xyz",
    ]

    print("=" * 60)
    print("ORB-SLAM3 Baseline Evaluation")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}")
    print(f"Results path: {BASELINE_PATH}")
    print()

    # Evaluate all sequences
    results = {}
    for seq in sequences:
        results[seq] = evaluate_sequence(seq)

    # Save results
    results_file = BASELINE_PATH / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Print summary table
    successful = [s for s, r in results.items() if r.get("status") == "success"]

    if successful:
        print(f"\n{'Sequence':<50} {'ATE RMSE':<12} {'RPE Trans':<12}")
        print("-" * 74)
        for seq in successful:
            ate_rmse = results[seq]["ate"]["rmse"]
            rpe_rmse = results[seq].get("rpe", {}).get("translation", {}).get("rmse", "N/A")
            if isinstance(rpe_rmse, float):
                print(f"{seq:<50} {ate_rmse:<12.4f} {rpe_rmse:<12.4f}")
            else:
                print(f"{seq:<50} {ate_rmse:<12.4f} {rpe_rmse:<12}")

    print(f"\n✓ Results saved: {results_file}")
    print(f"✓ Plots saved in: {PLOT_PATH}/")

    # Failed sequences
    failed = [s for s, r in results.items() if r.get("status") != "success"]
    if failed:
        print(f"\n⚠ Failed sequences: {len(failed)}")
        for seq in failed:
            print(f"  - {seq}: {results[seq].get('status', 'unknown')}")


if __name__ == "__main__":
    main()