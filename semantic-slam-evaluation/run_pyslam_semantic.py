"""
Run pySLAM with Semantic Feature Extractor
Processes TUM RGB-D sequences using trained semantic keypoint model
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
import cv2

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/workspace/baselines/pyslam/pyslam')

from semantic_feature_extractor import SemanticFeatureExtractor


class SimpleSLAM:
    """
    Simplified SLAM pipeline using semantic features.
    Focuses on frame-to-frame tracking for ATE evaluation.
    """

    def __init__(self, feature_extractor, camera_matrix, output_path):
        self.extractor = feature_extractor
        self.camera_matrix = camera_matrix
        self.output_path = output_path

        # Matcher for L2-normalized descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # State
        self.poses = []
        self.timestamps = []
        self.current_pose = np.eye(4)

        # Previous frame data
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_depth = None

    def process_frame(self, image, depth, timestamp):
        """Process a single frame"""
        # Extract features
        keypoints, descriptors = self.extractor.detectAndCompute(image)

        if self.prev_keypoints is None:
            # First frame - initialize
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_depth = depth
            self.poses.append(self.current_pose.copy())
            self.timestamps.append(timestamp)
            return True

        # Match with previous frame
        matches = self._match_features(self.prev_descriptors, descriptors)

        if len(matches) < 10:
            print(f"  ⚠ Only {len(matches)} matches - skipping frame")
            return False

        # Get 3D-2D correspondences
        pts_3d, pts_2d = self._get_correspondences(
            self.prev_keypoints, keypoints, matches, self.prev_depth
        )

        if len(pts_3d) < 10:
            print(f"  ⚠ Only {len(pts_3d)} valid 3D points - skipping")
            return False

        # Estimate pose using PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.camera_matrix, None,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None or len(inliers) < 10:
            print(f"  ⚠ PnP failed or too few inliers ({len(inliers) if inliers is not None else 0})")
            return False

        # Sanity check: translation should be reasonable (typically < 1m per frame)
        tvec_norm = np.linalg.norm(tvec)
        if tvec_norm > 5.0:  # More than 5 meters movement in one frame is suspicious
            print(f"  ⚠ Large motion detected ({tvec_norm:.2f}m) - possibly wrong")
            return False

        # Convert to transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = tvec.flatten()

        # Update global pose
        self.current_pose = self.current_pose @ T_rel

        # Store
        self.poses.append(self.current_pose.copy())
        self.timestamps.append(timestamp)

        # Update previous frame
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_depth = depth

        print(f"  ✓ Tracked: {len(matches)} matches, {len(inliers)} inliers")
        return True

    def _match_features(self, desc1, desc2):
        """Match descriptors using ratio test"""
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

        return good_matches

    def _get_correspondences(self, kpts1, kpts2, matches, depth1):
        """Get 3D-2D correspondences from matches"""
        pts_3d = []
        pts_2d = []

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        h, w = depth1.shape

        for match in matches:
            idx1 = match.queryIdx
            idx2 = match.trainIdx

            # Get 2D point in current frame
            pt2 = kpts2[idx2].pt

            # Get 3D point from previous frame
            pt1 = kpts1[idx1].pt
            x1, y1 = int(pt1[0]), int(pt1[1])

            # Check bounds
            if x1 < 0 or x1 >= w or y1 < 0 or y1 >= h:
                continue

            # Get depth (already in meters from TUM conversion)
            z = depth1[y1, x1]

            # Valid depth check (TUM: 0.5m to 5m is typical indoor range)
            if z <= 0.1 or z > 5.0:  # Reject too close (<10cm) or too far (>5m)
                continue

            # Back-project to 3D
            x = (x1 - cx) * z / fx
            y = (y1 - cy) * z / fy

            pts_3d.append([x, y, z])
            pts_2d.append(pt2)

        pts_3d_array = np.array(pts_3d, dtype=np.float32)
        pts_2d_array = np.array(pts_2d, dtype=np.float32)

        # Debug: print 3D point statistics for first frame
        if len(pts_3d_array) > 0 and len(self.poses) == 1:
            print(f"\n  3D points stats: "
                  f"X=[{pts_3d_array[:, 0].min():.2f}, {pts_3d_array[:, 0].max():.2f}], "
                  f"Y=[{pts_3d_array[:, 1].min():.2f}, {pts_3d_array[:, 1].max():.2f}], "
                  f"Z=[{pts_3d_array[:, 2].min():.2f}, {pts_3d_array[:, 2].max():.2f}]")

        return pts_3d_array, pts_2d_array

    def save_trajectory(self):
        """Save trajectory in TUM format"""
        with open(self.output_path, 'w') as f:
            for timestamp, pose in zip(self.timestamps, self.poses):
                # Extract translation
                tx, ty, tz = pose[:3, 3]

                # Sanity check: reasonable translation values (should be < 100m typically)
                if abs(tx) > 1000 or abs(ty) > 1000 or abs(tz) > 1000:
                    print(f"  ⚠ Warning: Large translation detected: [{tx:.2f}, {ty:.2f}, {tz:.2f}]")

                # Convert rotation to quaternion
                R = pose[:3, :3]
                qw, qx, qy, qz = self._rotation_to_quaternion(R)

                # Write in TUM format: timestamp tx ty tz qx qy qz qw
                f.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                       f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

        print(f"\n✓ Saved trajectory: {self.output_path}")
        print(f"  Frames tracked: {len(self.poses)}")

        # Print trajectory statistics for debugging
        if len(self.poses) > 0:
            translations = np.array([p[:3, 3] for p in self.poses])
            print(f"  Trajectory bounds:")
            print(f"    X: [{translations[:, 0].min():.3f}, {translations[:, 0].max():.3f}]")
            print(f"    Y: [{translations[:, 1].min():.3f}, {translations[:, 1].max():.3f}]")
            print(f"    Z: [{translations[:, 2].min():.3f}, {translations[:, 2].max():.3f}]")

    @staticmethod
    def _rotation_to_quaternion(R):
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return w, x, y, z


def load_tum_sequence(sequence_path, settings_path):
    """Load TUM RGB-D sequence data"""
    sequence_path = Path(sequence_path)

    # Load associations
    assoc_file = sequence_path / "associations.txt"
    associations = []
    with open(assoc_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                associations.append({
                    'timestamp': float(parts[0]),
                    'rgb_path': sequence_path / parts[1],
                    'depth_path': sequence_path / parts[3]
                })

    # Load camera calibration from settings
    # For TUM RGB-D, use standard calibration
    # TUM1: fx=517.3, fy=516.5, cx=318.6, cy=255.3
    # TUM3: fx=535.4, fy=539.2, cx=320.1, cy=247.6
    if 'TUM1' in settings_path or 'freiburg1' in str(sequence_path):
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    else:  # TUM3
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return associations, camera_matrix


def main():
    parser = argparse.ArgumentParser(description='Run pySLAM with semantic features')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained semantic model checkpoint')
    parser.add_argument('--sequence', type=str, required=True,
                       help='Path to TUM sequence directory')
    parser.add_argument('--settings', type=str, required=True,
                       help='Path to pySLAM settings file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output trajectory file path')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("PYSLAM WITH SEMANTIC FEATURES")
    print("="*70)

    # Load semantic feature extractor
    print("\n[1/4] Loading semantic feature extractor...")
    extractor = SemanticFeatureExtractor(args.checkpoint, device='cuda')

    # Load sequence
    print("\n[2/4] Loading TUM sequence...")
    associations, camera_matrix = load_tum_sequence(args.sequence, args.settings)
    print(f"  ✓ Loaded {len(associations)} frame pairs")
    print(f"  ✓ Camera matrix:")
    print(f"     fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
    print(f"     cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")

    # Initialize SLAM
    print("\n[3/4] Initializing SLAM pipeline...")
    slam = SimpleSLAM(extractor, camera_matrix, args.output)

    # Process sequence
    print("\n[4/4] Processing sequence...")
    print("-" * 70)

    num_tracked = 0
    for i, assoc in enumerate(associations):
        # Load RGB and depth
        rgb = cv2.imread(str(assoc['rgb_path']))
        depth_raw = cv2.imread(str(assoc['depth_path']), cv2.IMREAD_UNCHANGED)

        if rgb is None or depth_raw is None:
            print(f"Frame {i:4d}: ✗ Failed to load images")
            continue

        # CRITICAL: TUM depth is uint16 in millimeters, divide by 5000 to get meters
        # This converts from mm to meters (1000) and applies TUM scale factor (5)
        depth = depth_raw.astype(np.float32) / 5000.0

        # Sanity check depth values
        valid_depth = depth[(depth > 0) & (depth < 10)]  # Valid range 0-10m
        if len(valid_depth) > 0 and i == 0:
            print(f"Depth stats: min={valid_depth.min():.3f}m, max={valid_depth.max():.3f}m, mean={valid_depth.mean():.3f}m")

        # Process frame
        print(f"Frame {i:4d}/{len(associations)}: ", end='')
        success = slam.process_frame(rgb, depth, assoc['timestamp'])

        if success:
            num_tracked += 1

    print("-" * 70)
    print(f"\n✓ Processing complete!")
    print(f"  Total frames: {len(associations)}")
    print(f"  Successfully tracked: {num_tracked} ({100*num_tracked/len(associations):.1f}%)")

    # Save trajectory
    print("\n[5/5] Saving trajectory...")
    slam.save_trajectory()

    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()