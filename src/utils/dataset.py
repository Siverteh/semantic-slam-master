"""
TUM RGB-D Dataset utilities.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


class TUMDataset:
    """
    TUM RGB-D dataset loader.

    Dataset structure:
        sequence_name/
            rgb/
            depth/
            rgb.txt
            depth.txt
            groundtruth.txt
            associations.txt
    """

    def __init__(self, sequence_path: Path):
        """
        Args:
            sequence_path: Path to sequence directory
        """
        self.sequence_path = Path(sequence_path)
        self.rgb_dir = self.sequence_path / "rgb"
        self.depth_dir = self.sequence_path / "depth"

        # Load file lists
        self.rgb_list = self._load_file_list(self.sequence_path / "rgb.txt")
        self.depth_list = self._load_file_list(self.sequence_path / "depth.txt")

        # Load associations if exists
        assoc_file = self.sequence_path / "associations.txt"
        if assoc_file.exists():
            self.associations = self._load_associations(assoc_file)
        else:
            self.associations = None

        # Load ground truth if exists
        gt_file = self.sequence_path / "groundtruth.txt"
        if gt_file.exists():
            self.groundtruth = self._load_groundtruth(gt_file)
        else:
            self.groundtruth = None

    def _load_file_list(self, file_path: Path) -> List[Tuple[float, str]]:
        """Load timestamp-filename pairs from TUM format file."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    timestamp = float(parts[0])
                    filename = parts[1]
                    data.append((timestamp, filename))
        return data

    def _load_associations(self, file_path: Path) -> List[Tuple]:
        """Load RGB-D associations."""
        associations = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    rgb_time = float(parts[0])
                    rgb_file = parts[1]
                    depth_time = float(parts[2])
                    depth_file = parts[3]
                    associations.append((rgb_time, rgb_file, depth_time, depth_file))
        return associations

    def _load_groundtruth(self, file_path: Path) -> np.ndarray:
        """Load ground truth poses (timestamp, tx, ty, tz, qx, qy, qz, qw)."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    values = [float(x) for x in line.split()]
                    data.append(values)
        return np.array(data)

    def __len__(self) -> int:
        """Number of frames."""
        if self.associations:
            return len(self.associations)
        return min(len(self.rgb_list), len(self.depth_list))

    def __getitem__(self, idx: int) -> dict:
        """
        Get frame data.

        Returns:
            Dictionary with:
                - rgb: RGB image (H, W, 3)
                - depth: Depth image (H, W)
                - rgb_timestamp: RGB timestamp
                - depth_timestamp: Depth timestamp
                - pose: Ground truth pose if available (7,) [tx, ty, tz, qx, qy, qz, qw]
        """
        if self.associations:
            rgb_time, rgb_file, depth_time, depth_file = self.associations[idx]
        else:
            rgb_time, rgb_file = self.rgb_list[idx]
            depth_time, depth_file = self.depth_list[idx]

        # Load images
        rgb = cv2.imread(str(self.sequence_path / rgb_file))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(self.sequence_path / depth_file), cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 5000.0  # TUM depth scale factor

        result = {
            "rgb": rgb,
            "depth": depth,
            "rgb_timestamp": rgb_time,
            "depth_timestamp": depth_time,
        }

        # Add ground truth pose if available
        if self.groundtruth is not None:
            # Find closest ground truth pose
            gt_timestamps = self.groundtruth[:, 0]
            closest_idx = np.argmin(np.abs(gt_timestamps - rgb_time))
            if np.abs(gt_timestamps[closest_idx] - rgb_time) < 0.02:  # 20ms threshold
                result["pose"] = self.groundtruth[closest_idx, 1:]  # [tx, ty, tz, qx, qy, qz, qw]

        return result


def get_camera_intrinsics(sequence_name: str) -> dict:
    """
    Get camera intrinsics for TUM sequences.

    Returns:
        Dictionary with fx, fy, cx, cy
    """
    # Freiburg 1 camera
    if "freiburg1" in sequence_name:
        return {
            "fx": 517.3,
            "fy": 516.5,
            "cx": 318.6,
            "cy": 255.3,
            "width": 640,
            "height": 480,
        }
    # Freiburg 3 camera
    elif "freiburg3" in sequence_name:
        return {
            "fx": 535.4,
            "fy": 539.2,
            "cx": 320.1,
            "cy": 247.6,
            "width": 640,
            "height": 480,
        }
    else:
        raise ValueError(f"Unknown sequence: {sequence_name}")


def create_projection_matrix(intrinsics: dict) -> np.ndarray:
    """
    Create camera projection matrix K.

    Returns:
        3x3 intrinsic matrix
    """
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1]
    ])
    return K


if __name__ == "__main__":
    # Test dataset loading
    from pathlib import Path

    data_path = Path("/workspace/src/data/tum_rgbd")
    sequence = "rgbd_dataset_freiburg1_desk"

    dataset = TUMDataset(data_path / sequence)
    print(f"Dataset: {sequence}")
    print(f"Number of frames: {len(dataset)}")

    # Load first frame
    frame = dataset[0]
    print(f"RGB shape: {frame['rgb'].shape}")
    print(f"Depth shape: {frame['depth'].shape}")
    print(f"Timestamps: RGB={frame['rgb_timestamp']:.3f}, Depth={frame['depth_timestamp']:.3f}")

    if "pose" in frame:
        print(f"Ground truth pose available: {frame['pose'].shape}")

    # Camera intrinsics
    intrinsics = get_camera_intrinsics(sequence)
    print(f"Camera intrinsics: fx={intrinsics['fx']}, fy={intrinsics['fy']}")