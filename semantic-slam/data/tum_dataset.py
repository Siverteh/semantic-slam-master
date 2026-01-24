"""
TUM RGB-D Dataset Loader
FIXED: Added data augmentation for better generalization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional
import torchvision.transforms as transforms
import random


class TUMDataset(Dataset):
    """
    TUM RGB-D dataset for self-supervised training.
    Returns pairs of consecutive frames with depth and pose information.
    """

    def __init__(
        self,
        dataset_root: str,
        sequence: str,
        input_size: int = 448,
        frame_spacing: int = 1,
        max_frames: Optional[int] = None,
        augmentation: Optional[dict] = None,
        is_train: bool = True
    ):
        """
        Args:
            dataset_root: Path to TUM RGB-D dataset root
            sequence: Sequence name
            input_size: Image size for DINOv3
            frame_spacing: Spacing between consecutive frames
            max_frames: Maximum frames to use
            augmentation: Dict with augmentation params (only applied if is_train=True)
            is_train: Whether this is training (apply augmentation) or validation
        """
        self.dataset_root = Path(dataset_root)
        self.sequence = sequence
        self.input_size = input_size
        self.frame_spacing = frame_spacing
        self.is_train = is_train

        # Paths
        self.sequence_dir = self.dataset_root / sequence
        self.rgb_dir = self.sequence_dir / "rgb"
        self.depth_dir = self.sequence_dir / "depth"
        self.gt_file = self.sequence_dir / "groundtruth.txt"

        # Verify paths
        assert self.sequence_dir.exists(), f"Sequence not found: {self.sequence_dir}"
        assert self.rgb_dir.exists(), f"RGB directory not found: {self.rgb_dir}"
        assert self.depth_dir.exists(), f"Depth directory not found: {self.depth_dir}"

        # Load associations
        self.rgb_files, self.depth_files, self.timestamps = self._load_associations()

        # Load ground truth poses
        self.poses = self._load_groundtruth() if self.gt_file.exists() else None

        # Limit frames
        if max_frames is not None:
            self.rgb_files = self.rgb_files[:max_frames]
            self.depth_files = self.depth_files[:max_frames]
            self.timestamps = self.timestamps[:max_frames]
            if self.poses is not None:
                self.poses = self.poses[:max_frames]

        # Base transforms (no augmentation)
        self.base_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Augmentation transforms (following R2D2/SuperPoint)
        self.augmentation = augmentation if (augmentation and is_train) else None
        if self.augmentation and self.augmentation.get('enabled', False):
            self.color_jitter = transforms.ColorJitter(
                brightness=augmentation.get('brightness', 0.2),
                contrast=augmentation.get('contrast', 0.2),
                saturation=augmentation.get('saturation', 0.2),
                hue=augmentation.get('hue', 0.1)
            )
            self.blur_prob = augmentation.get('gaussian_blur', 0.3)
            self.gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

        self.depth_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])

        print(f"Loaded TUM sequence: {sequence}")
        print(f"  Frames: {len(self.rgb_files)}")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Frame spacing: {frame_spacing}")
        print(f"  Augmentation: {'enabled' if self.augmentation else 'disabled'}")

    def __len__(self) -> int:
        return max(0, len(self.rgb_files) - self.frame_spacing)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a pair of consecutive frames with metadata.
        """
        idx1 = idx
        idx2 = idx + self.frame_spacing

        # Load RGB images
        rgb1 = Image.open(self.rgb_dir / self.rgb_files[idx1]).convert("RGB")
        rgb2 = Image.open(self.rgb_dir / self.rgb_files[idx2]).convert("RGB")

        # Load depth maps
        depth1 = Image.open(self.depth_dir / self.depth_files[idx1])
        depth2 = Image.open(self.depth_dir / self.depth_files[idx2])

        # Convert depth to meters (TUM uses scale 5000)
        depth1_array = np.array(depth1).astype(np.float32) / 5000.0
        depth2_array = np.array(depth2).astype(np.float32) / 5000.0

        # Apply augmentation to RGB (not depth!)
        if self.augmentation:
            # Same augmentation for both frames (consistency)
            seed = random.randint(0, 2**32 - 1)

            # Frame 1
            random.seed(seed)
            rgb1 = self._apply_augmentation(rgb1)

            # Frame 2 (same augmentation)
            random.seed(seed)
            rgb2 = self._apply_augmentation(rgb2)

        # Transform to tensors
        rgb1_tensor = self.base_transform(rgb1)
        rgb2_tensor = self.base_transform(rgb2)

        depth1_tensor = torch.from_numpy(depth1_array).unsqueeze(0)
        depth2_tensor = torch.from_numpy(depth2_array).unsqueeze(0)

        # Resize depth
        depth1_tensor = torch.nn.functional.interpolate(
            depth1_tensor.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode='nearest'
        ).squeeze(0)

        depth2_tensor = torch.nn.functional.interpolate(
            depth2_tensor.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode='nearest'
        ).squeeze(0)

        # Prepare output
        output = {
            'rgb1': rgb1_tensor,
            'rgb2': rgb2_tensor,
            'depth1': depth1_tensor,
            'depth2': depth2_tensor,
            'timestamp1': self.timestamps[idx1],
            'timestamp2': self.timestamps[idx2]
        }

        # Add poses if available
        if self.poses is not None:
            pose1 = self.poses[idx1]
            pose2 = self.poses[idx2]

            # Relative pose: T_rel = T2 @ T1^-1
            relative_pose = pose2 @ np.linalg.inv(pose1)

            output['pose1'] = torch.from_numpy(pose1).float()
            output['pose2'] = torch.from_numpy(pose2).float()
            output['relative_pose'] = torch.from_numpy(relative_pose).float()

        return output

    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply color jitter and blur augmentation"""
        # Color jitter
        image = self.color_jitter(image)

        # Gaussian blur (with probability)
        if random.random() < self.blur_prob:
            image = self.gaussian_blur(image)

        return image

    def _load_associations(self) -> Tuple[list, list, list]:
        """Load RGB and depth associations"""
        rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])
        depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.png')])

        # Extract timestamps
        timestamps = [float(f.split('.')[0]) for f in rgb_files]

        # Ensure same length
        min_len = min(len(rgb_files), len(depth_files))
        rgb_files = rgb_files[:min_len]
        depth_files = depth_files[:min_len]
        timestamps = timestamps[:min_len]

        return rgb_files, depth_files, timestamps

    def _load_groundtruth(self) -> np.ndarray:
        """Load ground truth poses from groundtruth.txt"""
        poses = []
        timestamps_gt = []

        with open(self.gt_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue

                parts = line.strip().split()
                if len(parts) < 8:
                    continue

                timestamp = float(parts[0])
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])

                pose = self._quat_to_matrix(qx, qy, qz, qw, tx, ty, tz)

                timestamps_gt.append(timestamp)
                poses.append(pose)

        # Associate poses with RGB timestamps
        associated_poses = []
        for ts in self.timestamps:
            idx = np.argmin(np.abs(np.array(timestamps_gt) - ts))
            associated_poses.append(poses[idx])

        return np.array(associated_poses)

    @staticmethod
    def _quat_to_matrix(qx, qy, qz, qw, tx, ty, tz) -> np.ndarray:
        """Convert quaternion and translation to 4x4 transformation matrix"""
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        return T