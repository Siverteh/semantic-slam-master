"""
TUM RGB-D Dataset Loader
Loads RGB images, depth maps, and ground truth poses for training
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional
import torchvision.transforms as transforms


class TUMDataset(Dataset):
    """
    TUM RGB-D dataset for self-supervised training.
    Returns pairs of consecutive frames with depth and pose information.
    """
    
    def __init__(
        self,
        dataset_root: str,
        sequence: str,
        input_size: int = 518,
        frame_spacing: int = 1,
        max_frames: Optional[int] = None
    ):
        """
        Args:
            dataset_root: Path to TUM RGB-D dataset root (data/tum_rgbd/)
            sequence: Sequence name (e.g., "rgbd_dataset_freiburg1_desk")
            input_size: Image size for DINOv2 (518 for 37x37 patches)
            frame_spacing: Spacing between consecutive frames (1 = adjacent)
            max_frames: Maximum frames to use (None = use all)
        """
        self.dataset_root = Path(dataset_root)
        self.sequence = sequence
        self.input_size = input_size
        self.frame_spacing = frame_spacing
        
        # Paths
        self.sequence_dir = self.dataset_root / sequence
        self.rgb_dir = self.sequence_dir / "rgb"
        self.depth_dir = self.sequence_dir / "depth"
        self.gt_file = self.sequence_dir / "groundtruth.txt"
        
        # Verify paths exist
        assert self.sequence_dir.exists(), f"Sequence not found: {self.sequence_dir}"
        assert self.rgb_dir.exists(), f"RGB directory not found: {self.rgb_dir}"
        assert self.depth_dir.exists(), f"Depth directory not found: {self.depth_dir}"
        
        # Load associations (timestamp -> rgb/depth filenames)
        self.rgb_files, self.depth_files, self.timestamps = self._load_associations()
        
        # Load ground truth poses if available
        self.poses = self._load_groundtruth() if self.gt_file.exists() else None
        
        # Limit frames if requested
        if max_frames is not None:
            self.rgb_files = self.rgb_files[:max_frames]
            self.depth_files = self.depth_files[:max_frames]
            self.timestamps = self.timestamps[:max_frames]
            if self.poses is not None:
                self.poses = self.poses[:max_frames]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
        
        print(f"Loaded TUM sequence: {sequence}")
        print(f"  Frames: {len(self.rgb_files)}")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Frame spacing: {frame_spacing}")
    
    def __len__(self) -> int:
        # Return number of valid frame pairs
        return max(0, len(self.rgb_files) - self.frame_spacing)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a pair of consecutive frames with metadata.
        
        Returns:
            Dictionary containing:
                - rgb1: (3, H, W) RGB image at time t
                - rgb2: (3, H, W) RGB image at time t+spacing
                - depth1: (1, H, W) depth map at time t
                - depth2: (1, H, W) depth map at time t+spacing
                - pose1: (4, 4) camera pose at time t (if available)
                - pose2: (4, 4) camera pose at time t+spacing (if available)
                - relative_pose: (4, 4) relative pose from t to t+spacing
        """
        idx1 = idx
        idx2 = idx + self.frame_spacing
        
        # Load RGB images
        rgb1 = Image.open(self.rgb_dir / self.rgb_files[idx1]).convert("RGB")
        rgb2 = Image.open(self.rgb_dir / self.rgb_files[idx2]).convert("RGB")
        
        # Load depth maps
        depth1 = Image.open(self.depth_dir / self.depth_files[idx1])
        depth2 = Image.open(self.depth_dir / self.depth_files[idx2])
        
        # Convert depth to meters (TUM depth is in mm with scale factor 5000)
        depth1_array = np.array(depth1).astype(np.float32) / 5000.0
        depth2_array = np.array(depth2).astype(np.float32) / 5000.0
        
        # Transform
        rgb1_tensor = self.transform(rgb1)
        rgb2_tensor = self.transform(rgb2)
        
        depth1_tensor = torch.from_numpy(depth1_array).unsqueeze(0)
        depth2_tensor = torch.from_numpy(depth2_array).unsqueeze(0)
        
        # Resize depth to match input size
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
            
            # Compute relative pose: T_rel = T2 @ T1^-1
            relative_pose = pose2 @ np.linalg.inv(pose1)
            
            output['pose1'] = torch.from_numpy(pose1).float()
            output['pose2'] = torch.from_numpy(pose2).float()
            output['relative_pose'] = torch.from_numpy(relative_pose).float()
        
        return output
    
    def _load_associations(self) -> Tuple[list, list, list]:
        """Load RGB and depth associations"""
        # Simple approach: assume rgb/ and depth/ are synchronized
        rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])
        depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.png')])
        
        # Extract timestamps from filenames (format: timestamp.png)
        timestamps = [float(f.split('.')[0]) for f in rgb_files]
        
        # Ensure same length
        min_len = min(len(rgb_files), len(depth_files))
        rgb_files = rgb_files[:min_len]
        depth_files = depth_files[:min_len]
        timestamps = timestamps[:min_len]
        
        return rgb_files, depth_files, timestamps
    
    def _load_groundtruth(self) -> np.ndarray:
        """
        Load ground truth poses from groundtruth.txt
        Format: timestamp tx ty tz qx qy qz qw
        """
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
                
                # Convert quaternion + translation to 4x4 matrix
                pose = self._quat_to_matrix(qx, qy, qz, qw, tx, ty, tz)
                
                timestamps_gt.append(timestamp)
                poses.append(pose)
        
        # Associate poses with RGB timestamps (nearest neighbor)
        associated_poses = []
        for ts in self.timestamps:
            idx = np.argmin(np.abs(np.array(timestamps_gt) - ts))
            associated_poses.append(poses[idx])
        
        return np.array(associated_poses)
    
    @staticmethod
    def _quat_to_matrix(qx, qy, qz, qw, tx, ty, tz) -> np.ndarray:
        """Convert quaternion and translation to 4x4 transformation matrix"""
        # Normalize quaternion
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        # 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        
        return T