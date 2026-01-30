"""
Semantic Feature Extractor for pySLAM Integration
Loads trained DINOv3-based keypoint detector and descriptor refiner.
Provides cv2-compatible interface for pySLAM.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import sys

# Add semantic-slam to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'semantic-slam'))

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner


class SemanticFeatureExtractor:
    """
    Feature extractor using trained semantic keypoint detection model.
    Compatible with pySLAM's feature interface.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        num_keypoints: int = 500
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            num_keypoints: Number of keypoints to extract (default: 500)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_keypoints = num_keypoints

        print(f"Loading semantic feature extractor...")
        print(f"  Device: {self.device}")
        print(f"  Checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']

        # Initialize models
        self.backbone = DinoBackbone(
            model_name=config['model']['backbone'],
            input_size=config['model']['input_size'],
            freeze=True
        ).to(self.device)

        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            hidden_dim=config['model']['selector_hidden']
        ).to(self.device)

        self.refiner = DescriptorRefiner(
            input_dim=self.backbone.embed_dim,
            hidden_dim=config['model']['refiner_hidden'],
            output_dim=config['model']['descriptor_dim'],
            num_layers=config['model']['refiner_layers']
        ).to(self.device)

        # Load trained weights
        self.selector.load_state_dict(checkpoint['selector_state_dict'])
        self.refiner.load_state_dict(checkpoint['refiner_state_dict'])

        # Set to eval mode
        self.backbone.eval()
        self.selector.eval()
        self.refiner.eval()

        # Image preprocessing (same as training)
        self.input_size = config['model']['input_size']
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Input size: {self.input_size}x{self.input_size}")
        print(f"  ✓ Descriptor dim: {config['model']['descriptor_dim']}")
        print(f"  ✓ Target keypoints: {num_keypoints}")

    def detectAndCompute(self, image, mask=None):
        """
        Detect keypoints and compute descriptors.
        Compatible with OpenCV feature detector interface.

        Args:
            image: Input image (H, W, 3) in BGR format (OpenCV convention)
            mask: Optional mask (not used)

        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: numpy array (N, descriptor_dim)
        """
        # Store original dimensions
        orig_h, orig_w = image.shape[:2]

        # Preprocess image
        image_tensor = self._preprocess_image(image)

        with torch.no_grad():
            # Extract DINOv3 features
            features = self.backbone(image_tensor)  # (1, H, W, C)

            # Detect keypoints with sub-pixel offsets
            saliency, offsets = self.selector(features)
            keypoints_patch, scores, kpt_offsets = self.selector.select_keypoints(
                saliency, offsets, num_keypoints=self.num_keypoints
            )

            # Extract features at keypoint locations
            feat_at_kpts = self.backbone.extract_at_keypoints(features, keypoints_patch)

            # Refine descriptors
            descriptors_tensor = self.refiner(feat_at_kpts)  # (1, N, D)

        # Convert to numpy
        keypoints_patch = keypoints_patch[0].cpu().numpy()  # (N, 2) in patch coords
        scores = scores[0].cpu().numpy()  # (N,)
        descriptors = descriptors_tensor[0].cpu().numpy()  # (N, D)

        # Convert patch coordinates to pixel coordinates
        # Patch coords are in [0, grid_h-1] x [0, grid_w-1] where grid_h=grid_w=28
        # Pixel coords are in [0, orig_w-1] x [0, orig_h-1]

        # First: patch coords to input image coords (448x448)
        patch_size = self.backbone.patch_size  # 16
        pixel_coords_448 = keypoints_patch * patch_size + patch_size / 2

        # Then: scale to original image size
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        pixel_coords = pixel_coords_448.copy()
        pixel_coords[:, 0] *= scale_x  # x
        pixel_coords[:, 1] *= scale_y  # y

        # Convert to cv2.KeyPoint format
        cv_keypoints = []
        for i in range(len(pixel_coords)):
            kp = cv2.KeyPoint(
                x=float(pixel_coords[i, 0]),
                y=float(pixel_coords[i, 1]),
                size=20.0,  # Fixed size (doesn't matter for matching)
                angle=-1,   # No orientation
                response=float(scores[i]),
                octave=0,
                class_id=-1
            )
            cv_keypoints.append(kp)

        return cv_keypoints, descriptors

    def _preprocess_image(self, image):
        """
        Preprocess image for DINOv3.

        Args:
            image: BGR image (H, W, 3) from OpenCV

        Returns:
            tensor: (1, 3, input_size, input_size)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to input size
        image_resized = cv2.resize(
            image_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1]
        image_float = image_resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        image_norm = (image_float - self.mean) / self.std

        # Ensure float32 (numpy broadcasting can cause dtype issues)
        image_norm = image_norm.astype(np.float32)

        # Convert to tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)

        # Add batch dimension and ensure float32
        image_tensor = image_tensor.unsqueeze(0).float().to(self.device)

        return image_tensor

    def compute_descriptors_only(self, image, keypoints):
        """
        Compute descriptors for given keypoints.
        (Not typically used, but included for compatibility)

        Args:
            image: Input image (H, W, 3)
            keypoints: List of cv2.KeyPoint

        Returns:
            descriptors: numpy array (N, descriptor_dim)
        """
        # For simplicity, just run full detectAndCompute
        # In practice, you'd extract features at specific locations
        _, descriptors = self.detectAndCompute(image)
        return descriptors[:len(keypoints)]


def test_feature_extractor(checkpoint_path: str, test_image_path: str):
    """
    Test the semantic feature extractor on a single image.

    Args:
        checkpoint_path: Path to trained checkpoint
        test_image_path: Path to test image
    """
    print("\n" + "="*70)
    print("TESTING SEMANTIC FEATURE EXTRACTOR")
    print("="*70 + "\n")

    # Load extractor
    extractor = SemanticFeatureExtractor(checkpoint_path)

    # Load test image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Could not load image from {test_image_path}")
        return

    print(f"\nTest image: {test_image_path}")
    print(f"  Shape: {image.shape}")

    # Extract features
    print("\nExtracting features...")
    import time
    start = time.time()
    keypoints, descriptors = extractor.detectAndCompute(image)
    elapsed = time.time() - start

    print(f"\n✓ Extraction complete!")
    print(f"  Time: {elapsed*1000:.1f} ms ({1/elapsed:.1f} FPS)")
    print(f"  Keypoints: {len(keypoints)}")
    print(f"  Descriptor shape: {descriptors.shape}")
    print(f"  Descriptor range: [{descriptors.min():.3f}, {descriptors.max():.3f}]")

    # Visualize keypoints
    output_path = "test_keypoints.png"
    vis_image = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(output_path, vis_image)
    print(f"\n✓ Visualization saved: {output_path}")

    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test semantic feature extractor')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')

    args = parser.parse_args()

    test_feature_extractor(args.checkpoint, args.image)