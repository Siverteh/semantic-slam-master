"""
Inference Script for Semantic SLAM
Extract features from a single frame or image pair using trained DINOv3 heads
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import yaml

from models.dino_backbone import DinoBackbone
from models.keypoint_selector import KeypointSelector
from models.descriptor_refiner import DescriptorRefiner
from models.uncertainty_estimator import UncertaintyEstimator
import torchvision.transforms as transforms


class SemanticFeatureExtractor:
    """Extract semantic features from images using trained heads"""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load models
        self.backbone = DinoBackbone(
            model_name=self.config['model']['backbone'],
            input_size=self.config['model']['input_size'],
            freeze=True
        ).to(self.device)
        
        self.selector = KeypointSelector(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['selector_hidden'],
            num_layers=self.config['model']['selector_layers']
        ).to(self.device)
        
        self.refiner = DescriptorRefiner(
            input_dim=self.backbone.embed_dim,
            hidden_dim=self.config['model']['refiner_hidden'],
            output_dim=self.config['model']['descriptor_dim'],
            num_layers=self.config['model']['refiner_layers']
        ).to(self.device)
        
        self.estimator = UncertaintyEstimator(
            dino_dim=self.backbone.embed_dim,
            descriptor_dim=self.config['model']['descriptor_dim'],
            hidden_dim=self.config['model']['estimator_hidden']
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.selector.load_state_dict(checkpoint['selector_state_dict'])
        self.refiner.load_state_dict(checkpoint['refiner_state_dict'])
        self.estimator.load_state_dict(checkpoint['estimator_state_dict'])
        
        # Set to eval mode
        self.selector.eval()
        self.refiner.eval()
        self.estimator.eval()
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((self.config['model']['input_size'], 
                             self.config['model']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def extract_features(self, image_path: str) -> dict:
        """
        Extract semantic features from a single image using DINOv3.
        
        Returns:
            Dictionary with:
                - keypoints: (N, 2) numpy array
                - descriptors: (N, D) numpy array
                - confidence: (N,) numpy array
                - saliency_map: (H, W) numpy array
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract DINOv3 features
        dino_features = self.backbone(image_tensor)  # (1, H, W, C)
        
        # Keypoint selection
        saliency_map = self.selector(dino_features)  # (1, H, W, 1)
        keypoints, scores = self.selector.select_keypoints(
            saliency_map,
            num_keypoints=self.config['model']['num_keypoints']
        )
        
        # Extract features at keypoints
        features_at_kpts = self.backbone.extract_at_keypoints(dino_features, keypoints)
        
        # Refine descriptors
        descriptors = self.refiner(features_at_kpts)
        
        # Estimate confidence
        confidence = self.estimator(features_at_kpts, descriptors)
        
        # Convert to numpy
        return {
            'keypoints': keypoints[0].cpu().numpy(),
            'descriptors': descriptors[0].cpu().numpy(),
            'confidence': confidence[0, :, 0].cpu().numpy(),
            'scores': scores[0].cpu().numpy(),
            'saliency_map': saliency_map[0, :, :, 0].cpu().numpy()
        }
    
    def visualize_features(
        self,
        image_path: str,
        output_path: str = None,
        top_k: int = 100
    ):
        """Visualize extracted features on image"""
        # Extract features
        features = self.extract_features(image_path)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.config['model']['input_size'], 
                             self.config['model']['input_size']))
        image_np = np.array(image)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original image with keypoints
        axes[0].imshow(image_np)
        kpts = features['keypoints'][:top_k]
        conf = features['confidence'][:top_k]
        
        # Color keypoints by confidence
        scatter = axes[0].scatter(
            kpts[:, 0], kpts[:, 1],
            c=conf, cmap='hot', s=50, alpha=0.7
        )
        axes[0].set_title(f'Top {top_k} Keypoints (colored by confidence)')
        axes[0].axis('off')
        plt.colorbar(scatter, ax=axes[0])
        
        # 2. Saliency map
        axes[1].imshow(features['saliency_map'], cmap='hot')
        axes[1].set_title('Keypoint Saliency Map')
        axes[1].axis('off')
        
        # 3. Confidence histogram
        axes[2].hist(features['confidence'], bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Confidence Score')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Confidence Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print statistics
        print(f"\nFeature Statistics:")
        print(f"  Keypoints: {len(features['keypoints'])}")
        print(f"  Mean confidence: {features['confidence'].mean():.3f}")
        print(f"  Std confidence: {features['confidence'].std():.3f}")
        print(f"  Max saliency: {features['saliency_map'].max():.3f}")


def main():
    parser = argparse.ArgumentParser(description='Extract semantic SLAM features')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of top keypoints to visualize')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = SemanticFeatureExtractor(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    # Visualize
    extractor.visualize_features(
        image_path=args.image,
        output_path=args.output,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()