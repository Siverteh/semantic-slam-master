"""
Quick test to verify keypoint selector fix
Checks that keypoints are selected from HIGH saliency regions
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from models.keypoint_selector import KeypointSelector

# Create fake saliency map with clear high/low regions
B, H, W = 1, 28, 28

# Create saliency with bright spot in center, dark elsewhere
saliency_map = torch.zeros(B, H, W, 1)

# Make center region bright (0.8-1.0)
saliency_map[0, 10:18, 10:18, 0] = torch.rand(8, 8) * 0.2 + 0.8

# Make corners medium (0.3-0.5)
saliency_map[0, 0:5, 0:5, 0] = torch.rand(5, 5) * 0.2 + 0.3
saliency_map[0, 0:5, 23:28, 0] = torch.rand(5, 5) * 0.2 + 0.3

# Rest is low (<0.1)
saliency_map += torch.rand(B, H, W, 1) * 0.05

# Create selector
selector = KeypointSelector(input_dim=384)
selector.eval()

# Test keypoint selection
keypoints, scores = selector.select_keypoints(
    saliency_map,
    num_keypoints=100,
    nms_radius=2,
    threshold=0.3  # High threshold
)

print("="*70)
print("KEYPOINT SELECTOR TEST")
print("="*70)
print(f"Selected keypoints: {len(keypoints[0])}")
print(f"Score range: [{scores[0].min():.3f}, {scores[0].max():.3f}]")
print(f"Mean score: {scores[0].mean():.3f}")
print(f"Scores > 0.5: {(scores[0] > 0.5).sum().item()}")
print(f"Scores > 0.3: {(scores[0] > 0.3).sum().item()}")
print(f"Scores < 0.1: {(scores[0] < 0.1).sum().item()}")
print("="*70)

# Check if keypoints are in high saliency regions
kpts_np = keypoints[0].numpy()
sal_np = saliency_map[0, :, :, 0].numpy()

# Count keypoints in center (bright region)
center_mask = (kpts_np[:, 0] >= 10) & (kpts_np[:, 0] < 18) & \
              (kpts_np[:, 1] >= 10) & (kpts_np[:, 1] < 18)
pct_in_center = center_mask.sum() / len(kpts_np) * 100

print(f"\nKeypoints in center bright region: {center_mask.sum()} ({pct_in_center:.1f}%)")
print(f"Expected: >70% in center (where saliency is high)")

if pct_in_center > 70:
    print("✅ PASS: Keypoints correctly selected from high saliency!")
else:
    print("❌ FAIL: Keypoints not in high saliency regions!")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Saliency map
im = axes[0].imshow(sal_np, cmap='hot')
axes[0].set_title('Saliency Map\n(Bright center = high saliency)')
plt.colorbar(im, ax=axes[0])

# Keypoints on saliency
axes[1].imshow(sal_np, cmap='hot', alpha=0.5)
scatter = axes[1].scatter(
    kpts_np[:, 0], kpts_np[:, 1],
    c=scores[0].numpy(), cmap='viridis',
    s=50, edgecolors='white', linewidths=1
)
axes[1].set_title('Selected Keypoints\n(Should be in bright regions)')
plt.colorbar(scatter, ax=axes[1], label='Score')

plt.tight_layout()
plt.savefig('selector_test.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to selector_test.png")
print("Check that keypoints are concentrated in CENTER (bright region)!")