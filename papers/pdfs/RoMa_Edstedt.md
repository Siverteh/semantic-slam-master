# RoMa: Robust Dense Feature Matching

## Core Problem & Solution

Problem: Dense feature matching (finding ALL pixel correspondences between two images). Needs to be robust to:
Extreme scale changes
Illumination variations
Viewpoint changes
Low-texture regions

Previous approaches failed because:
Training from scratch = limited data, overfitting
ImageNet pretrained features = not robust enough
DINOv2 alone: too coarse (14x14)

RoMa's solution: Combine frozen DINOv2 (robust semantic features) with learned fine features (precise localization)

## Architecture Overview

Input Image Pair
    ↓
Frozen DINOv2 Encoder → Coarse features (14×14 patches, 1024-dim)
    ↓                     (frozen, pretrained)
Specialized VGG19 Encoder → Fine features (stride 1,2,4,8)
    ↓                        (learned from scratch)
Feature Pyramid: Multi-scale features
    ↓
Transformer Match Decoder → Anchor probabilities (NOT coordinates!)
    ↓
Coarse matching (low-res correspondences)
    ↓
Warp Refiners (5 levels) → Progressive refinement
    ↓
Dense correspondences + confidence maps

### Key Innovation: Two-Encoder Architecture

Why not use DINOv2 for everything?
DINOv2 has 14x14 patch size = too coarse for pixel-accurate matches
Cannot refine to sub-pixel precision

Solution:
Frozen DINOv2 for coarse matching (robust semantic understanding)
Learned VGG19 for fine features (precise localization)
No weight sharing - each encoder specialized for its task

This is directly relevant to thesis, it is similar to my descriptor refiner head.

## Three Main Contributions

### Frozen Foundation Model Features 

Finding: Frozen DINOv2 >> learned features from scratch 

Comparison (Table 1 in paper):

Frozen backbone   | PCK@1px  | PCK@3px  | PCK@5px
RN50 (ImageNet)   | 32.7%    | 72.3%    | 85.0%
VGG19 (ImageNet)  | 33.1%    | 72.7%    | 85.2%
DINOv2 (frozen)   | 42.8%    | 80.4%    | 90.1%  ← Best!

Key insight: DINOv2's self-supervised pretraining on diverse data makes it much more robust than supervised ImageNet features.

Why freeze instead of fine-tune?
Preserves generalization from massive pretraining dataset
Prevents overfitting to limited supervision data
Faster training (only learn fine features + decoder)

### Transformer Match Decoder with Anchor Probabilities

Problem with direct regression
Matching at coarse level is often multimodal (multiple plausible matches)
Regressing single coordinate can't express ambiguity
Forces network to average between modes (bad!)

RoMa's solution: Predict anchor probabilities
Instead of:
predicted_coord = decoder(features)  # Single (x,y) output

RoMa does:
anchor_probs = decoder(features)  # Probability over K anchors
predicted_coord = sum(anchor_probs[k] * anchor_positions[k])

Benefits:
Can express multimodal distributions
Softmax over anchors = proper probability distribution
Network learns which anchors are plausible

Architecture:
Transformer decoder (not CNN like previous work)
Coordinate-agnostic (doesn't directly regress coordinates)
8 transformer blocks
Outputs probabilities per location

### Two-Stage Loss Function

Key insight: Coarse matching != fine refinement, need different losses.

% Regression-by-classification
% Discretize coordinate space into bins
% Predict probability distribution over bins
loss_coarse = CrossEntropyLoss(predicted_distribution, target_bin)

% Robust regression (Huber loss)
% Once close to correct match, refine with regression
loss_fine = HuberLoss(predicted_offset, target_offset)

Why this matters:
Coarse stage: Multiple plausible matches -> classification loss
Fine stage: Single correct match -> regression loss
Better optimization, faster convergence

## Performance Results

### WxBS Benchmark (Extreme Conditions)

**What is WxBS?** Wide Baseline Stereo - extremely challenging:
- Seasonal changes (winter ↔ summer)
- Day ↔ night
- Extreme scale variations
- Viewpoint changes

**Results:**
```
Method              mAA@10px
LoFTR (CVPR 2021)   55.4%
DKM (CVPR 2023)     58.9%
RoMa (Ours)         80.1%  ← +36% improvement!
```

**This is where RoMa shines** - extreme robustness to challenging conditions.

### Other Benchmarks

**MegaDepth-1500 (outdoor scenes):**
```
Method    AUC@5°  AUC@10°  AUC@20°
DKM       60.4    74.9     85.1
RoMa      62.6    76.7     86.3  ← SOTA
```

**ScanNet-1500 (indoor scenes):**
```
Method    AUC@5°  AUC@10°  AUC@20°
DKM       29.4    50.7     68.3
RoMa      31.8    53.4     70.9  ← SOTA
```

**IMC2022 (Image Matching Challenge):**
```
Method           mAA@10
ASpanFormer      83.8%
DKM              83.1%
RoMa             88.0%  ← SOTA
```

**Consistent improvement across all benchmarks** - not overfitted to specific dataset.

---

## Ablation Study (Critical Insights)

**Table 2 in paper - Progressive improvements:**

Setup                              PCK@1px  PCK@3px  PCK@5px
I. Baseline (DKM, retrained)       37.0     75.8     87.1
II. Separate fine/coarse encoders  38.2     77.1     87.9
III. Replace RN50 → VGG19          38.8     77.6     88.3
IV. Add Transformer decoder        40.1     78.7     89.0
V. Add DINOv2 coarse features      42.0     79.8     89.8  ← Big jump!
VI. Regression-by-classification   43.5     81.0     90.5
VII. Robust regression (RoMa)      44.2     81.7     90.9
VIII. Remove Transformer decoder   41.8     79.5     89.6  ← Degrades

## Key Findings

DINOv2 gives biggest single improvement
Specialized encoders help
VGG19 > ResNet50 for fine features
Transformer decoder is critical with DINOv2
Loss improvements matter

## Direct Relevance to my Thesis

### How to Use Frozen DINOv2 Effectively

RoMa's approach:

Freeze DINOv2:
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
for param in dinov2.parameters():
    param.requires_grad = False

Extract coarse features:
coarse_features = dinov2.forward_features(image)

My approach should be similar
Freeze DINOv3 backbone
Use patch features (not CLS token)
Project to lower dimension (512-dim in RoMa)

### Specialized Fine Features are Essential

RoMa proves: DINOv2 alone is insufficient, need geometric features

My descriptor refiner head = exactly this
RoMa uses VGG19 for geometric features
I'm learning a refiner head to add geometric info
Same principle, different implementation

### Don't Share Weights Between Coarse/Fine

RoMa shows specialization matters:
Coarse encoder optimized for semantic understanding
Fine encoder optimized for precise localization
Separate weights = better performance

For my architecture:

%Good (RoMa approach):
self.dinov3 = load_frozen_dinov3()     # Coarse semantic
self.descriptor_refiner = RefinerHead()  # Fine geometric

%Bad (previous approaches):
self.shared_encoder = ...  # Trying to do both

### Two-Stage Loss Strategy

RoMa's finding: Coarse matching needs different loss thant refinement

For my self-supervised training:
% Coarse keypoint selection (multimodal)
loss_selector = FocalLoss(saliency_heatmap, target_heatmap)

% Fine descriptor matching (unimodal)  
loss_descriptor = RobustLoss(descriptors_t, descriptors_t+1)

% Uncertainty should help both
loss_uncertainty = CalibrationLoss(uncertainty, actual_error)

### Confidence/Uncertainty is important

RoMa outputs confidence maps with correspondences
Used to weight matches in RANSAC
Helps reject outliners
Improves pose estimation

My uncertainty estimator will do the same thing.

AspectRoMaYour ThesisTaskDense matching (all pixels)Sparse SLAM (keypoints only)SupervisionSupervised (depth maps)Self-supervised (photometric)Foundation modelDINOv2 (frozen)DINOv3 (frozen)Coarse featuresUse directlyUse for keypoint selectionFine featuresVGG19 encoderLearned descriptor refinerMatchingDense warp predictionSparse feature matchingOutputDense correspondence mapsCamera pose + sparse mapSpeed~200ms per pair20-30 FPS real-time