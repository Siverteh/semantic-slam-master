# DINOv3

## Core Improvements Over DINOv2

### Gram Anchoring (Main Innovation)

Problem DINOv2 had: Dense feature maps degrade during long training with large models. Features become "blurry" and lose spatial precision

DINOv3's solution: Gram anchoring, a technique to preserve dense feature quality even at scale.

Anchors the Gram matrix (feature correlation structure) during training
Prevents feature collapse that happens with very large models
Gives clean, sharp, dense features even at high resolution

This gives DINOv3 significantly better dense feature maps than DINOv2

### Scale-Up

Largest model: 7 billion parameters vs DINOv2's max og 1B
Trained on larger, more diverse datasets
Family of models from ViT-s to ViT-H+

### Better Data Curation

More careful dataset preparation
Includes aerial/satellite images
Better handling of domain-specific data

## Technical Architecture
Same backbone as DINOv2: Vision Transformer (ViT)

ViT-Small: 22M params (same as DINOv2-S)
ViT-Base: 86M params
ViT-Large: 304M params
ViT-Giant: 1.1B params
New: ViT-Huge+: 840M params
New: 7B parameter model

Patch size: Still 16x16
Training: Self-supervised (same SSL approach as DINOv2)

## Performance Improvements

Semantic Segmentation:
DINOv2 ViT-L: 54.0 mIoU
DINOv3 ViT-L: 55.9 mIoU (+1.9)
DINOv3 7B: Even higher

Depth estimation:
DINOv3 outperforms DINOv2 by 0.278 RMSE
Better dense features for geometric tasks

3D Correspondence Matching:
DINOv3: 60.2% recall
DINOv2: ~55% recall
~5% improvement in geometric matching

