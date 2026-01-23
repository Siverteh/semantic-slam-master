# ORB-SLAM3: An accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM

## What ORB-SLAM3 Is

The gold standard open-source SLAM system- most complete and accurate feature-based SLAM available.

Supports:
Monocular, Stereo, RGB-D cameras
Visual-only and Visual-Inertial modes
Pin-hole and fisheye lenses
Multi-map / Multi-session SLAM

My comparison: RGB-D monocular mode on TUM dataset

## Core Architecture

### Feature Detection and Matching

ORB Features (Oriented FAST and Rotated BRIEF):
Hand-crafted features (not learned)

FAST corner detectory for keypoints
BRIEF binary descriptors (256-bit)
Rotation invariant
FAST: 1-2 ms for 1000 features
Problems: Fails in low-texture environments

### Extraction Rates

1500 ORB points per image (monocular-inertial)
1000 points per image (stereo-inertial)

### System Architecture

Frame Input
    ↓
ORB Feature Extraction (1000-1500 points)
    ↓
Tracking Thread
    ├─ Frame-to-frame tracking (constant velocity model)
    ├─ Frame-to-map tracking (local map)
    └─ Relocalization (if lost)
    ↓
Local Mapping Thread
    ├─ Keyframe insertion
    ├─ Local BA (optimize recent keyframes)
    ├─ Point culling (remove unreliable points)
    └─ Keyframe culling
    ↓
Loop Closing Thread
    ├─ Place recognition (DBoW2)
    ├─ Loop detection
    ├─ Pose-graph optimization
    └─ Full BA (optimize entire map)

## Three Main Novelties

### Visual-Inertial SLAM with MAP Estimation

What they did:
Full Maximum-a-Posteriori (MAP) estimation even during IMU initialization
Fast IMU initialization (few seconds)

Performance:
2-10x more accurate thant previous VI-SLAM methods
EuRoC stereo-intertial: 3.5 cm average ATE
TUM-VI room sequences: 9 mm ATE (AR/VR quality!)

### Improved Place Recognition (Loop Closure)

Problem with DBoW2 (original):
Required 3 consecutive keyframes to match the same area (temporal consistency)
High precision, but low recall (slow to detect loops)
Misses many opportunities for map reuse

ORB-SLAM3's solution:
Check geometric consistency FIRST
Then local consistency with 3 covisible keyframes (not consecutive)
Higher recall = earlier loop closures = better accuracy

This results in more mid-term data association, denser map connections and better accuracy in loopy environments.

### Multi-Map SLAM

Novel contributions:
Multiple dosconnected maps
Seamless map merging when revisiting areas
Multi-session SLAM (use maps from previous runs)
Handles tracjubg kiss gracefully (starts a new map, merges later)

## Performance Results

EuRoC Dataset (indoor MAV flights)

Monocular SLAM:
MH01-05 (Machine Hall): 0.04-0.15 ATE
V1-V2 (Vicon Room): 0.02-0.08m ATE

Stereo SLAM:
Even better: 0.01-0.05 ATE
Essentially drift-free in mapped areas

TUM-VI Dataset (Handheld + Fisheye):
Room sequences (small indoor): <10 cm ATE
Corridor sequences 10-50cm ATE
Magistrale (long indoor): 1-5m ATE
outdoors (long): 10-70m ATE

## Three Types of Data Association

This is the key to ORB-SLAM3's accuracy.

### Short-Term Data Association

Matching in last few seconds
What VO systems do, like DINO-VO
Problem: Continuous drift

### Mid-Tern Data Association

Matching nearby map elements with small accumulated drift
Allows zero drift in mapped areas
ORB-SLAM3's advantage over pure VO
Uses covisibility graph to find matches

### Long-Term Data Association

Loop closure (DBoW2 place recognition)
Map merging (multi-map)
Relocalization (When tracking lost)
Corrects accumulated drift

Why this matters: My semantic features should improve all three types, but especially mid-term association in low-texture areas where ORB fails. 

## Where ORB-SLAM3 Struggles

### Low-Texture Environments

ORB-SLAM3 fails when:
Plain walls
Uniform surfaces
Repetitive textures
Very dark or very bright areas

Expected behavior:
Tracking loss
Drift accumulation
Fewer keypoints detected
Poor map quality

### Dynamic Objects

Orb has no semantic understanding
Moving people/objects create false matches
Need manual outlier rejection

DINOv3 semantic features can help distinguish static vs dynamic

### Lighting Changes

Hand-crafted features somewhat robust
But still affected by strong illumination changes
No learned adaptation

DINOv3 trained on massive data with lighting variations gives an advantage.

## Key Takeaways for the Thesis

ORB-SLAM3 is excellent in textured environments - you need to match this performance
ORB-SLAM3 fails in low-texture - this is where you win (fr1_plant)
Mid-term data association is critical - your semantic features should enable better matching in low-texture areas
Speed matters - 20-30 FPS is acceptable vs their 30-40 FPS
Statistical significance required - run 10 times, report mean ± std, use Wilcoxon test
Multi-map is powerful but not your focus - acknowledge as future work
Bundle adjustment with uncertainty - your novel contribution beyond just features