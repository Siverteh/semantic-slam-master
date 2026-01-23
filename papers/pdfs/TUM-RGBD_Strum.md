# A Benchmark for the Evaluation of RGB-D Slam Systems

## What is TUM RGB-D

The gold standard benchmark for evaluating RGB-D SLAM systems.

Created by: TUM (Technical University of Munich) Computer Vision Group + Freiburg Autonomous Intelligent Systems Lab
Released: 2012 (but still widely used today - that's how good it is!)
Key Innovation: First RGB-D dataset with highly accurate ground truth from motion capture system

## Dataset Specifications

### Recodring Setup

Sensor: Microsoft Kinect v1
resolution 640x480 (both RGB and depth)
Frame rate 30Hz
Color + depth synchronized

Ground Truth: Motion capture system
8 high-speed tracking cameras
Sampling rate: 100Hz
Accuracy: Sub-millimeter precision
Reflective markers on Kinect sensor

Environments:
fr1: office environments
fr2: Industrial hall
fr3: Additional office scenes

### Data Format

Each sequence contains:

rgb/          # Color images (PNG, 640×480)
depth/        # Depth images (PNG, 640×480, 5000 = 1 meter)
groundtruth.txt   # Camera poses (timestamp, x, y, z, qx, qy, qz, qw)
rgb.txt       # RGB timestamps + filenames
depth.txt     # Depth timestamps + filenames

Coordinate system: Standard ROS convention
x: right
y: down
z: forward

## Dataset Sequences (39 total)

### Main sequences

1. Validation/Debugging Sequences (Slow, Controlled Motion)
fr1/xyz       - Translation along axes (easy)
fr1/rpy       - Rotation around axes (easy)
Purpose: Verify your system works at all

2. Office Sequences (Typical Indoor SLAM)
fr1/desk      - Office desk, some texture ← Your baseline
fr1/desk2     - Second desk sequence
fr1/room      - Full room with loop closures
fr1/plant     - LOW TEXTURE vegetation ← YOUR TARGET!
Purpose: Realistic indoor navigation

3. Long Sequences (Drift Evaluation)
fr2/desk              - Longer desk trajectory
fr2/large_no_loop     - No loop closures (pure VO)
fr2/large_with_loop   - With loop closures
fr3/long_office_household - Very long
 trajectory
Purpose: Test drift accumulation
4. Dynamic Sequences (People Moving)
fr3/walking_static    - Camera static, person walks
fr3/walking_xyz       - Camera + person move
fr3/walking_rpy       - Rotation + dynamic objects
fr3/walking_half      - Half-sphere trajectory
Purpose: Robustness to dynamics

5. Special Challenging Sequences
fr3/nostructure_texture_near_withloop  - Low structure
fr3/nostructure_texture_far            - Very low structure
Purpose: Extreme challenges

6. Robot-Mounted Sequences
fr2/pioneer_360      - Robot rotation
fr2/pioneer_slam     - Robot navigation
Purpose: Different motion patterns (smoother than handheld)

### My Target Sequences

Primary sequences for main results:
Sequence          Length  Keyframes  Difficulty  Why Important
fr1/desk          ~20s    ~600       Easy        Textured baseline - should work well
fr1/plant         ~22s    ~1200      HARD        Low texture - YOUR MAIN TARGET
fr1/room          ~31s    ~1400      Medium      Loop closures, varied texture
fr3/long_office   ~200s   ~2500      Long        Drift + loop closures

Secondary sequences for ablations
fr1/desk2                 - Another textured baseline
fr3/nostructure_texture_* - Extreme low-texture challenge
fr3/walking_*             - Dynamic object robustness

### Recommended Evaluation Protocol

Minimum viable evaluation (for thesis):

✅ fr1/desk (baseline performance)
✅ fr1/plant (main contribution - low texture)
✅ fr1/room (loop closures)

Comprehensive evaluation:

✅ All above +
✅ fr3/long_office_household (drift)
✅ fr3/walking_xyz (dynamics)
✅ fr3/nostructure_texture_near_withloop (extreme)

## Bottom Line
TUM RGB-D is perfect for your thesis because:

✅ Standard benchmark - everyone uses it
✅ High-quality ground truth - motion capture accuracy
✅ Low-texture sequences exist - fr1/plant is your goldmine
✅ Established evaluation tools - evo toolkit
✅ Widely cited - reviewers will trust it
✅ Free and accessible - no licensing issues

fr1/plant is your star sequence - if you can beat ORB-SLAM3 there by 15%+, you have a thesis. Everything else is supporting evidence.
Focus your effort on:

Running clean baselines (ORB-SLAM3)
Multiple runs for statistics
Clear visualization
Honest reporting (including failures)