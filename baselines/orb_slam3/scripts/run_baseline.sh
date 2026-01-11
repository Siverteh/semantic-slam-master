#!/bin/bash
# Run ORB-SLAM3 on all TUM RGB-D sequences

set -e

# Resolve script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
ORB_SLAM3_DIR="$BASELINE_DIR/ORB_SLAM3"

# Paths
DATA_PATH="/workspace/data/tum_rgbd"
OUTPUT_PATH="/workspace/experiments/baselines/orb_slam3"
VOCAB_PATH="$ORB_SLAM3_DIR/Vocabulary/ORBvoc.txt"
EXECUTABLE="$ORB_SLAM3_DIR/Examples/RGB-D/rgbd_tum"

# Create output directories
mkdir -p "$OUTPUT_PATH/trajectories"
mkdir -p "$OUTPUT_PATH/logs"

# TUM RGB-D sequences
SEQUENCES=(
    "rgbd_dataset_freiburg1_desk"
    "rgbd_dataset_freiburg1_plant"
    "rgbd_dataset_freiburg1_room"
    "rgbd_dataset_freiburg3_long_office_household"
    "rgbd_dataset_freiburg3_walking_static"
    "rgbd_dataset_freiburg3_walking_xyz"
)

# TUM camera calibration files
FR1_CALIB="$ORB_SLAM3_DIR/Examples/RGB-D/TUM1.yaml"
FR3_CALIB="$ORB_SLAM3_DIR/Examples/RGB-D/TUM3.yaml"

echo "================================================"
echo "Running ORB-SLAM3 baseline on TUM RGB-D"
echo "================================================"
echo "ORB-SLAM3:   $ORB_SLAM3_DIR"
echo "Dataset:     $DATA_PATH"
echo "Output:      $OUTPUT_PATH"
echo ""

# Check for xvfb (needed for headless operation)
if ! command -v xvfb-run &> /dev/null; then
    echo "Installing xvfb for headless operation..."
    apt-get update && apt-get install -y xvfb
fi

for SEQ in "${SEQUENCES[@]}"; do
    echo ""
    echo "Processing: $SEQ"
    echo "----------------------------------------"

    # Determine calibration file
    if [[ $SEQ == *"freiburg1"* ]]; then
        CALIB_FILE=$FR1_CALIB
    else
        CALIB_FILE=$FR3_CALIB
    fi

    # Association file (should already exist in dataset)
    ASSOC_FILE="$DATA_PATH/$SEQ/associations.txt"

    # Check if association file exists
    if [ ! -f "$ASSOC_FILE" ]; then
        echo "⚠ Association file not found, generating..."
        if [ -f "$DATA_PATH/$SEQ/rgb.txt" ] && [ -f "$DATA_PATH/$SEQ/depth.txt" ]; then
            python3 /workspace/scripts/associate.py \
                "$DATA_PATH/$SEQ/rgb.txt" \
                "$DATA_PATH/$SEQ/depth.txt" \
                --output "$ASSOC_FILE"
            echo "✓ Generated association file"
        else
            echo "✗ Cannot find rgb.txt or depth.txt in $DATA_PATH/$SEQ"
            continue
        fi
    else
        echo "✓ Using existing association file"
    fi

    # Output files
    TRAJ_FILE="$OUTPUT_PATH/trajectories/${SEQ}_trajectory.txt"
    LOG_FILE="$OUTPUT_PATH/logs/${SEQ}.log"

    # Run ORB-SLAM3
    # Note: ORB-SLAM3 saves trajectory as KeyFrameTrajectory.txt in current directory
    cd "$OUTPUT_PATH/trajectories"

    # Use xvfb for headless operation (virtual display)
    # This prevents "Failed to open X display" error
    echo "Running ORB-SLAM3..."
    xvfb-run -a -s "-screen 0 640x480x24" \
        $EXECUTABLE \
        "$VOCAB_PATH" \
        "$CALIB_FILE" \
        "$DATA_PATH/$SEQ" \
        "$ASSOC_FILE" \
        2>&1 | tee "$LOG_FILE"

    # Rename output trajectory
    if [ -f "KeyFrameTrajectory.txt" ]; then
        mv KeyFrameTrajectory.txt "$TRAJ_FILE"
        echo "✓ Trajectory saved: $TRAJ_FILE"
    else
        echo "✗ Failed to generate trajectory for $SEQ"
    fi

    # Clean up intermediate files
    rm -f FrameTrajectory.txt CameraTrajectory.txt
done

echo ""
echo "================================================"
echo "✓ Baseline experiments complete!"
echo "================================================"
echo "Trajectories saved in: $OUTPUT_PATH/trajectories/"
echo ""
echo "Next: Run evaluation script"
echo "  python /workspace/scripts/evaluate_baseline.py"