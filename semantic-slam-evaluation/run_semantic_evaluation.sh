#!/bin/bash
# Run pySLAM with Semantic Feature Extractor on TUM RGB-D sequences
# Compares against baseline ORB-SLAM3 and vanilla pySLAM results

set -e

# Resolve paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYSLAM_DIR="/workspace/baselines/pyslam/pyslam"
SEMANTIC_SLAM_DIR="/workspace/semantic-slam"
DATA_PATH="/workspace/data/tum_rgbd"
OUTPUT_PATH="$SCRIPT_DIR/results"
CHECKPOINT_PATH="$SEMANTIC_SLAM_DIR/checkpoints/best_model.pth"

# Create output directories
mkdir -p "$OUTPUT_PATH/trajectories"
mkdir -p "$OUTPUT_PATH/logs"
mkdir -p "$OUTPUT_PATH/plots"

# TUM RGB-D sequences
SEQUENCES=(
    "rgbd_dataset_freiburg1_desk"
    "rgbd_dataset_freiburg1_plant"
    "rgbd_dataset_freiburg1_room"
    "rgbd_dataset_freiburg3_long_office_household"
    "rgbd_dataset_freiburg3_walking_static"
    "rgbd_dataset_freiburg3_walking_xyz"
)

echo "================================================"
echo "Running pySLAM with Semantic Feature Extractor"
echo "================================================"
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Dataset:     $DATA_PATH"
echo "Output:      $OUTPUT_PATH"
echo ""

# Check checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "✗ Checkpoint not found: $CHECKPOINT_PATH"
    echo "  Run training first: cd /workspace/semantic-slam && python train.py"
    exit 1
fi

# Check for xvfb (headless operation)
if ! command -v xvfb-run &> /dev/null; then
    echo "Installing xvfb for headless operation..."
    apt-get update -qq && apt-get install -y xvfb
fi

for SEQ in "${SEQUENCES[@]}"; do
    echo ""
    echo "Processing: $SEQ"
    echo "----------------------------------------"

    # Association file
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

    # Determine calibration settings
    if [[ $SEQ == *"freiburg1"* ]]; then
        SETTINGS="settings/TUM1.yaml"
    elif [[ $SEQ == *"freiburg2"* ]]; then
        SETTINGS="settings/TUM2.yaml"
    else
        SETTINGS="settings/TUM3.yaml"
    fi

    # Output files
    TRAJ_FILE="$OUTPUT_PATH/trajectories/${SEQ}_trajectory.txt"
    LOG_FILE="$OUTPUT_PATH/logs/${SEQ}.log"

    # Run pySLAM with semantic features
    echo "Running pySLAM with semantic features..."
    cd "$PYSLAM_DIR"

    # Use the Python runner script instead of main_slam.py
    xvfb-run -a -s "-screen 0 640x480x24" \
        python3 "$SCRIPT_DIR/run_pyslam_semantic.py" \
        --checkpoint "$CHECKPOINT_PATH" \
        --sequence "$DATA_PATH/$SEQ" \
        --settings "$SETTINGS" \
        --output "$TRAJ_FILE" \
        2>&1 | tee "$LOG_FILE"

    if [ -f "$TRAJ_FILE" ]; then
        echo "✓ Trajectory saved: $TRAJ_FILE"
    else
        echo "✗ Failed to generate trajectory for $SEQ"
    fi
done

echo ""
echo "================================================"
echo "✓ Semantic evaluation complete!"
echo "================================================"
echo "Trajectories saved in: $OUTPUT_PATH/trajectories/"
echo ""
echo "Next: Run evaluation to compute ATE"
echo "  python $SCRIPT_DIR/evaluate_semantic.py"
echo "================================================"