#!/bin/bash
# Run pySLAM on all TUM RGB-D sequences
# Modified to be consistent with ORB-SLAM3 run script

set -e

# Resolve script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
PYSLAM_DIR="$BASELINE_DIR/pyslam"

# Paths
DATA_PATH="/workspace/data/tum_rgbd"
OUTPUT_PATH="/workspace/experiments/baselines/pyslam"
EXECUTABLE="python3 $PYSLAM_DIR/main_slam.py"

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
echo "Running pySLAM baseline on TUM RGB-D"
echo "================================================"
echo "pySLAM:      $PYSLAM_DIR"
echo "Dataset:     $DATA_PATH"
echo "Output:      $OUTPUT_PATH"
echo ""

# Check for xvfb (needed for headless operation)
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

    # Create temporary config file
    CONFIG_FILE="/tmp/pyslam_config_${SEQ}.yaml"
    cp "$PYSLAM_DIR/config.yaml" "$CONFIG_FILE"
    
    # Update config for this sequence
    python3 <<EOF
import yaml
import os

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

if config is None:
    config = {}

# Set dataset type
if 'DATASET' not in config or config['DATASET'] is None:
    config['DATASET'] = {}
config['DATASET']['type'] = 'TUM_DATASET'

# Update TUM_DATASET section
if 'TUM_DATASET' not in config or config['TUM_DATASET'] is None:
    config['TUM_DATASET'] = {}

config['TUM_DATASET']['type'] = 'tum'
config['TUM_DATASET']['sensor_type'] = 'rgbd'
config['TUM_DATASET']['base_path'] = '$DATA_PATH'
config['TUM_DATASET']['name'] = '$SEQ'
config['TUM_DATASET']['settings'] = '$SETTINGS'
config['TUM_DATASET']['associations'] = 'associations.txt'
config['TUM_DATASET']['groundtruth_file'] = 'auto'

# Save trajectory settings
if 'SAVE_TRAJECTORY' not in config or config['SAVE_TRAJECTORY'] is None:
    config['SAVE_TRAJECTORY'] = {}

config['SAVE_TRAJECTORY']['save_trajectory'] = True
config['SAVE_TRAJECTORY']['format_type'] = 'tum'
config['SAVE_TRAJECTORY']['output_folder'] = '$OUTPUT_PATH'
config['SAVE_TRAJECTORY']['basename'] = '${SEQ}'

# Headless mode: ensure no GUIs are requested that might block
if 'GLOBAL_PARAMETERS' not in config or config['GLOBAL_PARAMETERS'] is None:
    config['GLOBAL_PARAMETERS'] = {}
config['GLOBAL_PARAMETERS']['show_viewer'] = False

with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f)
EOF

    # Output files
    TRAJ_FILE="$OUTPUT_PATH/trajectories/${SEQ}_trajectory.txt"
    LOG_FILE="$OUTPUT_PATH/logs/${SEQ}.log"

    # Run pySLAM
    echo "Running pySLAM..."
    cd "$PYSLAM_DIR"
    # Use xvfb-run for headless operation
    # Added --headless flag to ensure main_slam.py exits after dataset ends
    # Added --no_output_date to prevent timestamped folders
    xvfb-run -a -s "-screen 0 640x480x24" \
        $EXECUTABLE --config_path "$CONFIG_FILE" --headless --no_output_date \
        2>&1 | tee "$LOG_FILE"

    # Move and rename output trajectory
    # pySLAM saves it as $OUTPUT_PATH/${SEQ}_final.txt because we set output_folder=$OUTPUT_PATH
    TRAJ_GEN="$OUTPUT_PATH/${SEQ}_final.txt"
    TRAJ_ONLINE="$OUTPUT_PATH/${SEQ}_online.txt"
    
    if [ -f "$TRAJ_GEN" ]; then
        mv "$TRAJ_GEN" "$TRAJ_FILE"
        echo "✓ Final trajectory saved: $TRAJ_FILE"
        [ -f "$TRAJ_ONLINE" ] && rm "$TRAJ_ONLINE"
    elif [ -f "$TRAJ_ONLINE" ]; then
        mv "$TRAJ_ONLINE" "$TRAJ_FILE"
        echo "✓ Online trajectory saved (final not found): $TRAJ_FILE"
    else
        # Search fallback
        TRAJ_SAVED=$(ls "$OUTPUT_PATH/${SEQ}"* 2>/dev/null | grep -v "_trajectory.txt" | head -n 1)
        if [ -f "$TRAJ_SAVED" ]; then
            mv "$TRAJ_SAVED" "$TRAJ_FILE"
            echo "✓ Found and moved trajectory: $TRAJ_FILE"
        else
            echo "✗ Failed to generate trajectory for $SEQ"
        fi
    fi

    # Move plots
    # pySLAM saves plots in $OUTPUT_PATH/plot/ (because output_folder=$OUTPUT_PATH)
    if [ -d "$OUTPUT_PATH/plot" ]; then
        # Create sequence specific plot folder
        mkdir -p "$OUTPUT_PATH/plots/${SEQ}"
        mv "$OUTPUT_PATH/plot/"* "$OUTPUT_PATH/plots/${SEQ}/" 2>/dev/null || true
        rmdir "$OUTPUT_PATH/plot"
        echo "✓ Plots moved to: $OUTPUT_PATH/plots/${SEQ}/"
    fi
    
    # Also check if any plots were saved directly in the output folder
    mv "$OUTPUT_PATH/"*.png "$OUTPUT_PATH/plots/${SEQ}/" 2>/dev/null || true

    # Clean up temporary config
    rm -f "$CONFIG_FILE"
done

echo ""
echo "================================================"
echo "✓ Baseline experiments complete!"
echo "================================================"
echo "Trajectories saved in: $OUTPUT_PATH/trajectories/"
echo "Plots saved in:        $OUTPUT_PATH/plots/"
echo ""
echo "Next: Run evaluation script"
echo "  python /workspace/scripts/evaluate_baseline.py"
echo "================================================"
