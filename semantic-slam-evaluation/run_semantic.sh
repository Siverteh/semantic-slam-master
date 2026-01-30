#!/bin/bash
# Run pySLAM with semantic feature extractor on TUM RGB-D sequences

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYSLAM_DIR="$SCRIPT_DIR/pyslam"
SEMANTIC_SLAM_DIR="/workspace/semantic-slam"
DATA_PATH="/workspace/data/tum_rgbd"
OUTPUT_PATH="$SCRIPT_DIR/results"
CHECKPOINT_PATH="$SEMANTIC_SLAM_DIR/checkpoints/best_model.pth"
GTSAM_LIB_DIR="$PYSLAM_DIR/thirdparty/gtsam_local/install/lib"
export SEMANTIC_SLAM_EVAL_DIR="$SCRIPT_DIR"
export SEMANTIC_SLAM_DIR="$SEMANTIC_SLAM_DIR"
PYTHON_CANDIDATES=(
    "/workspace/semantic-slam-evaluation/.venv/bin/python"
    "/workspace/.venv/bin/python"
    "python3"
)

PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [ -x "$candidate" ] || [ "$candidate" = "python3" ]; then
        if "$candidate" - <<'PY' >/dev/null 2>&1
try:
    import yaml  # noqa: F401
    import torch  # noqa: F401
    import semantic_feature_extractor  # noqa: F401
    print("ok")
except Exception:
    raise SystemExit(1)
PY
        then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "✗ No Python interpreter with PyYAML + torch + semantic_feature_extractor found." >&2
    exit 1
fi

# Ensure C++ libs are discoverable by the Python package
CPP_BUILT_LIB="$PYSLAM_DIR/cpp/lib"
CPP_PKG_LIB="$PYSLAM_DIR/pyslam/slam/cpp/lib"
mkdir -p "$(dirname "$CPP_PKG_LIB")"
ln -sfn "$CPP_BUILT_LIB" "$CPP_PKG_LIB"

# Ensure GTSAM shared libraries are discoverable (libmetis-gtsam.so, libgtsam.so)
if [ -d "$GTSAM_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$GTSAM_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "$OUTPUT_PATH/trajectories" "$OUTPUT_PATH/logs" "$OUTPUT_PATH/plots"

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

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "✗ Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if ! command -v xvfb-run &> /dev/null; then
    echo "Installing xvfb for headless operation..."
    apt-get update -qq && apt-get install -y xvfb
fi

for SEQ in "${SEQUENCES[@]}"; do
    echo ""
    echo "Processing: $SEQ"
    echo "----------------------------------------"

    ASSOC_FILE="$DATA_PATH/$SEQ/associations.txt"
    if [ ! -f "$ASSOC_FILE" ]; then
        echo "⚠ Association file not found, generating..."
        if [ -f "$DATA_PATH/$SEQ/rgb.txt" ] && [ -f "$DATA_PATH/$SEQ/depth.txt" ]; then
            "$PYTHON_BIN" /workspace/scripts/associate.py \
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

    if [[ $SEQ == *"freiburg1"* ]]; then
        SETTINGS="settings/TUM1.yaml"
    elif [[ $SEQ == *"freiburg2"* ]]; then
        SETTINGS="settings/TUM2.yaml"
    else
        SETTINGS="settings/TUM3.yaml"
    fi

    CONFIG_FILE="/tmp/pyslam_config_${SEQ}.yaml"
    cp "$PYSLAM_DIR/config.yaml" "$CONFIG_FILE"

    "$PYTHON_BIN" - <<EOF
import yaml
with open("$CONFIG_FILE", "r") as f:
    config = yaml.safe_load(f) or {}
def ensure_dict(cfg, key):
    if not isinstance(cfg.get(key), dict):
        cfg[key] = {}
    return cfg[key]

ensure_dict(config, "DATASET")["type"] = "TUM_DATASET"
ensure_dict(config, "TUM_DATASET")
config["TUM_DATASET"]["type"] = "tum"
config["TUM_DATASET"]["sensor_type"] = "rgbd"
config["TUM_DATASET"]["base_path"] = "$DATA_PATH"
config["TUM_DATASET"]["name"] = "$SEQ"
config["TUM_DATASET"]["settings"] = "$SETTINGS"
config["TUM_DATASET"]["associations"] = "associations.txt"
config["TUM_DATASET"]["groundtruth_file"] = "auto"
ensure_dict(config, "SAVE_TRAJECTORY")
config["SAVE_TRAJECTORY"]["save_trajectory"] = False
config["SAVE_TRAJECTORY"]["format_type"] = "tum"
config["SAVE_TRAJECTORY"]["output_folder"] = "$OUTPUT_PATH"
config["SAVE_TRAJECTORY"]["basename"] = "$SEQ"
ensure_dict(config, "GLOBAL_PARAMETERS")
config["GLOBAL_PARAMETERS"]["show_viewer"] = False
with open("$CONFIG_FILE", "w") as f:
    yaml.safe_dump(config, f)
EOF

    TRAJ_FILE="$OUTPUT_PATH/trajectories/${SEQ}_trajectory.txt"
    LOG_FILE="$OUTPUT_PATH/logs/${SEQ}.log"

    echo "Running pySLAM with semantic features..."
    cd "$PYSLAM_DIR"

    xvfb-run -a -s "-screen 0 640x480x24" \
        "$PYTHON_BIN" - <<EOF 2>&1 | tee "$LOG_FILE"
import sys
from pathlib import Path
sys.path.insert(0, "$PYSLAM_DIR")
sys.path.insert(0, "$SCRIPT_DIR")
sys.path.insert(0, "$SEMANTIC_SLAM_DIR")

from pyslam.config import Config
from pyslam.slam.slam import Slam
from pyslam.slam import PinholeCamera
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType
from pyslam.io.trajectory_writer import TrajectoryWriter
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs

config = Config(config_path="$CONFIG_FILE")
dataset = dataset_factory(config)
camera = PinholeCamera(config)

feature_tracker_config = FeatureTrackerConfigs.SEMANTIC.copy()
feature_tracker_config["num_features"] = 500
feature_tracker_config["semantic_checkpoint"] = "$CHECKPOINT_PATH"
feature_tracker_config["semantic_device"] = "cuda"
feature_tracker_config["semantic_num_features"] = 500

loop_detection_config = LoopDetectorConfigs.OBINDEX2

slam = Slam(
    camera=camera,
    feature_tracker_config=feature_tracker_config,
    loop_detector_config=loop_detection_config,
    semantic_mapping_config=None,
    sensor_type=dataset.sensorType(),
    environment_type=dataset.environmentType(),
    config=config,
    headless=True,
)

img_id = 0
while dataset.is_ok:
    img = dataset.getImageColor(img_id)
    if img is None:
        break
    depth = None
    if dataset.sensorType() == SensorType.RGBD:
        depth = dataset.getDepth(img_id)
    timestamp = dataset.getTimestamp()
    slam.track(img, None, depth, img_id, timestamp)
    img_id += 1

est_poses, timestamps, _ = slam.get_final_trajectory()
Path("$OUTPUT_PATH/trajectories").mkdir(parents=True, exist_ok=True)
writer = TrajectoryWriter(format_type="tum", filename="$TRAJ_FILE")
writer.write_full_trajectory(est_poses, timestamps)
writer.close_file()
slam.quit()
EOF

    if [ -f "$TRAJ_FILE" ]; then
        echo "✓ Trajectory saved: $TRAJ_FILE"
    else
        echo "✗ Failed to generate trajectory for $SEQ"
    fi

    rm -f "$CONFIG_FILE"
done

echo ""
echo "================================================"
echo "✓ Semantic runs complete!"
echo "================================================"
echo "Trajectories saved in: $OUTPUT_PATH/trajectories/"
