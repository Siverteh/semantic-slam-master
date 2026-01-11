#!/bin/bash
# Configure pySLAM for headless mode (no GUI) and TUM dataset

set -e

PYSLAM_DIR="/workspace/baselines/pyslam/pyslam"
CONFIG_PARAMS="$PYSLAM_DIR/pyslam/config_parameters.py"
CONFIG_YAML="$PYSLAM_DIR/config.yaml"

echo "================================================"
echo "Configuring pySLAM for Headless Mode + TUM"
echo "================================================"

# PART 1: Modify config_parameters.py
if [ ! -f "$CONFIG_PARAMS" ]; then
    echo "✗ Error: config_parameters.py not found"
    exit 1
fi

if [ ! -f "$CONFIG_PARAMS.bak_headless" ]; then
    cp "$CONFIG_PARAMS" "$CONFIG_PARAMS.bak_headless"
    echo "✓ Backup created: config_parameters.py.bak_headless"
fi

echo ""
echo "[1/2] Disabling GUI in config_parameters.py..."

# Disable all visualization/GUI components
sed -i 's/kUseViewer3D = True/kUseViewer3D = False/g' "$CONFIG_PARAMS"
sed -i 's/kUseRerun = True/kUseRerun = False/g' "$CONFIG_PARAMS"
sed -i 's/kUseMplot2d = True/kUseMplot2d = False/g' "$CONFIG_PARAMS"
sed -i "s/kMplot2dBackend = 'TkAgg'/kMplot2dBackend = 'Agg'/g" "$CONFIG_PARAMS"

echo "  ✓ Disabled 3D viewer (Pangolin)"
echo "  ✓ Disabled Rerun"
echo "  ✓ Disabled interactive plots"
echo "  ✓ Set matplotlib backend to Agg"

# PART 2: Modify config.yaml
if [ ! -f "$CONFIG_YAML" ]; then
    echo "✗ Error: config.yaml not found"
    exit 1
fi

echo ""
echo "[2/2] Configuring config.yaml for TUM dataset..."

if [ ! -f "$CONFIG_YAML.bak" ]; then
    cp "$CONFIG_YAML" "$CONFIG_YAML.bak"
    echo "  ✓ Backup created: config.yaml.bak"
fi

# Update dataset type to TUM
sed -i 's/type: KITTI_DATASET/type: TUM_DATASET/g' "$CONFIG_YAML"
sed -i 's/type: VIDEO_DATASET/type: TUM_DATASET/g' "$CONFIG_YAML"

# Set TUM dataset path (will be overridden by wrapper for specific sequences)
sed -i "s|path: .*data/videos/kitti06|path: /workspace/data/tum_rgbd|g" "$CONFIG_YAML"

# Update sensor type for TUM (RGBD)
# Find the TUM_DATASET section and update sensor_type
awk '
/^TUM_DATASET:/ { in_tum=1 }
in_tum && /^[A-Z_]+DATASET:/ && !/^TUM_DATASET:/ { in_tum=0 }
in_tum && /sensor_type:/ {
    sub(/sensor_type: .*/, "sensor_type: rgbd")
    print
    next
}
{ print }
' "$CONFIG_YAML" > "$CONFIG_YAML.tmp" && mv "$CONFIG_YAML.tmp" "$CONFIG_YAML"

echo "  ✓ Changed dataset type to TUM_DATASET"
echo "  ✓ Set sensor_type to RGBD"

# Verify changes
echo ""
echo "Verifying configuration..."
if grep -q "kUseViewer3D = False" "$CONFIG_PARAMS" && \
   grep -q "type: TUM_DATASET" "$CONFIG_YAML"; then
    echo "✓ Configuration verified"
else
    echo "⚠ Warning: Could not verify all changes"
fi

echo ""
echo "================================================"
echo "✓ Configuration Complete"
echo "================================================"
echo ""
echo "Changes applied:"
echo "  - Disabled all GUI components"
echo "  - Changed dataset to TUM (RGBD)"
echo "  - Set matplotlib to non-interactive backend"
echo ""