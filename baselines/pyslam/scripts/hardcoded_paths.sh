#!/bin/bash
# Remove ALL hardcoded paths from config.yaml

set -e

PYSLAM_DIR="/workspace/baselines/pyslam/pyslam"
CONFIG_YAML="$PYSLAM_DIR/config.yaml"

echo "================================================"
echo "Fixing ALL Hardcoded Paths in config.yaml"
echo "================================================"

if [ ! -f "$CONFIG_YAML" ]; then
    echo "✗ Error: config.yaml not found"
    exit 1
fi

# Backup
if [ ! -f "$CONFIG_YAML.original" ]; then
    cp "$CONFIG_YAML" "$CONFIG_YAML.original"
    echo "✓ Original backup created"
fi

echo ""
echo "Removing hardcoded paths..."

# Replace ALL occurrences of Luigi's home directory
sed -i 's|/home/luigi/Work/datasets/rgbd_datasets/tum|/workspace/data/tum_rgbd|g' "$CONFIG_YAML"
sed -i 's|/home/luigi/.*tum/|/workspace/data/tum_rgbd/|g' "$CONFIG_YAML"

# Replace any other common hardcoded paths
sed -i 's|/home/[^/]*/.*datasets/|/workspace/data/|g' "$CONFIG_YAML"

echo "✓ Replaced all Luigi paths with /workspace/data/tum_rgbd"

# Show what was changed
echo ""
echo "Current TUM_DATASET section:"
awk '/^TUM_DATASET:/,/^[A-Z_]+DATASET:/' "$CONFIG_YAML" | head -20

echo ""
echo "================================================"
echo "✓ All Paths Fixed"
echo "================================================"