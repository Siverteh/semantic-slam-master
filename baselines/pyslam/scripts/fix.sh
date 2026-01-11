#!/bin/bash
# Comprehensive pySLAM compatibility patches
# 1. Fix rerun API incompatibility
# 2. Disable GTSAM in config
# 3. Make ALL GTSAM imports optional

set -e

PYSLAM_DIR="/workspace/baselines/pyslam/pyslam"
PYSLAM_SLAM_DIR="$PYSLAM_DIR/pyslam/slam"

echo "================================================"
echo "Applying pySLAM Compatibility Patches"
echo "================================================"

# PATCH 1: Fix rerun API
echo ""
echo "[1/4] Patching rerun_interface.py..."
RERUN_INTERFACE="$PYSLAM_DIR/pyslam/viz/rerun_interface.py"
if [ -f "$RERUN_INTERFACE" ]; then
    [ ! -f "$RERUN_INTERFACE.bak" ] && cp "$RERUN_INTERFACE" "$RERUN_INTERFACE.bak"
    sed -i 's/quaternion=rr\.Quaternion(xyzw=\[0\.0, 0\.0, 0\.0, 1\.0\])/quaternion=[0.0, 0.0, 0.0, 1.0]/g' "$RERUN_INTERFACE"
    echo "  ✓ Rerun patch applied"
else
    echo "  ⚠ rerun_interface.py not found"
fi

# PATCH 2: Disable GTSAM in config
echo ""
echo "[2/4] Disabling GTSAM in config..."
CONFIG_PARAMS="$PYSLAM_DIR/pyslam/config_parameters.py"
if [ -f "$CONFIG_PARAMS" ]; then
    [ ! -f "$CONFIG_PARAMS.bak" ] && cp "$CONFIG_PARAMS" "$CONFIG_PARAMS.bak"
    sed -i 's/kOptimizationFrontEndUseGtsam = True/kOptimizationFrontEndUseGtsam = False/g' "$CONFIG_PARAMS"
    sed -i 's/kOptimizationBundleAdjustUseGtsam = True/kOptimizationBundleAdjustUseGtsam = False/g' "$CONFIG_PARAMS"
    sed -i 's/kOptimizationLoopClosingUseGtsam = True/kOptimizationLoopClosingUseGtsam = False/g' "$CONFIG_PARAMS"
    echo "  ✓ GTSAM disabled in config"
else
    echo "  ⚠ config_parameters.py not found"
fi

# PATCH 3: Find and patch ALL files importing optimizer_gtsam
echo ""
echo "[3/4] Finding all optimizer_gtsam imports..."

# Search more broadly - in entire pyslam directory
FILES_TO_PATCH=$(grep -r "import optimizer_gtsam" "$PYSLAM_DIR/pyslam" --include="*.py" -l 2>/dev/null || true)

if [ -z "$FILES_TO_PATCH" ]; then
    echo "  ℹ No files found importing optimizer_gtsam"
else
    echo "  Found files to patch:"
    echo "$FILES_TO_PATCH" | while read -r file; do
        echo "    - $(basename $file)"
    done

    echo ""
    echo "  Patching files..."

    echo "$FILES_TO_PATCH" | while read -r file; do
        [ ! -f "${file}.bak" ] && cp "$file" "${file}.bak"

        # Check if already has try-except
        if grep -q "try:" "$file" && grep -q "import optimizer_gtsam" "$file" && grep -q "except ImportError" "$file"; then
            echo "    ✓ $(basename $file) - already patched"
        else
            # Patch different import patterns

            # Pattern 1: from . import optimizer_gtsam
            if grep -q "^from \. import optimizer_gtsam" "$file"; then
                sed -i '/^from \. import optimizer_gtsam$/c\
try:\
    from . import optimizer_gtsam\
except ImportError:\
    optimizer_gtsam = None' "$file"
                echo "    ✓ $(basename $file) - patched (from . import)"
            fi

            # Pattern 2: from pyslam.slam import optimizer_gtsam
            if grep -q "from pyslam.slam import optimizer_gtsam" "$file"; then
                sed -i '/from pyslam.slam import optimizer_gtsam/c\
try:\
    from pyslam.slam import optimizer_gtsam\
except ImportError:\
    optimizer_gtsam = None' "$file"
                echo "    ✓ $(basename $file) - patched (from pyslam.slam import)"
            fi

            # Pattern 3: import optimizer_gtsam (direct)
            if grep -q "^import optimizer_gtsam" "$file"; then
                sed -i '/^import optimizer_gtsam/c\
try:\
    import optimizer_gtsam\
except ImportError:\
    optimizer_gtsam = None' "$file"
                echo "    ✓ $(basename $file) - patched (direct import)"
            fi
        fi
    done
fi

# PATCH 4: Rename optimizer_gtsam.py to prevent accidental imports
echo ""
echo "[4/4] Disabling optimizer_gtsam.py..."
OPTIMIZER_GTSAM="$PYSLAM_SLAM_DIR/optimizer_gtsam.py"
if [ -f "$OPTIMIZER_GTSAM" ]; then
    if [ ! -f "${OPTIMIZER_GTSAM}.disabled" ]; then
        mv "$OPTIMIZER_GTSAM" "${OPTIMIZER_GTSAM}.disabled"
        echo "  ✓ Renamed optimizer_gtsam.py to optimizer_gtsam.py.disabled"
    else
        echo "  ✓ Already disabled"
    fi
else
    echo "  ℹ optimizer_gtsam.py not found"
fi

echo ""
echo "================================================"
echo "✓ All Patches Applied"
echo "================================================"
echo ""
echo "Changes:"
echo "  1. Fixed rerun API (Quaternion compatibility)"
echo "  2. Disabled GTSAM in config (using g2o)"
echo "  3. Made all optimizer_gtsam imports optional"
echo "  4. Disabled optimizer_gtsam.py module"
echo ""
echo "You can now run:"
echo "  ./baselines/pyslam/scripts/run_baseline.sh"
echo ""
echo "To restore GTSAM (requires fixing libmetis-gtsam.so):"
echo "  1. Restore config: cp $CONFIG_PARAMS.bak $CONFIG_PARAMS"
echo "  2. Restore slam files: find $PYSLAM_SLAM_DIR -name '*.bak' -exec bash -c 'cp \"\$0\" \"\${0%.bak}\"' {} \;"
echo "  3. Restore optimizer: mv ${OPTIMIZER_GTSAM}.disabled $OPTIMIZER_GTSAM"
echo ""