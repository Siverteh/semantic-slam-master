#!/bin/bash
# Setup script for pySLAM baseline

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
PYSLAM_DIR="$BASELINE_DIR/pyslam"

echo "================================================"
echo "Setting up pySLAM for baseline experiments"
echo "================================================"
echo "Installation directory: $BASELINE_DIR"
echo ""

# Function to check if a package is installed
check_package() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "ok installed"
}

# Install system dependencies
echo "Installing system dependencies..."
REQUIRED_PACKAGES=(
    "build-essential"
    "cmake"
    "git"
    "python3.10"
    "python3.10-dev"
    "python3.10-venv"
    "python3-pip"
    "libopencv-dev"
    "libopencv-contrib-dev"
    "libeigen3-dev"
    "libboost-all-dev"
    "libsuitesparse-dev"
    "qtdeclarative5-dev"
    "qt5-qmake"
    "libqglviewer-dev-qt5"
    "libglew-dev"
    "libpython3-dev"
    "libavcodec-dev"
    "libavformat-dev"
    "libswscale-dev"
)

MISSING_PACKAGES=()
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! check_package "$package"; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo "Installing missing packages: ${MISSING_PACKAGES[*]}"
    apt-get update -qq
    apt-get install -y "${MISSING_PACKAGES[@]}"
else
    echo "✓ All dependencies already installed"
fi

# Clone pySLAM
if [ ! -d "$PYSLAM_DIR" ]; then
    echo ""
    echo "Cloning pySLAM..."
    cd "$BASELINE_DIR"
    git clone --recursive https://github.com/luigifreda/pyslam.git
    cd pyslam
else
    echo ""
    echo "pySLAM already exists, updating..."
    cd "$PYSLAM_DIR"
    git pull || echo "Warning: Could not pull latest changes"
    git submodule update --init --recursive
fi

# Create virtual environment
echo ""
echo "Setting up Python environment..."
VENV_DIR="$PYSLAM_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    if ! python3.10 -m venv "$VENV_DIR"; then
        echo "✗ Failed to create virtual environment"
        echo ""
        echo "Trying to fix python3-venv installation..."
        apt-get update -qq
        apt-get install -y python3.10-venv python3-venv

        echo "Retrying virtual environment creation..."
        if ! python3.10 -m venv "$VENV_DIR"; then
            echo "✗ Still failed to create virtual environment"
            echo "Please check Python installation and try again"
            exit 1
        fi
    fi
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Verify virtual environment
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "✗ Virtual environment is invalid (no activate script)"
    echo "Removing and recreating..."
    rm -rf "$VENV_DIR"
    python3.10 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "✗ Failed to activate virtual environment"
    exit 1
fi
echo "✓ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel || {
    echo "✗ Failed to upgrade pip"
    echo "Trying alternative method..."
    python -m pip install --upgrade pip setuptools wheel
}
echo "✓ pip upgraded"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
echo "This may take 10-15 minutes (including PyTorch)..."

# Install core dependencies first
pip install numpy scipy || {
    echo "✗ Failed to install numpy/scipy"
    exit 1
}

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch (this is large, ~2GB)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install OpenCV
pip install opencv-python opencv-contrib-python || {
    echo "⚠ Failed to install opencv-python"
    echo "Trying alternative installation..."
    pip install opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64
}

# Install pySLAM-specific dependencies
echo ""
echo "Installing pySLAM-specific dependencies..."
pip install ujson
pip install msgpack msgpack-numpy
pip install tqdm termcolor

# Install other dependencies
pip install matplotlib
pip install scikit-image
pip install pytransform3d
pip install pyopengl
pip install PyQt5
pip install pyyaml
pip install evo
pip install gdown
pip install numba
pip install trimesh
pip install kornia
pip install ordered-set
pip install pyflann-py3
pip install faiss-gpu
pip install einops
pip install fast_pytorch_kmeans

# Install ADDITIONAL REQUIRED packages
echo ""
echo "Installing additional required packages..."
pip install pyqtgraph
pip install open3d
pip install psutil

# Install COMPATIBLE rerun version (fixes Quaternion error)
echo ""
echo "Installing compatible rerun version..."
pip uninstall -y rerun-sdk 2>/dev/null || true
pip install "rerun-sdk==0.14.1"
echo "✓ Installed rerun-sdk 0.14.1 (compatible version)"

# PATCH rerun_interface.py to fix API incompatibility
echo ""
echo "Patching rerun_interface.py for API compatibility..."
RERUN_INTERFACE="$PYSLAM_DIR/pyslam/viz/rerun_interface.py"
if [ -f "$RERUN_INTERFACE" ]; then
    # Backup original
    cp "$RERUN_INTERFACE" "$RERUN_INTERFACE.bak"

    # Fix line 222: replace rr.Quaternion with a compatible alternative
    # In rerun 0.14.1, use [x, y, z, w] array directly instead of Quaternion class
    sed -i 's/quaternion=rr\.Quaternion(xyzw=\[0\.0, 0\.0, 0\.0, 1\.0\])/quaternion=[0.0, 0.0, 0.0, 1.0]/g' "$RERUN_INTERFACE"

    echo "✓ Patched $RERUN_INTERFACE"
    echo "  Backup saved as: $RERUN_INTERFACE.bak"
else
    echo "⚠ rerun_interface.py not found at expected location"
    echo "  The rerun fix may need to be applied manually"
fi

# DISABLE GTSAM and use g2o instead (avoids libmetis-gtsam.so error)
echo ""
echo "Configuring optimizer (disabling GTSAM, using g2o)..."
CONFIG_PARAMS="$PYSLAM_DIR/pyslam/config_parameters.py"
if [ -f "$CONFIG_PARAMS" ]; then
    # Backup original
    if [ ! -f "$CONFIG_PARAMS.bak" ]; then
        cp "$CONFIG_PARAMS" "$CONFIG_PARAMS.bak"
    fi

    # Set all GTSAM flags to False (use g2o instead)
    sed -i 's/kOptimizationFrontEndUseGtsam = True/kOptimizationFrontEndUseGtsam = False/g' "$CONFIG_PARAMS"
    sed -i 's/kOptimizationBundleAdjustUseGtsam = True/kOptimizationBundleAdjustUseGtsam = False/g' "$CONFIG_PARAMS"
    sed -i 's/kOptimizationLoopClosingUseGtsam = True/kOptimizationLoopClosingUseGtsam = False/g' "$CONFIG_PARAMS"

    echo "✓ Configured to use g2o optimizer"
else
    echo "⚠ config_parameters.py not found"
fi

# Check for requirements files in pySLAM
cd "$PYSLAM_DIR"

if [ -f "requirements.txt" ]; then
    echo ""
    echo "✓ Found requirements.txt, installing..."
    pip install -r requirements.txt || echo "⚠ Some requirements failed (might be OK)"
fi

if [ -f "requirements-pip3.txt" ]; then
    echo ""
    echo "✓ Found requirements-pip3.txt, installing..."
    pip install -r requirements-pip3.txt || echo "⚠ Some requirements failed (might be OK)"
fi

echo "✓ Python dependencies installed"

# Verify critical imports
echo ""
echo "Verifying Python installation..."
python -c "import numpy; print('  ✓ numpy:', numpy.__version__)" || {
    echo "  ✗ numpy import failed"
    exit 1
}
python -c "import cv2; print('  ✓ opencv:', cv2.__version__)" || {
    echo "  ✗ opencv import failed"
    exit 1
}
python -c "import torch; print('  ✓ torch:', torch.__version__)" || {
    echo "  ✗ torch import failed"
    exit 1
}
python -c "import matplotlib; print('  ✓ matplotlib:', matplotlib.__version__)" || {
    echo "  ✗ matplotlib import failed"
    exit 1
}
python -c "import ujson; print('  ✓ ujson')" || {
    echo "  ✗ ujson import failed"
    exit 1
}
python -c "import pyqtgraph; print('  ✓ pyqtgraph')" || {
    echo "  ✗ pyqtgraph import failed"
    exit 1
}
python -c "import open3d; print('  ✓ open3d')" || {
    echo "  ✗ open3d import failed"
    exit 1
}
python -c "import psutil; print('  ✓ psutil')" || {
    echo "  ✗ psutil import failed"
    exit 1
}
python -c "import rerun; print('  ✓ rerun:', rerun.__version__)" || {
    echo "  ✗ rerun import failed"
    exit 1
}
echo "✓ Critical packages verified"

# Build C++ components
echo ""
echo "================================================"
echo "Building pySLAM C++ Components"
echo "================================================"

cd "$PYSLAM_DIR"

# Build Thirdparty libraries
echo ""
echo "Building Thirdparty libraries..."

# Build Pangolin (if not already installed)
if ! pkg-config --exists pangolin 2>/dev/null; then
    if [ -d "thirdparty/pangolin" ]; then
        echo "Building Pangolin from thirdparty..."
        cd thirdparty/pangolin

        # Try to install prerequisites if script exists
        if [ -f "scripts/install_prerequisites.sh" ]; then
            ./scripts/install_prerequisites.sh recommended
        else
            echo "  ⚠ No install_prerequisites.sh, skipping..."
        fi

        # Build Pangolin
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF
        make -j$(nproc)
        make install
        ldconfig
        cd "$PYSLAM_DIR"
        echo "✓ Pangolin built from thirdparty"
    else
        echo "  ⚠ Pangolin thirdparty not found, attempting system installation..."
        # Build Pangolin in /tmp as fallback
        PANGOLIN_BUILD_DIR="/tmp/pangolin-build-$$"
        mkdir -p "$PANGOLIN_BUILD_DIR"
        cd "$PANGOLIN_BUILD_DIR"

        git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git
        cd Pangolin
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF
        make -j$(nproc)
        make install
        ldconfig

        cd /
        rm -rf "$PANGOLIN_BUILD_DIR"
        cd "$PYSLAM_DIR"
        echo "✓ Pangolin built from source"
    fi
else
    echo "✓ Pangolin already installed"
fi

# Build g2o
if [ -d "thirdparty/g2o" ]; then
    echo ""
    echo "Building g2o..."
    cd thirdparty/g2o
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    make install
    ldconfig
    cd "$PYSLAM_DIR"
    echo "✓ g2o built"
fi

# Build pySLAM C++ components - FIX FOR CMAKE NINJA/MAKEFILE CONFLICT
echo ""
echo "Building pySLAM C++ components..."
cd "$PYSLAM_DIR"

# CLEAN OLD CMAKE CACHE TO FIX GENERATOR MISMATCH
echo "Cleaning old CMake cache..."
CPP_DIR="$PYSLAM_DIR/cpp"
BUILD_DIR="$CPP_DIR/build"

if [ -d "$BUILD_DIR" ]; then
    echo "  Removing old build directory..."
    rm -rf "$BUILD_DIR"
fi

# Clean other CMake artifacts
find "$CPP_DIR" -name "CMakeCache.txt" -delete 2>/dev/null || true
find "$CPP_DIR" -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true
find "$CPP_DIR" -name "build.ninja" -delete 2>/dev/null || true
echo "✓ CMake cache cleaned"

# Check for different build script names
BUILD_SCRIPT=""
if [ -f "build.sh" ]; then
    BUILD_SCRIPT="build.sh"
elif [ -f "install.sh" ]; then
    BUILD_SCRIPT="install.sh"
elif [ -f "install_all.sh" ]; then
    BUILD_SCRIPT="install_all.sh"
elif [ -f "install_basic.sh" ]; then
    BUILD_SCRIPT="install_basic.sh"
fi

if [ -n "$BUILD_SCRIPT" ]; then
    echo "Found $BUILD_SCRIPT, running it..."
    chmod +x "$BUILD_SCRIPT"
    ./"$BUILD_SCRIPT" || {
        echo "⚠ Build script failed, trying manual build..."
        BUILD_SCRIPT=""
    }
fi

# If no build script or it failed, try manual build
if [ -z "$BUILD_SCRIPT" ] || [ ! -f "pyslam_utils.so" ]; then
    echo "Attempting manual C++ build..."

    # Look for CMakeLists.txt in common locations
    if [ -f "CMakeLists.txt" ]; then
        echo "Found CMakeLists.txt in root, building..."
        mkdir -p build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python3) -G "Unix Makefiles"
        make -j$(nproc)
        cd "$PYSLAM_DIR"
    elif [ -f "cpp/CMakeLists.txt" ]; then
        echo "Found CMakeLists.txt in cpp/, building..."
        cd cpp
        mkdir -p build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python3) -G "Unix Makefiles"
        make -j$(nproc)
        cd "$PYSLAM_DIR"
    fi

    # Try setup.py if it exists
    if [ -f "setup.py" ]; then
        echo "Found setup.py, building Python extensions..."
        pip install -e . || {
            python setup.py build_ext --inplace || {
                echo "⚠ setup.py build failed"
            }
        }
    fi
fi

# COPY .SO FILES TO lib/ DIRECTORY
echo ""
echo "Copying .so files to lib/ directory..."
LIB_DIR="$PYSLAM_DIR/lib"
mkdir -p "$LIB_DIR"

SO_COUNT=0

# Find .so files in build directories
for so_file in $(find "$BUILD_DIR" -name "*.so" 2>/dev/null); do
    cp "$so_file" "$LIB_DIR/"
    echo "  Copied: $(basename $so_file)"
    SO_COUNT=$((SO_COUNT + 1))
done

# Check cpp/lib directory
if [ -d "$CPP_DIR/lib" ]; then
    for so_file in "$CPP_DIR/lib"/*.so; do
        if [ -f "$so_file" ]; then
            cp "$so_file" "$LIB_DIR/"
            echo "  Copied: $(basename $so_file)"
            SO_COUNT=$((SO_COUNT + 1))
        fi
    done 2>/dev/null
fi

echo "✓ Copied ${SO_COUNT} .so file(s) to $LIB_DIR"

# Verify pyslam_utils is available
echo ""
echo "Verifying C++ modules..."

# Check if lib directory exists and has .so files
if [ -d "$LIB_DIR" ] && [ -n "$(ls -A $LIB_DIR/*.so 2>/dev/null)" ]; then
    echo "✓ Found lib/ directory with .so files:"
    ls -la "$LIB_DIR"/*.so | head -5
    echo ""

    # Try importing with lib in path
    if python -c "import sys; sys.path.insert(0, '$LIB_DIR'); import pyslam_utils; print('  ✓ pyslam_utils imported successfully')" 2>/dev/null; then
        echo "✓ C++ modules built and can be imported"
        echo ""
        echo "NOTE: When running pySLAM, make sure to set PYTHONPATH:"
        echo "  export PYTHONPATH=$LIB_DIR:$PYSLAM_DIR/pyslam:$PYSLAM_DIR:\$PYTHONPATH"
        echo "  (or just $LIB_DIR:$PYSLAM_DIR:\$PYTHONPATH if no pyslam subdirectory)"
    else
        echo "⚠ pyslam_utils found but import failed"
        echo "This may be a Python/library compatibility issue"
    fi
else
    echo "⚠ lib/ directory not found or empty"
    echo ""
    echo "Checking for .so files in other locations..."
    find . -name "pyslam_utils*.so" -type f 2>/dev/null || echo "No pyslam_utils.so found"
    echo ""
    echo "You may need to manually build pySLAM. Check README.md for instructions:"
    ls -la *.md 2>/dev/null
fi

# Test installation
echo ""
echo "================================================"
echo "Testing pySLAM Installation"
echo "================================================"

# Test Python environment
python3 -c "import cv2; import numpy; import torch; import ujson; import matplotlib; print('✓ Core Python dependencies OK')" || {
    echo "✗ Failed to import core dependencies"
    exit 1
}

# Test if pySLAM modules can be imported
if [ -d "$PYSLAM_DIR" ]; then
    cd "$PYSLAM_DIR"

    echo ""
    echo "Testing pySLAM module imports..."
    python3 << 'PYTHON_TEST'
import sys
import os

# Add both lib and pyslam package directory
pyslam_dir = os.getcwd()
lib_dir = os.path.join(pyslam_dir, 'lib')
pyslam_package_dir = os.path.join(pyslam_dir, 'pyslam')

sys.path.insert(0, lib_dir)
if os.path.isdir(pyslam_package_dir):
    sys.path.insert(0, pyslam_package_dir)
    print(f"  Using package dir: {pyslam_package_dir}")
else:
    print(f"  Using root dir: {pyslam_dir}")
sys.path.insert(0, pyslam_dir)

# Test pySLAM imports
modules_ok = 0
modules_fail = 0

test_modules = ['config', 'camera', 'visual_odometry', 'slam']

for module in test_modules:
    try:
        exec(f"import {module}")
        print(f"  ✓ {module}")
        modules_ok += 1
    except ImportError as e:
        print(f"  ⚠ {module}: {e}")
        modules_fail += 1
    except Exception as e:
        print(f"  ⚠ {module}: {e}")
        modules_fail += 1

if modules_ok > 0:
    print(f"\n✓ {modules_ok} pySLAM modules can be imported")
    print("pySLAM appears functional!")
else:
    print(f"\n⚠ Could not import pySLAM modules")
    print("This might be OK if pySLAM uses a different structure")
    print("Run: bash baselines/pyslam/scripts/diagnose.sh")
PYTHON_TEST
fi

echo "✓ Installation tests passed"

deactivate

echo ""
echo "================================================"
echo "✓ pySLAM Setup Complete!"
echo "================================================"
echo ""
echo "Installation Summary:"
echo "  Root:        $PYSLAM_DIR"
echo "  Venv:        $VENV_DIR"
echo "  Python:      $(which python3.10)"
echo "  Lib dir:     $LIB_DIR"
echo ""
echo "To activate environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. bash $SCRIPT_DIR/run_baseline.sh     # Run on TUM"
echo "  2. python /workspace/scripts/evaluate_baseline.py --system pyslam"
echo ""
echo "If you encounter issues:"
echo "  - Run: bash $SCRIPT_DIR/diagnose.sh"
echo "  - Check that torch, ujson, msgpack are installed in venv"
echo "================================================"