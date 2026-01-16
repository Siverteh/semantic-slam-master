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

# 1. Clone pySLAM if it doesn't exist
if [ ! -d "$PYSLAM_DIR" ]; then
    echo "Cloning pySLAM repository..."
    git clone --recursive https://github.com/luigifreda/pyslam.git "$PYSLAM_DIR"
else
    echo "pySLAM repository already exists, updating..."
    cd "$PYSLAM_DIR"
    git pull
    git submodule update --init --recursive
    cd "$BASELINE_DIR"
fi

# 2. Patch and install Python dependencies
echo ""
echo "Installing Python dependencies..."
cd "$PYSLAM_DIR"

if [ -f "pyproject.toml" ]; then
    echo "Patching pyproject.toml..."
    sed -i 's/requires-python = ">=3.11.9"/requires-python = ">=3.10.0"/' pyproject.toml
    sed -i '/"onnxruntime>=1.22.0"/d' pyproject.toml
    sed -i '/"open3d"/d' pyproject.toml
    sed -i '/"pyqt5"/d' pyproject.toml
fi

# Upgrade pip
python3 -m pip install --upgrade pip setuptools wheel build

# Install core dependencies
echo "Installing core Python packages..."
# Uninstall pyflann if exists (it has Python 3 compatibility issues)
python3 -m pip uninstall -y pyflann || true
python3 -m pip install "numpy<2" "opencv-python" matplotlib scipy pyyaml pillow tqdm kornia==0.7.3 gdown hjson ujson timm evo trimesh munch plyfile glfw PyOpenGL PyGLM rich ruff configargparse numba scikit-learn scikit-image rerun-sdk pyflann-py3 faiss-cpu

# Install pyslam in editable mode
echo "Installing pySLAM in editable mode..."
python3 -m pip install --no-deps -e .

# 3. Build C++ components
echo ""
echo "Building C++ components..."
export WITH_PYTHON_INTERP_CHECK=ON

# Build main cpp bindings
mkdir -p "$PYSLAM_DIR/cpp/build"
cd "$PYSLAM_DIR/cpp/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build essential thirdparty components
echo "Building thirdparty components..."

# 1. orbslam2_features
cd "$PYSLAM_DIR/thirdparty/orbslam2_features"
./build.sh

# 2. pangolin
cd "$PYSLAM_DIR/thirdparty/pangolin"
./build.sh

# 3. g2opy
cd "$PYSLAM_DIR/thirdparty/g2opy"
# Remove sudo from build.sh if present
sed -i 's/sudo //g' build.sh 2>/dev/null || true
./build.sh

# 4. pydbow3
cd "$PYSLAM_DIR/thirdparty/pydbow3"
./build.sh

# 5. pyibow
cd "$PYSLAM_DIR/thirdparty/pyibow"
./build.sh

echo ""
echo "================================================"
echo "✓ pySLAM Setup Complete!"
echo "================================================"
