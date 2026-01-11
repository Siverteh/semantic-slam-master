#!/bin/bash
# Fixed setup script for ORB-SLAM3 baseline

set -e  # Exit on error

echo "================================================"
echo "Setting up ORB-SLAM3 for baseline experiments"
echo "================================================"

# Function to check if a package is installed
check_package() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "ok installed"
}

# Install all required system dependencies
echo "Installing system dependencies..."
REQUIRED_PACKAGES=(
    "build-essential"
    "cmake"
    "git"
    "libboost-all-dev"
    "libeigen3-dev"
    "libopencv-dev"
    "libopencv-contrib-dev"
    "libssl-dev"
    "libglew-dev"
    "libgl1-mesa-dev"
    "libglu1-mesa-dev"
    "pkg-config"
)

# Add Pangolin build dependencies (we'll build it from source)
PANGOLIN_DEPS=(
    "libglew-dev"
    "libpython3-dev"
    "libeigen3-dev"
    "libavcodec-dev"
    "libavutil-dev"
    "libavformat-dev"
    "libswscale-dev"
    "libjpeg-dev"
    "libpng-dev"
    "libtiff-dev"
)

REQUIRED_PACKAGES+=("${PANGOLIN_DEPS[@]}")

# Remove duplicates
REQUIRED_PACKAGES=($(printf "%s\n" "${REQUIRED_PACKAGES[@]}" | sort -u))

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

# Verify critical dependencies
echo ""
echo "Verifying critical dependencies..."

# Check OpenCV version
OPENCV_VERSION=$(pkg-config --modversion opencv4 2>/dev/null || echo "not found")
if [[ "$OPENCV_VERSION" == "not found" ]]; then
    echo "✗ OpenCV 4.x not found!"
    echo "Install with: apt-get install -y libopencv-dev libopencv-contrib-dev"
    exit 1
fi
echo "✓ OpenCV $OPENCV_VERSION"

# Check Eigen3
if [ ! -d "/usr/include/eigen3" ]; then
    echo "✗ Eigen3 not found!"
    exit 1
fi
echo "✓ Eigen3 installed"

# Check Pangolin (critical for ORB-SLAM3)
echo "Checking for Pangolin..."
PANGOLIN_FOUND=false

if pkg-config --exists pangolin 2>/dev/null; then
    PANGOLIN_VERSION=$(pkg-config --modversion pangolin)
    echo "✓ Pangolin $PANGOLIN_VERSION (via pkg-config)"
    PANGOLIN_FOUND=true
elif [ -f "/usr/local/lib/libpangolin.so" ] || [ -f "/usr/local/lib/libpangolin.a" ]; then
    echo "✓ Pangolin (installed in /usr/local)"
    PANGOLIN_FOUND=true
fi

if [ "$PANGOLIN_FOUND" = false ]; then
    echo "Pangolin not found - building from source..."
    echo "This is required for ORB-SLAM3 visualization"

    # Build Pangolin in /tmp
    PANGOLIN_BUILD_DIR="/tmp/pangolin-build-$"
    mkdir -p "$PANGOLIN_BUILD_DIR"
    cd "$PANGOLIN_BUILD_DIR"

    echo "Cloning Pangolin..."
    git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git

    cd Pangolin
    mkdir -p build && cd build

    echo "Building Pangolin..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTS=OFF

    make -j$(nproc)

    echo "Installing Pangolin..."
    make install
    ldconfig

    # Cleanup
    cd /
    rm -rf "$PANGOLIN_BUILD_DIR"

    # Verify installation
    if [ -f "/usr/local/lib/libpangolin.so" ] || [ -f "/usr/local/lib/libpangolin.a" ]; then
        echo "✓ Pangolin installed successfully"
    else
        echo "✗ Pangolin installation failed!"
        exit 1
    fi
fi

# Navigate to baselines directory
cd /workspace/src/baselines

# Clone ORB-SLAM3
if [ ! -d "ORB_SLAM3" ]; then
    echo ""
    echo "Cloning ORB-SLAM3..."
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
    cd ORB_SLAM3
else
    echo ""
    echo "ORB-SLAM3 already exists, updating..."
    cd ORB_SLAM3
    git pull || echo "Warning: Could not pull latest changes"
fi

# Download vocabulary first (needed before build)
echo ""
echo "Downloading ORB vocabulary..."
if [ ! -f "Vocabulary/ORBvoc.txt" ]; then
    mkdir -p Vocabulary
    cd Vocabulary
    if [ ! -f "ORBvoc.txt.tar.gz" ]; then
        wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz
    fi
    tar -xzf ORBvoc.txt.tar.gz
    rm ORBvoc.txt.tar.gz
    cd ..
    echo "✓ Vocabulary downloaded"
else
    echo "✓ Vocabulary already exists"
fi

# Clean any previous failed builds
echo ""
echo "Cleaning previous build artifacts..."
rm -rf Thirdparty/DBoW2/build
rm -rf Thirdparty/g2o/build
rm -rf build
rm -rf lib

# Build Thirdparty libraries
echo ""
echo "================================================"
echo "Building Third-party Libraries"
echo "================================================"

# Build DBoW2
echo ""
echo "[1/3] Building DBoW2..."
cd Thirdparty/DBoW2
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-std=c++14 -O3"
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "✗ DBoW2 build failed!"
    exit 1
fi
echo "✓ DBoW2 built successfully"

# Build g2o
echo ""
echo "[2/3] Building g2o..."
cd ../../g2o
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-std=c++14 -O3"
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "✗ g2o build failed!"
    exit 1
fi
echo "✓ g2o built successfully"

# Sophus is already included, no separate build needed
echo ""
echo "[3/3] Sophus (bundled - no build needed)"
echo "✓ Using bundled Sophus"

# Build ORB-SLAM3 main library
echo ""
echo "================================================"
echo "Building ORB-SLAM3 Main Library"
echo "================================================"
cd ../../..  # Back to ORB_SLAM3 root

# Create build directory
mkdir -p build && cd build

# Configure with CMake - use C++14 for compatibility
echo ""
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-Wall -O3" \
    -DCMAKE_CXX_FLAGS="-Wall -O3 -std=c++14" \
    -DCMAKE_CXX_STANDARD=14

# Build with better error output
echo ""
echo "Building ORB-SLAM3 (this may take 5-10 minutes)..."
echo "Building with $(nproc) cores..."

if ! make -j$(nproc) 2>&1 | tee build.log; then
    echo ""
    echo "✗ ORB-SLAM3 build failed!"
    echo "Check build.log for details"
    echo ""
    echo "Last 50 lines of build output:"
    tail -n 50 build.log
    echo ""
    echo "Common fixes:"
    echo "1. Try building with fewer cores: cd build && make -j4"
    echo "2. Check if Pangolin is properly installed: pkg-config --modversion pangolin"
    echo "3. Ensure OpenCV 4.x is installed: pkg-config --modversion opencv4"
    exit 1
fi

# Verify build outputs
echo ""
echo "Verifying build..."
cd ..

MISSING_FILES=()
EXPECTED_FILES=(
    "lib/libORB_SLAM3.so"
    "Examples/Monocular/mono_euroc"
    "Examples/RGB-D/rgbd_tum"
    "Examples/Stereo/stereo_euroc"
)

for file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "✗ Build incomplete - missing files:"
    printf '  - %s\n' "${MISSING_FILES[@]}"
    exit 1
fi

echo "✓ All expected files present"

# Test that the library can be loaded
echo ""
echo "Testing library loading..."
if ldd Examples/RGB-D/rgbd_tum | grep -q "not found"; then
    echo "✗ Missing runtime dependencies:"
    ldd Examples/RGB-D/rgbd_tum | grep "not found"
    exit 1
fi
echo "✓ All runtime dependencies satisfied"

# Create convenience symlinks
cd /workspace/src/baselines
if [ ! -L "ORBvoc.txt" ]; then
    ln -s ORB_SLAM3/Vocabulary/ORBvoc.txt ORBvoc.txt
    echo "✓ Created vocabulary symlink"
fi

echo ""
echo "================================================"
echo "✓ ORB-SLAM3 Setup Complete!"
echo "================================================"
echo ""
echo "Installation Summary:"
echo "  Root:        $(pwd)/ORB_SLAM3"
echo "  Library:     $(pwd)/ORB_SLAM3/lib/libORB_SLAM3.so"
echo "  Vocabulary:  $(pwd)/ORB_SLAM3/Vocabulary/ORBvoc.txt"
echo "  Examples:    $(pwd)/ORB_SLAM3/Examples/"
echo ""
echo "Available executables:"
echo "  - Monocular:  Examples/Monocular/mono_euroc"
echo "  - RGB-D:      Examples/RGB-D/rgbd_tum"
echo "  - Stereo:     Examples/Stereo/stereo_euroc"
echo ""
echo "Next steps:"
echo "  1. Run quick test: cd ORB_SLAM3 && ./Examples/RGB-D/rgbd_tum --help"
echo "  2. Run baseline:   bash scripts/run_orb_slam3_baseline.sh"
echo "  3. Evaluate:       python scripts/evaluate_baseline.py"
echo ""
echo "Pangolin Viewer: Enabled (GUI will open during SLAM)"
echo "================================================"