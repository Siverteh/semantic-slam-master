#!/bin/bash
# Check if development environment is correctly set up

echo "================================================"
echo "Checking Semantic SLAM Development Environment"
echo "================================================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass=0
check_fail=0

# Function to check command
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        ((check_pass++))
        return 0
    else
        echo -e "${RED}✗${NC} $1 is NOT installed"
        ((check_fail++))
        return 1
    fi
}

# Function to check file
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        ((check_pass++))
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        ((check_fail++))
        return 1
    fi
}

# Function to check directory
check_directory() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        ((check_pass++))
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Missing: $1"
        ((check_fail++))
        return 1
    fi
}

echo ""
echo "1. System Dependencies"
echo "----------------------"
check_command gcc
check_command g++
check_command cmake
check_command git
check_command python3
check_command pip3

echo ""
echo "2. GPU and CUDA"
echo "---------------"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA driver installed"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    ((check_pass++))
else
    echo -e "${RED}✗${NC} nvidia-smi not found"
    ((check_fail++))
fi

# Check CUDA
if [ -d "/usr/local/cuda" ]; then
    echo -e "${GREEN}✓${NC} CUDA toolkit installed"
    nvcc --version | grep "release"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} CUDA toolkit not found"
    ((check_fail++))
fi

echo ""
echo "3. Python Environment"
echo "--------------------"
python3 --version

# Check key Python packages
packages=("numpy" "torch" "cv2" "evo" "matplotlib")
for pkg in "${packages[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Python package: $pkg"
        ((check_pass++))
    else
        echo -e "${RED}✗${NC} Python package missing: $pkg"
        ((check_fail++))
    fi
done

# Check PyTorch CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} PyTorch CUDA available"
    python3 -c "import torch; print(f'  CUDA devices: {torch.cuda.device_count()}')"
    ((check_pass++))
else
    echo -e "${RED}✗${NC} PyTorch CUDA not available"
    ((check_fail++))
fi

echo ""
echo "4. Project Structure"
echo "-------------------"
check_directory "/workspace/src/data/tum_rgbd"
check_directory "/workspace/src/baselines"
check_directory "/workspace/scripts"
check_directory "/workspace/experiments"

echo ""
echo "5. TUM RGB-D Dataset"
echo "-------------------"
sequences=(
    "rgbd_dataset_freiburg1_desk"
    "rgbd_dataset_freiburg1_plant"
    "rgbd_dataset_freiburg1_room"
    "rgbd_dataset_freiburg3_long_office_household"
    "rgbd_dataset_freiburg3_walking_static"
    "rgbd_dataset_freiburg3_walking_xyz"
)

for seq in "${sequences[@]}"; do
    seq_path="/workspace/src/data/tum_rgbd/$seq"
    if [ -d "$seq_path" ]; then
        # Check essential files
        if [ -f "$seq_path/rgb.txt" ] && [ -f "$seq_path/depth.txt" ] && [ -f "$seq_path/groundtruth.txt" ]; then
            echo -e "${GREEN}✓${NC} $seq (complete)"
            ((check_pass++))
        else
            echo -e "${YELLOW}⚠${NC} $seq (missing files)"
            ((check_fail++))
        fi
    else
        echo -e "${RED}✗${NC} $seq (not found)"
        ((check_fail++))
    fi
done

echo ""
echo "6. ORB-SLAM3 Setup"
echo "-----------------"
if [ -d "/workspace/src/baselines/ORB_SLAM3" ]; then
    echo -e "${GREEN}✓${NC} ORB-SLAM3 repository cloned"
    ((check_pass++))

    # Check if built
    if [ -f "/workspace/src/baselines/ORB_SLAM3/Examples/RGB-D/rgbd_tum" ]; then
        echo -e "${GREEN}✓${NC} ORB-SLAM3 built successfully"
        ((check_pass++))
    else
        echo -e "${YELLOW}⚠${NC} ORB-SLAM3 not built yet"
        echo "  Run: bash scripts/setup_orb_slam3.sh"
        ((check_fail++))
    fi

    # Check vocabulary
    if [ -f "/workspace/src/baselines/ORB_SLAM3/Vocabulary/ORBvoc.txt" ]; then
        echo -e "${GREEN}✓${NC} ORB vocabulary downloaded"
        ((check_pass++))
    else
        echo -e "${YELLOW}⚠${NC} ORB vocabulary not found"
        echo "  Will be downloaded during setup"
        ((check_fail++))
    fi
else
    echo -e "${RED}✗${NC} ORB-SLAM3 not cloned"
    echo "  Run: bash scripts/setup_orb_slam3.sh"
    ((check_fail++))
fi

echo ""
echo "================================================"
echo "Summary"
echo "================================================"
echo -e "Passed: ${GREEN}$check_pass${NC}"
echo -e "Failed: ${RED}$check_fail${NC}"

if [ $check_fail -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Ready to start Phase 1:"
    echo "  1. bash scripts/setup_orb_slam3.sh"
    echo "  2. bash scripts/run_orb_slam3_baseline.sh"
    echo "  3. python scripts/evaluate_baseline.py"
    exit 0
else
    echo -e "${YELLOW}⚠ Some checks failed${NC}"
    echo "Please fix the issues above before proceeding."
    exit 1
fi