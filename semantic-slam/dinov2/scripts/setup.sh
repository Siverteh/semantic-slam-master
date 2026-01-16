#!/bin/bash
# Setup script for DINOv2

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DINOV2_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$DINOV2_ROOT/dinov2_repo"
MODELS_DIR="$DINOV2_ROOT/models"

echo "================================================"
echo "Setting up DINOv2 for Semantic SLAM"
echo "================================================"
echo "Installation directory: $DINOV2_ROOT"
echo ""

# 1. Clone the repository if it doesn't exist
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning DINOv2 repository..."
    git clone https://github.com/facebookresearch/dinov2.git "$REPO_DIR"
else
    echo "DINOv2 repository already exists, updating..."
    cd "$REPO_DIR"
    git pull
    cd "$DINOV2_ROOT"
fi

# 2. Install dependencies
echo ""
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install timm>=1.0.20 huggingface_hub pillow matplotlib pyyaml scikit-learn

# 3. Create models directory and download a base model
echo ""
echo "Setting up models directory..."
mkdir -p "$MODELS_DIR"

# Download DINOv2 ViT-S/14 via timm
MODEL_ID="vit_small_patch14_dinov2.lvd142m"
if [ ! -f "$MODELS_DIR/dinov2_vits14.pth" ]; then
    echo "Downloading $MODEL_ID via timm..."
    python3 -c "import torch; import timm; model = timm.create_model('$MODEL_ID', pretrained=True); torch.save(model.state_dict(), '$MODELS_DIR/dinov2_vits14.pth'); print('Model saved to $MODELS_DIR/dinov2_vits14.pth')"
else
    echo "Model already exists at $MODELS_DIR/dinov2_vits14.pth"
fi

echo ""
echo "================================================"
echo "✓ DINOv2 Setup Complete!"
echo "================================================"
echo "You can now run the test script:"
echo "python3 $SCRIPT_DIR/test_tum_rgbd.py"
echo "================================================"
