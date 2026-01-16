#!/bin/bash
# Setup script for DINOv3

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DINOV3_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$DINOV3_ROOT/DINOv3"
MODELS_DIR="$DINOV3_ROOT/models"

echo "================================================"
echo "Setting up DINOv3 for Semantic SLAM"
echo "================================================"
echo "Installation directory: $DINOV3_ROOT"
echo ""

# 1. Clone the repository if it doesn't exist
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning DINOv3 repository..."
    git clone https://github.com/facebookresearch/dinov3.git "$REPO_DIR"
else
    echo "DINOv3 repository already exists, updating..."
    cd "$REPO_DIR"
    git pull
    cd "$DINOV3_ROOT"
fi

# 2. Install dependencies
echo ""
echo "Installing Python dependencies..."
# We use python3 -m pip to ensure we use the correct pip
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # Default to CPU for setup, user can change to CUDA
python3 -m pip install transformers>=4.56.0 timm>=1.0.20 huggingface_hub pillow matplotlib pyyaml scikit-learn

# 3. Create models directory and download a base model
echo ""
echo "Setting up models directory..."
mkdir -p "$MODELS_DIR"

# Download DINOv3 ViT-S/16 (Small) as a lightweight base model for testing
# You can also use ViT-B/16 (Base) if you have more memory
MODEL_ID="facebook/dinov3-vits16-pretrain-lvd1689m"
MODEL_PATH="$MODELS_DIR/dinov3-vits16"

if [ ! -f "$MODELS_DIR/dinov3_vits16.pth" ]; then
    echo "Downloading $MODEL_ID via timm..."
    python3 -c "import torch; import timm; model = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True); torch.save(model.state_dict(), '$MODELS_DIR/dinov3_vits16.pth'); print('Model saved to $MODELS_DIR/dinov3_vits16.pth')"
else
    echo "Model already exists at $MODELS_DIR/dinov3_vits16.pth"
fi

echo ""
echo "================================================"
echo "✓ DINOv3 Setup Complete!"
echo "================================================"
echo "You can now run the test script:"
echo "python3 $SCRIPT_DIR/test_tum_rgbd.py"
echo "================================================"
