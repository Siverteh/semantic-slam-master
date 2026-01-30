#!/bin/bash
# Setup script for pySLAM baseline (copied from baselines)

set -e  # Exit on error
#!/bin/bash
# Setup script for isolated pySLAM + semantic features

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYSLAM_DIR="$SCRIPT_DIR/pyslam"

echo "================================================"
echo "Setting up pySLAM for semantic experiments"
echo "================================================"
echo "Installation directory: $SCRIPT_DIR"
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

python3 -m pip install --upgrade pip setuptools wheel build
python3 -m pip uninstall -y pyflann || true
python3 -m pip install "numpy<2" "opencv-python" matplotlib scipy pyyaml pillow tqdm kornia==0.7.3 gdown hjson ujson timm evo trimesh munch plyfile glfw PyOpenGL PyGLM rich ruff configargparse numba scikit-learn scikit-image rerun-sdk pyflann-py3 faiss-cpu

echo "Installing pySLAM in editable mode..."
python3 -m pip install --no-deps -e .

echo "Applying semantic feature integration..."
cat <<'PY' > /tmp/semantic_pyslam_patch.py
from pathlib import Path
import os

root = Path(os.environ["PYSLAM_DIR"])
feature_semantic = root / "pyslam" / "local_features" / "feature_semantic.py"
feature_types = root / "pyslam" / "local_features" / "feature_types.py"
feature_manager = root / "pyslam" / "local_features" / "feature_manager.py"
feature_tracker_configs = root / "pyslam" / "local_features" / "feature_tracker_configs.py"

semantic_source = '''"""
Semantic feature extractor wrapper for pySLAM.
Uses the custom semantic keypoint model from semantic-slam.
"""

from pathlib import Path
import sys
import os
import numpy as np

from .feature_base import BaseFeature2D
from pyslam.utilities.logging import Printer


def _add_workspace_paths():
    root = Path(__file__).resolve().parents[4]
    semantic_slam_path = root / "semantic-slam"
    semantic_eval_path = root / "semantic-slam-evaluation"
    if semantic_eval_path.exists():
        sys.path.insert(0, str(semantic_eval_path))
    if semantic_slam_path.exists():
        sys.path.insert(0, str(semantic_slam_path))


_add_workspace_paths()

try:
    from semantic_feature_extractor import SemanticFeatureExtractor
except Exception as exc:
    raise ImportError(
        "Failed to import SemanticFeatureExtractor. Ensure /workspace/semantic-slam-evaluation "
        "is available and dependencies are installed."
    ) from exc


class SemanticFeature2D(BaseFeature2D):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        num_features: int = 500,
        normalize_descriptors: bool = True,
    ):
        super().__init__(num_features=num_features, device=device)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Semantic checkpoint not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.normalize_descriptors = normalize_descriptors
        self.extractor = SemanticFeatureExtractor(
            checkpoint_path, device=device, num_keypoints=num_features
        )
        Printer.green(
            f"SemanticFeature2D: loaded checkpoint={checkpoint_path}, "
            f"num_features={num_features}, device={device}"
        )

    def setMaxFeatures(self, num_features):
        self.num_features = num_features
        if hasattr(self, "extractor"):
            self.extractor.num_keypoints = num_features

    def detectAndCompute(self, frame, mask=None):
        keypoints, descriptors = self.extractor.detectAndCompute(frame)
        if descriptors is None or len(keypoints) == 0:
            return [], None
        descriptors = descriptors.astype(np.float32)
        if self.normalize_descriptors:
            norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            descriptors = descriptors / norms
        return keypoints, descriptors

    def detect(self, frame, mask=None):
        keypoints, _ = self.detectAndCompute(frame)
        return keypoints

    def compute(self, frame, kps=None, mask=None):
        keypoints, descriptors = self.detectAndCompute(frame)
        return keypoints, descriptors
'''

feature_semantic.write_text(semantic_source)

text = feature_types.read_text()
if "SEMANTIC" not in text:
    text = text.replace(
        "KEYNETAFFNETHARDNET = 28  # [kornia-based] Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor. \"Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters\"\n",
        "KEYNETAFFNETHARDNET = 28  # [kornia-based] Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor. \"Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters\"\n    SEMANTIC = 29  # [end-to-end] joint detector-descriptor - custom semantic keypoint model\n",
    )
    text = text.replace(
        "KEYNETAFFNETHARDNET = 33  # [kornia-based] Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor. \"Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters\"\n",
        "KEYNETAFFNETHARDNET = 33  # [kornia-based] Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor. \"Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters\"\n    SEMANTIC = 34  # [end-to-end] joint detector-descriptor - custom semantic keypoint model\n",
    )
    text = text.replace(
        "max_descriptor_distance[FeatureDescriptorTypes.KEYNETAFFNETHARDNET] = (\n        2.40  # KEYNETAFFNETHARDNET\n    )\n    #\n",
        "max_descriptor_distance[FeatureDescriptorTypes.KEYNETAFFNETHARDNET] = (\n        2.40  # KEYNETAFFNETHARDNET\n    )\n    #\n    norm_type[FeatureDescriptorTypes.SEMANTIC] = cv2.NORM_L2\n    max_descriptor_distance[FeatureDescriptorTypes.SEMANTIC] = 2.0  # SEMANTIC\n    #\n",
    )
    feature_types.write_text(text)

text = feature_manager.read_text()
if "SemanticFeature2D" not in text:
    text = text.replace(
        "KeyNetAffNetHardNetFeature2D = import_from(\n    \"pyslam.local_features.feature_keynet_affnet_hardnet\", \"KeyNetAffNetHardNetFeature2D\"\n)\n",
        "KeyNetAffNetHardNetFeature2D = import_from(\n    \"pyslam.local_features.feature_keynet_affnet_hardnet\", \"KeyNetAffNetHardNetFeature2D\"\n)\nSemanticFeature2D = import_from(\"pyslam.local_features.feature_semantic\", \"SemanticFeature2D\")\n",
    )
    text = text.replace(
        "        elif self.detector_type == FeatureDetectorTypes.KEYNETAFFNETHARDNET:\n            # self.num_levels = - # internally recomputed\n            self._feature_detector = KeyNetAffNetHardNetFeature2D(num_features=self.num_features)\n            self.keypoint_filter_type = KeyPointFilterTypes.NONE\n            #\n            #\n",
        "        elif self.detector_type == FeatureDetectorTypes.KEYNETAFFNETHARDNET:\n            # self.num_levels = - # internally recomputed\n            self._feature_detector = KeyNetAffNetHardNetFeature2D(num_features=self.num_features)\n            self.keypoint_filter_type = KeyPointFilterTypes.NONE\n            #\n            #\n        elif self.detector_type == FeatureDetectorTypes.SEMANTIC:\n            self.num_levels = 1\n            self.need_color_image = True\n            self.oriented_features = False\n            semantic_checkpoint = kwargs.get('semantic_checkpoint')\n            semantic_device = kwargs.get('semantic_device', 'cuda')\n            semantic_num_features = kwargs.get('semantic_num_features', self.num_features)\n            semantic_normalize = kwargs.get('semantic_normalize_descriptors', True)\n            if semantic_checkpoint is None:\n                raise ValueError(\"Semantic detector requires 'semantic_checkpoint' in config\")\n            self._feature_detector = SemanticFeature2D(\n                checkpoint_path=semantic_checkpoint,\n                device=semantic_device,\n                num_features=semantic_num_features,\n                normalize_descriptors=semantic_normalize,\n            )\n            self.num_features = semantic_num_features\n            self.keypoint_filter_type = KeyPointFilterTypes.NONE\n            #\n            #\n",
    )
    text = text.replace(
        "            elif self.descriptor_type == FeatureDescriptorTypes.KEYNETAFFNETHARDNET:\n                if self.detector_type != FeatureDetectorTypes.KEYNETAFFNETHARDNET:\n                    raise ValueError(\n                        \"You cannot use KEYNETAFFNETHARDNET descriptor without KEYNETAFFNETHARDNET detector!\\nPlease, select KEYNETAFFNETHARDNET as both descriptor and detector!\"\n                    )\n                self._feature_descriptor = self._feature_detector  # reuse the same detector object\n                #\n                #\n",
        "            elif self.descriptor_type == FeatureDescriptorTypes.KEYNETAFFNETHARDNET:\n                if self.detector_type != FeatureDetectorTypes.KEYNETAFFNETHARDNET:\n                    raise ValueError(\n                        \"You cannot use KEYNETAFFNETHARDNET descriptor without KEYNETAFFNETHARDNET detector!\\nPlease, select KEYNETAFFNETHARDNET as both descriptor and detector!\"\n                    )\n                self._feature_descriptor = self._feature_detector  # reuse the same detector object\n                #\n                #\n            elif self.descriptor_type == FeatureDescriptorTypes.SEMANTIC:\n                self.oriented_features = False\n                self.need_color_image = True\n                if self.detector_type != FeatureDetectorTypes.SEMANTIC:\n                    raise ValueError(\n                        \"You cannot use SEMANTIC descriptor without SEMANTIC detector!\\nPlease, select SEMANTIC as both descriptor and detector!\"\n                    )\n                self._feature_descriptor = self._feature_detector\n                #\n                #\n",
    )
    feature_manager.write_text(text)

text = feature_tracker_configs.read_text()
if "SEMANTIC" not in text:
    insertion = """
    SEMANTIC = dict(
        num_features=kNumFeatures,
        num_levels=1,
        scale_factor=1.2,
        detector_type=FeatureDetectorTypes.SEMANTIC,
        descriptor_type=FeatureDescriptorTypes.SEMANTIC,
        sigma_level0=Parameters.kSigmaLevel0,
        match_ratio_test=0.8,
        tracker_type=kTrackerType,
        semantic_checkpoint=None,
        semantic_device="cuda",
        semantic_num_features=kNumFeatures,
        semantic_normalize_descriptors=True,
    )
"""
    if "XFEAT_XFEAT" in text:
        parts = text.split("XFEAT_XFEAT = dict(")
        text = parts[0] + "XFEAT_XFEAT = dict(" + parts[1] + insertion
    feature_tracker_configs.write_text(text)
PY
PYSLAM_DIR="$PYSLAM_DIR" python3 /tmp/semantic_pyslam_patch.py

echo "Patching AVX-512 helper for cpp_core build..."
python3 - <<'PY'
from pathlib import Path

path = Path("pyslam/slam/cpp/utils/descriptor_helpers.h")
text = path.read_text()
marker = "#elif defined(__AVX2__)"
if "#if defined(__AVX512F__)" in text and marker in text:
    avx512_block = text.split(marker)[0]
    if "hsum256_ps(__m256 v) noexcept" not in avx512_block:
        insertion = """
static inline float hsum256_ps(__m256 v) noexcept {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(sum);  // (sum3,sum3,sum1,sum1)
    __m128 sums = _mm_add_ps(sum, shuf); // (s3+s2, s3+s2, s1+s0, s1+s0)
    shuf = _mm_movehl_ps(shuf, sums);    // (   ,    , s3+s2,   )
    sums = _mm_add_ss(sums, shuf);       // s3+s2+s1+s0
    return _mm_cvtss_f32(sums);
}
"""
        text = text.replace(marker, insertion + marker, 1)
        path.write_text(text)
PY

# 3. Build C++ components
echo ""
echo "Building C++ components..."
export WITH_PYTHON_INTERP_CHECK=ON

echo "Installing system dependencies (nlohmann-json)..."
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y nlohmann-json3-dev
fi

if [ ! -e /usr/bin/python ] && [ -x /usr/bin/python3 ]; then
    ln -s /usr/bin/python3 /usr/bin/python
fi

echo "Building GTSAM..."
cd "$PYSLAM_DIR"
./scripts/install_gtsam.sh

echo "Building pySLAM C++ core (cpp_core)..."
./build_cpp_core.sh

mkdir -p "$PYSLAM_DIR/cpp/build"
cd "$PYSLAM_DIR/cpp/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "Building thirdparty components..."

cd "$PYSLAM_DIR/thirdparty/orbslam2_features"
./build.sh

cd "$PYSLAM_DIR/thirdparty/pangolin"
./build.sh

cd "$PYSLAM_DIR/thirdparty/g2opy"
sed -i 's/sudo //g' build.sh 2>/dev/null || true
./build.sh

cd "$PYSLAM_DIR/thirdparty/pydbow3"
./build.sh

cd "$PYSLAM_DIR/thirdparty/pyibow"
./build.sh

# Ensure C++ libs are discoverable by the Python package
CPP_BUILT_LIB="$PYSLAM_DIR/cpp/lib"
CPP_PKG_LIB="$PYSLAM_DIR/pyslam/slam/cpp/lib"
mkdir -p "$(dirname "$CPP_PKG_LIB")"
ln -sfn "$CPP_BUILT_LIB" "$CPP_PKG_LIB"

echo ""
echo "================================================"
echo "✓ pySLAM Setup Complete!"
echo "================================================"

