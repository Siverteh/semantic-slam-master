import torch
import timm
from PIL import Image
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from torchvision import transforms

def test_dinov2_on_tum(num_images=5, input_size=448):
    """
    Test DINOv2 features on multiple images from the TUM RGB-D dataset using timm.
    """
    print("================================================")
    print(f"DINOv2 TUM RGB-D Test Script (Input Size: {input_size})")
    print("================================================")

    # 1. Load Dataset Config
    config_path = '/workspace/configs/datasets/tum_rgbd.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Dataset config not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = config['dataset']['root']
    sequence = "rgbd_dataset_freiburg1_desk" # Default sequence
    image_dir = os.path.join(dataset_root, sequence, 'rgb')

    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return

    # 2. Get multiple images
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
    if not all_images:
        print(f"No images found in {image_dir}")
        return

    indices = np.linspace(0, min(100, len(all_images)-1), num_images, dtype=int)
    selected_images = [all_images[i] for i in indices]

    # 3. Load Model using timm
    # DINOv2 ViT-S/14
    model_name = 'vit_small_patch14_dinov2.lvd142m'
    print(f"Loading model: {model_name}")
    try:
        model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 4. Preprocessing
    data_config = timm.data.resolve_model_data_config(model)
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])

    patch_size = 14 # DINOv2 uses 14x14 patches
    grid_size = input_size // patch_size

    # Create directory for results
    results_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Processing {num_images} images...")

    for idx, img_name in enumerate(selected_images):
        image_path = os.path.join(image_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract features
            all_tokens = model.forward_features(img_tensor)

            # DINOv2 tokens in timm: [CLS, *patches] or [CLS, *registers, *patches]
            # Standard DINOv2 has no registers. Reg versions have 4.
            num_tokens = all_tokens.shape[1]
            num_patches = grid_size * grid_size

            # Auto-detect registers: if num_tokens > num_patches + 1
            n_extra = num_tokens - num_patches
            # Usually n_extra is 1 (CLS) or 5 (CLS + 4 Reg)

            patch_features = all_tokens[0, n_extra:, :].cpu().numpy()

            # PCA
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(patch_features)

            # Normalize and reshape
            pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
            pca_img = pca_features.reshape(grid_size, grid_size, 3)

            # Visualization
            plt.figure(figsize=(15, 7))

            plt.subplot(1, 2, 1)
            plt.imshow(image.resize((input_size, input_size)))
            plt.title(f"Original: {img_name}")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(pca_img, interpolation='nearest')
            plt.title("DINOv2 Semantic Features (PCA)")
            plt.axis('off')

            save_path = os.path.join(results_dir, f"result_{idx}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"  - Saved result {idx+1}/{num_images} to {save_path}")

    print(f"\n✓ All results saved in: {results_dir}")

if __name__ == "__main__":
    # Input size must be multiple of 14 for DINOv2
    # 14 * 32 = 448
    test_dinov2_on_tum(num_images=5, input_size=448)
