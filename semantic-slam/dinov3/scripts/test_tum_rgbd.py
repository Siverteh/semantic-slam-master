import torch
import timm
from PIL import Image
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from torchvision import transforms
import torch.nn.functional as F

def test_dinov3_on_tum(num_images=5, input_size=448):
    """
    Test DINOv3 features on multiple images from the TUM RGB-D dataset.
    Higher input_size provides finer resolution in the PCA map.
    """
    print("================================================")
    print(f"DINOv3 TUM RGB-D Test Script (Input Size: {input_size})")
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

    # Select a few images spread across the sequence
    indices = np.linspace(0, min(100, len(all_images)-1), num_images, dtype=int)
    selected_images = [all_images[i] for i in indices]

    # 3. Load Model using timm
    model_name = 'vit_small_patch16_dinov3.lvd1689m'
    print(f"Loading model: {model_name}")
    try:
        model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 4. Custom Preprocessing for higher resolution
    data_config = timm.data.resolve_model_data_config(model)
    # Override input size for finer features
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])

    patch_size = 16
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

        # Extract patch tokens
        # DINOv3 tokens: [CLS, *storage, *patches]
        n_storage_tokens = 4
        num_patches = grid_size * grid_size
        patch_features = all_tokens[0, 1 + n_storage_tokens : 1 + n_storage_tokens + num_patches, :].cpu().numpy()

        # PCA
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(patch_features)

        # Normalize and reshape
        pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        pca_img = pca_features.reshape(grid_size, grid_size, 3)

        # Visualization
        plt.figure(figsize=(15, 7))

        # Original Image (resized for comparison)
        plt.subplot(1, 2, 1)
        plt.imshow(image.resize((input_size, input_size)))
        plt.title(f"Original: {img_name}")
        plt.axis('off')

        # PCA Map (upsampled for better viewing)
        plt.subplot(1, 2, 2)
        plt.imshow(pca_img, interpolation='nearest')
        plt.title("DINOv3 Semantic Features (PCA)")
        plt.axis('off')

        save_path = os.path.join(results_dir, f"result_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  - Saved result {idx+1}/{num_images} to {save_path}")

    print(f"\n✓ All results saved in: {results_dir}")

if __name__ == "__main__":
    # Increase input size to 448 for much better semantic detail
    test_dinov3_on_tum(num_images=5, input_size=448)
