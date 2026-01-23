"""
Setup script for Semantic SLAM project
Creates necessary directories and verifies installation
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create all necessary directories"""
    dirs = [
        'models',
        'data',
        'losses',
        'configs',
        'checkpoints',
        'results',
        'logs',
    ]
    
    print("Creating directory structure...")
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"  ✓ {d}/")
    
    print("\nDirectory structure created!")


def verify_dependencies():
    """Check if required packages are installed"""
    print("\nVerifying dependencies...")
    
    required = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'yaml',
        'tqdm',
        'matplotlib'
    ]
    
    missing = []
    
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"  ✓ GPU memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 6:
                print("  ⚠️  Warning: GPU memory < 6GB, may need to reduce batch_size")
            
            return True
        else:
            print("  ✗ No GPU available (training will be very slow!)")
            print("  Consider using Google Colab or a cloud GPU")
            return False
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False


def check_dataset():
    """Check if TUM dataset is downloaded"""
    print("\nChecking dataset...")
    
    tum_dir = Path("data/tum_rgbd")
    
    if not tum_dir.exists():
        print("  ✗ TUM RGB-D dataset not found")
        print(f"  Run: python data/download_tum_rgbd.py")
        return False
    
    sequences = [
        "rgbd_dataset_freiburg1_desk",
        "rgbd_dataset_freiburg1_room",
        "rgbd_dataset_freiburg1_plant"
    ]
    
    found = 0
    for seq in sequences:
        seq_dir = tum_dir / seq
        if seq_dir.exists():
            rgb_dir = seq_dir / "rgb"
            if rgb_dir.exists():
                num_images = len(list(rgb_dir.glob("*.png")))
                print(f"  ✓ {seq} ({num_images} frames)")
                found += 1
            else:
                print(f"  ✗ {seq} (incomplete)")
        else:
            print(f"  ✗ {seq} (not found)")
    
    if found == 0:
        print("\n⚠️  No sequences found!")
        print("  Run: python data/download_tum_rgbd.py")
        return False
    elif found < len(sequences):
        print(f"\n⚠️  Only {found}/{len(sequences)} sequences found")
        print("  Some sequences missing, but you can start training with what you have")
        return True
    else:
        print("\n✓ All essential sequences downloaded!")
        return True


def test_imports():
    """Test if code files can be imported"""
    print("\nTesting imports...")
    
    try:
        # This would normally import from modules, but for setup we just check files exist
        files_to_check = [
            'models/dino_backbone.py',
            'models/keypoint_selector.py',
            'models/descriptor_refiner.py',
            'models/uncertainty_estimator.py',
            'data/tum_dataset.py',
            'losses/self_supervised.py',
            'train.py',
            'inference.py',
            'configs/train_config.yaml'
        ]
        
        all_exist = True
        for f in files_to_check:
            if Path(f).exists():
                print(f"  ✓ {f}")
            else:
                print(f"  ✗ {f} (missing)")
                all_exist = False
        
        if not all_exist:
            print("\n⚠️  Some code files missing!")
            print("  Make sure all files from the artifacts are created")
            return False
        else:
            print("\n✓ All code files present!")
            return True
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def print_summary():
    """Print setup summary and next steps"""
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\n📋 Next Steps:")
    print("\n1. Download dataset (if not done):")
    print("   python data/download_tum_rgbd.py")
    print("\n2. Configure training:")
    print("   Edit configs/train_config.yaml")
    print("   - Adjust batch_size for your GPU")
    print("   - Set loss weights")
    print("   - Enable/disable wandb logging")
    print("\n3. Start training:")
    print("   python train.py")
    print("\n4. Monitor progress:")
    print("   - Check wandb.ai (if enabled)")
    print("   - Or check checkpoints/ directory")
    print("\n5. Test inference:")
    print("   python inference.py --checkpoint checkpoints/best_model.pth \\")
    print("                       --image <path_to_image> \\")
    print("                       --output results/viz.png")
    print("\n" + "="*70)
    print("Good luck with your thesis! 🎓")
    print("="*70 + "\n")


def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("SEMANTIC SLAM - PROJECT SETUP")
    print("="*70 + "\n")
    
    # Run all checks
    create_directories()
    deps_ok = verify_dependencies()
    gpu_ok = check_gpu()
    dataset_ok = check_dataset()
    code_ok = test_imports()
    
    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    print(f"Dependencies: {'✓' if deps_ok else '✗'}")
    print(f"GPU:          {'✓' if gpu_ok else '✗ (optional but recommended)'}")
    print(f"Dataset:      {'✓' if dataset_ok else '✗ (run download script)'}")
    print(f"Code files:   {'✓' if code_ok else '✗ (create missing files)'}")
    
    if deps_ok and code_ok:
        print("\n✓ Setup successful! You can start training.")
        if not gpu_ok:
            print("⚠️  Warning: No GPU detected. Training will be VERY slow.")
        if not dataset_ok:
            print("⚠️  Warning: Dataset not complete. Run download script first.")
        
        print_summary()
    else:
        print("\n✗ Setup incomplete. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()