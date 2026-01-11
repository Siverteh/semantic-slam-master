"""
Simple Dataset Download Script for Semantic SLAM Master's Thesis
Downloads essential TUM RGB-D sequences and provides Replica instructions

Just run: python download.py
"""

import os
import urllib.request
import tarfile
from pathlib import Path


def download_with_progress(url, filepath):
    """Download file with simple progress reporting"""
    print(f"  Downloading from {url}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')

    urllib.request.urlretrieve(url, filepath, reporthook=progress)
    print()  # New line after download completes


def download_tum_sequence(sequence_name, url_path, dest_dir):
    """Download and extract a single TUM RGB-D sequence"""

    # Check if already exists
    sequence_folder = dest_dir / f"rgbd_dataset_{sequence_name}"
    if sequence_folder.exists():
        print(f"✓ {sequence_name} already exists, skipping")
        return True

    print(f"\n📥 Downloading {sequence_name}...")

    base_url = "https://cvg.cit.tum.de/rgbd/dataset"
    url = f"{base_url}/{url_path}"
    filename = url_path.split('/')[-1]
    filepath = dest_dir / filename

    try:
        # Download
        download_with_progress(url, filepath)

        # Extract
        print(f"📦 Extracting {sequence_name}...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(dest_dir)

        # Clean up compressed file
        filepath.unlink()

        print(f"✓ {sequence_name} complete!")
        return True

    except Exception as e:
        print(f"❌ Error with {sequence_name}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def download_tum_dataset():
    """Download essential TUM RGB-D sequences for the thesis"""

    print("\n" + "="*70)
    print("📚 Downloading TUM RGB-D Dataset")
    print("="*70)
    print("Essential sequences for semantic SLAM thesis (~12GB total)")
    print()

    dest_dir = Path("data/tum_rgbd")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Essential sequences with descriptions
    sequences = [
        ("freiburg1_desk", "freiburg1/rgbd_dataset_freiburg1_desk.tgz",
         "Static office desk - baseline sequence"),

        ("freiburg1_room", "freiburg1/rgbd_dataset_freiburg1_room.tgz",
         "Full room - more complex scene"),

        ("freiburg1_plant", "freiburg1/rgbd_dataset_freiburg1_plant.tgz",
         "Low texture - challenges feature matching"),

        ("freiburg3_long_office_household", "freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz",
         "Long sequence - tests drift and loop closure"),

        ("freiburg3_walking_xyz", "freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz",
         "Dynamic person walking - tests dynamic handling"),

        ("freiburg3_walking_static", "freiburg3/rgbd_dataset_freiburg3_walking_static.tgz",
         "Person static then dynamic - comparison sequence"),
    ]

    success_count = 0
    total = len(sequences)

    for i, (name, url_path, description) in enumerate(sequences, 1):
        print(f"\n[{i}/{total}] {description}")
        if download_tum_sequence(name, url_path, dest_dir):
            success_count += 1

    print("\n" + "="*70)
    print(f"✓ Downloaded {success_count}/{total} TUM sequences successfully")
    print("="*70)

    return success_count == total


def setup_replica_instructions():
    """Provide instructions for Replica dataset"""

    print("\n" + "="*70)
    print("📚 Replica Dataset Setup")
    print("="*70)
    print()
    print("Replica is a large dataset (~200GB full, ~30GB for essential scenes)")
    print()
    print("🎯 RECOMMENDED APPROACH:")
    print("-" * 70)
    print("1. Ask your supervisor if university has a local copy")
    print("   Many institutions have Replica cached at:")
    print("   /datasets/replica/ or /data/replica/ or similar")
    print()
    print("2. If available locally, create a symbolic link:")
    print("   ln -s /path/to/university/replica data/replica")
    print()
    print("-" * 70)
    print()
    print("🌐 DOWNLOAD YOURSELF (if needed):")
    print("-" * 70)
    print("Step 1: Clone the Replica repository")
    print("   git clone https://github.com/facebookresearch/Replica-Dataset.git")
    print()
    print("Step 2: Run their download script")
    print("   cd Replica-Dataset")
    print("   ./download.sh ../data/replica")
    print()
    print("Step 3: Wait for download to complete (~30GB for essential scenes)")
    print("-" * 70)
    print()
    print("📋 Essential scenes you need (download these first):")
    print("   • room_0      - Simple bedroom")
    print("   • office_0    - Standard office")
    print("   • office_1    - Another office variant")
    print("   • apartment_0 - Complex multi-room")
    print("   • hotel_0     - Challenging lighting")
    print()
    print("="*70)


def verify_downloads():
    """Verify what has been downloaded"""

    print("\n" + "="*70)
    print("🔍 Verifying Downloads")
    print("="*70)

    # Check TUM RGB-D
    tum_dir = Path("tum_rgbd")
    if tum_dir.exists():
        print("\n✓ TUM RGB-D Directory exists")
        sequences = [d for d in tum_dir.iterdir() if d.is_dir()]
        print(f"  Found {len(sequences)} sequences:")
        for seq in sorted(sequences):
            # Check for required files
            has_rgb = (seq / "rgb").exists()
            has_depth = (seq / "depth").exists()
            has_gt = (seq / "groundtruth.txt").exists()

            if has_rgb and has_depth and has_gt:
                status = "✓"
                # Count images
                rgb_count = len(list((seq / "rgb").glob("*.png")))
                print(f"    {status} {seq.name} ({rgb_count} frames)")
            else:
                print(f"    ❌ {seq.name} (incomplete)")
    else:
        print("\n❌ TUM RGB-D not found at data/tum_rgbd/")

    # Check Replica
    replica_dir = Path("replica")
    if replica_dir.exists():
        print("\n✓ Replica Directory exists")
        scenes = [d for d in replica_dir.iterdir() if d.is_dir()]
        print(f"  Found {len(scenes)} scenes:")
        for scene in sorted(scenes):
            has_mesh = (scene / "mesh.ply").exists()
            has_semantic = (scene / "habitat").exists()
            if has_mesh or has_semantic:
                print(f"    ✓ {scene.name}")
            else:
                print(f"    ⚠️  {scene.name} (incomplete)")
    else:
        print("\n⚠️  Replica not found at data/replica/")
        print("   Follow the instructions above to download Replica")

    print("\n" + "="*70)


def main():
    """Main download function"""

    print("\n" + "="*70)
    print("🎓 SEMANTIC SLAM MASTER'S THESIS - DATASET DOWNLOADER")
    print("="*70)
    print()
    print("This script will download:")
    print("  • TUM RGB-D: 6 essential sequences (~12GB)")
    print("  • Replica: Instructions for manual download")
    print()
    print("Estimated time: 30-60 minutes (depending on internet speed)")
    print("="*70)

    input("\nPress ENTER to start downloading...")

    # Download TUM RGB-D
    download_tum_dataset()

    # Replica instructions
    setup_replica_instructions()

    # Verify everything
    verify_downloads()

    # Final summary
    print("\n" + "="*70)
    print("📋 NEXT STEPS")
    print("="*70)
    print()
    print("1. ✓ TUM RGB-D downloaded - ready to use!")
    print()
    print("2. ⚠️  Download Replica following instructions above")
    print("   (or use university copy if available)")
    print()
    print("3. Start testing with:")
    print("   • fr1_desk - easiest sequence")
    print("   • fr1_room - more complex")
    print()
    print("4. Your data structure should look like:")
    print("   data/")
    print("   ├── tum_rgbd/")
    print("   │   ├── rgbd_dataset_freiburg1_desk/")
    print("   │   ├── rgbd_dataset_freiburg1_room/")
    print("   │   └── ...")
    print("   └── replica/")
    print("       ├── room_0/")
    print("       ├── office_0/")
    print("       └── ...")
    print()
    print("="*70)
    print("✓ Download script complete!")
    print("="*70)


if __name__ == "__main__":
    main()