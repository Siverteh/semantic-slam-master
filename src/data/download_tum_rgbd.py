"""
Complete TUM RGB-D Dataset Downloader
Downloads all TUM RGB-D sequences for SLAM research

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
    print()


def download_tum_sequence(sequence_name, url_path, dest_dir):
    """Download and extract a single TUM RGB-D sequence"""

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
        download_with_progress(url, filepath)

        print(f"📦 Extracting {sequence_name}...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(dest_dir)

        filepath.unlink()
        print(f"✓ {sequence_name} complete!")
        return True

    except Exception as e:
        print(f"❌ Error with {sequence_name}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def download_tum_dataset():
    """Download complete TUM RGB-D dataset"""

    print("\n" + "="*70)
    print("📚 TUM RGB-D COMPLETE DATASET DOWNLOADER")
    print("="*70)
    print("Downloading all 48 sequences (~60GB total)")
    print()

    dest_dir = Path("tum_rgbd")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # All sequences organized by category
    sequences = [
        # Testing and Debugging
        ("freiburg1_xyz", "freiburg1/rgbd_dataset_freiburg1_xyz.tgz"),
        ("freiburg1_rpy", "freiburg1/rgbd_dataset_freiburg1_rpy.tgz"),
        ("freiburg2_xyz", "freiburg2/rgbd_dataset_freiburg2_xyz.tgz"),
        ("freiburg2_rpy", "freiburg2/rgbd_dataset_freiburg2_rpy.tgz"),

        # Handheld SLAM
        ("freiburg1_360", "freiburg1/rgbd_dataset_freiburg1_360.tgz"),
        ("freiburg1_floor", "freiburg1/rgbd_dataset_freiburg1_floor.tgz"),
        ("freiburg1_desk", "freiburg1/rgbd_dataset_freiburg1_desk.tgz"),
        ("freiburg1_desk2", "freiburg1/rgbd_dataset_freiburg1_desk2.tgz"),
        ("freiburg1_room", "freiburg1/rgbd_dataset_freiburg1_room.tgz"),
        ("freiburg2_360_hemisphere", "freiburg2/rgbd_dataset_freiburg2_360_hemisphere.tgz"),
        ("freiburg2_360_kidnap", "freiburg2/rgbd_dataset_freiburg2_360_kidnap.tgz"),
        ("freiburg2_desk", "freiburg2/rgbd_dataset_freiburg2_desk.tgz"),
        ("freiburg2_large_no_loop", "freiburg2/rgbd_dataset_freiburg2_large_no_loop.tgz"),
        ("freiburg2_large_with_loop", "freiburg2/rgbd_dataset_freiburg2_large_with_loop.tgz"),
        ("freiburg3_long_office_household", "freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"),

        # Robot SLAM
        ("freiburg2_pioneer_360", "freiburg2/rgbd_dataset_freiburg2_pioneer_360.tgz"),
        ("freiburg2_pioneer_slam", "freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz"),
        ("freiburg2_pioneer_slam2", "freiburg2/rgbd_dataset_freiburg2_pioneer_slam2.tgz"),
        ("freiburg2_pioneer_slam3", "freiburg2/rgbd_dataset_freiburg2_pioneer_slam3.tgz"),

        # Structure vs. Texture
        ("freiburg3_nostructure_notexture_far", "freiburg3/rgbd_dataset_freiburg3_nostructure_notexture_far.tgz"),
        ("freiburg3_nostructure_notexture_near_withloop", "freiburg3/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop.tgz"),
        ("freiburg3_nostructure_texture_far", "freiburg3/rgbd_dataset_freiburg3_nostructure_texture_far.tgz"),
        ("freiburg3_nostructure_texture_near_withloop", "freiburg3/rgbd_dataset_freiburg3_nostructure_texture_near_withloop.tgz"),
        ("freiburg3_structure_notexture_far", "freiburg3/rgbd_dataset_freiburg3_structure_notexture_far.tgz"),
        ("freiburg3_structure_notexture_near", "freiburg3/rgbd_dataset_freiburg3_structure_notexture_near.tgz"),
        ("freiburg3_structure_texture_far", "freiburg3/rgbd_dataset_freiburg3_structure_texture_far.tgz"),
        ("freiburg3_structure_texture_near", "freiburg3/rgbd_dataset_freiburg3_structure_texture_near.tgz"),

        # Dynamic Objects
        ("freiburg2_desk_with_person", "freiburg2/rgbd_dataset_freiburg2_desk_with_person.tgz"),
        ("freiburg3_sitting_static", "freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz"),
        ("freiburg3_sitting_xyz", "freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz"),
        ("freiburg3_sitting_halfsphere", "freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere.tgz"),
        ("freiburg3_sitting_rpy", "freiburg3/rgbd_dataset_freiburg3_sitting_rpy.tgz"),
        ("freiburg3_walking_static", "freiburg3/rgbd_dataset_freiburg3_walking_static.tgz"),
        ("freiburg3_walking_xyz", "freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz"),
        ("freiburg3_walking_halfsphere", "freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz"),
        ("freiburg3_walking_rpy", "freiburg3/rgbd_dataset_freiburg3_walking_rpy.tgz"),

        # 3D Object Reconstruction
        ("freiburg1_plant", "freiburg1/rgbd_dataset_freiburg1_plant.tgz"),
        ("freiburg1_teddy", "freiburg1/rgbd_dataset_freiburg1_teddy.tgz"),
        ("freiburg2_coke", "freiburg2/rgbd_dataset_freiburg2_coke.tgz"),
        ("freiburg2_dishes", "freiburg2/rgbd_dataset_freiburg2_dishes.tgz"),
        ("freiburg2_flowerbouquet", "freiburg2/rgbd_dataset_freiburg2_flowerbouquet.tgz"),
        ("freiburg2_flowerbouquet_brownbackground", "freiburg2/rgbd_dataset_freiburg2_flowerbouquet_brownbackground.tgz"),
        ("freiburg2_metallic_sphere", "freiburg2/rgbd_dataset_freiburg2_metallic_sphere.tgz"),
        ("freiburg2_metallic_sphere2", "freiburg2/rgbd_dataset_freiburg2_metallic_sphere2.tgz"),
        ("freiburg3_cabinet", "freiburg3/rgbd_dataset_freiburg3_cabinet.tgz"),
        ("freiburg3_large_cabinet", "freiburg3/rgbd_dataset_freiburg3_large_cabinet.tgz"),
        ("freiburg3_teddy", "freiburg3/rgbd_dataset_freiburg3_teddy.tgz"),
    ]

    success_count = 0
    total = len(sequences)

    for i, (name, url_path) in enumerate(sequences, 1):
        print(f"\n[{i}/{total}]")
        if download_tum_sequence(name, url_path, dest_dir):
            success_count += 1

    print("\n" + "="*70)
    print(f"✓ Downloaded {success_count}/{total} sequences successfully")
    print("="*70)

    return success_count == total


def verify_downloads():
    """Verify downloaded sequences"""

    print("\n" + "="*70)
    print("🔍 Verifying Downloads")
    print("="*70)

    tum_dir = Path("tum_rgbd")
    if not tum_dir.exists():
        print("\n❌ TUM RGB-D directory not found")
        return

    sequences = [d for d in tum_dir.iterdir() if d.is_dir()]
    print(f"\n✓ Found {len(sequences)} sequences:")

    for seq in sorted(sequences):
        has_rgb = (seq / "rgb").exists()
        has_depth = (seq / "depth").exists()
        has_gt = (seq / "groundtruth.txt").exists()

        if has_rgb and has_depth and has_gt:
            rgb_count = len(list((seq / "rgb").glob("*.png")))
            print(f"  ✓ {seq.name} ({rgb_count} frames)")
        else:
            print(f"  ❌ {seq.name} (incomplete)")

    print("\n" + "="*70)


def main():
    """Main download function"""

    print("\n" + "="*70)
    print("🎓 TUM RGB-D COMPLETE DATASET DOWNLOADER")
    print("="*70)
    print()
    print("This script will download all 48 sequences (~60GB)")
    print("Estimated time: 2-4 hours (depending on internet speed)")
    print()
    print("Categories included:")
    print("  • Testing & Debugging (4 sequences)")
    print("  • Handheld SLAM (11 sequences)")
    print("  • Robot SLAM (4 sequences)")
    print("  • Structure vs. Texture (8 sequences)")
    print("  • Dynamic Objects (9 sequences)")
    print("  • 3D Object Reconstruction (12 sequences)")
    print("="*70)

    input("\nPress ENTER to start downloading...")

    download_tum_dataset()
    verify_downloads()

    print("\n" + "="*70)
    print("📋 NEXT STEPS")
    print("="*70)
    print()
    print("✓ Dataset downloaded to: tum_rgbd/")
    print()
    print("Start testing with:")
    print("  • freiburg1_xyz - simplest sequence")
    print("  • freiburg1_desk - standard benchmark")
    print("  • freiburg3_long_office_household - loop closure")
    print()
    print("Data structure:")
    print("  tum_rgbd/")
    print("  ├── rgbd_dataset_freiburg1_xyz/")
    print("  │   ├── rgb/")
    print("  │   ├── depth/")
    print("  │   └── groundtruth.txt")
    print("  ├── rgbd_dataset_freiburg1_desk/")
    print("  └── ...")
    print()
    print("="*70)
    print("✓ Download complete!")
    print("="*70)


if __name__ == "__main__":
    main()