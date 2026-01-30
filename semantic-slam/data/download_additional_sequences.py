"""
Download Additional TUM Sequences for Fixed Training
Run this to get fr2_desk and fr3_cabinet (needed for new training config)
"""

import urllib.request
import tarfile
from pathlib import Path


def download_with_progress(url, filepath):
    """Download file with progress"""
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


def download_sequence(sequence_name, url_path, dest_dir):
    """Download and extract sequence"""
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
        print(f"❌ Error: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def main():
    print("\n" + "="*70)
    print("📚 Download Additional TUM Sequences")
    print("="*70)
    print("Getting fr2_desk and fr3_cabinet for improved training")
    print()

    dest_dir = Path("data/tum_rgbd")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Additional sequences needed
    sequences = [
        ("freiburg2_desk", "freiburg2/rgbd_dataset_freiburg2_desk.tgz",
         "Freiburg 2 desk - different camera than fr1"),

        ("freiburg3_cabinet", "freiburg3/rgbd_dataset_freiburg3_cabinet.tgz",
         "Cabinet sequence - LOW TEXTURE like plant!"),
    ]

    success_count = 0
    for name, url_path, description in sequences:
        print(f"\n{description}")
        if download_sequence(name, url_path, dest_dir):
            success_count += 1

    print("\n" + "="*70)
    print(f"✓ Downloaded {success_count}/{len(sequences)} sequences")
    print("="*70)

    if success_count == len(sequences):
        print("\n✅ All sequences ready!")
        print("\nYou now have:")
        print("  • fr2_desk - training")
        print("  • fr3_cabinet - training (LOW TEXTURE)")
        print("  • fr3_walking_static - training (dynamic)")
        print("  • fr1_plant - validation (LOW TEXTURE)")
        print("\nStart training with:")
        print("  python train.py")
    else:
        print("\n❌ Some downloads failed. Please try again.")


if __name__ == "__main__":
    main()