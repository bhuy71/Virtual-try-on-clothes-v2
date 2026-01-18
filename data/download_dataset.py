"""
Dataset Download Script for Virtual Try-On

Supports:
- VITON dataset (256x192)
- VITON-HD dataset (1024x768)
- DressCode dataset

Usage:
    python download_dataset.py --dataset viton-hd --output_dir ./data
    
Note: Large datasets may require manual download from the official sources.
"""

import os
import argparse
import zipfile
import requests
from tqdm import tqdm

# Try to import gdown, but make it optional
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("Note: gdown not installed. Install with: pip install gdown")


DATASET_INFO = {
    "viton": {
        "description": "Original VITON dataset (256x192)",
        "gdrive_id": "1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo",
        "filename": "viton.zip",
        "size": "~2GB",
        "manual_url": "https://github.com/xthan/VITON"
    },
    "viton-hd": {
        "description": "VITON-HD dataset (1024x768)",
        "gdrive_id": "1-mQl3L8K0d8YNRZ3kRWGUwQQMsNRXhJL",  # Alternative ID
        "filename": "viton-hd.zip",
        "size": "~30GB",
        "manual_url": "https://github.com/shadow2496/VITON-HD"
    },
    "viton-hd-test": {
        "description": "VITON-HD test dataset only (for quick testing)",
        "gdrive_id": "1Y7uV0gomwWyVBpqAYnvkXVPOQ3mKLx4r",
        "filename": "viton-hd-test.zip",
        "size": "~3GB",
        "manual_url": "https://github.com/shadow2496/VITON-HD"
    }
}


def download_from_gdrive(file_id: str, output_path: str) -> bool:
    """Download a file from Google Drive."""
    if not GDOWN_AVAILABLE:
        return False
    
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        
        # Verify it's a valid zip file
        if os.path.exists(output_path):
            # Check if file is HTML (error page) instead of zip
            with open(output_path, 'rb') as f:
                header = f.read(4)
                if header[:2] != b'PK':  # ZIP files start with 'PK'
                    print("Downloaded file is not a valid ZIP archive.")
                    os.remove(output_path)
                    return False
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_file(url: str, output_path: str) -> bool:
    """Download a file from URL."""
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str) -> bool:
    """Extract a zip file."""
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file.")
        print("The file may be corrupted or the download link has expired.")
        return False


def print_manual_download_instructions(dataset_name: str, info: dict, output_dir: str):
    """Print instructions for manual download."""
    print("\n" + "="*60)
    print("⚠️  AUTOMATIC DOWNLOAD FAILED")
    print("="*60)
    print(f"\nThe {dataset_name} dataset is too large or requires manual download.")
    print("\n📥 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("-"*40)
    
    if dataset_name == "viton-hd":
        print("""
1. Visit the official VITON-HD repository:
   https://github.com/shadow2496/VITON-HD

2. Request access to the dataset (follow instructions in the repo)

3. Download the following files:
   - zalando-hd-resized.zip (train + test images)
   
4. Alternative: Use the preprocessed version from:
   https://www.kaggle.com/datasets/marquis03/viton-hd
   
5. Extract the downloaded files to:
   {output_dir}/viton_hd/

6. Run this script again with --skip_download:
   python download_dataset.py --dataset viton-hd --skip_download
""".format(output_dir=output_dir))

    elif dataset_name == "viton":
        print("""
1. Visit the official VITON repository:
   https://github.com/xthan/VITON

2. Download from the provided Google Drive links

3. Extract to:
   {output_dir}/viton/

4. Run this script again with --skip_download:
   python download_dataset.py --dataset viton --skip_download
""".format(output_dir=output_dir))

    else:
        print(f"""
1. Visit: {info.get('manual_url', 'the official repository')}

2. Follow the download instructions

3. Extract to:
   {output_dir}/{dataset_name.replace('-', '_')}/

4. Run this script again with --skip_download:
   python download_dataset.py --dataset {dataset_name} --skip_download
""")

    print("="*60 + "\n")


def setup_viton_structure(data_dir: str):
    """Set up the expected directory structure for VITON dataset."""
    expected_dirs = [
        "train/image",
        "train/cloth",
        "train/cloth-mask",
        "train/image-parse",
        "train/openpose-img",
        "train/openpose-json",
        "test/image",
        "test/cloth",
        "test/cloth-mask",
        "test/image-parse",
        "test/openpose-img",
        "test/openpose-json",
    ]
    
    for dir_path in expected_dirs:
        full_path = os.path.join(data_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        
    print("Directory structure created successfully!")
    print("\nExpected data structure:")
    print("""
    data/viton-hd/
    ├── train/
    │   ├── image/           # Person images
    │   ├── cloth/           # Clothing images
    │   ├── cloth-mask/      # Clothing masks
    │   ├── image-parse/     # Person parsing maps
    │   ├── openpose-img/    # Pose visualization
    │   └── openpose-json/   # Pose keypoints JSON
    ├── test/
    │   └── ... (same structure)
    ├── train_pairs.txt      # Training pairs (image, cloth)
    └── test_pairs.txt       # Testing pairs
    """)


def create_sample_pairs_file(data_dir: str, split: str = "train"):
    """Create a sample pairs file if not exists."""
    pairs_file = os.path.join(data_dir, f"{split}_pairs.txt")
    
    if os.path.exists(pairs_file):
        print(f"{pairs_file} already exists")
        return
        
    image_dir = os.path.join(data_dir, split, "image")
    
    if os.path.exists(image_dir):
        images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        
        with open(pairs_file, 'w') as f:
            for img in images[:100]:  # Create pairs for first 100 images
                # For testing, pair each person with a random cloth
                cloth = img.replace('_0.jpg', '_1.jpg')  # Standard naming convention
                f.write(f"{img} {cloth}\n")
                
        print(f"Created sample pairs file: {pairs_file}")
    else:
        print(f"Warning: {image_dir} does not exist. Please download the dataset first.")


def download_dataset(dataset_name: str, output_dir: str, skip_download: bool = False):
    """Download and set up a dataset."""
    if dataset_name not in DATASET_INFO:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(DATASET_INFO.keys())}")
        return
        
    info = DATASET_INFO[dataset_name]
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Description: {info['description']}")
    print(f"Size: {info['size']}")
    print(f"{'='*60}\n")
    
    dataset_dir = os.path.join(output_dir, dataset_name.replace('-', '_'))
    os.makedirs(dataset_dir, exist_ok=True)
    
    if not skip_download:
        zip_path = os.path.join(output_dir, info['filename'])
        
        # Try automatic download from Google Drive
        download_success = False
        if info.get('gdrive_id'):
            print(f"Attempting automatic download from Google Drive...")
            download_success = download_from_gdrive(info['gdrive_id'], zip_path)
        
        if not download_success:
            # Show manual download instructions
            print_manual_download_instructions(dataset_name, info, output_dir)
            return
        
        # Extract the zip file
        if not extract_zip(zip_path, dataset_dir):
            print_manual_download_instructions(dataset_name, info, output_dir)
            return
        
        # Clean up zip file
        os.remove(zip_path)
        print(f"Cleaned up {zip_path}")
    
    # Set up directory structure
    setup_viton_structure(dataset_dir)
    
    # Create pairs files
    create_sample_pairs_file(dataset_dir, "train")
    create_sample_pairs_file(dataset_dir, "test")
    
    print(f"\n✓ Dataset {dataset_name} is ready at: {dataset_dir}")


def download_pretrained_models(output_dir: str):
    """Download pretrained models for pose estimation and parsing."""
    models_dir = os.path.join(output_dir, "pretrained")
    os.makedirs(models_dir, exist_ok=True)
    
    # Placeholder for pretrained model URLs
    pretrained_models = {
        "openpose": "https://example.com/openpose.pth",
        "human_parse": "https://example.com/human_parse.pth",
    }
    
    print("\nPretrained models should be downloaded separately:")
    print("1. OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose")
    print("2. Human Parsing: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing")
    print("\nAlternatively, use MediaPipe for pose estimation (included in requirements)")


def main():
    parser = argparse.ArgumentParser(description="Download Virtual Try-On datasets")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="viton-hd",
        choices=list(DATASET_INFO.keys()),
        help="Dataset to download"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--setup_only",
        action="store_true",
        help="Only set up directory structure without downloading"
    )
    parser.add_argument(
        "--download_pretrained",
        action="store_true",
        help="Download pretrained models"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    download_dataset(
        args.dataset, 
        args.output_dir,
        skip_download=args.setup_only
    )
    
    if args.download_pretrained:
        download_pretrained_models(args.output_dir)
    
    print("\n" + "="*60)
    print("IMPORTANT NOTES:")
    print("="*60)
    print("""
1. For VITON-HD dataset, you may need to request access from:
   https://github.com/shadow2496/VITON-HD

2. For DressCode dataset, request access from:
   https://github.com/aimagelab/dress-code

3. After downloading, run the preprocessing script:
   python data/preprocess.py --data_dir ./data/viton_hd

4. The preprocessing will generate:
   - Pose keypoints (if not available)
   - Parsing maps (if not available)
   - Agnostic person representations
   - Cloth masks
    """)


if __name__ == "__main__":
    main()
