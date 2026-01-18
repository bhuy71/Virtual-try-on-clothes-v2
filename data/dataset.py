"""
PyTorch Dataset classes for Virtual Try-On

Supports:
- VITON dataset format
- VITON-HD dataset format
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Dict, Tuple, Optional, List


class VITONDataset(Dataset):
    """
    Dataset for VITON/VITON-HD virtual try-on.
    
    Expected directory structure:
    data_root/
    ├── train/
    │   ├── image/           # Person images
    │   ├── cloth/           # Clothing images
    │   ├── cloth-mask/      # Clothing masks
    │   ├── image-parse/     # Person parsing maps
    │   ├── agnostic/        # Agnostic person (optional)
    │   ├── openpose-img/    # Pose visualization
    │   └── openpose-json/   # Pose keypoints
    ├── test/
    │   └── ... (same structure)
    ├── train_pairs.txt
    └── test_pairs.txt
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (256, 192),
        transform: Optional[transforms.Compose] = None,
        semantic_nc: int = 13,
    ):
        """
        Args:
            data_root: Root directory of the dataset
            split: 'train' or 'test'
            image_size: Target size (height, width)
            transform: Optional transforms to apply
            semantic_nc: Number of semantic classes
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.semantic_nc = semantic_nc
        
        # Set up directories - support both standard and VITON-HD v3 naming
        self.image_dir = os.path.join(data_root, split, 'image')
        self.cloth_dir = os.path.join(data_root, split, 'cloth')
        self.cloth_mask_dir = os.path.join(data_root, split, 'cloth-mask')
        
        # Parse directory: try 'image-parse-v3' first, fallback to 'image-parse'
        parse_v3 = os.path.join(data_root, split, 'image-parse-v3')
        self.parse_dir = parse_v3 if os.path.exists(parse_v3) else os.path.join(data_root, split, 'image-parse')
        
        # Agnostic directory: try 'agnostic-v3.2' first, fallback to 'agnostic'
        agnostic_v3 = os.path.join(data_root, split, 'agnostic-v3.2')
        self.agnostic_dir = agnostic_v3 if os.path.exists(agnostic_v3) else os.path.join(data_root, split, 'agnostic')
        
        self.pose_dir = os.path.join(data_root, split, 'openpose_json')
        
        # Load pairs
        pairs_file = os.path.join(data_root, f'{split}_pairs.txt')
        self.pairs = self._load_pairs(pairs_file)
        
        # Data augmentation flag
        self.augment = (split == 'train')
        
        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # Augmentation transforms (applied before main transform)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        )
            
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
    def _load_pairs(self, pairs_file: str) -> List[Tuple[str, str]]:
        """
        Load image-cloth pairs.
        
        For TRAINING: Use paired data (same person-cloth ID) for supervised learning.
        For INFERENCE: Can use unpaired data (different person-cloth).
        
        VITON-HD structure:
        - image/00001_00.jpg (person wearing cloth)
        - cloth/00001_00.jpg (same cloth, extracted)
        """
        pairs = []
        
        # For training: Generate PAIRED data (same ID) instead of using unpaired pairs file
        # This is crucial for supervised learning with ground truth
        if self.split == 'train':
            # Generate paired data: each person with their own cloth
            if os.path.exists(self.image_dir):
                for img_file in sorted(os.listdir(self.image_dir)):
                    if img_file.endswith(('.jpg', '.png')):
                        # Use same filename for cloth (paired)
                        cloth_file = img_file
                        # Check if cloth exists
                        cloth_path = os.path.join(self.cloth_dir, cloth_file)
                        if os.path.exists(cloth_path):
                            pairs.append((img_file, cloth_file))
                print(f"Generated {len(pairs)} paired samples for training")
        else:
            # For test/inference: Use pairs file (can be unpaired)
            if os.path.exists(pairs_file):
                with open(pairs_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            pairs.append((parts[0], parts[1]))
            else:
                # Fallback to paired
                if os.path.exists(self.image_dir):
                    for img_file in sorted(os.listdir(self.image_dir)):
                        if img_file.endswith(('.jpg', '.png')):
                            pairs.append((img_file, img_file))
                        
        return pairs
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def _load_pose(self, pose_path: str) -> torch.Tensor:
        """Load pose keypoints and convert to heatmaps."""
        h, w = self.image_size
        n_keypoints = 18
        pose_heatmaps = torch.zeros(n_keypoints, h, w)
        
        if os.path.exists(pose_path):
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
                
            if 'people' in pose_data and len(pose_data['people']) > 0:
                keypoints = pose_data['people'][0].get('pose_keypoints_2d', [])
                
                # Reshape to (n_keypoints, 3)
                keypoints = np.array(keypoints).reshape(-1, 3)
                
                # Generate Gaussian heatmaps
                sigma = 6
                for i, (x, y, conf) in enumerate(keypoints[:n_keypoints]):
                    if conf > 0.1:
                        # Normalize coordinates to target size
                        x = int(x * w / 768)  # Assuming original width 768
                        y = int(y * h / 1024)  # Assuming original height 1024
                        
                        if 0 <= x < w and 0 <= y < h:
                            x_grid, y_grid = torch.meshgrid(
                                torch.arange(w), torch.arange(h), indexing='xy'
                            )
                            pose_heatmaps[i] = torch.exp(
                                -((x_grid - x).float()**2 + (y_grid - y).float()**2) / (2 * sigma**2)
                            )
                            
        return pose_heatmaps
        
    def _load_parse(self, parse_path: str) -> torch.Tensor:
        """
        Load parsing map and convert to one-hot encoding.
        
        Supports both:
        - Grayscale (L mode) with direct label values
        - Palette (P mode) where pixel values are palette indices
        
        VITON-HD v3 ATR parsing labels:
        0: Background, 2: Hair, 5: Upper-clothes, 9: Face, 
        10: Left-arm, 13: Left-leg, 14: Right-leg, 15: Left-shoe, etc.
        
        We use a fixed size of 20 channels to cover all possible labels.
        """
        h, w = self.image_size
        
        # Fixed number of channels to ensure consistent tensor size across batch
        # Use 20 to cover labels 0-19 (VITON-HD v3 uses labels up to ~15)
        num_channels = 20
        
        if os.path.exists(parse_path):
            parse_img = Image.open(parse_path)
            
            # Handle palette mode - don't convert to L, keep indices
            if parse_img.mode == 'P':
                parse_img = parse_img.resize((w, h), Image.NEAREST)
                parse_array = np.array(parse_img)  # This gives palette indices directly
            else:
                parse_img = parse_img.convert('L')
                parse_img = parse_img.resize((w, h), Image.NEAREST)
                parse_array = np.array(parse_img)
        else:
            parse_array = np.zeros((h, w), dtype=np.uint8)
            
        # Convert to one-hot encoding with fixed size
        parse_onehot = torch.zeros(num_channels, h, w)
        unique_labels = np.unique(parse_array)
        for label in unique_labels:
            if label < num_channels:
                parse_onehot[label] = torch.from_numpy((parse_array == label).astype(np.float32))
            
        return parse_onehot
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        image_name, cloth_name = self.pairs[idx]
        
        # Random horizontal flip for augmentation
        do_flip = self.augment and torch.rand(1).item() > 0.5
        
        # Load person image
        image_path = os.path.join(self.image_dir, image_name)
        person_image = Image.open(image_path).convert('RGB')
        if do_flip:
            person_image = person_image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.augment:
            person_image = self.color_jitter(person_image)
        person = self.transform(person_image)
        
        # Load cloth image
        cloth_path = os.path.join(self.cloth_dir, cloth_name)
        if os.path.exists(cloth_path):
            cloth_image = Image.open(cloth_path).convert('RGB')
        else:
            cloth_image = Image.new('RGB', self.image_size[::-1], (255, 255, 255))
        if do_flip:
            cloth_image = cloth_image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.augment:
            cloth_image = self.color_jitter(cloth_image)
        cloth = self.transform(cloth_image)
        
        # Load cloth mask
        cloth_mask_path = os.path.join(
            self.cloth_mask_dir, 
            cloth_name.replace('.jpg', '.png').replace('.JPG', '.png')
        )
        if os.path.exists(cloth_mask_path):
            cloth_mask = Image.open(cloth_mask_path).convert('L')
            if do_flip:
                cloth_mask = cloth_mask.transpose(Image.FLIP_LEFT_RIGHT)
            cloth_mask = self.mask_transform(cloth_mask)
        else:
            cloth_mask = torch.ones(1, *self.image_size)
            
        # Load agnostic representation
        agnostic_path = os.path.join(self.agnostic_dir, image_name)
        if os.path.exists(agnostic_path):
            agnostic = Image.open(agnostic_path).convert('RGB')
            if do_flip:
                agnostic = agnostic.transpose(Image.FLIP_LEFT_RIGHT)
            agnostic = self.transform(agnostic)
        else:
            agnostic = person.clone()
            
        # Load pose
        pose_path = os.path.join(
            self.pose_dir,
            image_name.replace('.jpg', '.json').replace('.png', '.json')
        )
        pose = self._load_pose(pose_path)
        if do_flip:
            pose = torch.flip(pose, dims=[2])  # Flip horizontally
            # Also swap left/right keypoints (indices need to be swapped)
        
        # Load parsing
        parse_path = os.path.join(
            self.parse_dir,
            image_name.replace('.jpg', '.png')
        )
        parse = self._load_parse(parse_path)
        if do_flip:
            parse = torch.flip(parse, dims=[2])  # Flip horizontally
        
        return {
            'image': person,
            'cloth': cloth,
            'cloth_mask': cloth_mask,
            'agnostic': agnostic,
            'pose': pose,
            'parse': parse,
            'image_name': image_name,
            'cloth_name': cloth_name,
        }


class VITONHDDataset(VITONDataset):
    """
    Dataset for VITON-HD with high-resolution support.
    Extends VITONDataset with additional processing for high-res images.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (1024, 768),
        **kwargs
    ):
        super().__init__(data_root, split, image_size, **kwargs)
        
        # Additional directories for HR-VITON
        self.densepose_dir = os.path.join(data_root, split, 'densepose')
        
    def _load_densepose(self, densepose_path: str) -> torch.Tensor:
        """Load DensePose representation."""
        h, w = self.image_size
        
        if os.path.exists(densepose_path):
            densepose = Image.open(densepose_path).convert('RGB')
            densepose = densepose.resize((w, h), Image.BILINEAR)
            densepose = transforms.ToTensor()(densepose)
        else:
            densepose = torch.zeros(3, h, w)
            
        return densepose
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with additional high-resolution features."""
        sample = super().__getitem__(idx)
        
        # Load DensePose if available
        image_name = sample['image_name']
        densepose_path = os.path.join(
            self.densepose_dir,
            image_name.replace('.jpg', '.png')
        )
        sample['densepose'] = self._load_densepose(densepose_path)
        
        return sample


def get_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 192),
    shuffle: bool = True,
    high_res: bool = False
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for virtual try-on dataset."""
    
    if high_res:
        dataset = VITONHDDataset(data_root, split, image_size)
    else:
        dataset = VITONDataset(data_root, split, image_size)
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


# Test the dataset
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Test with dummy data
    data_root = './data/viton'
    
    if os.path.exists(data_root):
        dataset = VITONDataset(data_root, split='train')
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print("Sample keys:", sample.keys())
            print("Image shape:", sample['image'].shape)
            print("Cloth shape:", sample['cloth'].shape)
            print("Pose shape:", sample['pose'].shape)
            print("Parse shape:", sample['parse'].shape)
    else:
        print(f"Data root not found: {data_root}")
        print("Please download the dataset first using: python data/download_dataset.py")
