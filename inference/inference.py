"""
Virtual Try-On Inference Pipeline

Unified inference interface for both CP-VTON and HR-VITON methods.

Usage:
    from inference import VITONInference
    
    # For CP-VTON (traditional method)
    model = VITONInference(
        method='cp_vton',
        checkpoint_path='checkpoints/cp_vton/best.pth',
        config_path='configs/cp_vton_config.yaml'
    )
    
    # For HR-VITON (state-of-the-art)
    model = VITONInference(
        method='hr_viton',
        checkpoint_path='checkpoints/hr_viton/best.pth',
        config_path='configs/hr_viton_config.yaml'
    )
    
    # Run inference
    result = model.try_on(
        person_image='path/to/person.jpg',
        cloth_image='path/to/cloth.jpg'
    )
    result.save('output.jpg')
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple, Optional, Dict, Any
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VITONInference:
    """Unified inference pipeline for virtual try-on."""
    
    SUPPORTED_METHODS = ['cp_vton', 'hr_viton']
    
    def __init__(
        self,
        method: str = 'hr_viton',
        checkpoint_path: str = None,
        config_path: str = None,
        device: str = 'auto'
    ):
        """
        Initialize inference pipeline.
        
        Args:
            method: Either 'cp_vton' or 'hr_viton'
            checkpoint_path: Path to model checkpoint
            config_path: Path to model configuration
            device: 'cuda', 'cpu', or 'auto' (default)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
            
        self.method = method
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
            
        # Get image size from config
        model_config = self.config.get('model', {})
        self.image_size = tuple(model_config.get('input_size', [256, 192] if method == 'cp_vton' else [1024, 768]))
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            print("Warning: No checkpoint provided. Model initialized with random weights.")
            
        # Initialize preprocessing modules
        self.person_parser = None
        self.pose_estimator = None
        
    def _get_default_config(self) -> dict:
        """Get default configuration based on method."""
        if self.method == 'cp_vton':
            return {
                'model': {
                    'input_size': [256, 192],
                    'input_channels': 22,
                    'output_channels': 3
                }
            }
        else:  # hr_viton
            return {
                'model': {
                    'input_size': [1024, 768],
                    'condition_generator': {
                        'input_channels': 3,
                        'output_channels': 3
                    },
                    'generator': {
                        'input_channels': 6,
                        'output_channels': 3,
                        'num_spade_blocks': 4
                    }
                }
            }
            
    def _build_model(self):
        """Build model based on method."""
        if self.method == 'cp_vton':
            from models.cp_vton import build_cp_vton
            return build_cp_vton(self.config)
        else:
            from models.hr_viton import build_hr_viton
            return build_hr_viton(self.config)
            
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        print("Checkpoint loaded successfully!")
        
    def _load_image(self, path: str, size: Tuple[int, int] = None) -> np.ndarray:
        """Load and resize image."""
        if size is None:
            size = self.image_size
            
        img = Image.open(path).convert('RGB')
        img = img.resize((size[1], size[0]), Image.LANCZOS)
        return np.array(img)
        
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalized tensor."""
        # HWC to CHW
        img = image.transpose(2, 0, 1).astype(np.float32)
        # Normalize to [-1, 1]
        img = (img / 255.0) * 2 - 1
        # Add batch dimension
        img = torch.from_numpy(img).unsqueeze(0)
        return img.to(self.device)
        
    def _generate_cloth_mask(self, cloth_image: np.ndarray) -> np.ndarray:
        """Generate binary mask for cloth image."""
        # Convert to grayscale
        gray = cv2.cvtColor(cloth_image, cv2.COLOR_RGB2GRAY)
        
        # Threshold to get mask (assuming white background)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
        
    def _generate_agnostic(self, person_image: np.ndarray, parse: np.ndarray = None) -> np.ndarray:
        """Generate agnostic person representation."""
        if parse is None:
            # Simple approximation: mask out the torso area
            h, w = person_image.shape[:2]
            agnostic = person_image.copy()
            
            # Create approximate mask for upper body
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Mask torso region (rough approximation)
            top = int(h * 0.1)
            bottom = int(h * 0.6)
            left = int(w * 0.2)
            right = int(w * 0.8)
            mask[top:bottom, left:right] = 255
            
            # Apply mask
            agnostic[mask > 0] = 128  # Gray color
        else:
            agnostic = self._apply_parse_mask(person_image, parse)
            
        return agnostic
        
    def _apply_parse_mask(self, image: np.ndarray, parse: np.ndarray) -> np.ndarray:
        """Apply parsing mask to create agnostic image."""
        # Clothing labels to mask out (upper body clothes)
        cloth_labels = [5, 6, 7]  # Typical labels for upper clothes
        
        mask = np.zeros(parse.shape[:2], dtype=np.uint8)
        for label in cloth_labels:
            mask[parse == label] = 255
            
        agnostic = image.copy()
        agnostic[mask > 0] = 128
        
        return agnostic
        
    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Convert output tensor to PIL Image."""
        # Remove batch dimension and move to CPU
        img = tensor.squeeze(0).cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        
        # CHW to HWC
        img = img.permute(1, 2, 0).numpy()
        
        # To uint8
        img = (img * 255).astype(np.uint8)
        
        return Image.fromarray(img)
        
    def try_on(
        self,
        person_image: Union[str, np.ndarray, Image.Image],
        cloth_image: Union[str, np.ndarray, Image.Image],
        person_parse: Optional[np.ndarray] = None,
        person_pose: Optional[np.ndarray] = None,
        return_all: bool = False
    ) -> Union[Image.Image, Dict[str, Image.Image]]:
        """
        Perform virtual try-on.
        
        Args:
            person_image: Path to person image or numpy array
            cloth_image: Path to cloth image or numpy array
            person_parse: Optional pre-computed person parsing
            person_pose: Optional pre-computed pose keypoints
            return_all: If True, return all intermediate results
            
        Returns:
            PIL Image of the result, or dict of all outputs if return_all=True
        """
        # Load images if paths provided
        if isinstance(person_image, str):
            person_np = self._load_image(person_image)
        elif isinstance(person_image, Image.Image):
            person_np = np.array(person_image.resize((self.image_size[1], self.image_size[0])))
        else:
            person_np = person_image
            
        if isinstance(cloth_image, str):
            cloth_np = self._load_image(cloth_image)
        elif isinstance(cloth_image, Image.Image):
            cloth_np = np.array(cloth_image.resize((self.image_size[1], self.image_size[0])))
        else:
            cloth_np = cloth_image
            
        # Preprocess
        person = self._preprocess_image(person_np)
        cloth = self._preprocess_image(cloth_np)
        
        # Generate cloth mask
        cloth_mask_np = self._generate_cloth_mask(cloth_np)
        cloth_mask = torch.from_numpy(cloth_mask_np).unsqueeze(0).unsqueeze(0).float()
        cloth_mask = cloth_mask / 255.0
        cloth_mask = cloth_mask.to(self.device)
        
        # Generate agnostic
        agnostic_np = self._generate_agnostic(person_np, person_parse)
        agnostic = self._preprocess_image(agnostic_np)
        
        # Generate parsing (placeholder if not provided)
        if person_parse is not None:
            parse = torch.from_numpy(person_parse).unsqueeze(0).float()
            parse = parse.to(self.device)
        else:
            parse = torch.zeros(1, 20, self.image_size[0], self.image_size[1]).to(self.device)
            
        # Inference
        with torch.no_grad():
            if self.method == 'cp_vton':
                output = self.model.inference(person, cloth, cloth_mask, agnostic, parse)
            else:
                output = self.model(person, cloth, cloth_mask, agnostic, parse)
                
        # Get main output
        result_tensor = output['output'] if isinstance(output, dict) else output
        result = self._postprocess(result_tensor)
        
        if return_all:
            results = {
                'output': result,
                'input_person': Image.fromarray(person_np),
                'input_cloth': Image.fromarray(cloth_np),
                'cloth_mask': Image.fromarray(cloth_mask_np),
                'agnostic': Image.fromarray(agnostic_np)
            }
            
            # Add warped cloth if available
            if isinstance(output, dict) and 'warped_cloth' in output:
                results['warped_cloth'] = self._postprocess(output['warped_cloth'])
                
            return results
        else:
            return result
            
    def batch_try_on(
        self,
        person_images: list,
        cloth_images: list,
        output_dir: str,
        save_individual: bool = True
    ) -> list:
        """
        Batch processing for multiple images.
        
        Args:
            person_images: List of paths to person images
            cloth_images: List of paths to cloth images (same length or single image)
            output_dir: Directory to save results
            save_individual: Save each result individually
            
        Returns:
            List of output PIL Images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle single cloth for all persons
        if len(cloth_images) == 1:
            cloth_images = cloth_images * len(person_images)
            
        assert len(person_images) == len(cloth_images), \
            "Must provide equal number of person and cloth images"
            
        results = []
        
        for i, (person_path, cloth_path) in enumerate(zip(person_images, cloth_images)):
            print(f"Processing {i+1}/{len(person_images)}...")
            
            result = self.try_on(person_path, cloth_path)
            results.append(result)
            
            if save_individual:
                person_name = os.path.splitext(os.path.basename(person_path))[0]
                cloth_name = os.path.splitext(os.path.basename(cloth_path))[0]
                output_path = os.path.join(output_dir, f"{person_name}_{cloth_name}.jpg")
                result.save(output_path)
                
        print(f"Results saved to: {output_dir}")
        return results


def main():
    """Demo inference script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Virtual Try-On Inference')
    parser.add_argument('--method', type=str, default='hr_viton',
                        choices=['cp_vton', 'hr_viton'],
                        help='Model method to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--person', type=str, required=True,
                        help='Path to person image')
    parser.add_argument('--cloth', type=str, required=True,
                        help='Path to cloth image')
    parser.add_argument('--output', type=str, default='output.jpg',
                        help='Output path')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Initialize model
    model = VITONInference(
        method=args.method,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Run inference
    print(f"Processing: {args.person} + {args.cloth}")
    result = model.try_on(args.person, args.cloth)
    
    # Save result
    result.save(args.output)
    print(f"Result saved to: {args.output}")


if __name__ == '__main__':
    main()
