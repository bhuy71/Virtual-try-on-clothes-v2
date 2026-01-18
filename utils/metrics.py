"""
Evaluation Metrics for Virtual Try-On

Implements standard metrics used for evaluating virtual try-on quality:
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)
- IS (Inception Score)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple
from PIL import Image
import cv2


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM).
    
    Measures the structural similarity between two images.
    Higher is better (max = 1.0).
    """
    
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
        
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM calculation."""
        def gaussian(size, sigma):
            x = torch.arange(size).float() - size // 2
            gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
            return gauss / gauss.sum()
            
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate SSIM between two images.
        
        Args:
            img1: First image tensor [B, C, H, W]
            img2: Second image tensor [B, C, H, W]
            
        Returns:
            SSIM value [B]
        """
        device = img1.device
        window = self.window.to(device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                   
        return ssim_map.mean(dim=[1, 2, 3])


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    
    Uses pretrained VGG features to measure perceptual similarity.
    Lower is better (min = 0.0).
    """
    
    def __init__(self, net: str = 'vgg'):
        super().__init__()
        
        # Use VGG16 features
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Extract feature layers
        self.slice1 = nn.Sequential(*list(vgg.features[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.features[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.features[9:16])) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg.features[16:23])) # relu4_3
        self.slice5 = nn.Sequential(*list(vgg.features[23:30])) # relu5_3
        
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Linear weights for combining features (from LPIPS paper)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input from [-1, 1] to ImageNet stats."""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        return (x - self.mean) / self.std
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate LPIPS distance.
        
        Args:
            img1: First image tensor [B, C, H, W] in range [-1, 1]
            img2: Second image tensor [B, C, H, W] in range [-1, 1]
            
        Returns:
            LPIPS distance [B]
        """
        x1 = self._normalize(img1)
        x2 = self._normalize(img2)
        
        lpips = 0.0
        
        for i, (slice_fn, weight) in enumerate(zip(
            [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5],
            self.weights
        )):
            x1 = slice_fn(x1)
            x2 = slice_fn(x2)
            
            # Normalize features
            x1_norm = F.normalize(x1, dim=1)
            x2_norm = F.normalize(x2, dim=1)
            
            # Compute difference
            diff = (x1_norm - x2_norm).pow(2)
            lpips += weight * diff.mean(dim=[1, 2, 3])
            
        return lpips


class InceptionFeatureExtractor(nn.Module):
    """Extract features from InceptionV3 for FID calculation."""
    
    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        
        # Use up to the last average pooling layer
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-d features from InceptionV3."""
        # Resize to 299x299
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            
        # Normalize
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = (x - self.mean) / self.std
        
        # Extract features
        features = self.blocks(x)
        return features.squeeze(-1).squeeze(-1)


class FID:
    """
    Fréchet Inception Distance.
    
    Measures the distance between feature distributions of real and generated images.
    Lower is better.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inception = InceptionFeatureExtractor().to(self.device)
        self.inception.eval()
        
    @torch.no_grad()
    def _get_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from images."""
        features = self.inception(images.to(self.device))
        return features.cpu().numpy()
        
    def _calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
        
    def _calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray
    ) -> float:
        """Calculate Fréchet distance between two Gaussians."""
        from scipy import linalg
        
        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
        
    def calculate(
        self,
        real_images: Union[torch.Tensor, List[torch.Tensor]],
        fake_images: Union[torch.Tensor, List[torch.Tensor]]
    ) -> float:
        """
        Calculate FID between real and generated images.
        
        Args:
            real_images: Real image tensor(s)
            fake_images: Generated image tensor(s)
            
        Returns:
            FID score (lower is better)
        """
        # Concatenate if list
        if isinstance(real_images, list):
            real_images = torch.cat(real_images, dim=0)
        if isinstance(fake_images, list):
            fake_images = torch.cat(fake_images, dim=0)
            
        # Get features
        real_features = self._get_features(real_images)
        fake_features = self._get_features(fake_images)
        
        # Calculate statistics
        mu_real, sigma_real = self._calculate_statistics(real_features)
        mu_fake, sigma_fake = self._calculate_statistics(fake_features)
        
        # Calculate FID
        return self._calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


class VITONMetrics:
    """
    Complete evaluation metrics suite for Virtual Try-On.
    
    Usage:
        metrics = VITONMetrics(device='cuda')
        results = metrics.evaluate(generated_images, real_images)
    """
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.ssim = SSIM().to(self.device)
        self.lpips = LPIPS().to(self.device)
        self.fid = FID(self.device)
        
    @torch.no_grad()
    def evaluate(
        self,
        generated: Union[torch.Tensor, List[torch.Tensor]],
        real: Union[torch.Tensor, List[torch.Tensor]]
    ) -> dict:
        """
        Evaluate generated images against real images.
        
        Args:
            generated: Generated image tensor(s) [B, 3, H, W] in range [-1, 1]
            real: Real image tensor(s) [B, 3, H, W] in range [-1, 1]
            
        Returns:
            Dictionary with all metrics
        """
        if isinstance(generated, list):
            generated = torch.cat(generated, dim=0)
        if isinstance(real, list):
            real = torch.cat(real, dim=0)
            
        generated = generated.to(self.device)
        real = real.to(self.device)
        
        # SSIM (higher is better)
        ssim_values = self.ssim(generated, real)
        ssim_mean = ssim_values.mean().item()
        
        # LPIPS (lower is better)
        lpips_values = self.lpips(generated, real)
        lpips_mean = lpips_values.mean().item()
        
        # FID (lower is better)
        fid_value = self.fid.calculate(real, generated)
        
        return {
            'ssim': ssim_mean,
            'lpips': lpips_mean,
            'fid': fid_value,
            'ssim_per_image': ssim_values.cpu().numpy(),
            'lpips_per_image': lpips_values.cpu().numpy()
        }
        
    def evaluate_from_paths(
        self,
        generated_paths: List[str],
        real_paths: List[str],
        image_size: Tuple[int, int] = (256, 256)
    ) -> dict:
        """
        Evaluate from image file paths.
        
        Args:
            generated_paths: List of paths to generated images
            real_paths: List of paths to real images
            image_size: Size to resize images to
            
        Returns:
            Dictionary with all metrics
        """
        def load_images(paths: List[str]) -> torch.Tensor:
            images = []
            for path in paths:
                img = Image.open(path).convert('RGB')
                img = img.resize((image_size[1], image_size[0]), Image.LANCZOS)
                img = np.array(img).transpose(2, 0, 1).astype(np.float32)
                img = (img / 255.0) * 2 - 1  # Normalize to [-1, 1]
                images.append(torch.from_numpy(img))
            return torch.stack(images)
            
        generated = load_images(generated_paths)
        real = load_images(real_paths)
        
        return self.evaluate(generated, real)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate SSIM between two numpy images.
    
    Args:
        img1: First image [H, W, C] in range [0, 255]
        img2: Second image [H, W, C] in range [0, 255]
        
    Returns:
        SSIM value
    """
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=2)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate PSNR between two numpy images.
    
    Args:
        img1: First image [H, W, C] in range [0, 255]
        img2: Second image [H, W, C] in range [0, 255]
        
    Returns:
        PSNR value in dB
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))
