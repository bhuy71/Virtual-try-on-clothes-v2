"""
Loss functions for Virtual Try-On models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    
    Computes L1 loss between VGG features of generated and target images.
    """
    
    def __init__(
        self,
        layers: List[str] = None,
        weights: List[float] = None,
        normalize_input: bool = True
    ):
        super().__init__()
        
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']
        if weights is None:
            weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
            
        self.layers = layers
        self.weights = weights
        self.normalize_input = normalize_input
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True)
        
        # Layer name to index mapping
        layer_indices = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
        }
        
        # Extract feature layers
        self.feature_layers = nn.ModuleList()
        prev_idx = 0
        for layer in layers:
            idx = layer_indices[layer]
            self.feature_layers.append(
                nn.Sequential(*list(vgg.features.children())[prev_idx:idx+1])
            )
            prev_idx = idx + 1
            
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # ImageNet normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input from [-1, 1] to ImageNet normalization."""
        if self.normalize_input:
            x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        return (x - self.mean) / self.std
        
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss."""
        gen_normalized = self.normalize(generated)
        tgt_normalized = self.normalize(target)
        
        loss = 0.0
        gen_feat = gen_normalized
        tgt_feat = tgt_normalized
        
        for i, layer in enumerate(self.feature_layers):
            gen_feat = layer(gen_feat)
            tgt_feat = layer(tgt_feat)
            loss += self.weights[i] * F.l1_loss(gen_feat, tgt_feat)
            
        return loss


class GANLoss(nn.Module):
    """
    GAN loss for generator and discriminator training.
    
    Supports:
    - vanilla: Standard GAN loss (BCE)
    - lsgan: Least Squares GAN loss (MSE)
    - hinge: Hinge loss
    """
    
    def __init__(
        self,
        mode: str = 'hinge',
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0
    ):
        super().__init__()
        
        self.mode = mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        if mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif mode == 'hinge':
            self.loss = None
        else:
            raise ValueError(f"Unknown GAN loss mode: {mode}")
            
    def get_target_tensor(
        self,
        prediction: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """Create target tensor matching prediction shape."""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
        
    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Compute GAN loss.
        
        Args:
            prediction: Discriminator output
            target_is_real: Whether the input is real or fake
            for_discriminator: Whether computing loss for D or G
        """
        if self.mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    loss = F.relu(1.0 - prediction).mean()
                else:
                    loss = F.relu(1.0 + prediction).mean()
            else:
                loss = -prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
            
        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for GAN training.
    
    Computes L1 loss between discriminator features of real and fake images.
    """
    
    def __init__(self, num_D: int = 1, n_layers: int = 3):
        super().__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        
    def forward(
        self,
        fake_features: List[List[torch.Tensor]],
        real_features: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """Compute feature matching loss."""
        loss = 0.0
        n_features = 0
        
        for d in range(self.num_D):
            for i in range(len(fake_features[d])):
                loss += F.l1_loss(fake_features[d][i], real_features[d][i].detach())
                n_features += 1
                
        return loss / n_features


class TVLoss(nn.Module):
    """Total Variation Loss for smoothness."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return (h_tv + w_tv) / batch_size


class WarpingLoss(nn.Module):
    """
    Loss for cloth warping quality.
    
    Combines:
    - L1 loss between warped and target cloth
    - Total variation loss for smooth flow
    - Second-order smoothness loss
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        tv_weight: float = 0.1,
        smooth_weight: float = 0.01
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.tv_weight = tv_weight
        self.smooth_weight = smooth_weight
        self.tv_loss = TVLoss()
        
    def forward(
        self,
        warped_cloth: torch.Tensor,
        target_cloth: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute warping loss."""
        # L1 reconstruction loss
        if mask is not None:
            l1_loss = F.l1_loss(warped_cloth * mask, target_cloth * mask)
        else:
            l1_loss = F.l1_loss(warped_cloth, target_cloth)
            
        loss = self.l1_weight * l1_loss
        
        # Total variation loss on flow
        if flow is not None and self.tv_weight > 0:
            loss += self.tv_weight * self.tv_loss(flow)
            
        return loss


class CompositionLoss(nn.Module):
    """
    Loss for image composition quality.
    
    Computes loss between:
    - Composite output and ground truth
    - Individual components (warped cloth, rendered person)
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        mask_weight: float = 1.0
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.mask_weight = mask_weight
        
        self.perceptual_loss = VGGPerceptualLoss()
        
    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        composition_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute composition loss."""
        # L1 loss
        loss = self.l1_weight * F.l1_loss(output, target)
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            loss += self.perceptual_weight * self.perceptual_loss(output, target)
            
        # Mask loss (if masks provided)
        if composition_mask is not None and target_mask is not None:
            if self.mask_weight > 0:
                loss += self.mask_weight * F.l1_loss(composition_mask, target_mask)
                
        return loss


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss."""
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
        
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / (2 * sigma**2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        channel = img1.shape[1]
        
        if channel == self.channel and self.window.device == img1.device:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel).to(img1.device)
            
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                   
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


import numpy as np  # Add this import at the top
