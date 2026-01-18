"""
Discriminator networks for HR-VITON

Multi-scale discriminator for high-resolution image synthesis.

Reference:
- "High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions" (ECCV 2022)
- "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs" (CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator.
    
    Classifies overlapping patches as real or fake.
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 4,
        norm: str = 'instance',
        use_sigmoid: bool = False
    ):
        super().__init__()
        
        # Normalization layer
        if norm == 'instance':
            NormLayer = nn.InstanceNorm2d
        elif norm == 'batch':
            NormLayer = nn.BatchNorm2d
        else:
            NormLayer = None
            
        # First layer (no normalization)
        sequence = [
            nn.Conv2d(input_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers
        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            
            layers = [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1)]
            if NormLayer:
                layers.append(NormLayer(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            sequence.extend(layers)
            
        # Last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        layers = [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1)]
        if NormLayer:
            layers.append(NormLayer(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        sequence.extend(layers)
        
        # Final prediction
        sequence.append(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))
        
        if use_sigmoid:
            sequence.append(nn.Sigmoid())
            
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NLayerDiscriminatorWithFeatures(nn.Module):
    """
    PatchGAN discriminator that also returns intermediate features.
    
    Used for feature matching loss.
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 4,
        norm: str = 'instance'
    ):
        super().__init__()
        
        self.n_layers = n_layers
        
        # Normalization layer
        if norm == 'instance':
            NormLayer = nn.InstanceNorm2d
        elif norm == 'batch':
            NormLayer = nn.BatchNorm2d
        else:
            NormLayer = nn.Identity
            
        # Build layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        # Intermediate layers
        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            
            self.layers.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1),
                NormLayer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            
        # Second to last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        self.layers.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1),
            NormLayer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        # Final prediction
        self.final = nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning prediction and intermediate features.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            prediction: Patch predictions (B, 1, H', W')
            features: List of intermediate features
        """
        features = []
        
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            
        prediction = self.final(x)
        
        return prediction, features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator.
    
    Applies multiple discriminators at different image scales
    for better high-resolution discrimination.
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 4,
        num_D: int = 2,
        norm: str = 'instance',
        get_features: bool = True
    ):
        super().__init__()
        
        self.num_D = num_D
        self.get_features = get_features
        
        # Create discriminators
        self.discriminators = nn.ModuleList()
        
        for i in range(num_D):
            if get_features:
                self.discriminators.append(
                    NLayerDiscriminatorWithFeatures(input_nc, ndf, n_layers, norm)
                )
            else:
                self.discriminators.append(
                    NLayerDiscriminator(input_nc, ndf, n_layers, norm)
                )
                
        # Downsampler for multi-scale
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Multi-scale forward pass.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            predictions: List of predictions from each scale
            features: List of feature lists from each scale
        """
        predictions = []
        all_features = []
        
        for i, D in enumerate(self.discriminators):
            if self.get_features:
                pred, features = D(x)
                all_features.append(features)
            else:
                pred = D(x)
                
            predictions.append(pred)
            
            # Downsample for next scale
            if i < self.num_D - 1:
                x = self.downsample(x)
                
        return predictions, all_features


class ConditionalDiscriminator(nn.Module):
    """
    Conditional discriminator that takes both image and condition.
    
    Used for semantic-aware discrimination.
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        condition_nc: int = 13,  # segmentation map
        ndf: int = 64,
        n_layers: int = 4,
        num_D: int = 2
    ):
        super().__init__()
        
        # Multi-scale discriminator
        self.disc = MultiScaleDiscriminator(
            input_nc=input_nc + condition_nc,
            ndf=ndf,
            n_layers=n_layers,
            num_D=num_D,
            get_features=True
        )
        
    def forward(
        self,
        image: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Conditional discrimination.
        
        Args:
            image: Image to discriminate (B, 3, H, W)
            condition: Condition (e.g., segmentation) (B, condition_nc, H, W)
            
        Returns:
            predictions and features from multi-scale discrimination
        """
        # Resize condition if needed
        if condition.shape[2:] != image.shape[2:]:
            condition = F.interpolate(condition, size=image.shape[2:], mode='nearest')
            
        # Concatenate and discriminate
        x = torch.cat([image, condition], dim=1)
        return self.disc(x)


class SpectralNormDiscriminator(nn.Module):
    """
    Discriminator with spectral normalization for training stability.
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 4
    ):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(input_nc, ndf, 4, 2, 1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        # Last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)
        ))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_discriminator(
    config: dict,
    device: torch.device = None
) -> nn.Module:
    """Factory function to create discriminator from config."""
    
    disc_type = config.get('type', 'multi_scale')
    input_nc = config.get('input_nc', 3)
    ndf = config.get('ndf', 64)
    n_layers = config.get('n_layers', 4)
    num_D = config.get('num_D', 2)
    
    if disc_type == 'multi_scale':
        disc = MultiScaleDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers,
            num_D=num_D
        )
    elif disc_type == 'conditional':
        condition_nc = config.get('condition_nc', 13)
        disc = ConditionalDiscriminator(
            input_nc=input_nc,
            condition_nc=condition_nc,
            ndf=ndf,
            n_layers=n_layers,
            num_D=num_D
        )
    elif disc_type == 'spectral':
        disc = SpectralNormDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers
        )
    else:
        disc = NLayerDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers
        )
        
    if device:
        disc = disc.to(device)
        
    return disc
