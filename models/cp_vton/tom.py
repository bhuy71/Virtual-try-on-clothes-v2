"""
Try-On Module (TOM) for CP-VTON

The TOM synthesizes the final try-on result by:
1. Blending warped clothing with person image
2. Generating a composition mask
3. Refining details using perceptual losses

Reference:
- "Toward Characteristic-Preserving Image-based Virtual Try-On Network" (ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class UNetDown(nn.Module):
    """U-Net downsampling block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
            
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    """U-Net upsampling block with skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        return torch.cat([x, skip], dim=1)


class TOM(nn.Module):
    """
    Try-On Module
    
    U-Net based generator that synthesizes the final try-on image
    by blending warped clothing with person representation.
    """
    
    def __init__(
        self,
        input_nc: int = 25,  # person (3) + warped_cloth (3) + pose (18) + shape (1)
        output_nc: int = 4,   # RGB (3) + composition mask (1)
        ngf: int = 64,
        n_downsample: int = 6,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        
        # Encoder (downsampling)
        self.down1 = UNetDown(input_nc, ngf, normalize=False)
        self.down2 = UNetDown(ngf, ngf * 2)
        self.down3 = UNetDown(ngf * 2, ngf * 4)
        self.down4 = UNetDown(ngf * 4, ngf * 8)
        self.down5 = UNetDown(ngf * 8, ngf * 8)
        self.down6 = UNetDown(ngf * 8, ngf * 8, normalize=False)
        
        # Decoder (upsampling with skip connections)
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=dropout)
        self.up2 = UNetUp(ngf * 16, ngf * 8, dropout=dropout)
        self.up3 = UNetUp(ngf * 16, ngf * 4, dropout=dropout)
        self.up4 = UNetUp(ngf * 8, ngf * 2)
        self.up5 = UNetUp(ngf * 4, ngf)
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
        )
        
        # Separate heads for RGB and mask
        self.rgb_activation = nn.Tanh()
        self.mask_activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Concatenated input (person + warped_cloth + pose + shape)
            
        Returns:
            Tuple of (rendered_person, composition_mask)
        """
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        # Decoder with skip connections
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        
        # Final output
        output = self.final(u5)
        
        # Split into RGB and mask
        rgb = self.rgb_activation(output[:, :3])
        mask = self.mask_activation(output[:, 3:4])
        
        return rgb, mask


class TOMWithRefinement(nn.Module):
    """
    Enhanced TOM with additional refinement network.
    
    Two-stage approach:
    1. Coarse generation
    2. Detail refinement
    """
    
    def __init__(
        self,
        input_nc: int = 25,
        output_nc: int = 4,
        ngf: int = 64
    ):
        super().__init__()
        
        # Coarse generator
        self.coarse = TOM(input_nc, output_nc, ngf, n_downsample=5)
        
        # Refinement network (takes coarse output + warped cloth)
        refinement_input_nc = 7  # coarse_rgb (3) + warped_cloth (3) + mask (1)
        
        self.refine = nn.Sequential(
            # Encoder
            nn.Conv2d(refinement_input_nc, ngf, 7, 1, 3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            
            # Residual blocks
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            ResidualBlock(ngf * 4),
            
            # Decoder
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf, 3, 7, 1, 3),
            nn.Tanh()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        warped_cloth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (refined_output, coarse_rgb, mask)
        """
        # Coarse generation
        coarse_rgb, mask = self.coarse(x)
        
        # Refinement
        refine_input = torch.cat([coarse_rgb, warped_cloth, mask], dim=1)
        refined = self.refine(refine_input)
        
        return refined, coarse_rgb, mask


class ResidualBlock(nn.Module):
    """Residual block for refinement network."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CompositionModule(nn.Module):
    """
    Final composition module that blends warped cloth with rendered person.
    
    output = mask * warped_cloth + (1 - mask) * rendered_person
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        warped_cloth: torch.Tensor,
        rendered_person: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compose final try-on image.
        
        Args:
            warped_cloth: Warped clothing image (B, 3, H, W)
            rendered_person: Rendered person (B, 3, H, W)
            mask: Composition mask (B, 1, H, W)
            
        Returns:
            Final try-on image (B, 3, H, W)
        """
        return mask * warped_cloth + (1 - mask) * rendered_person


class TOMComplete(nn.Module):
    """
    Complete Try-On Module with composition.
    """
    
    def __init__(
        self,
        input_nc: int = 25,
        ngf: int = 64,
        use_refinement: bool = False
    ):
        super().__init__()
        
        self.use_refinement = use_refinement
        
        if use_refinement:
            self.generator = TOMWithRefinement(input_nc, 4, ngf)
        else:
            self.generator = TOM(input_nc, 4, ngf)
            
        self.composition = CompositionModule()
        
    def forward(
        self,
        agnostic: torch.Tensor,
        warped_cloth: torch.Tensor,
        pose: torch.Tensor,
        parse_shape: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Generate try-on result.
        
        Args:
            agnostic: Agnostic person representation (B, 3, H, W)
            warped_cloth: Warped clothing from GMM (B, 3, H, W)
            pose: Pose heatmaps (B, 18, H, W)
            parse_shape: Body shape mask (B, 1, H, W)
            
        Returns:
            Dictionary with outputs
        """
        # Concatenate inputs
        if parse_shape is not None:
            x = torch.cat([agnostic, warped_cloth, pose, parse_shape], dim=1)
        else:
            x = torch.cat([agnostic, warped_cloth, pose], dim=1)
            
        # Generate
        if self.use_refinement:
            refined, coarse, mask = self.generator(x, warped_cloth)
            
            # Compose
            output = self.composition(warped_cloth, refined, mask)
            
            return {
                'output': output,
                'refined': refined,
                'coarse': coarse,
                'mask': mask,
            }
        else:
            rendered, mask = self.generator(x)
            
            # Compose
            output = self.composition(warped_cloth, rendered, mask)
            
            return {
                'output': output,
                'rendered': rendered,
                'mask': mask,
            }
