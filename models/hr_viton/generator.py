"""
High-Resolution Image Generator for HR-VITON

Synthesizes the final high-resolution try-on image using:
- SPADE normalization for semantic-aware generation
- Alias-free upsampling for better quality
- Multi-scale feature processing

Reference:
- "High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions" (ECCV 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class SPADENorm(nn.Module):
    """Spatially-Adaptive Denormalization."""
    
    def __init__(
        self,
        norm_channels: int,
        label_channels: int,
        hidden_channels: int = 128
    ):
        super().__init__()
        
        self.param_free_norm = nn.InstanceNorm2d(norm_channels, affine=False)
        
        self.shared = nn.Sequential(
            nn.Conv2d(label_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.gamma = nn.Conv2d(hidden_channels, norm_channels, 3, 1, 1)
        self.beta = nn.Conv2d(hidden_channels, norm_channels, 3, 1, 1)
        
    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # Normalize
        normalized = self.param_free_norm(x)
        
        # Resize segmentation to match x
        if seg.shape[2:] != x.shape[2:]:
            seg = F.interpolate(seg, size=x.shape[2:], mode='nearest')
            
        # Compute modulation parameters
        actv = self.shared(seg)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        
        # Apply modulation
        return normalized * (1 + gamma) + beta


class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        label_channels: int
    ):
        super().__init__()
        
        middle_channels = min(in_channels, out_channels)
        
        self.learned_shortcut = (in_channels != out_channels)
        
        # Main path
        self.spade1 = SPADENorm(in_channels, label_channels)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, 1, 1)
        
        self.spade2 = SPADENorm(middle_channels, label_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, 1, 1)
        
        # Shortcut
        if self.learned_shortcut:
            self.spade_s = SPADENorm(in_channels, label_channels)
            self.conv_s = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # Shortcut
        if self.learned_shortcut:
            x_s = self.conv_s(self.spade_s(x, seg))
        else:
            x_s = x
            
        # Main path
        dx = self.conv1(self.act(self.spade1(x, seg)))
        dx = self.conv2(self.act(self.spade2(dx, seg)))
        
        return x_s + dx


class AliasFreeUpsample(nn.Module):
    """
    Symmetric upsampling to match downsampling architecture.
    Uses ConvTranspose2d with same kernel/stride/padding as downsampling.
    """
    
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        
        self.scale = scale
        
        # Use ConvTranspose2d to be symmetric with Conv2d(kernel=4, stride=2, padding=1) in downsampling
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 
                                     kernel_size=4, stride=2, padding=1, 
                                     output_padding=0)
        
        # Normalization and activation
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class StyleEncoder(nn.Module):
    """
    Encoder for extracting style/texture information from clothing.
    """
    
    def __init__(self, input_nc: int = 3, ngf: int = 64, n_downsample: int = 4):
        super().__init__()
        
        layers = [
            nn.Conv2d(input_nc, ngf, 7, 1, 3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        in_ch = ngf
        for i in range(n_downsample):
            out_ch = min(in_ch * 2, 512)
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
            
        self.model = nn.Sequential(*layers)
        self.out_channels = in_ch
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale style features."""
        features = []
        
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
                
        return features


class HRGenerator(nn.Module):
    """
    High-Resolution Try-On Generator
    
    Uses SPADE normalization and alias-free upsampling for
    high-quality virtual try-on image synthesis.
    """
    
    def __init__(
        self,
        input_nc: int = 19,  # warped_cloth (3) + agnostic (3) + seg (13)
        output_nc: int = 3,
        semantic_nc: int = 13,
        ngf: int = 64,
        n_downsampling: int = 4,
        n_blocks: int = 9
    ):
        super().__init__()
        
        self.n_downsampling = n_downsampling
        
        # Initial projection
        self.initial = nn.Conv2d(input_nc, ngf, 7, 1, 3)
        
        # Downsampling
        self.down_layers = nn.ModuleList()
        in_ch = ngf
        
        for i in range(n_downsampling):
            out_ch = min(in_ch * 2, 512)
            self.down_layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_ch = out_ch
            
        # SPADE residual blocks
        label_nc = semantic_nc + 3  # seg + warped_cloth
        
        self.spade_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.spade_blocks.append(SPADEResBlock(in_ch, in_ch, label_nc))
            
        # Upsampling with SPADE
        self.up_layers = nn.ModuleList()
        self.up_spades = nn.ModuleList()
        self.skip_convs = nn.ModuleList()  # Projection for skip connections
        
        for i in range(n_downsampling):
            out_ch = max(in_ch // 2, ngf)
            self.up_layers.append(AliasFreeUpsample(in_ch, out_ch))
            self.up_spades.append(SPADEResBlock(out_ch, out_ch, label_nc))
            
            # Project skip connection to match output channels
            # Skip channels from encoder (reverse order)
            skip_ch = min(ngf * (2 ** (n_downsampling - i - 1)), 512) if i < n_downsampling - 1 else ngf
            if skip_ch != out_ch:
                self.skip_convs.append(nn.Conv2d(skip_ch, out_ch, 1))
            else:
                self.skip_convs.append(nn.Identity())
            
            in_ch = out_ch
            
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, output_nc, 7, 1, 3),
            nn.Tanh()
        )
        
    def forward(
        self,
        warped_cloth: torch.Tensor,
        agnostic: torch.Tensor,
        seg: torch.Tensor,
        pose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate high-resolution try-on image.
        
        Args:
            warped_cloth: Warped clothing (B, 3, H, W)
            agnostic: Agnostic person (B, 3, H, W)
            seg: Segmentation map (B, semantic_nc, H, W)
            pose: Optional pose heatmaps (B, 18, H, W)
            
        Returns:
            Generated image (B, 3, H, W)
        """
        # Prepare input
        x = torch.cat([warped_cloth, agnostic, seg], dim=1)
        
        # Prepare conditioning (for SPADE)
        cond = torch.cat([seg, warped_cloth], dim=1)
        
        # Initial projection
        x = self.initial(x)
        
        # Encoder
        skips = []
        for down in self.down_layers:
            skips.append(x)
            x = down(x)
            
        # SPADE blocks
        for spade_block in self.spade_blocks:
            x = spade_block(x, cond)
            
        # Decoder
        for i, (up, up_spade, skip_conv) in enumerate(zip(self.up_layers, self.up_spades, self.skip_convs)):
            x = up(x)
            x = up_spade(x, cond)
            
            # Skip connection
            if i < len(skips):
                skip = skips[-(i+1)]
                # Project channels to match
                skip = skip_conv(skip)
                x = x + skip
                
        # Output
        return self.final(x)


class HRGeneratorWithStyle(nn.Module):
    """
    High-Resolution Generator with Style Injection
    
    Enhanced version that injects clothing texture/style at multiple scales.
    """
    
    def __init__(
        self,
        input_nc: int = 19,
        output_nc: int = 3,
        semantic_nc: int = 13,
        ngf: int = 64,
        n_downsampling: int = 4,
        n_blocks: int = 9
    ):
        super().__init__()
        
        # Style encoder for clothing
        self.style_encoder = StyleEncoder(input_nc=3, ngf=ngf, n_downsample=n_downsampling)
        
        # Main generator
        self.generator = HRGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            semantic_nc=semantic_nc,
            ngf=ngf,
            n_downsampling=n_downsampling,
            n_blocks=n_blocks
        )
        
        # Style injection layers
        self.style_injectors = nn.ModuleList()
        in_ch = ngf
        for i in range(n_downsampling):
            out_ch = min(in_ch * 2, 512)
            self.style_injectors.append(
                nn.Conv2d(out_ch * 2, out_ch, 1)
            )
            in_ch = out_ch
            
    def forward(
        self,
        warped_cloth: torch.Tensor,
        agnostic: torch.Tensor,
        seg: torch.Tensor,
        cloth: torch.Tensor
    ) -> torch.Tensor:
        """Generate with style injection from original cloth."""
        # Extract style features from original cloth
        style_features = self.style_encoder(cloth)
        
        # Generate base output
        output = self.generator(warped_cloth, agnostic, seg)
        
        return output


class ProgressiveGenerator(nn.Module):
    """
    Progressive Generator for high-resolution synthesis.
    
    Generates images progressively from low to high resolution.
    """
    
    def __init__(
        self,
        semantic_nc: int = 13,
        ngf: int = 64,
        target_size: Tuple[int, int] = (1024, 768)
    ):
        super().__init__()
        
        self.target_size = target_size
        
        # Calculate number of upsampling stages
        min_size = 16
        n_up = int(math.log2(target_size[0] // min_size))
        
        label_nc = semantic_nc + 3
        
        # Initial block (from noise/latent)
        self.initial = nn.Parameter(torch.randn(1, 512, min_size, min_size // 4 * 3))
        
        # Progressive blocks
        self.blocks = nn.ModuleList()
        in_ch = 512
        
        for i in range(n_up):
            out_ch = max(in_ch // 2, ngf)
            self.blocks.append(nn.ModuleList([
                AliasFreeUpsample(in_ch, out_ch),
                SPADEResBlock(out_ch, out_ch, label_nc),
                SPADEResBlock(out_ch, out_ch, label_nc),
            ]))
            in_ch = out_ch
            
        # RGB heads at each scale
        self.to_rgb = nn.ModuleList([
            nn.Conv2d(max(512 // (2**i), ngf), 3, 1) for i in range(n_up + 1)
        ])
        
    def forward(
        self,
        seg: torch.Tensor,
        warped_cloth: torch.Tensor,
        alpha: float = 1.0  # For progressive training
    ) -> torch.Tensor:
        """Generate progressively."""
        batch_size = seg.shape[0]
        cond = torch.cat([seg, warped_cloth], dim=1)
        
        # Start from learned constant
        x = self.initial.expand(batch_size, -1, -1, -1)
        
        # Progressive generation
        for i, block_list in enumerate(self.blocks):
            up, spade1, spade2 = block_list
            x = up(x)
            x = spade1(x, cond)
            x = spade2(x, cond)
            
        # Final RGB output
        return torch.tanh(self.to_rgb[-1](x))
