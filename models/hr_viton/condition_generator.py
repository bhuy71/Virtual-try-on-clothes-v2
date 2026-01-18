"""
Condition Generator for HR-VITON

Generates segmentation map and misalignment-aware conditions
for the high-resolution try-on generation.

Reference:
- "High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions" (ECCV 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResBlock(nn.Module):
    """Residual block with optional normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = 'instance'
    ):
        super().__init__()
        
        self.learned_shortcut = (in_channels != out_channels)
        
        # Normalization
        if norm == 'instance':
            NormLayer = nn.InstanceNorm2d
        elif norm == 'batch':
            NormLayer = nn.BatchNorm2d
        else:
            NormLayer = nn.Identity
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = NormLayer(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = NormLayer(out_channels)
        
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, 1)
            
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            shortcut = self.conv_s(x)
        else:
            shortcut = x
            
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        return self.act(out + shortcut)


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(self, in_channels: int, out_channels: int, norm: str = 'instance'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
        
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
            
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    """Upsampling block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = 'instance',
        use_dropout: bool = False
    ):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
            
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            
        return self.dropout(self.act(self.norm(self.conv(x))))


class AppearanceFlowEstimator(nn.Module):
    """
    Appearance Flow Estimation Network
    
    Estimates dense flow field to warp clothing to match person's pose.
    """
    
    def __init__(
        self,
        input_nc: int = 6,  # agnostic (3) + cloth (3)
        ngf: int = 64,
        n_downsample: int = 4
    ):
        super().__init__()
        
        # Encoder
        self.down_layers = nn.ModuleList()
        
        # Initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 7, 1, 3),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Downsampling
        in_ch = ngf
        for i in range(n_downsample):
            out_ch = min(in_ch * 2, 512)
            self.down_layers.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(in_ch, in_ch),
            ResBlock(in_ch, in_ch),
            ResBlock(in_ch, in_ch),
        )
        
        # Decoder with skip connections
        self.up_layers = nn.ModuleList()
        for i in range(n_downsample):
            out_ch = max(in_ch // 2, ngf)
            # Skip connection doubles input channels (current + skip from encoder)
            skip_ch = in_ch if i < n_downsample - 1 else 0
            self.up_layers.append(UpBlock(in_ch + skip_ch, out_ch))
            in_ch = out_ch
            
        # Flow prediction
        self.flow_conv = nn.Conv2d(ngf, 2, 7, 1, 3)
        
    def forward(
        self,
        agnostic: torch.Tensor,
        cloth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate appearance flow.
        
        Returns:
            flow: Dense flow field (B, 2, H, W)
            warped_cloth: Warped clothing (B, 3, H, W)
        """
        x = torch.cat([agnostic, cloth], dim=1)
        
        # Encoder
        x = self.initial(x)
        skips = [x]
        
        for down in self.down_layers:
            x = down(x)
            skips.append(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections from encoder
        for i, up in enumerate(self.up_layers):
            # Get corresponding skip connection from encoder
            skip_idx = -(i + 2)  # Skip from encoder (reverse order)
            skip = skips[skip_idx] if -skip_idx <= len(skips) else None
            
            # Concatenate with skip if available
            if skip is not None and i < len(self.up_layers) - 1:
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            
            # Apply upsampling block
            x = up(x, None)  # Skip already concatenated above
                
        # Predict flow
        flow = self.flow_conv(x)
        flow = torch.tanh(flow) * 2  # Scale flow to reasonable range
        
        # Warp cloth using flow
        warped_cloth = self.warp(cloth, flow)
        
        return flow, warped_cloth
        
    def warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp image using flow field."""
        b, c, h, w = x.shape
        
        # Resize flow to match input dimensions if needed
        if flow.shape[2:] != (h, w):
            flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=True)
        
        # Create sampling grid
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, w, device=x.device),
            torch.linspace(-1, 1, h, device=x.device),
            indexing='xy'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        grid = grid.expand(b, -1, -1, -1)  # (B, H, W, 2)
        
        # Add flow to grid
        flow = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        sample_grid = grid + flow
        
        # Sample
        warped = F.grid_sample(
            x, sample_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped


class SegmentationGenerator(nn.Module):
    """
    Segmentation Map Generator
    
    Predicts the target segmentation map for the try-on result.
    """
    
    def __init__(
        self,
        input_nc: int = 16,  # agnostic (3) + parse (13)
        output_nc: int = 13,
        ngf: int = 64,
        n_downsample: int = 4,
        n_blocks: int = 6
    ):
        super().__init__()
        
        # Encoder
        layers = [
            nn.Conv2d(input_nc, ngf, 7, 1, 3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_ch = ngf
        for i in range(n_downsample):
            out_ch = min(in_ch * 2, 512)
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
            
        # Residual blocks
        for i in range(n_blocks):
            layers.append(ResBlock(in_ch, in_ch))
            
        # Upsampling
        for i in range(n_downsample):
            out_ch = max(in_ch // 2, ngf)
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
            
        # Output
        layers += [
            nn.Conv2d(ngf, output_nc, 7, 1, 3),
        ]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate segmentation map.
        
        Args:
            x: Concatenated input (agnostic + warped_cloth + parse)
            
        Returns:
            Predicted segmentation logits (B, output_nc, H, W)
        """
        return self.model(x)


class ConditionGenerator(nn.Module):
    """
    Complete Condition Generator for HR-VITON
    
    Generates:
    1. Warped clothing via appearance flow
    2. Target segmentation map
    3. Misalignment-aware mask
    """
    
    def __init__(
        self,
        semantic_nc: int = 13,
        ngf: int = 64
    ):
        super().__init__()
        
        # Appearance flow for cloth warping
        self.flow_estimator = AppearanceFlowEstimator(
            input_nc=3 + 3,  # agnostic + cloth
            ngf=ngf
        )
        
        # Segmentation generator
        # Input: agnostic (3) + warped_cloth (3) + parse (semantic_nc)
        self.seg_generator = SegmentationGenerator(
            input_nc=3 + 3 + semantic_nc,
            output_nc=semantic_nc,
            ngf=ngf
        )
        
        # Misalignment mask predictor
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(semantic_nc + 3, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        agnostic: torch.Tensor,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
        parse: torch.Tensor
    ) -> dict:
        """
        Generate conditions for image synthesis.
        
        Args:
            agnostic: Agnostic person representation (B, 3, H, W)
            cloth: Clothing image (B, 3, H, W)
            cloth_mask: Clothing mask (B, 1, H, W)
            parse: Person parsing map (B, semantic_nc, H, W)
            
        Returns:
            Dictionary with warped_cloth, flow, seg, and mask
        """
        # Estimate appearance flow and warp cloth
        flow, warped_cloth = self.flow_estimator(agnostic, cloth)
        
        # Warp mask as well
        warped_mask = self.flow_estimator.warp(cloth_mask, flow)
        
        # Generate segmentation
        seg_input = torch.cat([agnostic, warped_cloth, parse], dim=1)
        seg_logits = self.seg_generator(seg_input)
        seg = F.softmax(seg_logits, dim=1)
        
        # Predict misalignment mask
        mask_input = torch.cat([seg, warped_cloth], dim=1)
        misalign_mask = self.mask_predictor(mask_input)
        
        return {
            'warped_cloth': warped_cloth,
            'warped_mask': warped_mask,
            'flow': flow,
            'seg': seg,
            'seg_logits': seg_logits,
            'misalign_mask': misalign_mask,
        }
