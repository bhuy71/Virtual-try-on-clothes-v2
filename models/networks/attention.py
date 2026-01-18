"""
Attention mechanisms for Virtual Try-On models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelfAttention(nn.Module):
    """Self-attention mechanism for feature refinement."""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.shape
        
        # Compute query, key, value
        q = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h * w)
        v = self.value(x).view(batch, -1, h * w)
        
        # Attention weights
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch, c, h, w)
        
        return self.gamma * out + x


class CrossAttention(nn.Module):
    """Cross-attention between two feature maps."""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        reduced_channels = max(in_channels // reduction, 1)
        
        self.query = nn.Conv2d(in_channels, reduced_channels, 1)
        self.key = nn.Conv2d(in_channels, reduced_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Query feature map (B, C, H, W)
            context: Key/Value feature map (B, C, H', W')
        """
        batch, c, h, w = x.shape
        _, _, h_ctx, w_ctx = context.shape
        
        # Compute query from x, key/value from context
        q = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.key(context).view(batch, -1, h_ctx * w_ctx)
        v = self.value(context).view(batch, -1, h_ctx * w_ctx)
        
        # Attention weights
        attn = torch.bmm(q, k) / math.sqrt(q.shape[-1])
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch, c, h, w)
        
        return self.gamma * out + x


class ChannelAttention(nn.Module):
    """Channel attention module (Squeeze-and-Excitation)."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, _, _ = x.shape
        
        # Average and max pooling
        avg_out = self.avg_pool(x).view(batch, c)
        max_out = self.max_pool(x).view(batch, c)
        
        # Channel attention
        attn = self.fc(avg_out) + self.fc(max_out)
        attn = attn.view(batch, c, 1, 1)
        
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)."""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class FlowAttention(nn.Module):
    """
    Attention module for appearance flow estimation.
    
    Computes correspondence between source (cloth) and target (person) features.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Feature projection
        self.source_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.target_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-based correspondence.
        
        Args:
            source: Source features (cloth) - (B, C, Hs, Ws)
            target: Target features (person) - (B, C, Ht, Wt)
            
        Returns:
            Warped source features and attention weights
        """
        batch, c, hs, ws = source.shape
        _, _, ht, wt = target.shape
        
        # Project features
        src_feat = self.source_proj(source)
        tgt_feat = self.target_proj(target)
        
        # Reshape for attention computation
        src_feat = src_feat.view(batch, -1, hs * ws)  # (B, C, Hs*Ws)
        tgt_feat = tgt_feat.view(batch, -1, ht * wt)  # (B, C, Ht*Wt)
        
        # Normalize for cosine similarity
        src_feat = F.normalize(src_feat, dim=1)
        tgt_feat = F.normalize(tgt_feat, dim=1)
        
        # Compute attention: (B, Ht*Wt, Hs*Ws)
        attn = torch.bmm(tgt_feat.permute(0, 2, 1), src_feat) / self.temperature
        attn = F.softmax(attn, dim=-1)
        
        # Warp source features using attention
        source_flat = source.view(batch, c, hs * ws)  # (B, C, Hs*Ws)
        warped = torch.bmm(source_flat, attn.permute(0, 2, 1))  # (B, C, Ht*Wt)
        warped = warped.view(batch, c, ht, wt)
        
        # Reshape attention for visualization
        attn_map = attn.view(batch, ht, wt, hs, ws)
        
        return warped, attn_map


class MultiScaleAttention(nn.Module):
    """Multi-scale attention for handling different receptive fields."""
    
    def __init__(self, in_channels: int, scales: list = [1, 2, 4]):
        super().__init__()
        
        self.scales = scales
        self.attentions = nn.ModuleList([
            SelfAttention(in_channels) for _ in scales
        ])
        
        self.fusion = nn.Conv2d(in_channels * len(scales), in_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.shape
        
        outputs = []
        for scale, attn in zip(self.scales, self.attentions):
            if scale > 1:
                # Downsample
                x_scaled = F.avg_pool2d(x, scale)
                # Apply attention
                out = attn(x_scaled)
                # Upsample back
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            else:
                out = attn(x)
            outputs.append(out)
            
        # Fuse multi-scale outputs
        fused = torch.cat(outputs, dim=1)
        return self.fusion(fused)


class ParsingAttention(nn.Module):
    """
    Attention guided by semantic parsing.
    
    Uses parsing map to guide attention for cloth-body alignment.
    """
    
    def __init__(self, in_channels: int, parsing_nc: int):
        super().__init__()
        
        self.parsing_encoder = nn.Sequential(
            nn.Conv2d(parsing_nc, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        self.feature_conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        
    def forward(
        self,
        features: torch.Tensor,
        parsing: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply parsing-guided attention.
        
        Args:
            features: Input features (B, C, H, W)
            parsing: Semantic parsing map (B, parsing_nc, H, W)
        """
        # Resize parsing if needed
        if parsing.shape[2:] != features.shape[2:]:
            parsing = F.interpolate(
                parsing, size=features.shape[2:], mode='nearest'
            )
            
        # Generate attention from parsing
        attn = self.parsing_encoder(parsing)
        
        # Apply attention and combine with original
        attended = features * attn
        combined = torch.cat([features, attended], dim=1)
        
        return self.feature_conv(combined)
