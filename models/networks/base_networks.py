"""
Base network components shared across models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: str = 'batch',
        activation: str = 'relu',
        bias: bool = True
    ):
        super().__init__()
        
        layers = []
        
        # Convolution
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        ))
        
        # Normalization
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm == 'layer':
            layers.append(nn.GroupNorm(1, out_channels))
            
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(
        self,
        channels: int,
        norm: str = 'batch',
        use_dropout: bool = False
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels) if norm == 'batch' else nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        layers.extend([
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels) if norm == 'batch' else nn.InstanceNorm2d(channels),
        ])
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownsampleBlock(nn.Module):
    """Downsampling block with strided convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = 'batch'
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
        
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
            
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UpsampleBlock(nn.Module):
    """Upsampling block with transposed convolution or interpolation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = 'batch',
        use_dropout: bool = False,
        mode: str = 'transpose'
    ):
        super().__init__()
        
        if mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            )
            
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
            
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.norm(self.upsample(x))))


class UNetEncoder(nn.Module):
    """U-Net encoder."""
    
    def __init__(
        self,
        in_channels: int,
        ngf: int = 64,
        n_downsampling: int = 4
    ):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 7, 1, 3),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling layers
        self.down_layers = nn.ModuleList()
        channels = ngf
        for i in range(n_downsampling):
            out_channels = min(channels * 2, 512)
            self.down_layers.append(DownsampleBlock(channels, out_channels))
            channels = out_channels
            
        self.out_channels = channels
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        x = self.initial(x)
        features.append(x)
        
        for down in self.down_layers:
            x = down(x)
            features.append(x)
            
        return x, features[:-1]  # Return bottleneck and skip features


class UNetDecoder(nn.Module):
    """U-Net decoder with skip connections."""
    
    def __init__(
        self,
        out_channels: int,
        ngf: int = 64,
        n_upsampling: int = 4,
        use_skip: bool = True
    ):
        super().__init__()
        
        self.use_skip = use_skip
        
        # Calculate channel dimensions
        channels = min(ngf * (2 ** n_upsampling), 512)
        
        self.up_layers = nn.ModuleList()
        for i in range(n_upsampling):
            in_ch = channels * 2 if use_skip and i > 0 else channels
            out_ch = max(channels // 2, ngf)
            self.up_layers.append(UpsampleBlock(in_ch, out_ch, use_dropout=(i < 3)))
            channels = out_ch
            
        self.final = nn.Sequential(
            nn.Conv2d(channels * 2 if use_skip else channels, out_channels, 7, 1, 3),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor, skip_features: List[torch.Tensor] = None) -> torch.Tensor:
        for i, up in enumerate(self.up_layers):
            x = up(x)
            if self.use_skip and skip_features and i < len(skip_features):
                skip = skip_features[-(i+1)]
                # Handle size mismatch
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
                
        return self.final(x)


class UNet(nn.Module):
    """Complete U-Net architecture."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ngf: int = 64,
        n_downsampling: int = 4
    ):
        super().__init__()
        
        self.encoder = UNetEncoder(in_channels, ngf, n_downsampling)
        self.decoder = UNetDecoder(out_channels, ngf, n_downsampling)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skip_features = self.encoder(x)
        output = self.decoder(bottleneck, skip_features)
        return output


class FeatureExtractor(nn.Module):
    """Feature extractor for correlation computation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, out_channels, stride=2),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CorrelationLayer(nn.Module):
    """Compute correlation between two feature maps."""
    
    def __init__(self, max_displacement: int = 4, stride: int = 1):
        super().__init__()
        self.max_displacement = max_displacement
        self.stride = stride
        
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation between two feature maps.
        
        Args:
            feat1: (B, C, H, W) - First feature map
            feat2: (B, C, H, W) - Second feature map
            
        Returns:
            Correlation volume (B, (2*d+1)^2, H, W)
        """
        b, c, h, w = feat1.shape
        d = self.max_displacement
        
        # Normalize features
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        # Pad feat2 for displacement
        feat2_padded = F.pad(feat2, [d, d, d, d], mode='zeros')
        
        # Compute correlation
        corr = []
        for dy in range(-d, d + 1, self.stride):
            for dx in range(-d, d + 1, self.stride):
                feat2_shifted = feat2_padded[:, :, d+dy:d+dy+h, d+dx:d+dx+w]
                corr.append((feat1 * feat2_shifted).sum(dim=1, keepdim=True))
                
        return torch.cat(corr, dim=1)


class SPADEBlock(nn.Module):
    """SPADE (Spatially-Adaptive Denormalization) block."""
    
    def __init__(
        self,
        norm_channels: int,
        label_channels: int,
        hidden_channels: int = 128
    ):
        super().__init__()
        
        self.param_free_norm = nn.InstanceNorm2d(norm_channels, affine=False)
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.mlp_gamma = nn.Conv2d(hidden_channels, norm_channels, 3, 1, 1)
        self.mlp_beta = nn.Conv2d(hidden_channels, norm_channels, 3, 1, 1)
        
    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        
        # Resize segmentation map to match x
        seg = F.interpolate(seg, size=x.shape[2:], mode='nearest')
        
        actv = self.mlp_shared(seg)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
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
        
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, 1, 1)
        
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            
        self.spade1 = SPADEBlock(in_channels, label_channels)
        self.spade2 = SPADEBlock(middle_channels, label_channels)
        
        if self.learned_shortcut:
            self.spade_s = SPADEBlock(in_channels, label_channels)
            
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        x_s = self.shortcut(x, seg)
        
        dx = self.conv1(self.act(self.spade1(x, seg)))
        dx = self.conv2(self.act(self.spade2(dx, seg)))
        
        return x_s + dx
        
    def shortcut(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            x_s = self.conv_s(self.spade_s(x, seg))
        else:
            x_s = x
        return x_s
