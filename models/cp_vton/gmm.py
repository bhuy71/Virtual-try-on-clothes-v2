"""
Geometric Matching Module (GMM) for CP-VTON

The GMM learns to warp the clothing item to match the person's pose and body shape
using Thin-Plate Spline (TPS) transformation.

Reference:
- "Toward Characteristic-Preserving Image-based Virtual Try-On Network" (ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class FeatureExtraction(nn.Module):
    """Feature extraction network for GMM."""
    
    def __init__(self, input_nc: int, ngf: int = 64, num_layers: int = 4):
        super().__init__()
        
        layers = [
            nn.Conv2d(input_nc, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        
        nf = ngf
        for i in range(1, num_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, 2, 1),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            
        self.model = nn.Sequential(*layers)
        self.output_nc = nf
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FeatureCorrelation(nn.Module):
    """Compute correlation between feature maps."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, feat_A: torch.Tensor, feat_B: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation between two feature maps.
        
        Args:
            feat_A: (B, C, H, W) - Feature map A
            feat_B: (B, C, H, W) - Feature map B
            
        Returns:
            Correlation tensor (B, H*W, H, W)
        """
        b, c, h, w = feat_A.shape
        
        # Normalize features
        feat_A = F.normalize(feat_A.view(b, c, -1), dim=1)  # (B, C, H*W)
        feat_B = F.normalize(feat_B.view(b, c, -1), dim=1)  # (B, C, H*W)
        
        # Compute correlation
        corr = torch.bmm(feat_A.permute(0, 2, 1), feat_B)  # (B, H*W, H*W)
        corr = corr.view(b, h * w, h, w)
        
        return corr


class FeatureRegression(nn.Module):
    """Regression network to predict TPS parameters from correlation."""
    
    def __init__(self, input_nc: int, output_size: int = 2 * 5 * 5, input_size: Tuple[int, int] = (16, 12)):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Calculate fc input size based on input_size after 2 stride-2 convs
        # Input: (h, w) -> after 2x stride-2: (h//4, w//4)
        fc_h = input_size[0] // 4
        fc_w = input_size[1] // 4
        self.fc_input_size = 64 * fc_h * fc_w
        
        self.fc = nn.Linear(self.fc_input_size, output_size)
        
        # Initialize to identity transformation
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        theta = self.fc(x)
        return theta


class TPSTransform(nn.Module):
    """Thin-Plate Spline transformation module."""
    
    def __init__(self, height: int, width: int, grid_size: int = 5):
        super().__init__()
        
        self.height = height
        self.width = width
        self.grid_size = grid_size
        
        # Create regular grid of control points
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        
        # Control points: (grid_size * grid_size, 2)
        self.register_buffer('control_points', 
            torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        )
        
        # Create sampling grid
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        
        # Sampling points: (height * width, 2)
        self.register_buffer('sampling_grid',
            torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        )
        
        # Precompute TPS kernel matrix
        self._compute_tps_kernel()
        
    def _compute_tps_kernel(self):
        """Precompute TPS kernel matrices."""
        n = self.grid_size * self.grid_size
        P = self.control_points
        
        # Kernel matrix K
        diff = P.unsqueeze(0) - P.unsqueeze(1)  # (n, n, 2)
        dist_sq = (diff ** 2).sum(dim=2)  # (n, n)
        dist_sq = torch.clamp(dist_sq, min=1e-8)
        K = 0.5 * dist_sq * torch.log(dist_sq)
        K[dist_sq == 0] = 0
        
        # Matrix L = [K P; P^T 0]
        P_ones = torch.cat([torch.ones(n, 1), P], dim=1)  # (n, 3)
        L = torch.zeros(n + 3, n + 3)
        L[:n, :n] = K
        L[:n, n:] = P_ones
        L[n:, :n] = P_ones.t()
        
        # Inverse of L
        self.register_buffer('L_inv', torch.inverse(L))
        
    def compute_tps_weights(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute TPS weights from deformation parameters.
        
        Args:
            theta: Deformation parameters (B, grid_size*grid_size*2)
            
        Returns:
            TPS weights (B, n+3, 2)
        """
        batch_size = theta.shape[0]
        n = self.grid_size * self.grid_size
        
        # Reshape theta to target control points
        target = theta.view(batch_size, n, 2)  # (B, n, 2)
        
        # Add the offset to original control points
        target = target + self.control_points.unsqueeze(0)
        
        # Solve for TPS weights
        # [w; a] = L_inv @ [target; 0]
        target_padded = torch.cat([
            target,
            torch.zeros(batch_size, 3, 2, device=theta.device)
        ], dim=1)  # (B, n+3, 2)
        
        weights = torch.bmm(
            self.L_inv.unsqueeze(0).expand(batch_size, -1, -1),
            target_padded
        )  # (B, n+3, 2)
        
        return weights
        
    def forward(
        self,
        source: torch.Tensor,
        theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply TPS transformation to source image.
        
        Args:
            source: Source image (B, C, H, W)
            theta: TPS parameters (B, grid_size*grid_size*2)
            
        Returns:
            Warped image and sampling grid
        """
        batch_size = source.shape[0]
        n = self.grid_size * self.grid_size
        
        # Compute TPS weights
        weights = self.compute_tps_weights(theta)  # (B, n+3, 2)
        
        # Compute deformed grid
        P = self.sampling_grid  # (H*W, 2)
        P_control = self.control_points  # (n, 2)
        
        # Kernel values at sampling points
        diff = P.unsqueeze(1) - P_control.unsqueeze(0)  # (H*W, n, 2)
        dist_sq = (diff ** 2).sum(dim=2)  # (H*W, n)
        dist_sq = torch.clamp(dist_sq, min=1e-8)
        U = 0.5 * dist_sq * torch.log(dist_sq)
        U[dist_sq == 0] = 0  # (H*W, n)
        
        # Polynomial terms
        P_poly = torch.cat([
            torch.ones(self.height * self.width, 1, device=source.device),
            P
        ], dim=1)  # (H*W, 3)
        
        # Full design matrix
        F_matrix = torch.cat([U, P_poly], dim=1)  # (H*W, n+3)
        F_matrix = F_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H*W, n+3)
        
        # Compute deformed positions
        deformed = torch.bmm(F_matrix, weights)  # (B, H*W, 2)
        deformed = deformed.view(batch_size, self.height, self.width, 2)
        
        # Sample from source using deformed grid
        warped = F.grid_sample(
            source, deformed, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped, deformed


class GMM(nn.Module):
    """
    Geometric Matching Module
    
    Takes person representation and clothing image as input,
    predicts TPS transformation to warp the clothing.
    """
    
    def __init__(
        self,
        input_nc_person: int = 22,  # Pose heatmaps + body shape
        input_nc_cloth: int = 3,
        ngf: int = 64,
        grid_size: int = 5,
        image_size: Tuple[int, int] = (256, 192)
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.image_size = image_size
        
        # Feature extraction for person
        self.feature_person = FeatureExtraction(input_nc_person, ngf)
        
        # Feature extraction for cloth
        self.feature_cloth = FeatureExtraction(input_nc_cloth, ngf)
        
        # Feature correlation
        self.correlation = FeatureCorrelation()
        
        # Regression to TPS parameters
        # Input: correlation feature map (h*w channels)
        h = image_size[0] // 16  # After 4 downsampling in FeatureExtraction
        w = image_size[1] // 16
        self.regression = FeatureRegression(
            input_nc=h * w,
            output_size=2 * grid_size * grid_size,
            input_size=(h, w)  # Pass feature map size for fc layer calculation
        )
        
        # TPS transformation
        self.tps = TPSTransform(image_size[0], image_size[1], grid_size)
        
    def forward(
        self,
        person: torch.Tensor,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor = None
    ) -> dict:
        """
        Forward pass.
        
        Args:
            person: Person representation (B, input_nc_person, H, W)
            cloth: Clothing image (B, 3, H, W)
            cloth_mask: Optional clothing mask (B, 1, H, W)
            
        Returns:
            Dictionary containing:
                - warped_cloth: Warped clothing image
                - warped_mask: Warped clothing mask (if provided)
                - theta: TPS parameters
                - grid: Sampling grid
        """
        # Extract features
        feat_person = self.feature_person(person)
        feat_cloth = self.feature_cloth(cloth)
        
        # Compute correlation
        corr = self.correlation(feat_person, feat_cloth)
        
        # Regress TPS parameters
        theta = self.regression(corr)
        
        # Apply TPS transformation
        warped_cloth, grid = self.tps(cloth, theta)
        
        result = {
            'warped_cloth': warped_cloth,
            'theta': theta,
            'grid': grid,
        }
        
        # Warp mask if provided
        if cloth_mask is not None:
            warped_mask, _ = self.tps(cloth_mask, theta)
            result['warped_mask'] = warped_mask
            
        return result


# Alternative: Affine Transformation Module (simpler baseline)
class AffineGMM(nn.Module):
    """Simpler GMM using affine transformation instead of TPS."""
    
    def __init__(
        self,
        input_nc_person: int = 22,
        input_nc_cloth: int = 3,
        ngf: int = 64,
        image_size: Tuple[int, int] = (256, 192)
    ):
        super().__init__()
        
        # Feature extraction
        self.feature_person = FeatureExtraction(input_nc_person, ngf)
        self.feature_cloth = FeatureExtraction(input_nc_cloth, ngf)
        
        # Correlation
        self.correlation = FeatureCorrelation()
        
        # Regress affine parameters (6 values for 2x3 matrix)
        h = image_size[0] // 16
        w = image_size[1] // 16
        
        self.regression = nn.Sequential(
            nn.Conv2d(h * w, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 6)
        )
        
        # Initialize to identity transformation
        self.regression[-1].weight.data.zero_()
        self.regression[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
        
    def forward(
        self,
        person: torch.Tensor,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor = None
    ) -> dict:
        # Extract features
        feat_person = self.feature_person(person)
        feat_cloth = self.feature_cloth(cloth)
        
        # Compute correlation
        corr = self.correlation(feat_person, feat_cloth)
        
        # Regress affine parameters
        theta = self.regression(corr)
        theta = theta.view(-1, 2, 3)
        
        # Create affine grid
        grid = F.affine_grid(theta, cloth.shape, align_corners=True)
        
        # Warp cloth
        warped_cloth = F.grid_sample(
            cloth, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        result = {
            'warped_cloth': warped_cloth,
            'theta': theta,
            'grid': grid,
        }
        
        if cloth_mask is not None:
            warped_mask = F.grid_sample(
                cloth_mask, grid, mode='bilinear', padding_mode='zeros', align_corners=True
            )
            result['warped_mask'] = warped_mask
            
        return result
