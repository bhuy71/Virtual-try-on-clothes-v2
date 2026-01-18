"""
Complete CP-VTON Model

Characteristic-Preserving Virtual Try-On Network

Two-stage architecture:
1. GMM (Geometric Matching Module): Warps clothing to match person's pose
2. TOM (Try-On Module): Synthesizes final try-on result

Reference:
- "Toward Characteristic-Preserving Image-based Virtual Try-On Network" (ECCV 2018)
  Authors: Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .gmm import GMM, AffineGMM
from .tom import TOM, TOMComplete


class CPVTON(nn.Module):
    """
    Complete CP-VTON Model
    
    A two-stage pipeline for image-based virtual try-on:
    1. GMM warps the clothing to match person's body shape and pose
    2. TOM blends the warped clothing with person to generate final result
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 192),
        pose_nc: int = 18,
        semantic_nc: int = 1,
        ngf: int = 64,
        grid_size: int = 5,
        use_tps: bool = True,
        use_refinement: bool = False
    ):
        """
        Args:
            image_size: Target image size (height, width)
            pose_nc: Number of pose keypoint channels
            semantic_nc: Number of semantic parsing channels
            ngf: Base number of generator filters
            grid_size: TPS grid size
            use_tps: Use TPS transformation (True) or affine (False)
            use_refinement: Use refinement network in TOM
        """
        super().__init__()
        
        self.image_size = image_size
        self.pose_nc = pose_nc
        
        # Input channels for GMM: pose + body shape
        gmm_person_nc = pose_nc + semantic_nc
        
        # GMM: Geometric Matching Module
        if use_tps:
            self.gmm = GMM(
                input_nc_person=gmm_person_nc,
                input_nc_cloth=3,
                ngf=ngf,
                grid_size=grid_size,
                image_size=image_size
            )
        else:
            self.gmm = AffineGMM(
                input_nc_person=gmm_person_nc,
                input_nc_cloth=3,
                ngf=ngf,
                image_size=image_size
            )
            
        # TOM: Try-On Module
        # Input: agnostic (3) + warped_cloth (3) + pose (18) + shape (1)
        tom_input_nc = 3 + 3 + pose_nc + semantic_nc
        
        self.tom = TOMComplete(
            input_nc=tom_input_nc,
            ngf=ngf,
            use_refinement=use_refinement
        )
        
    def forward(
        self,
        person: torch.Tensor,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
        pose: torch.Tensor,
        agnostic: torch.Tensor,
        parse_shape: torch.Tensor,
        stage: str = 'full'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            person: Person image (B, 3, H, W)
            cloth: Clothing image (B, 3, H, W)
            cloth_mask: Clothing mask (B, 1, H, W)
            pose: Pose heatmaps (B, 18, H, W)
            agnostic: Agnostic person (B, 3, H, W)
            parse_shape: Body shape mask (B, 1, H, W)
            stage: 'gmm', 'tom', or 'full'
            
        Returns:
            Dictionary with all intermediate and final outputs
        """
        result = {}
        
        # Stage 1: Geometric Matching
        if stage in ['gmm', 'full']:
            # Create GMM input
            gmm_input = torch.cat([pose, parse_shape], dim=1)
            
            gmm_output = self.gmm(gmm_input, cloth, cloth_mask)
            
            result['warped_cloth'] = gmm_output['warped_cloth']
            result['warped_mask'] = gmm_output.get('warped_mask')
            result['theta'] = gmm_output['theta']
            result['grid'] = gmm_output['grid']
            
        # Stage 2: Try-On Synthesis
        if stage in ['tom', 'full']:
            warped_cloth = result.get('warped_cloth', cloth)
            
            tom_output = self.tom(
                agnostic=agnostic,
                warped_cloth=warped_cloth,
                pose=pose,
                parse_shape=parse_shape
            )
            
            result['output'] = tom_output['output']
            result['rendered'] = tom_output.get('rendered')
            result['mask'] = tom_output['mask']
            
            if 'refined' in tom_output:
                result['refined'] = tom_output['refined']
                result['coarse'] = tom_output['coarse']
                
        return result
    
    def gmm_forward(
        self,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
        pose: torch.Tensor,
        parse_shape: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """GMM-only forward pass."""
        gmm_input = torch.cat([pose, parse_shape], dim=1)
        return self.gmm(gmm_input, cloth, cloth_mask)
    
    def tom_forward(
        self,
        agnostic: torch.Tensor,
        warped_cloth: torch.Tensor,
        pose: torch.Tensor,
        parse_shape: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """TOM-only forward pass."""
        return self.tom(agnostic, warped_cloth, pose, parse_shape)


class CPVTONTrainer:
    """
    Training utilities for CP-VTON.
    
    Supports:
    - Two-stage training (GMM then TOM)
    - End-to-end training
    - Loss computation
    """
    
    def __init__(
        self,
        model: CPVTON,
        device: torch.device,
        lr: float = 0.0001,
        betas: Tuple[float, float] = (0.5, 0.999)
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizers for separate training
        self.optimizer_gmm = torch.optim.Adam(
            model.gmm.parameters(), lr=lr, betas=betas
        )
        self.optimizer_tom = torch.optim.Adam(
            model.tom.parameters(), lr=lr, betas=betas
        )
        
        # Full optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas
        )
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        
    def compute_gmm_loss(
        self,
        warped_cloth: torch.Tensor,
        target_cloth: torch.Tensor,
        warped_mask: torch.Tensor = None,
        target_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute GMM loss."""
        # L1 loss on warped cloth
        loss = self.l1_loss(warped_cloth, target_cloth)
        
        # Optional mask loss
        if warped_mask is not None and target_mask is not None:
            loss += self.l1_loss(warped_mask, target_mask)
            
        return loss
    
    def compute_tom_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        target_mask: torch.Tensor,
        perceptual_loss_fn = None,
        weights: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute TOM loss."""
        if weights is None:
            weights = {'l1': 1.0, 'perceptual': 1.0, 'mask': 1.0}
            
        losses = {}
        
        # L1 reconstruction loss
        losses['l1'] = weights['l1'] * self.l1_loss(output, target)
        
        # Perceptual loss
        if perceptual_loss_fn is not None:
            losses['perceptual'] = weights['perceptual'] * perceptual_loss_fn(output, target)
            
        # Mask loss
        losses['mask'] = weights['mask'] * self.l1_loss(mask, target_mask)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


def build_cpvton(config: dict) -> CPVTON:
    """Build CP-VTON model from config."""
    return CPVTON(
        image_size=tuple(config.get('image_size', (256, 192))),
        pose_nc=config.get('pose_nc', 18),
        semantic_nc=config.get('semantic_nc', 1),
        ngf=config.get('ngf', 64),
        grid_size=config.get('grid_size', 5),
        use_tps=config.get('use_tps', True),
        use_refinement=config.get('use_refinement', False)
    )
