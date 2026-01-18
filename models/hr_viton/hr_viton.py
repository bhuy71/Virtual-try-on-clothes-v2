"""
Complete HR-VITON Model

High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions

Key Features:
1. Appearance flow for accurate cloth warping
2. Segmentation prediction for semantic awareness
3. SPADE-based high-resolution synthesis
4. Multi-scale discrimination

Reference:
- "High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions" (ECCV 2022)
  Authors: Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, Jaegul Choo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .condition_generator import ConditionGenerator
from .generator import HRGenerator
from .discriminator import MultiScaleDiscriminator, ConditionalDiscriminator


class HRVITON(nn.Module):
    """
    Complete HR-VITON Model
    
    End-to-end high-resolution virtual try-on system.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 768),
        semantic_nc: int = 20,  # VITON-HD v3 uses 20 labels (changed from 13)
        pose_nc: int = 18,
        ngf: int = 64,
        ndf: int = 64
    ):
        """
        Args:
            image_size: Target image size (height, width)
            semantic_nc: Number of semantic parsing classes (20 for VITON-HD v3)
            pose_nc: Number of pose keypoint channels
            ngf: Base number of generator filters
            ndf: Base number of discriminator filters
        """
        super().__init__()
        
        self.image_size = image_size
        self.semantic_nc = semantic_nc
        
        # Condition Generator: produces warped cloth, flow, and segmentation
        self.condition_generator = ConditionGenerator(
            semantic_nc=semantic_nc,
            ngf=ngf
        )
        
        # Image Generator: synthesizes final try-on image
        self.generator = HRGenerator(
            input_nc=3 + 3 + semantic_nc,  # warped_cloth + agnostic + seg
            output_nc=3,
            semantic_nc=semantic_nc,
            ngf=ngf,
            n_downsampling=4,
            n_blocks=9
        )
        
        # Discriminators
        self.discriminator = MultiScaleDiscriminator(
            input_nc=3,
            ndf=ndf,
            n_layers=4,
            num_D=2
        )
        
        self.seg_discriminator = ConditionalDiscriminator(
            input_nc=3,
            condition_nc=semantic_nc,
            ndf=ndf,
            n_layers=3,
            num_D=1
        )
        
    def forward(
        self,
        person: torch.Tensor,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
        agnostic: torch.Tensor,
        parse: torch.Tensor,
        pose: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            person: Person image (B, 3, H, W) - Ground truth for training
            cloth: Clothing image (B, 3, H, W)
            cloth_mask: Clothing mask (B, 1, H, W)
            agnostic: Agnostic person representation (B, 3, H, W)
            parse: Person parsing map (B, semantic_nc, H, W)
            pose: Optional pose heatmaps (B, pose_nc, H, W)
            
        Returns:
            Dictionary with all outputs
        """
        # Generate conditions
        cond_output = self.condition_generator(
            agnostic, cloth, cloth_mask, parse
        )
        
        warped_cloth = cond_output['warped_cloth']
        seg = cond_output['seg']
        
        # Generate final image
        output = self.generator(warped_cloth, agnostic, seg, pose)
        
        return {
            'output': output,
            'warped_cloth': warped_cloth,
            'warped_mask': cond_output['warped_mask'],
            'flow': cond_output['flow'],
            'seg': seg,
            'seg_logits': cond_output['seg_logits'],
            'misalign_mask': cond_output['misalign_mask'],
        }
        
    def generate(
        self,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
        agnostic: torch.Tensor,
        parse: torch.Tensor,
        pose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inference-only generation.
        
        Args:
            cloth: Clothing to try on
            cloth_mask: Clothing mask
            agnostic: Agnostic person
            parse: Person parsing
            pose: Optional pose
            
        Returns:
            Generated try-on image
        """
        with torch.no_grad():
            output = self.forward(
                person=None,  # Not needed for inference
                cloth=cloth,
                cloth_mask=cloth_mask,
                agnostic=agnostic,
                parse=parse,
                pose=pose
            )
            return output['output']


class HRVITONLoss(nn.Module):
    """
    Loss functions for HR-VITON training.
    
    Includes:
    - Adversarial loss
    - Feature matching loss
    - Perceptual loss
    - L1 reconstruction loss
    - Segmentation loss
    - Flow regularization loss
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        device: torch.device = None
    ):
        super().__init__()
        
        self.weights = weights or {
            'adversarial': 1.0,
            'feature_matching': 10.0,
            'perceptual': 10.0,
            'l1': 10.0,
            'seg': 1.0,
            'flow': 0.1,
        }
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Perceptual loss (VGG)
        from ..networks.losses import VGGPerceptualLoss
        self.perceptual = VGGPerceptualLoss().to(self.device)
        
        # Cross entropy for segmentation
        self.ce_loss = nn.CrossEntropyLoss()
        
        # L1 loss
        self.l1_loss = nn.L1Loss()
        
    def adversarial_loss(
        self,
        pred: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        """Hinge adversarial loss."""
        if for_discriminator:
            if target_is_real:
                return F.relu(1.0 - pred).mean()
            else:
                return F.relu(1.0 + pred).mean()
        else:
            return -pred.mean()
            
    def feature_matching_loss(
        self,
        fake_features: List[List[torch.Tensor]],
        real_features: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """Feature matching loss."""
        loss = 0.0
        n_features = 0
        
        for fake_f, real_f in zip(fake_features, real_features):
            for ff, rf in zip(fake_f, real_f):
                loss += self.l1_loss(ff, rf.detach())
                n_features += 1
                
        return loss / n_features if n_features > 0 else loss
        
    def compute_generator_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        disc_fake_pred: List[torch.Tensor],
        disc_real_features: List[List[torch.Tensor]],
        disc_fake_features: List[List[torch.Tensor]],
        seg_logits: torch.Tensor,
        seg_target: torch.Tensor,
        flow: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute generator losses.
        
        Returns:
            Dictionary of individual losses and total
        """
        losses = {}
        
        # Adversarial loss
        g_adv = 0.0
        for pred in disc_fake_pred:
            g_adv += self.adversarial_loss(pred, True, for_discriminator=False)
        g_adv /= len(disc_fake_pred)
        losses['adversarial'] = self.weights['adversarial'] * g_adv
        
        # Feature matching loss
        losses['feature_matching'] = self.weights['feature_matching'] * \
            self.feature_matching_loss(disc_fake_features, disc_real_features)
            
        # Perceptual loss
        losses['perceptual'] = self.weights['perceptual'] * \
            self.perceptual(output, target)
            
        # L1 reconstruction loss
        losses['l1'] = self.weights['l1'] * self.l1_loss(output, target)
        
        # Segmentation loss
        if seg_target.dim() == 4:  # One-hot encoded
            seg_target_labels = seg_target.argmax(dim=1)
        else:
            seg_target_labels = seg_target
        losses['seg'] = self.weights['seg'] * self.ce_loss(seg_logits, seg_target_labels)
        
        # Flow regularization (total variation)
        flow_tv = (
            torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean() +
            torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
        )
        losses['flow'] = self.weights['flow'] * flow_tv
        
        # Total
        losses['total'] = sum(losses.values())
        
        return losses
        
    def compute_discriminator_loss(
        self,
        real_pred: List[torch.Tensor],
        fake_pred: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute discriminator loss."""
        d_loss = 0.0
        
        for rp, fp in zip(real_pred, fake_pred):
            d_loss += self.adversarial_loss(rp, True, for_discriminator=True)
            d_loss += self.adversarial_loss(fp, False, for_discriminator=True)
            
        return d_loss / (2 * len(real_pred))


class HRVITONTrainer:
    """
    Training utilities for HR-VITON.
    """
    
    def __init__(
        self,
        model: HRVITON,
        device: torch.device,
        lr_g: float = 0.0001,
        lr_d: float = 0.0004,
        betas: Tuple[float, float] = (0.0, 0.9)
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            list(model.condition_generator.parameters()) + 
            list(model.generator.parameters()),
            lr=lr_g, betas=betas
        )
        
        self.optimizer_D = torch.optim.Adam(
            list(model.discriminator.parameters()) +
            list(model.seg_discriminator.parameters()),
            lr=lr_d, betas=betas
        )
        
        # Loss
        self.criterion = HRVITONLoss(device=device)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
                
        person = batch['image']
        cloth = batch['cloth']
        cloth_mask = batch['cloth_mask']
        agnostic = batch['agnostic']
        parse = batch['parse']
        pose = batch.get('pose')
        
        # Forward pass
        output = self.model(person, cloth, cloth_mask, agnostic, parse, pose)
        
        # Discriminator forward
        with torch.no_grad():
            real_pred, real_features = self.model.discriminator(person)
        fake_pred, fake_features = self.model.discriminator(output['output'])
        
        # ----- Train Discriminator -----
        self.optimizer_D.zero_grad()
        
        # Re-compute with gradients for D
        real_pred_d, _ = self.model.discriminator(person)
        fake_pred_d, _ = self.model.discriminator(output['output'].detach())
        
        d_loss = self.criterion.compute_discriminator_loss(real_pred_d, fake_pred_d)
        d_loss.backward()
        self.optimizer_D.step()
        
        # ----- Train Generator -----
        self.optimizer_G.zero_grad()
        
        g_losses = self.criterion.compute_generator_loss(
            output=output['output'],
            target=person,
            disc_fake_pred=fake_pred,
            disc_real_features=real_features,
            disc_fake_features=fake_features,
            seg_logits=output['seg_logits'],
            seg_target=parse,
            flow=output['flow']
        )
        
        g_losses['total'].backward()
        self.optimizer_G.step()
        
        # Return losses as floats
        return {
            'd_loss': d_loss.item(),
            **{k: v.item() for k, v in g_losses.items()}
        }


def build_hr_viton(config: dict) -> HRVITON:
    """Build HR-VITON model from config."""
    return HRVITON(
        image_size=tuple(config.get('image_size', (1024, 768))),
        semantic_nc=config.get('semantic_nc', 13),
        pose_nc=config.get('pose_nc', 18),
        ngf=config.get('ngf', 64),
        ndf=config.get('ndf', 64)
    )
