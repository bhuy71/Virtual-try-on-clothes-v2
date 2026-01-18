"""
Training script for CP-VTON (Method 1: Traditional)

Two-stage training:
1. GMM: Geometric Matching Module
2. TOM: Try-On Module

Usage:
    # Train GMM
    python train_cp_vton.py --stage gmm --config ../configs/cp_vton_config.yaml

    # Train TOM (after GMM)
    python train_cp_vton.py --stage tom --config ../configs/cp_vton_config.yaml --gmm_checkpoint ./checkpoints/gmm/best.pth

    # Train both stages
    python train_cp_vton.py --stage full --config ../configs/cp_vton_config.yaml
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add parent directory to path - fix for Kaggle/notebook imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from models.cp_vton import CPVTON, GMM, TOMComplete
from models.networks.losses import VGGPerceptualLoss
from data.dataset import VITONDataset
from train.trainer import BaseTrainer, load_config


class GMMTrainer(BaseTrainer):
    """Trainer for Geometric Matching Module."""
    
    def __init__(
        self,
        model: GMM,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: torch.device = None,
        config: dict = None
    ):
        super().__init__(model, train_loader, val_loader, device, config)
        
        # Optimizer
        training_config = config.get('training', {}).get('gmm', {})
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_config.get('lr', 0.0001),
            betas=(training_config.get('beta1', 0.5), training_config.get('beta2', 0.999))
        )
        
        # Loss
        self.l1_loss = nn.L1Loss()
        
    def train_step(self, batch: dict) -> dict:
        """Single GMM training step."""
        # Move to device
        cloth = batch['cloth'].to(self.device)
        cloth_mask = batch['cloth_mask'].to(self.device)
        pose = batch['pose'].to(self.device)
        parse = batch['parse'].to(self.device)
        image = batch['image'].to(self.device)  # Ground truth
        
        # Create GMM input (pose + body shape)
        # Body shape: upper body region from parsing
        body_shape = parse[:, 4:5]  # Upper clothes channel as body shape
        gmm_input = torch.cat([pose, body_shape], dim=1)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(gmm_input, cloth, cloth_mask)
        
        # Loss: L1 between warped cloth and ground truth cloth region
        warped_cloth = output['warped_cloth']
        
        # Use the cloth region from ground truth image
        # In practice, we use the cloth in the original position
        loss = self.l1_loss(warped_cloth, cloth)  # Self-reconstruction
        
        # Optional: Add regularization on theta
        theta = output['theta']
        theta_reg = torch.mean(theta ** 2) * 0.01
        
        total_loss = loss + theta_reg
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total': total_loss.item(),
            'l1': loss.item(),
            'theta_reg': theta_reg.item()
        }
        
    def validate_step(self, batch: dict) -> dict:
        """Single GMM validation step."""
        cloth = batch['cloth'].to(self.device)
        cloth_mask = batch['cloth_mask'].to(self.device)
        pose = batch['pose'].to(self.device)
        parse = batch['parse'].to(self.device)
        
        body_shape = parse[:, 4:5]
        gmm_input = torch.cat([pose, body_shape], dim=1)
        
        output = self.model(gmm_input, cloth, cloth_mask)
        warped_cloth = output['warped_cloth']
        
        loss = self.l1_loss(warped_cloth, cloth)
        
        return {'l1': loss.item()}


class TOMTrainer(BaseTrainer):
    """Trainer for Try-On Module."""
    
    def __init__(
        self,
        model: TOMComplete,
        gmm: GMM,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: torch.device = None,
        config: dict = None
    ):
        super().__init__(model, train_loader, val_loader, device, config)
        
        self.gmm = gmm.to(self.device)
        self.gmm.eval()  # Freeze GMM
        
        # Optimizer
        training_config = config.get('training', {}).get('tom', {})
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_config.get('lr', 0.0001),
            betas=(training_config.get('beta1', 0.5), training_config.get('beta2', 0.999))
        )
        
        # Losses
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss().to(self.device)
        
        # Loss weights
        loss_config = config.get('loss', {}).get('tom', {})
        self.l1_weight = loss_config.get('l1_weight', 1.0)
        self.perceptual_weight = loss_config.get('perceptual_weight', 1.0)
        self.mask_weight = loss_config.get('composition_weight', 1.0)
        
    def train_step(self, batch: dict) -> dict:
        """Single TOM training step."""
        # Move to device
        image = batch['image'].to(self.device)  # Ground truth
        cloth = batch['cloth'].to(self.device)
        cloth_mask = batch['cloth_mask'].to(self.device)
        agnostic = batch['agnostic'].to(self.device)
        pose = batch['pose'].to(self.device)
        parse = batch['parse'].to(self.device)
        
        # Get warped cloth from frozen GMM
        with torch.no_grad():
            body_shape = parse[:, 4:5]
            gmm_input = torch.cat([pose, body_shape], dim=1)
            gmm_output = self.gmm(gmm_input, cloth, cloth_mask)
            warped_cloth = gmm_output['warped_cloth']
            
        # Forward pass through TOM
        self.optimizer.zero_grad()
        
        tom_output = self.model(
            agnostic=agnostic,
            warped_cloth=warped_cloth,
            pose=pose,
            parse_shape=body_shape
        )
        
        output = tom_output['output']
        mask = tom_output['mask']
        
        # Losses
        l1_loss = self.l1_loss(output, image)
        perceptual_loss = self.perceptual_loss(output, image)
        
        # Mask loss: encourage mask to match cloth region
        target_mask = (parse[:, 4:5] > 0.5).float()  # Upper clothes region
        mask_loss = self.l1_loss(mask, target_mask)
        
        total_loss = (
            self.l1_weight * l1_loss +
            self.perceptual_weight * perceptual_loss +
            self.mask_weight * mask_loss
        )
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item(),
            'mask': mask_loss.item()
        }
        
    def validate_step(self, batch: dict) -> dict:
        """Single TOM validation step."""
        image = batch['image'].to(self.device)
        cloth = batch['cloth'].to(self.device)
        cloth_mask = batch['cloth_mask'].to(self.device)
        agnostic = batch['agnostic'].to(self.device)
        pose = batch['pose'].to(self.device)
        parse = batch['parse'].to(self.device)
        
        body_shape = parse[:, 4:5]
        gmm_input = torch.cat([pose, body_shape], dim=1)
        gmm_output = self.gmm(gmm_input, cloth, cloth_mask)
        warped_cloth = gmm_output['warped_cloth']
        
        tom_output = self.model(
            agnostic=agnostic,
            warped_cloth=warped_cloth,
            pose=pose,
            parse_shape=body_shape
        )
        
        output = tom_output['output']
        
        l1_loss = self.l1_loss(output, image)
        perceptual_loss = self.perceptual_loss(output, image)
        
        return {
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item()
        }


def train_gmm(config: dict, device: torch.device):
    """Train GMM stage."""
    print("\n" + "="*60)
    print("Training GMM (Geometric Matching Module)")
    print("="*60 + "\n")
    
    # Create dataset and dataloader
    image_size = tuple(config.get('model', {}).get('input_size', [256, 192]))
    
    train_dataset = VITONDataset(
        data_root=config['data']['data_root'],
        split='train',
        image_size=image_size
    )
    
    val_dataset = VITONDataset(
        data_root=config['data']['data_root'],
        split='test',
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['gmm']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['gmm']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # Create GMM model
    gmm_config = config.get('gmm', {})
    model = GMM(
        input_nc_person=gmm_config.get('input_nc', 19),
        input_nc_cloth=3,
        ngf=gmm_config.get('ngf', 64),
        grid_size=gmm_config.get('grid_size', 5),
        image_size=image_size
    )
    
    # Create trainer
    trainer_config = {
        **config,
        'checkpoint_dir': os.path.join(config['checkpoint']['save_dir'], 'gmm'),
        'log_dir': os.path.join(config['checkpoint']['save_dir'], 'logs/gmm'),
    }
    
    trainer = GMMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=trainer_config
    )
    
    # Train
    trainer.train(
        num_epochs=config['training']['gmm']['epochs'],
        resume_from=config['checkpoint'].get('resume')
    )
    
    return trainer.model


def train_tom(config: dict, device: torch.device, gmm_checkpoint: str = None):
    """Train TOM stage."""
    print("\n" + "="*60)
    print("Training TOM (Try-On Module)")
    print("="*60 + "\n")
    
    image_size = tuple(config.get('model', {}).get('input_size', [256, 192]))
    
    # Load datasets
    train_dataset = VITONDataset(
        data_root=config['data']['data_root'],
        split='train',
        image_size=image_size
    )
    
    val_dataset = VITONDataset(
        data_root=config['data']['data_root'],
        split='test',
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['tom']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['tom']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # Load GMM
    gmm_config = config.get('gmm', {})
    gmm = GMM(
        input_nc_person=gmm_config.get('input_nc', 19),
        input_nc_cloth=3,
        ngf=gmm_config.get('ngf', 64),
        grid_size=gmm_config.get('grid_size', 5),
        image_size=image_size
    )
    
    if gmm_checkpoint:
        checkpoint = torch.load(gmm_checkpoint, map_location=device)
        gmm.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded GMM from {gmm_checkpoint}")
    else:
        # Try to load from default location
        default_gmm_path = os.path.join(config['checkpoint']['save_dir'], 'gmm/best.pth')
        if os.path.exists(default_gmm_path):
            checkpoint = torch.load(default_gmm_path, map_location=device)
            gmm.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded GMM from {default_gmm_path}")
        else:
            print("Warning: No GMM checkpoint found. Using random initialization.")
            
    # Create TOM model
    tom_config = config.get('tom', {})
    model = TOMComplete(
        input_nc=tom_config.get('input_nc', 22),  # pose(18) + shape(1) + agnostic(3)
        ngf=tom_config.get('ngf', 64),
        use_refinement=False
    )
    
    # Create trainer
    trainer_config = {
        **config,
        'checkpoint_dir': os.path.join(config['checkpoint']['save_dir'], 'tom'),
        'log_dir': os.path.join(config['checkpoint']['save_dir'], 'logs/tom'),
    }
    
    trainer = TOMTrainer(
        model=model,
        gmm=gmm,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=trainer_config
    )
    
    # Train
    trainer.train(
        num_epochs=config['training']['tom']['epochs'],
        resume_from=config['checkpoint'].get('resume')
    )
    
    return trainer.model


def main():
    parser = argparse.ArgumentParser(description='Train CP-VTON')
    parser.add_argument('--stage', type=str, default='full',
                        choices=['gmm', 'tom', 'full'],
                        help='Training stage')
    parser.add_argument('--config', type=str, default='configs/cp_vton_config.yaml',
                        help='Path to config file')
    parser.add_argument('--gmm_checkpoint', type=str, default=None,
                        help='Path to GMM checkpoint (for TOM training)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.resume:
        config['checkpoint']['resume'] = args.resume
        
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train
    if args.stage == 'gmm':
        train_gmm(config, device)
    elif args.stage == 'tom':
        train_tom(config, device, args.gmm_checkpoint)
    elif args.stage == 'full':
        # Train both stages sequentially
        gmm = train_gmm(config, device)
        
        # Save GMM path for TOM
        gmm_checkpoint = os.path.join(config['checkpoint']['save_dir'], 'gmm/best.pth')
        train_tom(config, device, gmm_checkpoint)
        
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
