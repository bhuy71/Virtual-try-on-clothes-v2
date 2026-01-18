"""
Training script for HR-VITON (Method 2: State-of-the-Art)

End-to-end high-resolution virtual try-on training.

Usage:
    python train_hr_viton.py --config ../configs/hr_viton_config.yaml
    
    # With Weights & Biases logging
    python train_hr_viton.py --config ../configs/hr_viton_config.yaml --use_wandb
    
    # Resume training
    python train_hr_viton.py --config ../configs/hr_viton_config.yaml --resume ./checkpoints/hr_viton/epoch_50.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from typing import Dict, Any, Optional

# Add parent directory to path - fix for Kaggle/notebook imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from models.hr_viton import HRVITON, HRVITONLoss, build_hr_viton
from data.dataset import VITONHDDataset
from train.trainer import BaseTrainer, load_config
import torchvision.utils as vutils


class HRVITONTrainer(BaseTrainer):
    """Trainer for HR-VITON model."""
    
    def __init__(
        self,
        model: HRVITON,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: torch.device = None,
        config: dict = None
    ):
        super().__init__(model, train_loader, val_loader, device, config)
        
        # Optimizers
        training_config = config.get('training', {})
        opt_config = training_config.get('optimizer', {})
        
        # Generator optimizer
        self.optimizer_G = torch.optim.Adam(
            list(self.model.condition_generator.parameters()) +
            list(self.model.generator.parameters()),
            lr=opt_config.get('lr_g', 0.0001),
            betas=(opt_config.get('beta1', 0.0), opt_config.get('beta2', 0.9))
        )
        
        # Discriminator optimizer
        self.optimizer_D = torch.optim.Adam(
            list(self.model.discriminator.parameters()) +
            list(self.model.seg_discriminator.parameters()),
            lr=opt_config.get('lr_d', 0.0004),
            betas=(opt_config.get('beta1', 0.0), opt_config.get('beta2', 0.9))
        )
        
        # Learning rate schedulers
        scheduler_config = training_config.get('scheduler', {})
        if scheduler_config.get('type') == 'step':
            self.scheduler_G = torch.optim.lr_scheduler.StepLR(
                self.optimizer_G,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.5)
            )
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(
                self.optimizer_D,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        else:
            self.scheduler_G = None
            self.scheduler_D = None
            
        # Loss function with default weights
        default_weights = {
            'adversarial': 1.0,
            'feature_matching': 10.0,
            'perceptual': 10.0,
            'l1': 10.0,
            'seg': 1.0,
            'flow': 0.1,
        }
        # Merge config weights with defaults
        loss_weights = {**default_weights, **config.get('loss', {})}
        self.criterion = HRVITONLoss(weights=loss_weights, device=self.device)
        
        # Sample directory for visualization
        self.sample_dir = os.path.join(
            config.get('checkpoint', {}).get('save_dir', './checkpoints'),
            'samples'
        )
        os.makedirs(self.sample_dir, exist_ok=True)
        
    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        # Move to device
        image = batch['image'].to(self.device)
        cloth = batch['cloth'].to(self.device)
        cloth_mask = batch['cloth_mask'].to(self.device)
        agnostic = batch['agnostic'].to(self.device)
        parse = batch['parse'].to(self.device)
        pose = batch.get('pose')
        if pose is not None:
            pose = pose.to(self.device)
            
        # Forward pass through full model
        output = self.model(image, cloth, cloth_mask, agnostic, parse, pose)
        
        generated = output['output']
        
        # ---------- Train Discriminator ----------
        self.optimizer_D.zero_grad()
        
        # Real images
        real_pred, real_features = self.model.discriminator(image)
        
        # Fake images (detached)
        fake_pred_d, _ = self.model.discriminator(generated.detach())
        
        # Discriminator loss
        d_loss = 0.0
        for rp, fp in zip(real_pred, fake_pred_d):
            d_loss += F.relu(1.0 - rp).mean() + F.relu(1.0 + fp).mean()
        d_loss = d_loss / (2 * len(real_pred))
        
        d_loss.backward()
        self.optimizer_D.step()
        
        # ---------- Train Generator ----------
        self.optimizer_G.zero_grad()
        
        # Discriminator on fake (with gradient)
        fake_pred, fake_features = self.model.discriminator(generated)
        
        # Generator losses
        g_losses = self.criterion.compute_generator_loss(
            output=generated,
            target=image,
            disc_fake_pred=fake_pred,
            disc_real_features=real_features,
            disc_fake_features=fake_features,
            seg_logits=output['seg_logits'],
            seg_target=parse,
            flow=output['flow']
        )
        
        g_losses['total'].backward()
        self.optimizer_G.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_losses['total'].item(),
            'l1': g_losses['l1'].item(),
            'perceptual': g_losses['perceptual'].item(),
            'adversarial': g_losses['adversarial'].item(),
        }
        
    def validate_step(self, batch: dict) -> dict:
        """Single validation step."""
        image = batch['image'].to(self.device)
        cloth = batch['cloth'].to(self.device)
        cloth_mask = batch['cloth_mask'].to(self.device)
        agnostic = batch['agnostic'].to(self.device)
        parse = batch['parse'].to(self.device)
        pose = batch.get('pose')
        if pose is not None:
            pose = pose.to(self.device)
            
        output = self.model(image, cloth, cloth_mask, agnostic, parse, pose)
        generated = output['output']
        
        # L1 and perceptual loss
        l1_loss = F.l1_loss(generated, image)
        perceptual_loss = self.criterion.perceptual(generated, image)
        
        return {
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item()
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with visualization."""
        epoch_losses = super().train_epoch()
        
        # Update learning rate
        if self.scheduler_G:
            self.scheduler_G.step()
        if self.scheduler_D:
            self.scheduler_D.step()
            
        # Save sample images
        if self.current_epoch % self.config.get('logging', {}).get('sample_interval', 5) == 0:
            self.save_samples()
            
        return epoch_losses
        
    def save_samples(self):
        """Save sample generated images."""
        self.model.eval()
        
        # Get a batch from validation set
        if self.val_loader is not None:
            batch = next(iter(self.val_loader))
        else:
            batch = next(iter(self.train_loader))
            
        with torch.no_grad():
            image = batch['image'].to(self.device)
            cloth = batch['cloth'].to(self.device)
            cloth_mask = batch['cloth_mask'].to(self.device)
            agnostic = batch['agnostic'].to(self.device)
            parse = batch['parse'].to(self.device)
            
            output = self.model(image, cloth, cloth_mask, agnostic, parse)
            
        # Create visualization grid
        n = min(4, image.size(0))
        
        # Denormalize images
        def denorm(x):
            return (x + 1) / 2
            
        grid = torch.cat([
            denorm(agnostic[:n]),       # Agnostic person
            denorm(cloth[:n]),          # Target cloth
            denorm(output['warped_cloth'][:n]),  # Warped cloth
            denorm(output['output'][:n]),        # Generated
            denorm(image[:n]),          # Ground truth
        ], dim=0)
        
        # Save grid
        save_path = os.path.join(self.sample_dir, f'epoch_{self.current_epoch}.png')
        vutils.save_image(grid, save_path, nrow=n)
        
        # Also add to tensorboard
        self.writer.add_image('samples', vutils.make_grid(grid, nrow=n), self.current_epoch)
        
        self.model.train()


def main():
    parser = argparse.ArgumentParser(description='Train HR-VITON')
    parser.add_argument('--config', type=str, default='configs/hr_viton_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize W&B if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.get('logging', {}).get('project_name', 'virtual-tryon'),
                config=config
            )
        except ImportError:
            print("Warning: wandb not installed. Skipping W&B logging.")
            
    # Create datasets
    image_size = tuple(config.get('model', {}).get('input_size', [1024, 768]))
    
    print(f"Loading datasets with image size: {image_size}")
    
    train_dataset = VITONHDDataset(
        data_root=config['data']['data_root'],
        split='train',
        image_size=image_size
    )
    
    val_dataset = VITONHDDataset(
        data_root=config['data']['data_root'],
        split='test',
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 8),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 8),
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = build_hr_viton(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer_config = {
        **config,
        'checkpoint_dir': config['checkpoint']['save_dir'],
        'log_dir': os.path.join(config['checkpoint']['save_dir'], 'logs'),
    }
    
    trainer = HRVITONTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=trainer_config
    )
    
    # Train
    trainer.train(
        num_epochs=config['training']['epochs'],
        resume_from=args.resume
    )
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
