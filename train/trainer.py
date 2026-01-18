"""
Base Trainer class for Virtual Try-On models.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Callable
from tqdm import tqdm
import yaml
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Base trainer class with common functionality.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = None,
        config: Dict[str, Any] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Extract nested config values
        training_config = self.config.get('training', {})
        checkpoint_config = self.config.get('checkpoint', {})
        
        # Logging
        log_dir = self.config.get('log_dir', training_config.get('log_dir', './logs'))
        self.writer = SummaryWriter(log_dir)
        
        # Checkpoint directory - support both flat and nested config
        self.checkpoint_dir = checkpoint_config.get('save_dir', 
                              self.config.get('checkpoint_dir', './checkpoints'))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save interval
        self.save_interval = training_config.get('save_interval', 
                             self.config.get('save_interval', 10))
        
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step. Must be implemented by subclass."""
        pass
        
    @abstractmethod
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step. Must be implemented by subclass."""
        pass
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            losses = self.train_step(batch)
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({k: f'{v:.4f}' for k, v in losses.items()})
            
            # Log to tensorboard
            if self.global_step % self.config.get('log_interval', 100) == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                    
            self.global_step += 1
            
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            
        return epoch_losses
        
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        
        val_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                losses = self.validate_step(batch)
                
                for k, v in losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v
                num_batches += 1
                
        # Average losses
        for k in val_losses:
            val_losses[k] /= num_batches
            self.writer.add_scalar(f'val/{k}', val_losses[k], self.current_epoch)
            
        return val_losses
        
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Optional checkpoint path to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)
            
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            print(f"Epoch {epoch} train losses: {train_losses}")
            
            # Validate
            val_losses = self.validate()
            if val_losses:
                print(f"Epoch {epoch} val losses: {val_losses}")
                
            # Save checkpoint at intervals
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')
                
            # Save best model based on loss
            metric = val_losses.get('total', train_losses.get('total', float('inf')))
            if metric < self.best_metric:
                self.best_metric = metric
                self.save_checkpoint('best.pth')
                print(f"New best model saved with metric: {metric:.4f}")
                
        # Save final model
        self.save_checkpoint('final.pth')
        self.writer.close()
        
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        # Add optimizer states if available
        if hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if hasattr(self, 'optimizer_G'):
            checkpoint['optimizer_G_state_dict'] = self.optimizer_G.state_dict()
        if hasattr(self, 'optimizer_D'):
            checkpoint['optimizer_D_state_dict'] = self.optimizer_D.state_dict()
            
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        # Load optimizer states if available
        if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hasattr(self, 'optimizer_G') and 'optimizer_G_state_dict' in checkpoint:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        if hasattr(self, 'optimizer_D') and 'optimizer_D_state_dict' in checkpoint:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(
    config: Dict[str, Any],
    high_res: bool = False
) -> tuple:
    """Create train and validation dataloaders from config."""
    from data.dataset import get_dataloader
    
    data_config = config.get('data', {})
    
    train_loader = get_dataloader(
        data_root=data_config.get('data_root', './data/viton'),
        split='train',
        batch_size=config.get('training', {}).get('batch_size', 8),
        num_workers=data_config.get('num_workers', 4),
        image_size=tuple(config.get('model', {}).get('input_size', (256, 192))),
        shuffle=True,
        high_res=high_res
    )
    
    val_loader = get_dataloader(
        data_root=data_config.get('data_root', './data/viton'),
        split='test',
        batch_size=config.get('training', {}).get('batch_size', 8),
        num_workers=data_config.get('num_workers', 4),
        image_size=tuple(config.get('model', {}).get('input_size', (256, 192))),
        shuffle=False,
        high_res=high_res
    )
    
    return train_loader, val_loader
