"""Distributed training with multi-GPU support using PyTorch DDP.

Production-grade distributed training for scaling across multiple GPUs.
Supports both single-node multi-GPU and multi-node training.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import socket


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl',
                      master_addr: Optional[str] = None, master_port: Optional[str] = None):
    """Initialize distributed process group.

    Args:
        rank: Rank of current process
        world_size: Total number of processes
        backend: Backend for distributed training ('nccl' for GPU, 'gloo' for CPU)
        master_addr: Master node address (defaults to 'localhost' for single-node)
        master_port: Master node port (defaults to '12355')

    Note:
        For multi-node training, you MUST provide master_addr as the IP/hostname
        of the master node. Default 'localhost' only works for single-node multi-GPU.
    """
    # Set environment variables for distributed training with validation
    if 'MASTER_ADDR' not in os.environ:
        addr = master_addr if master_addr is not None else 'localhost'
        os.environ['MASTER_ADDR'] = addr
        if addr == 'localhost' and world_size > torch.cuda.device_count():
            print(f"[Rank {rank}] WARNING: Using MASTER_ADDR='localhost' with world_size={world_size} "
                  f"but only {torch.cuda.device_count()} GPUs detected. "
                  f"For multi-node training, set master_addr to the master node IP/hostname.")

    if 'MASTER_PORT' not in os.environ:
        port = master_port if master_port is not None else '12355'
        os.environ['MASTER_PORT'] = port

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Initialized distributed process group (world_size={world_size})")


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_world_info() -> Dict[str, int]:
    """Get information about distributed world.

    Returns:
        Dictionary with rank, world_size, and local_rank
    """
    if not dist.is_initialized():
        return {'rank': 0, 'world_size': 1, 'local_rank': 0}

    return {
        'rank': dist.get_rank(),
        'world_size': dist.get_world_size(),
        'local_rank': int(os.environ.get('LOCAL_RANK', 0))
    }


class DistributedTrainer:
    """Distributed trainer with multi-GPU support using DDP.

    Features:
    - Multi-GPU training with DistributedDataParallel
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Automatic data distribution across GPUs
    - Synchronized batch normalization
    - Gradient clipping and warmup

    Example:
        >>> # Launch with torchrun:
        >>> # torchrun --nproc_per_node=4 train_distributed.py
        >>>
        >>> trainer = DistributedTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     rank=rank,
        ...     world_size=world_size
        ... )
        >>> history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        rank: int,
        world_size: int,
        device: Optional[torch.device] = None,
        scheduler: Optional[Callable] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = './experiments/distributed',
        model_name: str = 'unknown',
        num_classes: int = 2,
        # Advanced features
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clip_value: Optional[float] = None,
        warmup_epochs: int = 0,
        sync_bn: bool = True,
        find_unused_parameters: bool = False
    ):
        """Initialize distributed trainer.

        Args:
            model: Model to train (will be wrapped in DDP)
            train_loader: Training data loader (should use DistributedSampler)
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            rank: Rank of current process
            world_size: Total number of processes
            device: Device to use (defaults to cuda:rank)
            scheduler: Optional learning rate scheduler
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            model_name: Name of model architecture
            num_classes: Number of output classes
            mixed_precision: Whether to use mixed precision (FP16)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            gradient_clip_value: Max gradient norm (None = no clipping)
            warmup_epochs: Number of warmup epochs
            sync_bn: Whether to convert BatchNorm to SyncBatchNorm
            find_unused_parameters: Whether to find unused parameters in DDP
        """
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        # Device setup
        if device is None:
            self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Model setup with DDP
        model = model.to(self.device)

        # Convert BatchNorm to SyncBatchNorm for better multi-GPU training
        if sync_bn and world_size > 1 and torch.cuda.is_available():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if self.is_main_process:
                print("✓ Converted BatchNorm to SyncBatchNorm")

        # Wrap model with DDP
        if world_size > 1:
            self.model = DDP(
                model,
                device_ids=[rank] if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused_parameters
            )
            if self.is_main_process:
                print(f"✓ Wrapped model with DistributedDataParallel (world_size={world_size})")
        else:
            self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.num_classes = num_classes

        # Advanced features
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_value = gradient_clip_value
        self.warmup_epochs = warmup_epochs

        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None

        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        self.global_step = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Create checkpoint directory (only on main process)
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nDistributed Trainer initialized:")
            print(f"  - Rank: {self.rank}/{self.world_size}")
            print(f"  - Device: {self.device}")
            print(f"  - Mixed Precision: {self.mixed_precision}")
            print(f"  - Gradient Accumulation: {self.gradient_accumulation_steps} steps")
            print(f"  - Gradient Clipping: {self.gradient_clip_value}")
            print(f"  - Warmup Epochs: {self.warmup_epochs}")
            print(f"  - SyncBatchNorm: {sync_bn}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with distributed training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Set epoch for DistributedSampler (ensures different shuffling each epoch)
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        # Only show progress bar on main process
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.current_epoch}')
        else:
            pbar = self.train_loader

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Mixed precision forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights after accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip_value is not None:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_value
                    )

                # Optimizer step
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Statistics
            running_loss += loss.item() * images.size(0) * self.gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar (main process only)
            if self.is_main_process and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'acc': f'{100 * correct / total:.2f}%',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

        # Gather statistics across all processes
        total_tensor = torch.tensor([total], device=self.device)
        correct_tensor = torch.tensor([correct], device=self.device)
        loss_tensor = torch.tensor([running_loss], device=self.device)

        if self.world_size > 1:
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        epoch_loss = loss_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0
        epoch_acc = correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def validate(self) -> Dict[str, float]:
        """Validate the model with distributed support."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            if self.is_main_process:
                pbar = tqdm(self.val_loader, desc=f'Validation Epoch {self.current_epoch}')
            else:
                pbar = self.val_loader

            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Mixed precision inference
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar (main process only)
                if self.is_main_process and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * correct / total:.2f}%'
                    })

        # Gather statistics across all processes
        total_tensor = torch.tensor([total], device=self.device)
        correct_tensor = torch.tensor([correct], device=self.device)
        loss_tensor = torch.tensor([running_loss], device=self.device)

        if self.world_size > 1:
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        epoch_loss = loss_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0
        epoch_acc = correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return

        # Get model state dict (unwrap DDP if necessary)
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'arch': self.model_name,
            'num_classes': self.num_classes,
            'global_step': self.global_step,
            'world_size': self.world_size,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'✓ Saved best model to {best_path}')

    def train(self) -> Dict[str, list]:
        """Run full distributed training loop."""
        if self.is_main_process:
            print(f'\n{"=" * 80}')
            print(f'Starting distributed training on {self.world_size} GPUs')
            print(f'{"=" * 80}\n')

        # Store initial learning rates for warmup
        if self.warmup_epochs > 0:
            for param_group in self.optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1

            # Learning rate warmup
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['initial_lr'] * warmup_factor

            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])

            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # Learning rate scheduling
            if self.scheduler is not None and epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Print epoch summary (main process only)
            if self.is_main_process:
                print(f'\nEpoch {self.current_epoch}/{self.num_epochs} Summary:')
                print(f'  Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["acc"]:.4f}')
                print(f'  Val   - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["acc"]:.4f}')
                print(f'  LR: {current_lr:.2e}')

            # Check for improvement
            is_best = val_metrics['acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['acc']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                if self.is_main_process:
                    print(f'  ✓ New best model! Val Acc: {self.best_val_acc:.4f}')
            else:
                self.patience_counter += 1
                if self.is_main_process:
                    print(f'  Patience: {self.patience_counter}/{self.early_stopping_patience}')

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                if self.is_main_process:
                    print(f'\n⚠ Early stopping triggered after {epoch + 1} epochs')
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        if self.is_main_process:
            print(f'\n{"=" * 80}')
            print('Distributed training completed!')
            print(f'Best validation accuracy: {self.best_val_acc:.4f}')
            print(f'Best validation loss: {self.best_val_loss:.4f}')
            print(f'{"=" * 80}\n')

        return self.history
