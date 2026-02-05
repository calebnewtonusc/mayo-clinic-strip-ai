"""Advanced training loop with mixed precision and gradient accumulation.

Production-grade training features for better performance and efficiency.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
from tqdm import tqdm
import numpy as np
from pathlib import Path


class AdvancedTrainer:
    """Advanced trainer with mixed precision and gradient accumulation.

    Features:
    - Mixed precision training (FP16) for 2-3x speedup
    - Gradient accumulation for larger effective batch sizes
    - Gradient clipping for training stability
    - Learning rate warmup
    - Advanced metrics tracking
    - TensorBoard integration

    Example:
        >>> trainer = AdvancedTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     device=device,
        ...     mixed_precision=True,
        ...     gradient_accumulation_steps=4
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
        device: torch.device,
        scheduler: Optional[Callable] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = './experiments/checkpoints',
        model_name: str = 'unknown',
        num_classes: int = 2,
        # Advanced features
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clip_value: Optional[float] = None,
        warmup_epochs: int = 0,
        use_tensorboard: bool = False,
        log_interval: int = 10
    ):
        """Initialize advanced trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
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
            use_tensorboard: Whether to log to TensorBoard
            log_interval: How often to log metrics
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
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
        self.use_tensorboard = use_tensorboard
        self.log_interval = log_interval

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

        # TensorBoard writer
        self.writer = None
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'logs'))

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Advanced Trainer initialized:")
        print(f"  - Mixed Precision: {self.mixed_precision}")
        print(f"  - Gradient Accumulation: {self.gradient_accumulation_steps} steps")
        print(f"  - Gradient Clipping: {self.gradient_clip_value}")
        print(f"  - Warmup Epochs: {self.warmup_epochs}")
        print(f"  - TensorBoard: {self.use_tensorboard}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with advanced features."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.current_epoch}')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Mixed precision forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass with mixed precision
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

                # Log to TensorBoard
                if self.writer and self.global_step % self.log_interval == 0:
                    self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Train/LearningRate', lr, self.global_step)

            # Statistics
            running_loss += loss.item() * images.size(0) * self.gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1

            # Update progress bar
            if total > 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'acc': f'{100 * correct / total:.2f}%',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validation Epoch {self.current_epoch}')
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

                # Update progress bar
                if total > 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * correct / total:.2f}%'
                    })

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def _adjust_learning_rate_warmup(self, epoch: int):
        """Linear warmup for first few epochs."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with advanced trainer state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'arch': self.model_name,
            'num_classes': self.num_classes,
            'model_type': type(self.model).__name__,
            'global_step': self.global_step,
            # Advanced trainer state
            'mixed_precision': self.mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
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
        """Run full training loop with advanced features."""
        print(f'\n{"=" * 80}')
        print(f'Starting advanced training for {self.num_epochs} epochs')
        print(f'{"=" * 80}\n')

        # Store initial learning rates for warmup
        if self.warmup_epochs > 0:
            for param_group in self.optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1

            # Learning rate warmup
            if self.warmup_epochs > 0:
                self._adjust_learning_rate_warmup(epoch)

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

            # Learning rate scheduling (after warmup)
            if self.scheduler is not None and epoch >= self.warmup_epochs:
                self.scheduler.step()

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Epoch/TrainLoss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/TrainAcc', train_metrics['acc'], epoch)
                self.writer.add_scalar('Epoch/ValLoss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Epoch/ValAcc', val_metrics['acc'], epoch)
                self.writer.add_scalar('Epoch/LearningRate', current_lr, epoch)

            # Print epoch summary
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
                print(f'  ✓ New best model! Val Acc: {self.best_val_acc:.4f}')
            else:
                self.patience_counter += 1
                print(f'  Patience: {self.patience_counter}/{self.early_stopping_patience}')

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f'\n⚠ Early stopping triggered after {epoch + 1} epochs')
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        print(f'\n{"=" * 80}')
        print('Training completed!')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        print(f'Best validation loss: {self.best_val_loss:.4f}')
        print(f'{"=" * 80}\n')

        return self.history
