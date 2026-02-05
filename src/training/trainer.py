"""Training loop implementation."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Dict, Optional, Callable
from tqdm import tqdm
import numpy as np
from pathlib import Path


class Trainer:
    """Training and validation loop handler."""

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
        num_classes: int = 2
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        # Convert checkpoint_dir to absolute path to avoid issues when running from different directories
        self.checkpoint_dir = str(Path(checkpoint_dir).resolve())
        self.model_name = model_name
        self.num_classes = num_classes

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%' if total > 0 else '0.00%'
                })

        # Handle empty batches
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        return {'loss': epoch_loss, 'acc': epoch_acc}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'arch': self.model_name,
            'num_classes': self.num_classes,
            'model_type': type(self.model).__name__
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        checkpoint_path = f'{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = f'{self.checkpoint_dir}/best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'Saved best model to {best_path}')

    def train(self) -> Dict[str, list]:
        """Run full training loop."""
        print(f'Starting training for {self.num_epochs} epochs...')

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            print(f'\nEpoch {self.current_epoch}/{self.num_epochs}')
            print('-' * 50)

            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])

            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Print epoch summary
            print(f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["acc"]:.4f}')
            print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["acc"]:.4f}')

            # Check for improvement
            is_best = val_metrics['acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['acc']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        print('\nTraining completed!')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')

        return self.history
