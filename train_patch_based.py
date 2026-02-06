#!/usr/bin/env python3
"""
Patch-based training with Hugging Face vision transformers.
Better for whole-slide pathology images - extracts patches instead of resizing.
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTConfig
from tqdm import tqdm

from src.data.patch_dataset import PatchDataset, get_patch_transforms


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    return running_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser(description='Patch-based training')
    parser.add_argument('--config', type=str, default='config/mayo_patch_config.yaml')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create datasets
    train_transform = get_patch_transforms(config, train=True)
    val_transform = get_patch_transforms(config, train=False)

    train_dataset = PatchDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        patch_size=config['data']['patch_size'],
        num_patches_per_image=config['data']['num_patches_per_image'],
        transform=train_transform,
        mode=config['data']['patch_mode']
    )

    val_dataset = PatchDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        patch_size=config['data']['patch_size'],
        num_patches_per_image=config['data']['num_patches_per_image'],
        transform=val_transform,
        mode=config['data']['patch_mode']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # Create model - Vision Transformer
    print(f"Loading model: {config['model']['model_name']}")

    if config['model']['pretrained']:
        model = ViTForImageClassification.from_pretrained(
            config['model']['model_name'],
            num_labels=2,
            ignore_mismatched_sizes=True
        )
    else:
        vit_config = ViTConfig(
            image_size=config['data']['patch_size'],
            num_labels=2
        )
        model = ViTForImageClassification(vit_config)

    model = model.to(device)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Loss function with class weights
    if config['loss']['class_weights']:
        class_weights = torch.tensor(config['loss']['class_weights'], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted cross-entropy with class weights: {config['loss']['class_weights']}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training'].get('min_learning_rate', 1e-6)
    )

    # Resume from checkpoint if provided
    start_epoch = 1
    best_val_acc = 0.0
    patience_counter = 0

    if args.resume:
        print(f'Loading checkpoint from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_acc']
        print(f'Resuming from epoch {start_epoch} (best val acc: {best_val_acc:.4f})')

    # Training loop
    os.makedirs('experiments/patch_based/checkpoints', exist_ok=True)

    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        print(f'\nEpoch {epoch}/{config["training"]["num_epochs"]}')
        print('-' * 50)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Learning Rate: {current_lr:.6f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, 'experiments/patch_based/checkpoints/best_model.pth')
            print(f'✓ Saved best model (val_acc: {val_acc:.4f})')
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f'\nEarly stopping triggered after {epoch} epochs')
            break

    print(f'\nTraining complete!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()
