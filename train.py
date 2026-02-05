"""Main training script for stroke classification."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import StrokeDataset
from src.data.augmentation import get_train_augmentation, get_val_augmentation
from src.models.cnn import ResNetClassifier, SimpleCNN
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, load_config, get_device, create_experiment_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train stroke classification model')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--experiment_name', type=str, default='experiment_001',
                        help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Create experiment directory
    exp_dir = create_experiment_dir('experiments', args.experiment_name)
    print(f'Experiment directory: {exp_dir}')

    # Create datasets
    print('Loading datasets...')
    train_transform = get_train_augmentation(
        image_size=config['data']['image_size']
    )
    val_transform = get_val_augmentation(
        image_size=config['data']['image_size']
    )

    train_dataset = StrokeDataset(
        data_dir=args.data_dir,
        split='train',
        transform=train_transform
    )

    val_dataset = StrokeDataset(
        data_dir=args.data_dir,
        split='val',
        transform=val_transform
    )

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    # Create data loaders
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

    # Create model
    print(f"Creating model: {config['model']['architecture']}")

    if config['model']['architecture'] == 'simple_cnn':
        model = SimpleCNN(
            in_channels=config['model']['in_channels'],
            num_classes=config['model']['num_classes']
        )
    else:
        model = ResNetClassifier(
            arch=config['model']['architecture'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            freeze_backbone=config['model']['freeze_backbone'],
            in_channels=config['model']['in_channels']
        )

    model = model.to(device)

    # Print model info
    from src.utils.helpers import count_parameters
    print(f'Total parameters: {count_parameters(model):,}')

    # Loss function
    if config['loss']['type'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss']['type'] == 'weighted_cross_entropy':
        if config['loss']['class_weights'] is not None:
            class_weights = torch.tensor(config['loss']['class_weights'], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted cross-entropy with class weights: {config['loss']['class_weights']}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Warning: weighted_cross_entropy specified but no class_weights provided, using standard CE")
    else:
        raise ValueError(f"Unknown loss type: {config['loss']['type']}")

    # Optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")

    # Learning rate scheduler
    scheduler = None
    scheduler_type = config['training'].get('scheduler', 'none').lower()

    if scheduler_type == 'step':
        # Validate scheduler_params exists
        if 'scheduler_params' not in config['training']:
            raise ValueError(
                "Scheduler type 'step' requires 'scheduler_params' in config with 'step_size' and 'gamma'"
            )
        params = config['training']['scheduler_params']
        if 'step_size' not in params or 'gamma' not in params:
            raise ValueError(
                f"StepLR scheduler requires 'step_size' and 'gamma' in scheduler_params. "
                f"Got: {list(params.keys())}"
            )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params['step_size'],
            gamma=params['gamma']
        )
        print(f"Using StepLR scheduler with step_size={params['step_size']}, gamma={params['gamma']}")
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
        print(f"Using CosineAnnealingLR scheduler with T_max={config['training']['num_epochs']}")
    elif scheduler_type in ['none', '']:
        print("No learning rate scheduler")
    else:
        raise ValueError(
            f"Unknown scheduler type: '{scheduler_type}'. "
            f"Supported: 'step', 'cosine', 'none'"
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        checkpoint_dir=str(exp_dir / 'checkpoints'),
        model_name=config['model']['architecture'],
        num_classes=config['model']['num_classes']
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Train model
    print('\nStarting training...')
    history = trainer.train()

    # Save final results
    import json
    results_path = exp_dir / 'results' / 'training_history.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining completed! Results saved to {results_path}')


if __name__ == '__main__':
    main()
