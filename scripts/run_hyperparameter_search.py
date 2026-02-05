"""Run hyperparameter search for model optimization."""

import argparse
import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.dataset import StrokeDataset
from src.data.augmentation import get_train_augmentation, get_val_augmentation
from src.models.cnn import ResNetClassifier, SimpleCNN
from src.training.trainer import Trainer
from src.training.hyperparameter_search import GridSearch, RandomSearch, sample_loguniform, sample_choice, sample_int
from src.utils.helpers import load_config, get_device, set_seed


def train_with_params(params: dict, base_config: dict, data_loaders: dict, device):
    """Training function for hyperparameter search.

    Args:
        params: Hyperparameters to test
        base_config: Base configuration
        data_loaders: Dict with train and val loaders
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    # Update config with search params
    config = base_config.copy()
    config['training'].update(params)

    # Create model
    if config['model']['architecture'] == 'simple_cnn':
        model = SimpleCNN(
            in_channels=config['model']['in_channels'],
            num_classes=config['model']['num_classes']
        )
    else:
        model = ResNetClassifier(
            arch=config['model']['architecture'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained']
        )

    model = model.to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )

    # Scheduler
    scheduler = None
    if config['training'].get('scheduler') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )

    # Create trainer with reduced epochs for search
    trainer = Trainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        num_epochs=min(config['training']['num_epochs'], 30),  # Cap at 30 for search
        early_stopping_patience=10,
        checkpoint_dir='experiments/hp_search/temp'
    )

    # Train
    history = trainer.train()

    # Return best validation metrics
    best_val_acc = max(history['val_acc'])
    best_val_loss = min(history['val_loss'])

    return {
        'val_acc': best_val_acc,
        'val_loss': best_val_loss,
        'train_acc': history['train_acc'][-1],
        'train_loss': history['train_loss'][-1]
    }


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Base config file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--search_type', type=str, default='grid',
                        choices=['grid', 'random'],
                        help='Search strategy')
    parser.add_argument('--n_iterations', type=int, default=20,
                        help='Number of iterations for random search')
    parser.add_argument('--output_dir', type=str, default='experiments/hp_search',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load base config
    config = load_config(args.config)
    device = get_device()

    # Load datasets
    print("Loading datasets...")
    train_transform = get_train_augmentation(config['data']['image_size'])
    val_transform = get_val_augmentation(config['data']['image_size'])

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    data_loaders = {'train': train_loader, 'val': val_loader}

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Define training function
    def train_fn(params):
        return train_with_params(params, config, data_loaders, device)

    # Perform search
    if args.search_type == 'grid':
        print("\n" + "="*60)
        print("GRID SEARCH")
        print("="*60)

        # Define parameter grid
        param_grid = {
            'learning_rate': [1e-4, 5e-4, 1e-3],
            'weight_decay': [0, 1e-4, 1e-3],
            'optimizer': ['adam', 'adamw']
        }

        search = GridSearch(
            train_fn=train_fn,
            param_grid=param_grid,
            metric='val_acc',
            maximize=True
        )

    else:  # random search
        print("\n" + "="*60)
        print("RANDOM SEARCH")
        print("="*60)

        # Define parameter distributions
        param_distributions = {
            'learning_rate': sample_loguniform(1e-5, 1e-2),
            'weight_decay': sample_loguniform(1e-6, 1e-2),
            'optimizer': sample_choice(['adam', 'adamw', 'sgd'])
        }

        search = RandomSearch(
            train_fn=train_fn,
            param_distributions=param_distributions,
            n_iterations=args.n_iterations,
            metric='val_acc',
            maximize=True,
            seed=args.seed
        )

    # Run search
    best_params = search.search()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    search.save_results(output_dir / f'{args.search_type}_search_results.json')

    # Print summary
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*60)
    print(f"\nBest Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    best_result = search.get_best_params()
    best_metrics = [r for r in search.results if r['params'] == best_result][0]['metrics']

    print(f"\nBest Validation Accuracy: {best_metrics['val_acc']:.4f}")
    print(f"Best Validation Loss: {best_metrics['val_loss']:.4f}")

    print(f"\n✅ Results saved to {output_dir}")
    print("\nNext steps:")
    print("1. Update your config file with best parameters")
    print("2. Train full model: python train.py --config config/default_config.yaml")


if __name__ == '__main__':
    main()
