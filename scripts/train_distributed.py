"""Launch script for distributed training across multiple GPUs.

Usage:
    # Single machine, 4 GPUs:
    torchrun --nproc_per_node=4 scripts/train_distributed.py --config config/train_config.yaml

    # Multiple machines (2 machines, 4 GPUs each):
    # On machine 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=12355 scripts/train_distributed.py --config config/train_config.yaml
    # On machine 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=12355 scripts/train_distributed.py --config config/train_config.yaml
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.distributed_trainer import DistributedTrainer, setup_distributed, cleanup_distributed, get_world_info
from src.models.cnn import ResNetClassifier, EfficientNetClassifier, DenseNetClassifier
from src.utils.helpers import load_config, set_seed
from torch.utils.data.distributed import DistributedSampler


def create_distributed_dataloaders(config, rank, world_size):
    """Create dataloaders with DistributedSampler.

    Args:
        config: Configuration dictionary
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    from src.data.dataset import StrokeDataset
    from src.data.augmentation import get_train_transforms, get_val_transforms

    # Load data splits
    data_dir = Path(config['data']['data_dir'])
    train_split = data_dir / 'train_split.txt'
    val_split = data_dir / 'val_split.txt'
    test_split = data_dir / 'test_split.txt'

    # Create datasets
    train_dataset = StrokeDataset(
        str(train_split),
        transform=get_train_transforms(config.get('augmentation', {}))
    )
    val_dataset = StrokeDataset(
        str(val_split),
        transform=get_val_transforms()
    )
    test_dataset = StrokeDataset(
        str(test_split),
        transform=get_val_transforms()
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.get('seed', 42)
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Important for DDP
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Distributed training for Mayo Clinic STRIP AI')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Distributed backend')
    args = parser.parse_args()

    # Get distributed info from environment (set by torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Setup distributed training
    setup_distributed(rank, world_size, backend=args.backend)

    is_main_process = (rank == 0)

    # Load config
    config = load_config(args.config)

    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed + rank)  # Different seed per process

    if is_main_process:
        print(f"\n{'=' * 80}")
        print(f"Distributed Training Configuration")
        print(f"{'=' * 80}")
        print(f"World Size: {world_size}")
        print(f"Backend: {args.backend}")
        print(f"Model: {config['model']['architecture']}")
        print(f"Batch Size per GPU: {config['training']['batch_size']}")
        print(f"Effective Batch Size: {config['training']['batch_size'] * world_size}")
        print(f"{'=' * 80}\n")

    # Create model
    model_config = config['model']
    arch = model_config['architecture']
    num_classes = model_config.get('num_classes', 2)
    pretrained = model_config.get('pretrained', True)

    if 'resnet' in arch.lower():
        model = ResNetClassifier(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif 'efficientnet' in arch.lower():
        model = EfficientNetClassifier(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif 'densenet' in arch.lower():
        model = DenseNetClassifier(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Create distributed dataloaders
    train_loader, val_loader, test_loader = create_distributed_dataloaders(config, rank, world_size)

    if is_main_process:
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 0.0001)
        )
    elif optimizer_config['type'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0.0001)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")

    # Learning rate scheduler
    scheduler = None
    if 'scheduler' in config['training']:
        scheduler_config = config['training']['scheduler']
        if scheduler_config['type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs']
            )
        elif scheduler_config['type'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )

    # Create distributed trainer
    trainer = DistributedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        rank=rank,
        world_size=world_size,
        scheduler=scheduler,
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training'].get('early_stopping_patience', 10),
        checkpoint_dir=f"experiments/{config['experiment']['name']}/checkpoints",
        model_name=arch,
        num_classes=num_classes,
        mixed_precision=config['training'].get('mixed_precision', True),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        gradient_clip_value=config['training'].get('gradient_clip_value', None),
        warmup_epochs=config['training'].get('warmup_epochs', 0),
        sync_bn=config['training'].get('sync_bn', True)
    )

    # Train
    history = trainer.train()

    # Cleanup
    cleanup_distributed()

    if is_main_process:
        print("\n✓ Distributed training completed successfully!")


if __name__ == '__main__':
    main()
