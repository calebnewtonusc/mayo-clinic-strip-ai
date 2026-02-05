"""Evaluation script for trained models."""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json

from src.data.dataset import StrokeDataset
from src.data.augmentation import get_val_augmentation
from src.models.cnn import ResNetClassifier, SimpleCNN
from src.evaluation.metrics import (
    calculate_metrics, calculate_patient_level_metrics,
    print_classification_report, calculate_confidence_intervals
)
from src.utils.helpers import load_config, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate stroke classification model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='experiments/evaluation',
                        help='Directory to save evaluation results')
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset."""
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get device
    device = get_device()

    # Load model
    print(f'Loading model from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model architecture
    if config['model']['architecture'] == 'simple_cnn':
        model = SimpleCNN(
            in_channels=config['model']['in_channels'],
            num_classes=config['model']['num_classes']
        )
    else:
        model = ResNetClassifier(
            arch=config['model']['architecture'],
            num_classes=config['model']['num_classes'],
            pretrained=False,  # Don't need pretrained weights for evaluation
            in_channels=config['model']['in_channels']
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create dataset
    print(f'Loading {args.split} dataset...')
    val_transform = get_val_augmentation(image_size=config['data']['image_size'])

    dataset = StrokeDataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=val_transform
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    print(f'Evaluating on {len(dataset)} samples...')

    # Evaluate
    y_true, y_pred, y_prob = evaluate_model(model, data_loader, device)

    # Calculate metrics
    print('\n' + '='*60)
    print('IMAGE-LEVEL EVALUATION')
    print('='*60)

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value:.4f}')

    print_classification_report(y_true, y_pred, class_names=['CE', 'LAA'])

    # Calculate confidence intervals for accuracy
    from sklearn.metrics import accuracy_score
    mean_acc, lower_ci, upper_ci = calculate_confidence_intervals(
        y_true, y_pred, accuracy_score, n_bootstrap=1000
    )
    print(f'\nAccuracy 95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]')

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': len(dataset),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'confidence_intervals': {
            'accuracy': {
                'mean': float(mean_acc),
                'lower': float(lower_ci),
                'upper': float(upper_ci)
            }
        }
    }

    results_path = output_dir / f'evaluation_results_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
