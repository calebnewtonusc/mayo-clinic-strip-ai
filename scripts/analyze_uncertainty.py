"""Analyze model uncertainty using Monte Carlo dropout and test-time augmentation."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('..')

from torch.utils.data import DataLoader
from src.data.dataset import StrokeDataset
from src.data.augmentation import get_train_augmentation, get_val_augmentation
from src.models.cnn import ResNetClassifier, SimpleCNN, EfficientNetClassifier
from src.evaluation.uncertainty import (
    monte_carlo_dropout, test_time_augmentation,
    calculate_confidence_metrics, calibration_curve,
    expected_calibration_error, identify_uncertain_samples
)
from src.utils.helpers import load_config, get_device


def load_model(checkpoint_path, config):
    """Load trained model."""
    device = get_device()

    # Create model
    arch = config['model']['architecture']
    if arch == 'simple_cnn':
        model = SimpleCNN(
            in_channels=config['model']['in_channels'],
            num_classes=config['model']['num_classes']
        )
    elif 'efficientnet' in arch:
        model = EfficientNetClassifier(
            arch=arch,
            num_classes=config['model']['num_classes'],
            pretrained=False
        )
    else:
        model = ResNetClassifier(
            arch=arch,
            num_classes=config['model']['num_classes'],
            pretrained=False,
            in_channels=config['model']['in_channels']
        )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model, device


def analyze_mc_dropout(model, dataloader, device, n_iterations=30):
    """Analyze uncertainty using Monte Carlo dropout."""
    print(f"\nPerforming Monte Carlo Dropout ({n_iterations} iterations)...")

    all_mean_probs = []
    all_std_probs = []
    all_predictions_mc = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)

        # MC dropout
        mean_probs, std_probs, predictions_mc = monte_carlo_dropout(
            model, images, n_iterations=n_iterations, device=device
        )

        all_mean_probs.append(mean_probs)
        all_std_probs.append(std_probs)
        all_predictions_mc.append(predictions_mc)
        all_labels.append(labels.numpy())

    # Concatenate
    mean_probs = np.concatenate(all_mean_probs, axis=0)
    std_probs = np.concatenate(all_std_probs, axis=0)
    predictions_mc = np.concatenate(all_predictions_mc, axis=1)  # (n_iter, n_samples, n_classes)
    labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    metrics = calculate_confidence_metrics(mean_probs, predictions_mc)

    print("\nMonte Carlo Dropout Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Identify uncertain samples
    uncertain = identify_uncertain_samples(mean_probs, predictions_mc)
    print(f"\nUncertain samples: {uncertain.sum()} / {len(uncertain)} ({100*uncertain.sum()/len(uncertain):.1f}%)")

    return mean_probs, std_probs, predictions_mc, labels, uncertain


def analyze_tta(model, dataset, device, n_augmentations=10):
    """Analyze uncertainty using test-time augmentation."""
    print(f"\nPerforming Test-Time Augmentation ({n_augmentations} augmentations)...")

    augmentation_fn = get_train_augmentation(image_size=224, p=1.0)

    all_mean_probs = []
    all_std_probs = []
    all_labels = []

    # Sample subset for TTA (it's expensive)
    indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)

    for idx in indices:
        image, label = dataset[idx]
        image = image.unsqueeze(0)

        # TTA
        mean_probs, std_probs = test_time_augmentation(
            model, image, augmentation_fn, n_augmentations=n_augmentations, device=device
        )

        all_mean_probs.append(mean_probs)
        all_std_probs.append(std_probs)
        all_labels.append(label)

    mean_probs = np.concatenate(all_mean_probs, axis=0)
    std_probs = np.concatenate(all_std_probs, axis=0)
    labels = np.array(all_labels)

    # Calculate metrics
    metrics = calculate_confidence_metrics(mean_probs)

    print("\nTest-Time Augmentation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    return mean_probs, std_probs, labels


def plot_uncertainty_analysis(
    mean_probs,
    std_probs,
    labels,
    uncertain_mask,
    output_dir
):
    """Create uncertainty visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get predictions
    predictions = mean_probs.argmax(axis=1)
    confidences = mean_probs.max(axis=1)

    # Plot 1: Confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(confidences[predictions == labels], bins=30, alpha=0.7, label='Correct', density=True)
    axes[0].hist(confidences[predictions != labels], bins=30, alpha=0.7, label='Incorrect', density=True)
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Uncertainty vs Accuracy
    axes[1].scatter(confidences[predictions == labels], np.ones(sum(predictions == labels)),
                   alpha=0.5, label='Correct', s=20)
    axes[1].scatter(confidences[predictions != labels], np.zeros(sum(predictions != labels)),
                   alpha=0.5, label='Incorrect', s=20)
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Correctness')
    axes[1].set_title('Confidence vs Correctness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Calibration curve
    y_true_binary = (predictions == labels).astype(int)
    bin_confs, bin_accs = calibration_curve(y_true_binary, confidences, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(bin_confs, bin_accs, 'o-', label='Model')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calculate ECE
    ece = expected_calibration_error(y_true_binary, confidences)
    ax.text(0.05, 0.95, f'ECE: {ece:.4f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[checkmark.circle] Saved uncertainty analysis plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze model uncertainty')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--split', type=str, default='val',
                        help='Which split to use')
    parser.add_argument('--output_dir', type=str, default='experiments/uncertainty',
                        help='Output directory')
    parser.add_argument('--mc_iterations', type=int, default=30,
                        help='Number of MC dropout iterations')
    parser.add_argument('--tta_augmentations', type=int, default=10,
                        help='Number of TTA augmentations')
    parser.add_argument('--skip_tta', action='store_true',
                        help='Skip test-time augmentation (faster)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, device = load_model(args.checkpoint, config)

    # Create dataset
    print(f"Loading {args.split} dataset")
    dataset = StrokeDataset(
        data_dir=args.data_dir,
        split=args.split,
        split_file=f'data/splits/{args.split}.json',
        transform=get_val_augmentation(config['data']['image_size'])
    )

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    # Create dataloader for MC dropout
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # Monte Carlo Dropout Analysis
    print("="*60)
    print("MONTE CARLO DROPOUT ANALYSIS")
    print("="*60)

    mean_probs, std_probs, predictions_mc, labels, uncertain = analyze_mc_dropout(
        model, dataloader, device, n_iterations=args.mc_iterations
    )

    # Plot uncertainty analysis
    plot_uncertainty_analysis(
        mean_probs, std_probs, labels, uncertain,
        output_dir=Path(args.output_dir) / 'mc_dropout'
    )

    # Test-Time Augmentation Analysis (optional)
    if not args.skip_tta:
        print("\n" + "="*60)
        print("TEST-TIME AUGMENTATION ANALYSIS")
        print("="*60)

        tta_mean_probs, tta_std_probs, tta_labels = analyze_tta(
            model, dataset, device, n_augmentations=args.tta_augmentations
        )

        # Simple metrics
        tta_predictions = tta_mean_probs.argmax(axis=1)
        tta_accuracy = (tta_predictions == tta_labels).mean()
        print(f"\nTTA Accuracy: {tta_accuracy:.4f}")

    print(f"\n[checkmark.circle] Uncertainty analysis complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
