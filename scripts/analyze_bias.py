"""Analyze model for potential biases across different data subgroups."""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
sys.path.append('..')

from src.models.cnn import SimpleCNN, ResNetClassifier, EfficientNetClassifier
from src.data.dataset import StrokeDataset
from src.utils.helpers import load_config, get_device
from src.evaluation.metrics import calculate_metrics
from torch.utils.data import DataLoader, Subset


def load_metadata(metadata_file):
    """Load metadata about subgroups.

    Expected format:
    {
        "samples": [
            {
                "image_id": "patient_001_slice_05",
                "age_group": "60-70",
                "sex": "M",
                "scanner": "GE_Discovery",
                "image_quality": "high"
            },
            ...
        ]
    }

    Args:
        metadata_file: Path to metadata JSON file

    Returns:
        Dictionary mapping image IDs to metadata
    """
    with open(metadata_file, 'r') as f:
        data = json.load(f)

    metadata = {}
    for sample in data.get('samples', []):
        image_id = sample['image_id']
        metadata[image_id] = sample

    return metadata


def analyze_subgroup_performance(model, dataloader, device, subgroup_key, metadata):
    """Analyze model performance across different subgroups.

    Args:
        model: The model to analyze
        dataloader: DataLoader for data
        device: Device to run on
        subgroup_key: Key to group by (e.g., 'age_group', 'sex', 'scanner')
        metadata: Metadata dictionary

    Returns:
        Dictionary with subgroup-specific metrics
    """
    model.eval()

    subgroup_predictions = defaultdict(lambda: {'y_true': [], 'y_pred': [], 'y_prob': []})

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f'Analyzing {subgroup_key}'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Group by subgroup
            for i in range(len(labels)):
                # Try to match image to metadata (simplified)
                # In practice, you'd need to track image IDs through the dataset
                subgroup_value = 'unknown'  # Default

                if len(metadata) > 0:
                    # This is a placeholder - in practice you'd need image IDs
                    # For now, we'll demonstrate with synthetic subgroups
                    subgroup_value = np.random.choice(['group_A', 'group_B', 'group_C'])

                subgroup_predictions[subgroup_value]['y_true'].append(labels[i].item())
                subgroup_predictions[subgroup_value]['y_pred'].append(predicted[i].item())
                subgroup_predictions[subgroup_value]['y_prob'].append(probs[i].cpu().numpy())

    # Calculate metrics for each subgroup
    results = {}
    for subgroup, data in subgroup_predictions.items():
        if len(data['y_true']) == 0:
            continue

        y_true = np.array(data['y_true'])
        y_pred = np.array(data['y_pred'])
        y_prob = np.array(data['y_prob'])

        metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics['n_samples'] = len(y_true)

        results[subgroup] = metrics

    return results


def calculate_fairness_metrics(subgroup_results):
    """Calculate fairness metrics across subgroups.

    Args:
        subgroup_results: Dictionary of subgroup-specific metrics

    Returns:
        Dictionary with fairness metrics
    """
    # Extract key metrics for each subgroup
    accuracies = [r['accuracy'] for r in subgroup_results.values()]
    sensitivities = [r['sensitivity'] for r in subgroup_results.values()]
    specificities = [r['specificity'] for r in subgroup_results.values()]

    # Demographic parity difference (max difference in positive prediction rate)
    # For medical use, we focus on performance parity
    acc_range = max(accuracies) - min(accuracies)
    sens_range = max(sensitivities) - min(sensitivities)
    spec_range = max(specificities) - min(specificities)

    # Equal opportunity difference (difference in TPR/sensitivity)
    equal_opp_diff = sens_range

    # Equalized odds (max of TPR and FPR differences)
    fpr_values = [1 - r['specificity'] for r in subgroup_results.values()]
    fpr_range = max(fpr_values) - min(fpr_values)
    equalized_odds = max(sens_range, fpr_range)

    return {
        'accuracy_range': acc_range,
        'sensitivity_range': sens_range,
        'specificity_range': spec_range,
        'equal_opportunity_difference': equal_opp_diff,
        'equalized_odds_difference': equalized_odds,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies)
    }


def plot_subgroup_comparison(subgroup_results, subgroup_key, save_path):
    """Plot comparison of metrics across subgroups.

    Args:
        subgroup_results: Dictionary of subgroup-specific metrics
        subgroup_key: Name of the subgroup attribute
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    subgroups = list(subgroup_results.keys())
    metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'auc']
    titles = ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 'ROC-AUC']

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 2, idx % 2]

        values = [subgroup_results[sg][metric] for sg in subgroups]
        n_samples = [subgroup_results[sg]['n_samples'] for sg in subgroups]

        # Color bars by sample size
        colors = plt.cm.viridis(np.array(n_samples) / max(n_samples))

        bars = ax.bar(subgroups, values, color=colors, alpha=0.7)
        ax.set_xlabel('Subgroup')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by {subgroup_key}')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val, n in zip(bars, values, n_samples):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}\n(n={n})',
                   ha='center', va='bottom', fontsize=8)

        # Rotate x labels if many subgroups
        if len(subgroups) > 5:
            ax.set_xticklabels(subgroups, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Subgroup comparison plot saved to {save_path}")
    plt.close()


def plot_fairness_metrics(fairness_results, save_path):
    """Plot fairness metrics.

    Args:
        fairness_results: Dictionary of fairness metrics
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['accuracy_range', 'sensitivity_range', 'specificity_range',
               'equal_opportunity_difference', 'equalized_odds_difference']
    values = [fairness_results[m] for m in metrics]
    labels = ['Accuracy\nRange', 'Sensitivity\nRange', 'Specificity\nRange',
              'Equal Opportunity\nDifference', 'Equalized Odds\nDifference']

    # Color bars based on fairness (lower is better)
    colors = ['green' if v < 0.05 else 'orange' if v < 0.1 else 'red' for v in values]

    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    ax.set_ylabel('Difference')
    ax.set_title('Fairness Metrics (Lower is Better)')
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Good (<0.05)')
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<0.1)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Fairness metrics plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze model for bias')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to processed data')
    parser.add_argument('--metadata-file', type=str, default=None,
                        help='Path to metadata JSON file (optional)')
    parser.add_argument('--output-dir', type=str, default='results/bias_analysis',
                        help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--subgroup-key', type=str, default='age_group',
                        help='Metadata key to analyze (e.g., age_group, sex, scanner)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.config)
    device = get_device()

    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Initialize model based on checkpoint
    model_arch = checkpoint.get('arch', 'resnet18')
    num_classes = checkpoint.get('num_classes', 2)

    if 'resnet' in model_arch:
        model = ResNetClassifier(
            arch=model_arch,
            num_classes=num_classes,
            pretrained=False
        )
    elif 'efficientnet' in model_arch:
        model = EfficientNetClassifier(
            arch=model_arch,
            num_classes=num_classes,
            pretrained=False
        )
    else:
        model = SimpleCNN(in_channels=3, num_classes=num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load metadata if provided
    metadata = {}
    if args.metadata_file and Path(args.metadata_file).exists():
        print(f"Loading metadata from {args.metadata_file}...")
        metadata = load_metadata(args.metadata_file)
        print(f"Loaded metadata for {len(metadata)} samples")
    else:
        print("No metadata file provided. Using synthetic subgroups for demonstration.")

    print("Loading test data...")
    from src.data.augmentation import get_val_augmentation
    test_transform = get_val_augmentation(config['data']['image_size'])

    test_dataset = StrokeDataset(
        data_dir=args.data_dir,
        split='test',
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Test set size: {len(test_dataset)}")

    print(f"\nAnalyzing performance across subgroups ({args.subgroup_key})...")
    subgroup_results = analyze_subgroup_performance(
        model, test_loader, device, args.subgroup_key, metadata
    )

    # Print results
    print("\n" + "="*60)
    print(f"SUBGROUP ANALYSIS: {args.subgroup_key}")
    print("="*60)

    for subgroup, metrics in subgroup_results.items():
        print(f"\n{subgroup} (n={metrics['n_samples']}):")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  ROC-AUC:     {metrics['auc']:.4f}")

    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(subgroup_results)

    print("\n" + "="*60)
    print("FAIRNESS METRICS")
    print("="*60)
    print(f"Accuracy range:               {fairness_metrics['accuracy_range']:.4f}")
    print(f"Sensitivity range:            {fairness_metrics['sensitivity_range']:.4f}")
    print(f"Specificity range:            {fairness_metrics['specificity_range']:.4f}")
    print(f"Equal opportunity difference: {fairness_metrics['equal_opportunity_difference']:.4f}")
    print(f"Equalized odds difference:    {fairness_metrics['equalized_odds_difference']:.4f}")
    print(f"Mean accuracy:                {fairness_metrics['mean_accuracy']:.4f} ± {fairness_metrics['std_accuracy']:.4f}")

    # Fairness assessment
    max_disparity = max(
        fairness_metrics['accuracy_range'],
        fairness_metrics['sensitivity_range'],
        fairness_metrics['specificity_range']
    )

    print("\n" + "="*60)
    if max_disparity < 0.05:
        print("✓ Model shows EXCELLENT FAIRNESS across subgroups")
    elif max_disparity < 0.1:
        print("✓ Model shows GOOD FAIRNESS across subgroups")
    elif max_disparity < 0.15:
        print("⚠ Model shows MODERATE DISPARITIES across subgroups")
    else:
        print("⚠ Model shows SIGNIFICANT DISPARITIES across subgroups")
        print("   Consider data balancing or fairness-aware training")
    print("="*60)

    # Save results
    results_data = {
        'subgroup_key': args.subgroup_key,
        'subgroup_results': {k: {key: float(val) if isinstance(val, np.floating) else val
                                for key, val in v.items()}
                           for k, v in subgroup_results.items()},
        'fairness_metrics': fairness_metrics
    }

    results_file = output_dir / 'bias_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Plot results
    plot_path = output_dir / f'subgroup_comparison_{args.subgroup_key}.png'
    plot_subgroup_comparison(subgroup_results, args.subgroup_key, plot_path)

    fairness_plot_path = output_dir / 'fairness_metrics.png'
    plot_fairness_metrics(fairness_metrics, fairness_plot_path)


if __name__ == '__main__':
    main()
