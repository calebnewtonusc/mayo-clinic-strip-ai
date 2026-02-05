"""Analyze model robustness to different perturbations and corruptions."""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.models.cnn import SimpleCNN, ResNetClassifier, EfficientNetClassifier
from src.data.dataset import StrokeDataset
from src.utils.helpers import load_config, get_device
from torch.utils.data import DataLoader


def add_gaussian_noise(images, std=0.1):
    """Add Gaussian noise to images.

    Args:
        images: Input images tensor
        std: Standard deviation of noise

    Returns:
        Noisy images
    """
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0, 1)


def add_salt_pepper_noise(images, amount=0.05):
    """Add salt and pepper noise to images.

    Args:
        images: Input images tensor
        amount: Fraction of pixels to corrupt

    Returns:
        Noisy images
    """
    noisy = images.clone()

    # Salt noise
    num_salt = int(amount * images.numel() * 0.5)
    salt_coords = [torch.randint(0, i, (num_salt,)) for i in images.shape]
    noisy[salt_coords] = 1

    # Pepper noise
    num_pepper = int(amount * images.numel() * 0.5)
    pepper_coords = [torch.randint(0, i, (num_pepper,)) for i in images.shape]
    noisy[pepper_coords] = 0

    return noisy


def add_blur(images, kernel_size=5):
    """Add Gaussian blur to images.

    Args:
        images: Input images tensor
        kernel_size: Size of blur kernel

    Returns:
        Blurred images
    """
    from torchvision.transforms.functional import gaussian_blur
    return gaussian_blur(images, kernel_size)


def adjust_brightness(images, factor=1.5):
    """Adjust image brightness.

    Args:
        images: Input images tensor
        factor: Brightness adjustment factor (>1 = brighter, <1 = darker)

    Returns:
        Adjusted images
    """
    return torch.clamp(images * factor, 0, 1)


def adjust_contrast(images, factor=1.5):
    """Adjust image contrast.

    Args:
        images: Input images tensor
        factor: Contrast adjustment factor

    Returns:
        Adjusted images
    """
    mean = images.mean(dim=[2, 3], keepdim=True)
    return torch.clamp((images - mean) * factor + mean, 0, 1)


def test_robustness(model, dataloader, device, corruption_fn, corruption_name):
    """Test model robustness to a specific corruption.

    Args:
        model: The model to test
        dataloader: DataLoader for test data
        device: Device to run on
        corruption_fn: Function to apply corruption
        corruption_name: Name of the corruption

    Returns:
        Dictionary with results
    """
    model.eval()

    correct_clean = 0
    correct_corrupted = 0
    total = 0

    all_clean_probs = []
    all_corrupted_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f'Testing {corruption_name}'):
            images = images.to(device)
            labels = labels.to(device)

            # Clean predictions
            outputs_clean = model(images)
            probs_clean = F.softmax(outputs_clean, dim=1)
            _, predicted_clean = torch.max(outputs_clean, 1)
            correct_clean += (predicted_clean == labels).sum().item()

            # Corrupted predictions
            images_corrupted = corruption_fn(images)
            outputs_corrupted = model(images_corrupted)
            probs_corrupted = F.softmax(outputs_corrupted, dim=1)
            _, predicted_corrupted = torch.max(outputs_corrupted, 1)
            correct_corrupted += (predicted_corrupted == labels).sum().item()

            total += labels.size(0)

            all_clean_probs.append(probs_clean.cpu().numpy())
            all_corrupted_probs.append(probs_corrupted.cpu().numpy())

    acc_clean = 100 * correct_clean / total
    acc_corrupted = 100 * correct_corrupted / total
    acc_drop = acc_clean - acc_corrupted

    all_clean_probs = np.concatenate(all_clean_probs)
    all_corrupted_probs = np.concatenate(all_corrupted_probs)

    # Calculate confidence changes
    clean_conf = all_clean_probs.max(axis=1).mean()
    corrupted_conf = all_corrupted_probs.max(axis=1).mean()
    conf_drop = clean_conf - corrupted_conf

    return {
        'corruption': corruption_name,
        'accuracy_clean': acc_clean,
        'accuracy_corrupted': acc_corrupted,
        'accuracy_drop': acc_drop,
        'confidence_clean': float(clean_conf),
        'confidence_corrupted': float(corrupted_conf),
        'confidence_drop': float(conf_drop)
    }


def plot_robustness_results(results, save_path):
    """Plot robustness test results.

    Args:
        results: List of result dictionaries
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    corruptions = [r['corruption'] for r in results]
    acc_clean = [r['accuracy_clean'] for r in results]
    acc_corrupted = [r['accuracy_corrupted'] for r in results]
    acc_drop = [r['accuracy_drop'] for r in results]

    # Accuracy comparison
    x = np.arange(len(corruptions))
    width = 0.35

    ax1.bar(x - width/2, acc_clean, width, label='Clean', alpha=0.8)
    ax1.bar(x + width/2, acc_corrupted, width, label='Corrupted', alpha=0.8)
    ax1.set_xlabel('Corruption Type')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy: Clean vs Corrupted')
    ax1.set_xticks(x)
    ax1.set_xticklabels(corruptions, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy drop
    colors = ['red' if drop > 10 else 'orange' if drop > 5 else 'green'
              for drop in acc_drop]
    ax2.bar(corruptions, acc_drop, color=colors, alpha=0.7)
    ax2.set_xlabel('Corruption Type')
    ax2.set_ylabel('Accuracy Drop (%)')
    ax2.set_title('Accuracy Drop per Corruption')
    ax2.set_xticklabels(corruptions, rotation=45, ha='right')
    ax2.axhline(y=5, color='orange', linestyle='--', label='5% threshold')
    ax2.axhline(y=10, color='red', linestyle='--', label='10% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Robustness plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test model robustness')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to processed data')
    parser.add_argument('--output-dir', type=str, default='results/robustness',
                        help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for testing')
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

    # Define corruptions to test
    corruptions = [
        (lambda x: add_gaussian_noise(x, std=0.05), 'Gaussian Noise (σ=0.05)'),
        (lambda x: add_gaussian_noise(x, std=0.1), 'Gaussian Noise (σ=0.1)'),
        (lambda x: add_salt_pepper_noise(x, amount=0.05), 'Salt & Pepper (5%)'),
        (lambda x: add_blur(x, kernel_size=5), 'Gaussian Blur (k=5)'),
        (lambda x: adjust_brightness(x, factor=1.3), 'Brightness (+30%)'),
        (lambda x: adjust_brightness(x, factor=0.7), 'Brightness (-30%)'),
        (lambda x: adjust_contrast(x, factor=1.5), 'Contrast (+50%)'),
        (lambda x: adjust_contrast(x, factor=0.5), 'Contrast (-50%)'),
    ]

    print("\nTesting robustness to corruptions...")
    results = []

    for corruption_fn, corruption_name in corruptions:
        result = test_robustness(
            model, test_loader, device,
            corruption_fn, corruption_name
        )
        results.append(result)

        print(f"\n{corruption_name}:")
        print(f"  Clean accuracy: {result['accuracy_clean']:.2f}%")
        print(f"  Corrupted accuracy: {result['accuracy_corrupted']:.2f}%")
        print(f"  Accuracy drop: {result['accuracy_drop']:.2f}%")
        print(f"  Confidence drop: {result['confidence_drop']:.4f}")

    # Save results
    results_file = output_dir / 'robustness_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Plot results
    plot_path = output_dir / 'robustness_plot.png'
    plot_robustness_results(results, plot_path)

    # Summary statistics
    avg_acc_drop = np.mean([r['accuracy_drop'] for r in results])
    max_acc_drop = np.max([r['accuracy_drop'] for r in results])

    print("\n" + "="*60)
    print("ROBUSTNESS SUMMARY")
    print("="*60)
    print(f"Average accuracy drop: {avg_acc_drop:.2f}%")
    print(f"Maximum accuracy drop: {max_acc_drop:.2f}%")

    if max_acc_drop < 5:
        print("✓ Model is HIGHLY ROBUST")
    elif max_acc_drop < 10:
        print("✓ Model is MODERATELY ROBUST")
    else:
        print("⚠ Model shows SIGNIFICANT SENSITIVITY to corruptions")
    print("="*60)


if __name__ == '__main__':
    main()
