"""Visualize model predictions on sample images."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('..')

from src.data.dataset import StrokeDataset
from src.data.augmentation import get_val_augmentation
from src.models.cnn import ResNetClassifier, SimpleCNN
from src.utils.helpers import get_device


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    device = get_device()

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
            pretrained=False,
            in_channels=config['model']['in_channels']
        )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, device


def visualize_predictions(
    model,
    dataset,
    device,
    num_samples=16,
    save_path=None
):
    """Visualize predictions on sample images."""
    class_names = ['CE', 'LAA']

    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    # Calculate grid size
    n_cols = 4
    n_rows = (len(indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Get image and true label
        image, true_label = dataset[idx]

        # Make prediction
        with torch.no_grad():
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            prob = torch.softmax(output, dim=1)
            pred_label = torch.argmax(output, dim=1).item()
            confidence = prob[0, pred_label].item()

        # Prepare image for display
        img_display = image.permute(1, 2, 0).cpu().numpy()

        # Denormalize if needed
        if img_display.max() <= 1.0:
            # Assume ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = std * img_display + mean
            img_display = np.clip(img_display, 0, 1)

        # Display image
        ax.imshow(img_display)

        # Color code: green if correct, red if wrong
        color = 'green' if pred_label == true_label else 'red'

        # Title with prediction info
        title = f"True: {class_names[true_label]}\n"
        title += f"Pred: {class_names[pred_label]} ({confidence:.2%})"

        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')

    # Hide empty subplots
    for i in range(len(indices), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.suptitle('Model Predictions (Green=Correct, Red=Incorrect)', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Which split to visualize')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization (if not provided, shows interactive plot)')
    args = parser.parse_args()

    # Load config
    from src.utils.helpers import load_config
    config = load_config(args.config)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, device = load_model(args.checkpoint, config)

    # Create dataset
    print(f"Loading {args.split} dataset from {args.data_dir}")
    dataset = StrokeDataset(
        data_dir=args.data_dir,
        split=args.split,
        split_file=f'data/splits/{args.split}.json',
        transform=get_val_augmentation(config['data']['image_size'])
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    # Visualize
    print(f"Visualizing {args.num_samples} predictions...")
    visualize_predictions(
        model=model,
        dataset=dataset,
        device=device,
        num_samples=args.num_samples,
        save_path=args.output
    )

    print("Done!")


if __name__ == '__main__':
    main()
