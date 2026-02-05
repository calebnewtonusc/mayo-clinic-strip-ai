"""Generate interpretability visualizations for trained models."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('..')

from src.data.dataset import StrokeDataset
from src.data.augmentation import get_val_augmentation
from src.models.cnn import ResNetClassifier, SimpleCNN, EfficientNetClassifier
from src.visualization.gradcam import (
    GradCAM, GradCAMPlusPlus, get_target_layer,
    overlay_heatmap_on_image, compute_guided_backprop
)
from src.utils.helpers import load_config, get_device


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint."""
    device = get_device()

    # Create model
    arch = config['model']['architecture']
    if arch == 'simple_cnn':
        model = SimpleCNN(
            in_channels=config['model']['in_channels'],
            num_classes=config['model']['num_classes']
        )
        model_type = 'simple_cnn'
    elif 'efficientnet' in arch:
        model = EfficientNetClassifier(
            arch=arch,
            num_classes=config['model']['num_classes'],
            pretrained=False
        )
        model_type = 'efficientnet'
    else:
        model = ResNetClassifier(
            arch=arch,
            num_classes=config['model']['num_classes'],
            pretrained=False,
            in_channels=config['model']['in_channels']
        )
        model_type = 'resnet'

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, model_type, device


def generate_gradcam_visualizations(
    model,
    model_type,
    dataset,
    device,
    output_dir,
    num_samples=10,
    use_gradcam_plus=False
):
    """Generate Grad-CAM visualizations for sample images.

    Args:
        model: Trained model
        model_type: Type of model architecture
        dataset: Dataset to visualize
        device: Device to run on
        output_dir: Output directory
        num_samples: Number of samples to visualize
        use_gradcam_plus: Whether to use Grad-CAM++ instead of Grad-CAM
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get target layer
    target_layer = get_target_layer(model, model_type)

    # Initialize Grad-CAM
    if use_gradcam_plus:
        gradcam = GradCAMPlusPlus(model, target_layer)
        method_name = "GradCAM++"
    else:
        gradcam = GradCAM(model, target_layer)
        method_name = "GradCAM"

    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    class_names = ['CE', 'LAA']

    # Generate visualizations
    for i, idx in enumerate(indices):
        image, true_label = dataset[idx]

        # Prepare input
        image_input = image.unsqueeze(0).to(device)

        # Generate CAM
        cam = gradcam.generate_cam(image_input)

        # Get prediction
        with torch.no_grad():
            output = model(image_input)
            pred_label = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_label].item()

        # Prepare image for visualization
        img_display = image.permute(1, 2, 0).cpu().numpy()

        # Denormalize
        if img_display.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = std * img_display + mean
            img_display = np.clip(img_display, 0, 1)

        # Convert to uint8
        img_display = (img_display * 255).astype(np.uint8)

        # Overlay heatmap
        overlayed = overlay_heatmap_on_image(img_display, cam, alpha=0.4)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Heatmap only
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title(f'{method_name} Heatmap')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(overlayed)
        axes[2].set_title('Overlayed')
        axes[2].axis('off')

        # Overall title
        color = 'green' if pred_label == true_label else 'red'
        title = f"True: {class_names[true_label]} | Pred: {class_names[pred_label]} ({confidence:.2%})"
        fig.suptitle(title, fontsize=14, fontweight='bold', color=color)

        plt.tight_layout()

        # Save
        save_path = output_dir / f"sample_{i:03d}_{method_name.lower()}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved {save_path}")

    print(f"\n✅ Generated {len(indices)} {method_name} visualizations in {output_dir}")


def generate_feature_visualizations(
    model,
    dataloader,
    device,
    output_dir
):
    """Generate feature space visualizations using t-SNE and UMAP.

    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: Device to run on
        output_dir: Output directory
    """
    from src.visualization.features import (
        extract_features, compute_tsne, compute_pca, plot_embedding,
        analyze_feature_separability
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nExtracting features...")
    features, labels = extract_features(model, dataloader, device)

    print(f"Feature shape: {features.shape}")

    # Analyze separability
    print("\nAnalyzing feature separability...")
    separability = analyze_feature_separability(features, labels)

    print("Separability Metrics:")
    for metric, value in separability.items():
        print(f"  {metric}: {value:.4f}")

    # Compute t-SNE
    print("\nComputing t-SNE...")
    tsne_embedded = compute_tsne(features)
    plot_embedding(
        tsne_embedded,
        labels,
        title="t-SNE Feature Embedding",
        save_path=output_dir / "tsne_embedding.png"
    )

    # Compute PCA
    print("Computing PCA...")
    pca_embedded = compute_pca(features)
    plot_embedding(
        pca_embedded,
        labels,
        title="PCA Feature Embedding",
        save_path=output_dir / "pca_embedding.png"
    )

    # Try UMAP if available
    try:
        from src.visualization.features import compute_umap, UMAP_AVAILABLE
        if UMAP_AVAILABLE:
            print("Computing UMAP...")
            umap_embedded = compute_umap(features)
            plot_embedding(
                umap_embedded,
                labels,
                title="UMAP Feature Embedding",
                save_path=output_dir / "umap_embedding.png"
            )
    except:
        print("UMAP not available, skipping...")

    print(f"\n✅ Generated feature visualizations in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate interpretability visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Which split to use')
    parser.add_argument('--output_dir', type=str, default='experiments/interpretability',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples for Grad-CAM')
    parser.add_argument('--gradcam_plus', action='store_true',
                        help='Use Grad-CAM++ instead of Grad-CAM')
    parser.add_argument('--skip_features', action='store_true',
                        help='Skip feature visualization')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, model_type, device = load_model(args.checkpoint, config)

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

    # Generate Grad-CAM visualizations
    print("\n" + "="*60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*60)

    generate_gradcam_visualizations(
        model=model,
        model_type=model_type,
        dataset=dataset,
        device=device,
        output_dir=Path(args.output_dir) / "gradcam",
        num_samples=args.num_samples,
        use_gradcam_plus=args.gradcam_plus
    )

    # Generate feature visualizations
    if not args.skip_features:
        print("\n" + "="*60)
        print("GENERATING FEATURE VISUALIZATIONS")
        print("="*60)

        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )

        generate_feature_visualizations(
            model=model,
            dataloader=dataloader,
            device=device,
            output_dir=Path(args.output_dir) / "features"
        )

    print("\n✅ All interpretability visualizations generated!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
