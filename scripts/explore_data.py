"""Explore and visualize dataset."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import sys
sys.path.append('..')

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

from src.utils.logging_config import setup_logging


def load_image(image_path: Path):
    """Load medical image from various formats."""
    suffix = image_path.suffix.lower()

    if suffix == '.dcm' and PYDICOM_AVAILABLE:
        ds = pydicom.dcmread(str(image_path))
        return ds.pixel_array

    elif suffix in ['.nii', '.gz'] and NIBABEL_AVAILABLE:
        img = nib.load(str(image_path))
        return img.get_fdata()

    elif suffix in ['.png', '.jpg', '.jpeg']:
        from PIL import Image
        return np.array(Image.open(image_path))

    else:
        return None


def analyze_image_properties(data_dir: Path):
    """Analyze image properties across dataset."""
    logger = setup_logging()
    logger.info("Analyzing image properties...")

    properties = {
        'shapes': [],
        'dtypes': defaultdict(int),
        'intensity_stats': {
            'means': [],
            'stds': [],
            'mins': [],
            'maxs': []
        },
        'by_class': {
            'CE': {'shapes': [], 'intensities': []},
            'LAA': {'shapes': [], 'intensities': []}
        }
    }

    for class_name in ['CE', 'LAA']:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        patient_dirs = [d for d in class_dir.iterdir() if d.is_dir()]

        for patient_dir in patient_dirs[:10]:  # Sample first 10 patients per class
            # Find images
            image_extensions = ['.dcm', '.nii', '.nii.gz', '.png', '.jpg']
            images = []
            for ext in image_extensions:
                images.extend(list(patient_dir.glob(f'*{ext}')))

            for img_path in images[:5]:  # Sample first 5 images per patient
                try:
                    img = load_image(img_path)
                    if img is None:
                        continue

                    # Collect properties
                    properties['shapes'].append(img.shape)
                    properties['dtypes'][str(img.dtype)] += 1
                    properties['intensity_stats']['means'].append(np.mean(img))
                    properties['intensity_stats']['stds'].append(np.std(img))
                    properties['intensity_stats']['mins'].append(np.min(img))
                    properties['intensity_stats']['maxs'].append(np.max(img))

                    # By class
                    properties['by_class'][class_name]['shapes'].append(img.shape)
                    properties['by_class'][class_name]['intensities'].append(np.mean(img))

                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")

    return properties


def visualize_statistics(properties, output_dir: Path):
    """Create visualization plots."""
    logger = setup_logging()
    logger.info("Creating visualizations...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Image dimensions distribution
    if properties['shapes']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        heights = [s[0] if len(s) >= 1 else 0 for s in properties['shapes']]
        widths = [s[1] if len(s) >= 2 else 0 for s in properties['shapes']]
        depths = [s[2] if len(s) >= 3 else 0 for s in properties['shapes'] if len(s) >= 3]

        axes[0].hist(heights, bins=20, edgecolor='black')
        axes[0].set_title('Height Distribution')
        axes[0].set_xlabel('Height (pixels)')
        axes[0].set_ylabel('Count')

        axes[1].hist(widths, bins=20, edgecolor='black')
        axes[1].set_title('Width Distribution')
        axes[1].set_xlabel('Width (pixels)')
        axes[1].set_ylabel('Count')

        if depths:
            axes[2].hist(depths, bins=20, edgecolor='black')
            axes[2].set_title('Depth Distribution')
            axes[2].set_xlabel('Depth (slices)')
            axes[2].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(output_dir / 'image_dimensions.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 2: Intensity statistics
    if properties['intensity_stats']['means']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].hist(properties['intensity_stats']['means'], bins=30, edgecolor='black')
        axes[0, 0].set_title('Mean Intensity Distribution')
        axes[0, 0].set_xlabel('Mean Intensity')
        axes[0, 0].set_ylabel('Count')

        axes[0, 1].hist(properties['intensity_stats']['stds'], bins=30, edgecolor='black')
        axes[0, 1].set_title('Standard Deviation Distribution')
        axes[0, 1].set_xlabel('Std')
        axes[0, 1].set_ylabel('Count')

        axes[1, 0].hist(properties['intensity_stats']['mins'], bins=30, edgecolor='black')
        axes[1, 0].set_title('Min Intensity Distribution')
        axes[1, 0].set_xlabel('Min')
        axes[1, 0].set_ylabel('Count')

        axes[1, 1].hist(properties['intensity_stats']['maxs'], bins=30, edgecolor='black')
        axes[1, 1].set_title('Max Intensity Distribution')
        axes[1, 1].set_xlabel('Max')
        axes[1, 1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(output_dir / 'intensity_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 3: Class comparison
    if all(properties['by_class'][cls]['intensities'] for cls in ['CE', 'LAA']):
        fig, ax = plt.subplots(figsize=(8, 6))

        data = [
            properties['by_class']['CE']['intensities'],
            properties['by_class']['LAA']['intensities']
        ]

        ax.boxplot(data, labels=['CE', 'LAA'])
        ax.set_title('Mean Intensity by Class')
        ax.set_ylabel('Mean Intensity')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'class_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Explore dataset and create visualizations')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/exploration',
                        help='Directory to save plots')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Analyze properties
    properties = analyze_image_properties(data_dir)

    # Create visualizations
    visualize_statistics(properties, output_dir)

    # Print summary
    logger = setup_logging()
    logger.info("\n" + "="*60)
    logger.info("IMAGE PROPERTIES SUMMARY")
    logger.info("="*60)

    if properties['shapes']:
        logger.info(f"\nNumber of images analyzed: {len(properties['shapes'])}")
        logger.info(f"\nData types found: {dict(properties['dtypes'])}")

        logger.info("\nIntensity Statistics:")
        logger.info(f"  Mean intensity: {np.mean(properties['intensity_stats']['means']):.2f} ± {np.std(properties['intensity_stats']['means']):.2f}")
        logger.info(f"  Min value range: [{np.min(properties['intensity_stats']['mins']):.2f}, {np.max(properties['intensity_stats']['mins']):.2f}]")
        logger.info(f"  Max value range: [{np.min(properties['intensity_stats']['maxs']):.2f}, {np.max(properties['intensity_stats']['maxs']):.2f}]")

    logger.info("\n" + "="*60)


if __name__ == '__main__':
    main()
