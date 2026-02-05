"""Preprocess medical images for training."""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM loading disabled.")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. NIfTI loading disabled.")

from PIL import Image
from src.data.preprocessing import (
    normalize_intensity, resize_image, apply_windowing, remove_outliers
)
from src.utils.logging_config import setup_logging


def load_medical_image(image_path: Path):
    """Load medical image from various formats.

    Args:
        image_path: Path to image file

    Returns:
        numpy array of image data
    """
    suffix = image_path.suffix.lower()

    try:
        if suffix == '.dcm' and PYDICOM_AVAILABLE:
            ds = pydicom.dcmread(str(image_path))
            image = ds.pixel_array.astype(np.float32)
            return image

        elif (suffix == '.gz' or suffix == '.nii') and NIBABEL_AVAILABLE:
            img = nib.load(str(image_path))
            image = img.get_fdata().astype(np.float32)
            # Take middle slice if 3D
            if image.ndim == 3:
                image = image[:, :, image.shape[2] // 2]
            return image

        elif suffix in ['.png', '.jpg', '.jpeg']:
            image = np.array(Image.open(image_path)).astype(np.float32)
            # Convert to grayscale if RGB
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            return image

        else:
            print(f"Unsupported format: {suffix}")
            return None

    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def preprocess_image(
    image: np.ndarray,
    target_size: tuple = (224, 224),
    normalize_method: str = 'zscore',
    remove_outliers_flag: bool = True
):
    """Preprocess a single image.

    Args:
        image: Input image array
        target_size: Target (height, width)
        normalize_method: Normalization method
        remove_outliers_flag: Whether to remove outliers

    Returns:
        Preprocessed image
    """
    # Remove outliers
    if remove_outliers_flag:
        image = remove_outliers(image, percentile_range=(1, 99))

    # Normalize intensity
    image = normalize_intensity(image, method=normalize_method)

    # Resize
    image = resize_image(image, target_size, maintain_aspect_ratio=True)

    return image


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    target_size: tuple = (224, 224),
    normalize_method: str = 'zscore',
    save_format: str = 'png'
):
    """Process entire dataset.

    Args:
        input_dir: Input data directory
        output_dir: Output directory
        target_size: Target image size
        normalize_method: Normalization method
        save_format: Output format ('png' or 'npy')
    """
    logger = setup_logging()
    logger.info("Starting data preprocessing...")

    for class_name in ['CE', 'LAA']:
        class_input_dir = input_dir / class_name
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        if not class_input_dir.exists():
            logger.warning(f"Class directory not found: {class_input_dir}")
            continue

        patient_dirs = [d for d in class_input_dir.iterdir() if d.is_dir()]
        logger.info(f"\nProcessing {len(patient_dirs)} patients in {class_name}...")

        for patient_dir in tqdm(patient_dirs, desc=f"Processing {class_name}"):
            patient_output_dir = class_output_dir / patient_dir.name
            patient_output_dir.mkdir(parents=True, exist_ok=True)

            # Find all images
            image_extensions = ['.dcm', '.nii', '.nii.gz', '.png', '.jpg', '.jpeg']
            images = []
            for ext in image_extensions:
                images.extend(list(patient_dir.glob(f'*{ext}')))
                images.extend(list(patient_dir.glob(f'**/*{ext}')))

            images = list(set(images))  # Remove duplicates

            for img_path in images:
                # Load image
                image = load_medical_image(img_path)
                if image is None:
                    continue

                # Preprocess
                processed = preprocess_image(
                    image,
                    target_size=target_size,
                    normalize_method=normalize_method
                )

                # Save
                output_name = img_path.stem
                if save_format == 'png':
                    # Convert to uint8 for PNG
                    processed_uint8 = ((processed - processed.min()) / (processed.max() - processed.min() + 1e-8) * 255).astype(np.uint8)
                    output_path = patient_output_dir / f"{output_name}.png"
                    Image.fromarray(processed_uint8).save(output_path)
                elif save_format == 'npy':
                    output_path = patient_output_dir / f"{output_name}.npy"
                    np.save(output_path, processed)

    logger.info("\nPreprocessing complete!")


def main():
    parser = argparse.ArgumentParser(description='Preprocess medical images')
    parser.add_argument('--input_dir', type=str, default='data/raw',
                        help='Input data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        help='Target image size (height width)')
    parser.add_argument('--normalize', type=str, default='zscore',
                        choices=['zscore', 'minmax', 'percentile'],
                        help='Normalization method')
    parser.add_argument('--save_format', type=str, default='png',
                        choices=['png', 'npy'],
                        help='Output format')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_size = tuple(args.target_size)

    process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        target_size=target_size,
        normalize_method=args.normalize,
        save_format=args.save_format
    )


if __name__ == '__main__':
    main()
