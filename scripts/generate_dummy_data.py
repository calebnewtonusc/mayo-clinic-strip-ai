"""Generate dummy medical imaging data for testing the pipeline."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
from tqdm import tqdm


def generate_synthetic_medical_image(size=(256, 256), complexity='medium'):
    """Generate a synthetic medical-looking image.

    Args:
        size: Image size (height, width)
        complexity: 'simple', 'medium', or 'complex'

    Returns:
        numpy array of synthetic image
    """
    h, w = size

    # Base noise
    image = np.random.randn(h, w) * 20 + 100

    if complexity in ['medium', 'complex']:
        # Add some structure (simulating tissue)
        y, x = np.ogrid[0:h, 0:w]

        # Circular structures (simulating vessels/clots)
        num_circles = np.random.randint(3, 8)
        for _ in range(num_circles):
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            r = np.random.randint(10, 40)
            intensity = np.random.uniform(50, 200)

            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask = dist < r
            image[mask] = intensity + np.random.randn(mask.sum()) * 10

    if complexity == 'complex':
        # Add gradient (simulating intensity variation)
        gradient = np.linspace(0, 50, w)
        image += gradient[None, :]

        # Add some texture
        texture = np.random.randn(h, w) * 30
        from scipy.ndimage import gaussian_filter
        texture = gaussian_filter(texture, sigma=2)
        image += texture

    # Clip to valid range
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def generate_dummy_dataset(
    output_dir: Path,
    num_patients_per_class: int = 20,
    images_per_patient: tuple = (3, 8),
    image_size: tuple = (256, 256)
):
    """Generate a complete dummy dataset.

    Args:
        output_dir: Output directory
        num_patients_per_class: Number of patients per class
        images_per_patient: (min, max) images per patient
        image_size: Image size (height, width)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        'num_patients': num_patients_per_class * 2,
        'classes': ['CE', 'LAA'],
        'patients': {}
    }

    print(f"Generating dummy dataset in {output_dir}")
    print(f"Patients per class: {num_patients_per_class}")
    print(f"Images per patient: {images_per_patient[0]}-{images_per_patient[1]}")

    for class_name in ['CE', 'LAA']:
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        print(f"\nGenerating {class_name} data...")

        for patient_idx in tqdm(range(num_patients_per_class)):
            patient_id = f"patient_{class_name}_{patient_idx:03d}"
            patient_dir = class_dir / patient_id
            patient_dir.mkdir(exist_ok=True)

            # Random number of images for this patient
            num_images = np.random.randint(images_per_patient[0], images_per_patient[1] + 1)

            patient_info = {
                'class': class_name,
                'num_images': num_images,
                'images': []
            }

            for img_idx in range(num_images):
                # Generate image with some variation
                complexity = np.random.choice(['simple', 'medium', 'complex'], p=[0.1, 0.6, 0.3])
                image = generate_synthetic_medical_image(image_size, complexity)

                # Add slight class-specific characteristics
                if class_name == 'CE':
                    # CE: Slightly more uniform, darker
                    image = (image * 0.9).astype(np.uint8)
                else:
                    # LAA: Slightly more textured, brighter
                    image = np.clip(image * 1.1 + 10, 0, 255).astype(np.uint8)

                # Save image
                img_path = patient_dir / f"image_{img_idx:03d}.png"
                Image.fromarray(image).save(img_path)

                patient_info['images'].append(str(img_path.relative_to(output_dir)))

            dataset_info['patients'][patient_id] = patient_info

    # Save dataset info
    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n✅ Dummy dataset generated successfully!")
    print(f"📁 Location: {output_dir}")
    print(f"📊 Total patients: {dataset_info['num_patients']}")
    print(f"📊 Total images: {sum(p['num_images'] for p in dataset_info['patients'].values())}")
    print(f"ℹ️  Dataset info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate dummy medical imaging data')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--num_patients', type=int, default=20,
                        help='Number of patients per class')
    parser.add_argument('--min_images', type=int, default=3,
                        help='Minimum images per patient')
    parser.add_argument('--max_images', type=int, default=8,
                        help='Maximum images per patient')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='Image size (height width)')
    args = parser.parse_args()

    # Check if scipy is available for complex images
    try:
        import scipy
    except ImportError:
        print("Warning: scipy not installed. Install with 'pip install scipy' for better dummy images.")
        print("Continuing with simpler image generation...\n")

    generate_dummy_dataset(
        output_dir=Path(args.output_dir),
        num_patients_per_class=args.num_patients,
        images_per_patient=(args.min_images, args.max_images),
        image_size=tuple(args.image_size)
    )

    print("\n🚀 Next steps:")
    print("1. Validate data: python scripts/validate_data.py --data_dir data/raw")
    print("2. Preprocess data: python scripts/preprocess_data.py")
    print("3. Create splits: python scripts/create_splits.py")
    print("4. Test DataLoader: python scripts/test_dataloader.py")
    print("5. Train model: python train.py")


if __name__ == '__main__':
    main()
