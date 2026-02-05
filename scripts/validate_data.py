"""Validate data quality and structure."""

import argparse
from pathlib import Path
import json
from collections import defaultdict
import sys
sys.path.append('..')

from src.utils.logging_config import setup_logging


def validate_directory_structure(data_dir: Path):
    """Validate that data directory has correct structure."""
    logger = setup_logging()
    logger.info("Validating directory structure...")

    issues = []

    # Check if data directory exists
    if not data_dir.exists():
        issues.append(f"Data directory not found: {data_dir}")
        return issues

    # Check for class directories
    ce_dir = data_dir / 'CE'
    laa_dir = data_dir / 'LAA'

    if not ce_dir.exists():
        issues.append("CE directory not found")
    if not laa_dir.exists():
        issues.append("LAA directory not found")

    if issues:
        return issues

    # Check for patient directories
    ce_patients = list(ce_dir.glob('*'))
    laa_patients = list(laa_dir.glob('*'))

    logger.info(f"Found {len(ce_patients)} CE patient directories")
    logger.info(f"Found {len(laa_patients)} LAA patient directories")

    if len(ce_patients) == 0:
        issues.append("No patient directories found in CE")
    if len(laa_patients) == 0:
        issues.append("No patient directories found in LAA")

    return issues


def scan_dataset(data_dir: Path):
    """Scan dataset and collect statistics."""
    logger = setup_logging()
    logger.info("Scanning dataset...")

    stats = {
        'classes': {},
        'total_patients': 0,
        'total_images': 0,
        'file_extensions': defaultdict(int),
        'patients_by_class': {},
        'images_by_class': {}
    }

    for class_name in ['CE', 'LAA']:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        patient_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        num_patients = len(patient_dirs)

        images_in_class = 0
        images_per_patient = []

        for patient_dir in patient_dirs:
            # Count images (common medical imaging formats)
            image_extensions = ['.dcm', '.nii', '.nii.gz', '.png', '.jpg', '.jpeg']
            images = []
            for ext in image_extensions:
                images.extend(list(patient_dir.glob(f'*{ext}')))
                images.extend(list(patient_dir.glob(f'**/*{ext}')))

            images = list(set(images))  # Remove duplicates
            num_images = len(images)
            images_per_patient.append(num_images)
            images_in_class += num_images

            # Track file extensions
            for img in images:
                if img.suffix == '.gz' and img.stem.endswith('.nii'):
                    stats['file_extensions']['.nii.gz'] += 1
                else:
                    stats['file_extensions'][img.suffix] += 1

        stats['classes'][class_name] = {
            'num_patients': num_patients,
            'num_images': images_in_class,
            'images_per_patient': {
                'mean': sum(images_per_patient) / len(images_per_patient) if images_per_patient else 0,
                'min': min(images_per_patient) if images_per_patient else 0,
                'max': max(images_per_patient) if images_per_patient else 0
            }
        }

        stats['total_patients'] += num_patients
        stats['total_images'] += images_in_class
        stats['patients_by_class'][class_name] = num_patients
        stats['images_by_class'][class_name] = images_in_class

    return stats


def print_statistics(stats):
    """Print dataset statistics."""
    logger = setup_logging()

    logger.info("\n" + "="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)

    logger.info(f"\nTotal Patients: {stats['total_patients']}")
    logger.info(f"Total Images: {stats['total_images']}")

    logger.info("\nClass Distribution:")
    for class_name, class_stats in stats['classes'].items():
        logger.info(f"\n  {class_name}:")
        logger.info(f"    Patients: {class_stats['num_patients']}")
        logger.info(f"    Images: {class_stats['num_images']}")
        logger.info(f"    Images per patient:")
        logger.info(f"      Mean: {class_stats['images_per_patient']['mean']:.2f}")
        logger.info(f"      Min: {class_stats['images_per_patient']['min']}")
        logger.info(f"      Max: {class_stats['images_per_patient']['max']}")

    # Class imbalance
    if len(stats['patients_by_class']) == 2:
        classes = list(stats['patients_by_class'].keys())
        ratio = stats['patients_by_class'][classes[0]] / stats['patients_by_class'][classes[1]]
        logger.info(f"\nPatient-level class imbalance ratio: {ratio:.2f}:1")

        ratio = stats['images_by_class'][classes[0]] / stats['images_by_class'][classes[1]]
        logger.info(f"Image-level class imbalance ratio: {ratio:.2f}:1")

    logger.info("\nFile Extensions:")
    for ext, count in stats['file_extensions'].items():
        logger.info(f"  {ext}: {count}")

    logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Validate dataset structure and quality')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Path to data directory')
    parser.add_argument('--output', type=str, default='data/dataset_stats.json',
                        help='Path to save statistics JSON')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Validate structure
    issues = validate_directory_structure(data_dir)

    if issues:
        print("VALIDATION ERRORS:")
        for issue in issues:
            print(f"  - {issue}")
        return

    print("Directory structure validation passed!")

    # Scan dataset
    stats = scan_dataset(data_dir)

    # Print statistics
    print_statistics(stats)

    # Save statistics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to {output_path}")


if __name__ == '__main__':
    main()
