"""Create train/validation/test splits with patient-level separation."""

import argparse
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import sys
sys.path.append('..')

from src.utils.logging_config import setup_logging


def collect_patient_data(data_dir: Path):
    """Collect patient IDs and their images.

    Args:
        data_dir: Data directory path

    Returns:
        Dictionary mapping patient_id -> (class_label, image_paths)
    """
    patient_data = {}

    for class_idx, class_name in enumerate(['CE', 'LAA']):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        patient_dirs = [d for d in class_dir.iterdir() if d.is_dir()]

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name

            # Find all images for this patient
            image_extensions = ['.png', '.jpg', '.jpeg', '.npy', '.dcm', '.nii']
            images = []
            for ext in image_extensions:
                images.extend(list(patient_dir.glob(f'*{ext}')))

            if images:
                # Store relative paths
                relative_paths = [str(img.relative_to(data_dir)) for img in images]
                patient_data[patient_id] = {
                    'class': class_name,
                    'class_idx': class_idx,
                    'images': relative_paths,
                    'num_images': len(images)
                }

    return patient_data


def create_stratified_splits(
    patient_data: dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """Create stratified train/val/test splits at patient level.

    Args:
        patient_data: Dictionary of patient data
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with split information
    """
    logger = setup_logging()

    # Extract patient IDs and labels
    patient_ids = list(patient_data.keys())
    labels = [patient_data[pid]['class_idx'] for pid in patient_ids]

    logger.info(f"Total patients: {len(patient_ids)}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # First split: train vs (val + test)
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        patient_ids,
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_seed
    )

    # Second split: val vs test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids, val_labels, test_labels = train_test_split(
        temp_ids,
        temp_labels,
        test_size=(1 - relative_val_ratio),
        stratify=temp_labels,
        random_state=random_seed
    )

    # Create split dictionaries
    splits = {
        'train': {
            'patient_ids': train_ids,
            'patients': {pid: patient_data[pid] for pid in train_ids},
            'num_patients': len(train_ids),
            'num_images': sum(patient_data[pid]['num_images'] for pid in train_ids)
        },
        'val': {
            'patient_ids': val_ids,
            'patients': {pid: patient_data[pid] for pid in val_ids},
            'num_patients': len(val_ids),
            'num_images': sum(patient_data[pid]['num_images'] for pid in val_ids)
        },
        'test': {
            'patient_ids': test_ids,
            'patients': {pid: patient_data[pid] for pid in test_ids},
            'num_patients': len(test_ids),
            'num_images': sum(patient_data[pid]['num_images'] for pid in test_ids)
        }
    }

    # Log statistics
    logger.info("\nSplit Statistics:")
    for split_name, split_data in splits.items():
        logger.info(f"\n{split_name.upper()}:")
        logger.info(f"  Patients: {split_data['num_patients']}")
        logger.info(f"  Images: {split_data['num_images']}")

        # Class distribution
        class_counts = defaultdict(int)
        for pid in split_data['patient_ids']:
            class_counts[patient_data[pid]['class']] += 1
        logger.info(f"  Class distribution: {dict(class_counts)}")

    return splits


def save_splits(splits: dict, output_dir: Path):
    """Save splits to JSON files.

    Args:
        splits: Dictionary containing split information
        output_dir: Output directory
    """
    logger = setup_logging()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved {split_name} split to {output_path}")

    # Save summary
    summary = {
        'num_splits': len(splits),
        'splits': {
            name: {
                'num_patients': data['num_patients'],
                'num_images': data['num_images']
            }
            for name, data in splits.items()
        }
    }

    summary_path = output_dir / 'splits_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")


def verify_no_leakage(splits: dict):
    """Verify no patient appears in multiple splits.

    Args:
        splits: Dictionary containing split information

    Returns:
        Boolean indicating if verification passed
    """
    logger = setup_logging()
    logger.info("\nVerifying no patient leakage...")

    all_patients = set()
    for split_name, split_data in splits.items():
        split_patients = set(split_data['patient_ids'])

        # Check for overlap with previous splits
        overlap = all_patients & split_patients
        if overlap:
            logger.error(f"Patient leakage detected in {split_name}: {overlap}")
            return False

        all_patients.update(split_patients)

    logger.info("No patient leakage detected!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                        help='Output directory for splits')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Collect patient data
    patient_data = collect_patient_data(data_dir)

    if not patient_data:
        print("No patient data found! Make sure data is in the correct format.")
        return

    # Create splits
    splits = create_stratified_splits(
        patient_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    # Verify no leakage
    if not verify_no_leakage(splits):
        print("ERROR: Patient leakage detected! Splits not saved.")
        return

    # Save splits
    save_splits(splits, output_dir)

    print("\nSplits created successfully!")


if __name__ == '__main__':
    main()
