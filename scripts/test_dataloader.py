"""Test DataLoader performance and correctness."""

import argparse
import time
from pathlib import Path
import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader

from src.data.dataset import StrokeDataset, PatientLevelDataset
from src.data.augmentation import get_train_augmentation, get_val_augmentation
from src.utils.logging_config import setup_logging


def test_basic_loading(data_dir: str, split_file: str):
    """Test basic dataset loading."""
    logger = setup_logging()
    logger.info("Testing basic dataset loading...")

    # Create dataset
    dataset = StrokeDataset(
        data_dir=data_dir,
        split='train',
        split_file=split_file,
        transform=None
    )

    logger.info(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        # Load first sample
        image, label = dataset[0]
        logger.info(f"Sample image shape: {image.shape}")
        logger.info(f"Sample label: {label}")
        logger.info("Basic loading test PASSED")
    else:
        logger.warning("Dataset is empty!")


def test_with_augmentation(data_dir: str, split_file: str):
    """Test dataset with augmentation."""
    logger = setup_logging()
    logger.info("\nTesting dataset with augmentation...")

    # Create dataset with augmentation
    transform = get_train_augmentation(image_size=224)
    dataset = StrokeDataset(
        data_dir=data_dir,
        split='train',
        split_file=split_file,
        transform=transform
    )

    if len(dataset) > 0:
        # Load sample
        image, label = dataset[0]
        logger.info(f"Augmented image shape: {image.shape}")
        logger.info(f"Augmented image dtype: {image.dtype}")
        logger.info("Augmentation test PASSED")
    else:
        logger.warning("Dataset is empty!")


def test_dataloader_speed(data_dir: str, split_file: str, batch_size: int = 32, num_workers: int = 4):
    """Benchmark DataLoader speed."""
    logger = setup_logging()
    logger.info(f"\nBenchmarking DataLoader (batch_size={batch_size}, num_workers={num_workers})...")

    transform = get_train_augmentation(image_size=224)
    dataset = StrokeDataset(
        data_dir=data_dir,
        split='train',
        split_file=split_file,
        transform=transform
    )

    if len(dataset) == 0:
        logger.warning("Dataset is empty! Cannot benchmark.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Warmup
    logger.info("Warming up...")
    for i, (images, labels) in enumerate(dataloader):
        if i >= 2:
            break

    # Benchmark
    logger.info("Benchmarking...")
    start_time = time.time()
    total_samples = 0

    for i, (images, labels) in enumerate(dataloader):
        total_samples += images.size(0)
        if i >= 20:  # Test first 20 batches
            break

    elapsed_time = time.time() - start_time
    samples_per_sec = total_samples / elapsed_time

    logger.info(f"Loaded {total_samples} samples in {elapsed_time:.2f} seconds")
    logger.info(f"Speed: {samples_per_sec:.2f} samples/second")
    logger.info(f"Batch loading time: {elapsed_time / (i + 1):.3f} seconds/batch")


def test_patient_level_dataset(data_dir: str, split_file: str):
    """Test patient-level dataset."""
    logger = setup_logging()
    logger.info("\nTesting patient-level dataset...")

    dataset = PatientLevelDataset(
        data_dir=data_dir,
        split='train',
        split_file=split_file,
        transform=get_val_augmentation(224)
    )

    logger.info(f"Number of patients: {len(dataset)}")

    if len(dataset) > 0:
        # Load first patient
        images, label = dataset[0]
        logger.info(f"Patient has {len(images)} images")
        if len(images) > 0:
            logger.info(f"Image shape: {images[0].shape}")
        logger.info(f"Label: {label}")
        logger.info("Patient-level dataset test PASSED")
    else:
        logger.warning("Dataset is empty!")


def test_class_balance(data_dir: str, split_file: str):
    """Test class balance in dataset."""
    logger = setup_logging()
    logger.info("\nTesting class balance...")

    dataset = StrokeDataset(
        data_dir=data_dir,
        split='train',
        split_file=split_file,
        transform=None
    )

    if len(dataset) == 0:
        logger.warning("Dataset is empty!")
        return

    # Count classes
    class_counts = {0: 0, 1: 0}
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_counts[label] += 1

    logger.info(f"Class 0 (CE): {class_counts[0]} samples")
    logger.info(f"Class 1 (LAA): {class_counts[1]} samples")

    if class_counts[0] > 0 and class_counts[1] > 0:
        ratio = max(class_counts[0], class_counts[1]) / min(class_counts[0], class_counts[1])
        logger.info(f"Imbalance ratio: {ratio:.2f}:1")


def main():
    parser = argparse.ArgumentParser(description='Test DataLoader functionality')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--split_file', type=str, default='data/splits/train.json',
                        help='Split file path')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for speed test')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader')
    args = parser.parse_args()

    print("="*60)
    print("DataLoader Testing Suite")
    print("="*60)

    # Run tests
    test_basic_loading(args.data_dir, args.split_file)
    test_with_augmentation(args.data_dir, args.split_file)
    test_class_balance(args.data_dir, args.split_file)
    test_patient_level_dataset(args.data_dir, args.split_file)
    test_dataloader_speed(args.data_dir, args.split_file, args.batch_size, args.num_workers)

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == '__main__':
    main()
