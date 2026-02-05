"""Unit tests for dataset classes."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')

from src.data.dataset import StrokeDataset, PatientLevelDataset


class TestStrokeDataset:
    """Tests for StrokeDataset class."""

    def test_dataset_initialization(self):
        """Test dataset can be initialized."""
        dataset = StrokeDataset(
            data_dir='data/processed',
            split='train',
            transform=None
        )
        assert dataset is not None

    def test_dataset_length(self):
        """Test dataset length is non-negative."""
        dataset = StrokeDataset(
            data_dir='data/processed',
            split='train'
        )
        assert len(dataset) >= 0

    def test_getitem_returns_correct_format(self):
        """Test __getitem__ returns (image, label) tuple."""
        dataset = StrokeDataset(
            data_dir='data/processed',
            split='train'
        )

        if len(dataset) > 0:
            image, label = dataset[0]

            # Check types
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, (int, np.integer))

            # Check shapes
            assert image.ndim == 3  # (C, H, W)
            assert label in [0, 1]  # Binary classification

    def test_dataset_with_transform(self):
        """Test dataset works with transforms."""
        from src.data.augmentation import get_val_augmentation

        transform = get_val_augmentation(224)
        dataset = StrokeDataset(
            data_dir='data/processed',
            split='train',
            transform=transform
        )

        if len(dataset) > 0:
            image, label = dataset[0]
            assert image.shape[1] == 224  # Height
            assert image.shape[2] == 224  # Width


class TestPatientLevelDataset:
    """Tests for PatientLevelDataset class."""

    def test_patient_dataset_initialization(self):
        """Test patient dataset initialization."""
        dataset = PatientLevelDataset(
            data_dir='data/processed',
            split='train'
        )
        assert dataset is not None

    def test_patient_dataset_returns_list(self):
        """Test patient dataset returns list of images."""
        dataset = PatientLevelDataset(
            data_dir='data/processed',
            split='train'
        )

        if len(dataset) > 0:
            images, label = dataset[0]

            assert isinstance(images, list)
            assert len(images) > 0
            assert isinstance(label, (int, np.integer))

            # Check first image
            assert isinstance(images[0], torch.Tensor)
            assert images[0].ndim == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
