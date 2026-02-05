"""PyTorch Dataset classes for medical image data."""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, List


class StrokeDataset(Dataset):
    """Dataset for stroke blood clot classification.

    Args:
        data_dir: Path to data directory
        split: One of 'train', 'val', or 'test'
        transform: Optional transform to be applied on images
        target_size: Tuple of (height, width) for resizing images
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size

        # TODO: Implement data loading logic
        self.image_paths = []
        self.labels = []

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, label)
        """
        # TODO: Implement image loading and preprocessing
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image (placeholder)
        image = np.zeros((*self.target_size, 3), dtype=np.float32)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1)

        return image, label


class PatientLevelDataset(Dataset):
    """Dataset that groups images by patient for patient-level prediction.

    Args:
        data_dir: Path to data directory
        split: One of 'train', 'val', or 'test'
        transform: Optional transform to be applied on images
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # TODO: Implement patient-level grouping
        self.patient_ids = []
        self.patient_image_paths = {}  # patient_id -> list of image paths
        self.patient_labels = {}  # patient_id -> label

    def __len__(self) -> int:
        """Return the number of patients in the dataset."""
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:
        """Get all images for a patient.

        Args:
            idx: Index of the patient

        Returns:
            Tuple of (list of images, label)
        """
        patient_id = self.patient_ids[idx]
        image_paths = self.patient_image_paths[patient_id]
        label = self.patient_labels[patient_id]

        # Load all images for this patient
        images = []
        for image_path in image_paths:
            # TODO: Implement image loading
            image = np.zeros((224, 224, 3), dtype=np.float32)

            if self.transform:
                image = self.transform(image)

            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).permute(2, 0, 1)

            images.append(image)

        return images, label
