"""PyTorch Dataset classes for medical image data."""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import json
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class StrokeDataset(Dataset):
    """Dataset for stroke blood clot classification.

    Args:
        data_dir: Path to data directory
        split: One of 'train', 'val', or 'test'
        split_file: Path to split JSON file (e.g., 'data/splits/train.json')
        transform: Optional transform to be applied on images
        target_size: Tuple of (height, width) for resizing images
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        split_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size

        # Load split information
        if split_file is None:
            split_file = f'data/splits/{split}.json'

        self.image_paths = []
        self.labels = []

        # Try to load from split file
        split_path = Path(split_file)
        if split_path.exists():
            self._load_from_split_file(split_path)
        else:
            # Fallback: scan directory structure
            logger.warning(f"Split file {split_file} not found. Scanning directory structure...")
            self._scan_directory()

        # Validate that dataset is not empty
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No valid images found in {self.data_dir} for split '{split}'. "
                f"Please ensure data directory exists and contains images."
            )

    def _load_from_split_file(self, split_path: Path):
        """Load data paths from split JSON file."""
        with open(split_path, 'r') as f:
            split_data = json.load(f)

        class_map = {'CE': 0, 'LAA': 1}

        for patient_id, patient_info in split_data['patients'].items():
            class_name = patient_info['class']
            label = class_map[class_name]

            for img_path in patient_info['images']:
                full_path = self.data_dir / img_path
                if full_path.exists():
                    self.image_paths.append(str(full_path))
                    self.labels.append(label)

    def _scan_directory(self):
        """Fallback: scan directory for images."""
        class_map = {'CE': 0, 'LAA': 1}

        for class_name, label in class_map.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            # Find all images
            for patient_dir in class_dir.iterdir():
                if not patient_dir.is_dir():
                    continue

                for img_path in patient_dir.glob('*'):
                    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.npy']:
                        self.image_paths.append(str(img_path))
                        self.labels.append(label)

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
        image_path = Path(self.image_paths[idx])
        label = self.labels[idx]

        # Load image with error handling
        try:
            if image_path.suffix == '.npy':
                image = np.load(image_path).astype(np.float32)
            else:
                image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a blank image as fallback
            image = np.zeros((224, 224, 3), dtype=np.float32)
            logger.warning(f"Returning blank image for corrupted file: {image_path}")

        # Apply transforms
        if self.transform:
            if hasattr(self.transform, '__call__'):
                # Albumentations format
                transformed = self.transform(image=image)
                if isinstance(transformed, dict):
                    image = transformed['image']
                else:
                    image = transformed

        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            if image.ndim == 2:  # Grayscale
                image = torch.from_numpy(image).unsqueeze(0)
            elif image.ndim == 3:
                image = torch.from_numpy(image).permute(2, 0, 1)

        return image, label


class PatientLevelDataset(Dataset):
    """Dataset that groups images by patient for patient-level prediction.

    Args:
        data_dir: Path to data directory
        split: One of 'train', 'val', or 'test'
        split_file: Path to split JSON file
        transform: Optional transform to be applied on images
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        split_file: Optional[str] = None,
        transform: Optional[Callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load split information
        if split_file is None:
            split_file = f'data/splits/{split}.json'

        self.patient_ids = []
        self.patient_image_paths = {}  # patient_id -> list of image paths
        self.patient_labels = {}  # patient_id -> label

        split_path = Path(split_file)
        if split_path.exists():
            self._load_from_split_file(split_path)
        else:
            print(f"Warning: Split file {split_file} not found.")

    def _load_from_split_file(self, split_path: Path):
        """Load patient data from split JSON file."""
        with open(split_path, 'r') as f:
            split_data = json.load(f)

        class_map = {'CE': 0, 'LAA': 1}

        for patient_id, patient_info in split_data['patients'].items():
            class_name = patient_info['class']
            label = class_map[class_name]

            # Store patient info
            self.patient_ids.append(patient_id)
            self.patient_labels[patient_id] = label

            # Store image paths
            image_paths = []
            for img_path in patient_info['images']:
                full_path = self.data_dir / img_path
                if full_path.exists():
                    image_paths.append(str(full_path))

            self.patient_image_paths[patient_id] = image_paths

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
        for img_path_str in image_paths:
            img_path = Path(img_path_str)

            # Load image
            if img_path.suffix == '.npy':
                image = np.load(img_path).astype(np.float32)
            else:
                image = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)

            # Apply transforms
            if self.transform:
                if hasattr(self.transform, '__call__'):
                    transformed = self.transform(image=image)
                    if isinstance(transformed, dict):
                        image = transformed['image']
                    else:
                        image = transformed

            # Convert to tensor if not already
            if not isinstance(image, torch.Tensor):
                if image.ndim == 2:  # Grayscale
                    image = torch.from_numpy(image).unsqueeze(0)
                elif image.ndim == 3:
                    image = torch.from_numpy(image).permute(2, 0, 1)

            images.append(image)

        return images, label
