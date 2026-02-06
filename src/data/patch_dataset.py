"""
Patch-based dataset for whole-slide pathology images.
Extracts multiple patches per image instead of resizing.
"""

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PatchDataset(Dataset):
    """
    Patch-based dataset that extracts multiple patches from whole-slide images.

    Args:
        data_dir: Root directory containing CE/ and LAA/ subdirectories
        split: 'train', 'val', or 'test'
        patch_size: Size of patches to extract (default: 512)
        num_patches_per_image: Number of random patches to extract per image
        transform: Albumentations transform pipeline
        mode: 'random' (random patches) or 'grid' (grid-based patches)
    """

    def __init__(
        self,
        data_dir,
        split,
        patch_size=512,
        num_patches_per_image=16,
        transform=None,
        mode='random'
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.transform = transform
        self.mode = mode

        # Load split file
        splits_dir = self.data_dir / 'splits'
        split_file = splits_dir / f'{split}.txt'

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            patient_ids = [line.strip() for line in f]

        # Collect all images for patients in this split
        self.samples = []
        self.class_to_idx = {'CE': 0, 'LAA': 1}

        for class_name in ['CE', 'LAA']:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            for patient_id in patient_ids:
                patient_dir = class_dir / patient_id
                if not patient_dir.exists():
                    continue

                # Find all image files for this patient
                for img_file in patient_dir.glob('*.jpg'):
                    self.samples.append({
                        'path': img_file,
                        'label': self.class_to_idx[class_name],
                        'patient_id': patient_id,
                        'class': class_name
                    })

        print(f"{split.capitalize()} set: {len(self.samples)} images from {len(patient_ids)} patients")

    def __len__(self):
        return len(self.samples) * self.num_patches_per_image

    def __getitem__(self, idx):
        # Map flat index to (image_idx, patch_idx)
        image_idx = idx // self.num_patches_per_image
        patch_idx = idx % self.num_patches_per_image

        sample = self.samples[image_idx]

        # Load image
        image = Image.open(sample['path']).convert('RGB')
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Extract patch
        if self.mode == 'random':
            patch = self._extract_random_patch(image_np, h, w)
        else:  # grid
            patch = self._extract_grid_patch(image_np, h, w, patch_idx)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=patch)
            patch = transformed['image']

        return patch, sample['label']

    def _extract_random_patch(self, image, h, w):
        """Extract a random patch from the image."""
        if h < self.patch_size or w < self.patch_size:
            # Image smaller than patch size, resize
            image = Image.fromarray(image)
            image = image.resize((self.patch_size, self.patch_size), Image.LANCZOS)
            return np.array(image)

        # Random crop
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)

        patch = image[y:y+self.patch_size, x:x+self.patch_size]
        return patch

    def _extract_grid_patch(self, image, h, w, patch_idx):
        """Extract a patch from a regular grid."""
        # Calculate grid dimensions
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size

        if grid_h == 0 or grid_w == 0:
            # Image too small, resize
            image = Image.fromarray(image)
            image = image.resize((self.patch_size, self.patch_size), Image.LANCZOS)
            return np.array(image)

        # Get grid position
        grid_y = (patch_idx // grid_w) % grid_h
        grid_x = patch_idx % grid_w

        y = grid_y * self.patch_size
        x = grid_x * self.patch_size

        patch = image[y:y+self.patch_size, x:x+self.patch_size]
        return patch

    def get_patient_samples(self, patient_id):
        """Get all samples for a specific patient (for patient-level aggregation)."""
        patient_samples = []
        for i, sample in enumerate(self.samples):
            if sample['patient_id'] == patient_id:
                # Get all patches for this image
                for patch_idx in range(self.num_patches_per_image):
                    idx = i * self.num_patches_per_image + patch_idx
                    patient_samples.append(idx)
        return patient_samples


def get_patch_transforms(config, train=True):
    """Create augmentation pipeline for patch-based training."""

    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            ], p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
