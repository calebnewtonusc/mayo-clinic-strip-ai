"""Data augmentation for medical images."""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


def get_train_augmentation(
    image_size: int = 224,
    p: float = 0.5
) -> A.Compose:
    """Get training data augmentation pipeline.

    Args:
        image_size: Target image size
        p: Probability of applying augmentations

    Returns:
        Albumentations composition of transforms
    """
    return A.Compose([
        # Resize
        A.Resize(height=image_size, width=image_size),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=p),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=p
        ),

        # Elastic deformation (common in medical imaging)
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.3
        ),

        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=p
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=p),

        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats as default
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_augmentation(image_size: int = 224) -> A.Compose:
    """Get validation/test data augmentation pipeline.

    Args:
        image_size: Target image size

    Returns:
        Albumentations composition of transforms
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_strong_augmentation(image_size: int = 224) -> A.Compose:
    """Get strong augmentation for handling limited data.

    Args:
        image_size: Target image size

    Returns:
        Albumentations composition of transforms
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),

        # Stronger geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.7
        ),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),

        # Stronger intensity augmentations
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(70, 130), p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.5),

        # Additional augmentations
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            p=0.5
        ),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(p=0.3),

        # Noise
        A.GaussNoise(var_limit=(10.0, 100.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),

        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


# TODO: Add medical-specific augmentations
# - Simulate imaging artifacts
# - Intensity non-uniformity simulation
# - MixUp/CutMix implementations
