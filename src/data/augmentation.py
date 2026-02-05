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


def mixup_data(x, y, alpha=1.0):
    """Apply MixUp augmentation.

    MixUp: Beyond Empirical Risk Minimization (Zhang et al., 2017)
    Combines two samples and their labels using a convex combination.

    Args:
        x: Batch of images (B, C, H, W)
        y: Batch of labels (B,)
        alpha: MixUp interpolation strength (typically 1.0)

    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda)
        where mixed_x = lambda * x + (1 - lambda) * x_shuffled
    """
    import torch

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for MixUp training.

    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: MixUp lambda

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation.

    CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)
    Cuts and pastes patches between images.

    Args:
        x: Batch of images (B, C, H, W)
        y: Batch of labels (B,)
        alpha: CutMix interpolation strength (typically 1.0)

    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda)
    """
    import torch

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Get bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix.

    Args:
        size: Image size (B, C, H, W)
        lam: Lambda from beta distribution

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling of center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# Medical-specific augmentation helpers
def simulate_imaging_artifacts(image, artifact_type='motion'):
    """Simulate common medical imaging artifacts.

    Args:
        image: Input image (numpy array)
        artifact_type: Type of artifact ('motion', 'noise', 'bias_field')

    Returns:
        Image with simulated artifacts
    """
    # Placeholder for future implementation
    # Would include motion blur, field inhomogeneity, etc.
    return image
