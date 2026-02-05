"""Image preprocessing utilities for medical images."""

import numpy as np
from typing import Tuple, Optional
import cv2


def normalize_intensity(
    image: np.ndarray,
    method: str = 'zscore',
    clip_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """Normalize image intensities.

    Args:
        image: Input image array
        method: Normalization method ('zscore', 'minmax', 'percentile')
        clip_range: Optional range to clip values before normalization

    Returns:
        Normalized image
    """
    image = image.astype(np.float32)

    if clip_range is not None:
        image = np.clip(image, clip_range[0], clip_range[1])

    if method == 'zscore':
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        return image

    elif method == 'minmax':
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        return image

    elif method == 'percentile':
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1)
        return image

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    maintain_aspect_ratio: bool = False,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resize image to target size.

    Args:
        image: Input image
        target_size: Target (height, width)
        maintain_aspect_ratio: If True, pad to maintain aspect ratio
        interpolation: OpenCV interpolation method

    Returns:
        Resized image
    """
    if maintain_aspect_ratio:
        # Calculate scaling factor
        h, w = image.shape[:2]
        target_h, target_w = target_size
        scale = min(target_h / h, target_w / w)

        # Resize with aspect ratio
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # Pad to target size
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )
        return padded
    else:
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)


def apply_windowing(
    image: np.ndarray,
    window_center: float,
    window_width: float
) -> np.ndarray:
    """Apply intensity windowing (common in medical imaging).

    Args:
        image: Input image
        window_center: Center of the intensity window
        window_width: Width of the intensity window

    Returns:
        Windowed image in range [0, 1]
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / window_width

    return windowed


def remove_outliers(
    image: np.ndarray,
    percentile_range: Tuple[float, float] = (1, 99)
) -> np.ndarray:
    """Remove outlier pixel values by clipping to percentiles.

    Args:
        image: Input image
        percentile_range: Tuple of (lower, upper) percentiles

    Returns:
        Image with outliers clipped
    """
    lower, upper = np.percentile(image, percentile_range)
    return np.clip(image, lower, upper)


# TODO: Add more preprocessing functions
# - DICOM loading and preprocessing
# - NIfTI loading and preprocessing
# - Bias field correction
# - Noise reduction filters
# - Image quality assessment
