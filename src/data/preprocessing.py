"""Image preprocessing utilities for medical images."""

import numpy as np
from typing import Tuple, Optional, Dict
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


def load_dicom(dicom_path: str) -> Tuple[np.ndarray, dict]:
    """Load DICOM image file.

    Args:
        dicom_path: Path to DICOM file

    Returns:
        Tuple of (image array, metadata dictionary)

    Raises:
        ImportError: If pydicom is not installed
        FileNotFoundError: If DICOM file doesn't exist
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM loading. "
            "Install with: pip install pydicom"
        )

    import os
    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    # Read DICOM file
    ds = pydicom.dcmread(dicom_path)

    # Extract pixel array
    image = ds.pixel_array.astype(np.float32)

    # Apply rescale slope and intercept if available (Hounsfield units for CT)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        image = image * ds.RescaleSlope + ds.RescaleIntercept

    # Extract metadata
    metadata = {
        'patient_id': str(ds.PatientID) if hasattr(ds, 'PatientID') else 'unknown',
        'study_date': str(ds.StudyDate) if hasattr(ds, 'StudyDate') else 'unknown',
        'modality': str(ds.Modality) if hasattr(ds, 'Modality') else 'unknown',
        'pixel_spacing': list(ds.PixelSpacing) if hasattr(ds, 'PixelSpacing') else None,
        'slice_thickness': float(ds.SliceThickness) if hasattr(ds, 'SliceThickness') else None,
        'window_center': float(ds.WindowCenter) if hasattr(ds, 'WindowCenter') else None,
        'window_width': float(ds.WindowWidth) if hasattr(ds, 'WindowWidth') else None,
    }

    return image, metadata


def load_nifti(nifti_path: str, slice_idx: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """Load NIfTI image file.

    Args:
        nifti_path: Path to NIfTI file (.nii or .nii.gz)
        slice_idx: If provided, extract specific slice. If None and 3D, takes middle slice.

    Returns:
        Tuple of (image array, metadata dictionary)

    Raises:
        ImportError: If nibabel is not installed
        FileNotFoundError: If NIfTI file doesn't exist
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI loading. "
            "Install with: pip install nibabel"
        )

    import os
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    # Load NIfTI file
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)

    # Handle 3D volumes
    if data.ndim == 3:
        if slice_idx is not None:
            image = data[:, :, slice_idx]
        else:
            # Take middle slice
            image = data[:, :, data.shape[2] // 2]
    elif data.ndim == 4:
        # 4D data (e.g., time series) - take first timepoint and middle slice
        if slice_idx is not None:
            image = data[:, :, slice_idx, 0]
        else:
            image = data[:, :, data.shape[2] // 2, 0]
    else:
        image = data

    # Extract metadata
    header = img.header
    metadata = {
        'dimensions': list(data.shape),
        'voxel_size': list(header.get_zooms()),
        'data_type': str(header.get_data_dtype()),
        'affine': img.affine.tolist(),
    }

    return image, metadata


def correct_bias_field(
    image: np.ndarray,
    n_iterations: int = 50,
    convergence_threshold: float = 0.001
) -> np.ndarray:
    """Correct intensity non-uniformity (bias field) using N4ITK-like algorithm.

    This is a simplified version of N4ITK bias field correction.
    For full N4ITK, use SimpleITK.

    Args:
        image: Input image with bias field
        n_iterations: Number of iterations
        convergence_threshold: Convergence threshold

    Returns:
        Bias-corrected image
    """
    from scipy.ndimage import gaussian_filter

    # Ensure float type
    image = image.astype(np.float32)

    # Avoid division by zero
    image = np.where(image == 0, 1e-6, image)

    # Initial estimate - smoothed version of image
    bias_field = gaussian_filter(image, sigma=20)

    for iteration in range(n_iterations):
        # Estimate current corrected image
        corrected = image / (bias_field + 1e-6)

        # Smooth the corrected image to get new bias estimate
        new_bias = gaussian_filter(corrected, sigma=20)

        # Update bias field
        old_bias = bias_field.copy()
        bias_field = image / (new_bias + 1e-6)
        bias_field = gaussian_filter(bias_field, sigma=20)

        # Check convergence
        change = np.abs(bias_field - old_bias).mean()
        if change < convergence_threshold:
            break

    # Final correction
    corrected_image = image / (bias_field + 1e-6)

    # Normalize to original range
    corrected_image = corrected_image * (np.mean(image) / np.mean(corrected_image))

    return corrected_image


def denoise_nlm(
    image: np.ndarray,
    patch_size: int = 5,
    patch_distance: int = 6,
    h: float = 0.1
) -> np.ndarray:
    """Non-local means denoising for medical images.

    Args:
        image: Input noisy image
        patch_size: Size of patches to compare
        patch_distance: Maximum distance to search for similar patches
        h: Filtering parameter (higher = more smoothing)

    Returns:
        Denoised image
    """
    # Normalize h based on image intensity range
    img_range = image.max() - image.min()
    h_scaled = h * img_range

    denoised = cv2.fastNlMeansDenoising(
        image.astype(np.uint8) if image.max() <= 255 else (image / image.max() * 255).astype(np.uint8),
        None,
        h=h_scaled,
        templateWindowSize=patch_size,
        searchWindowSize=patch_distance * 2 + 1
    )

    # Scale back if needed
    if image.max() > 255:
        denoised = denoised.astype(np.float32) / 255.0 * image.max()

    return denoised.astype(np.float32)


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """Bilateral filtering for edge-preserving denoising.

    Args:
        image: Input noisy image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Denoised image
    """
    # Normalize to 0-255 range for cv2
    img_min, img_max = image.min(), image.max()
    normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    # Apply bilateral filter
    denoised = cv2.bilateralFilter(normalized, d, sigma_color, sigma_space)

    # Scale back to original range
    denoised = denoised.astype(np.float32) / 255.0 * (img_max - img_min) + img_min

    return denoised


def denoise_gaussian(
    image: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """Gaussian smoothing for noise reduction.

    Args:
        image: Input noisy image
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Denoised image
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma=sigma)


def denoise_median(
    image: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """Median filtering for salt-and-pepper noise removal.

    Args:
        image: Input noisy image
        kernel_size: Size of median filter kernel

    Returns:
        Denoised image
    """
    return cv2.medianBlur(
        image.astype(np.float32),
        kernel_size
    )


def assess_image_quality(image: np.ndarray) -> Dict[str, float]:
    """Assess medical image quality metrics.

    Args:
        image: Input image

    Returns:
        Dictionary of quality metrics
    """
    from scipy.stats import entropy

    # Ensure float type
    image = image.astype(np.float32)

    # 1. Signal-to-Noise Ratio (SNR)
    signal = np.mean(image)
    noise = np.std(image)
    snr = signal / (noise + 1e-6)

    # 2. Contrast-to-Noise Ratio (CNR)
    # Approximate by comparing different regions
    h, w = image.shape[:2]
    center_region = image[h//4:3*h//4, w//4:3*w//4]
    background_region = np.concatenate([
        image[0:h//4, :].flatten(),
        image[3*h//4:, :].flatten()
    ])

    if len(background_region) > 0:
        cnr = (np.mean(center_region) - np.mean(background_region)) / (np.std(background_region) + 1e-6)
    else:
        cnr = 0.0

    # 3. Entropy (information content)
    hist, _ = np.histogram(image.flatten(), bins=256, range=(image.min(), image.max()))
    hist = hist / (hist.sum() + 1e-6)
    img_entropy = entropy(hist + 1e-10)

    # 4. Sharpness (gradient magnitude)
    if len(image.shape) == 2:
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness = np.mean(gradient_magnitude)
    else:
        sharpness = 0.0

    # 5. Dynamic range
    dynamic_range = image.max() - image.min()

    return {
        'snr': float(snr),
        'cnr': float(cnr),
        'entropy': float(img_entropy),
        'sharpness': float(sharpness),
        'dynamic_range': float(dynamic_range),
        'mean_intensity': float(np.mean(image)),
        'std_intensity': float(np.std(image)),
    }


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize_method: str = 'zscore',
    denoise_method: Optional[str] = None,
    correct_bias: bool = False
) -> np.ndarray:
    """Complete preprocessing pipeline for medical images.

    Args:
        image: Input image
        target_size: Target size for resizing
        normalize_method: Normalization method ('zscore', 'minmax', 'percentile')
        denoise_method: Denoising method ('gaussian', 'bilateral', 'nlm', 'median', None)
        correct_bias: Whether to apply bias field correction

    Returns:
        Preprocessed image
    """
    # 1. Bias field correction if requested
    if correct_bias:
        image = correct_bias_field(image)

    # 2. Denoising if requested
    if denoise_method == 'gaussian':
        image = denoise_gaussian(image, sigma=1.0)
    elif denoise_method == 'bilateral':
        image = denoise_bilateral(image)
    elif denoise_method == 'nlm':
        image = denoise_nlm(image)
    elif denoise_method == 'median':
        image = denoise_median(image)

    # 3. Remove outliers
    image = remove_outliers(image)

    # 4. Normalize intensity
    image = normalize_intensity(image, method=normalize_method)

    # 5. Resize
    image = resize_image(image, target_size)

    return image
