"""Unit tests for preprocessing functions."""

import pytest
import numpy as np
import sys
sys.path.append('..')

from src.data.preprocessing import (
    normalize_intensity,
    resize_image,
    apply_windowing,
    remove_outliers
)


class TestNormalization:
    """Tests for intensity normalization."""

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        image = np.random.rand(100, 100) * 255

        normalized = normalize_intensity(image, method='zscore')

        # Check mean and std
        assert abs(np.mean(normalized)) < 1e-5  # Mean should be ~0
        assert abs(np.std(normalized) - 1.0) < 1e-5  # Std should be ~1

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        image = np.random.rand(100, 100) * 255

        normalized = normalize_intensity(image, method='minmax')

        # Check range
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_percentile_normalization(self):
        """Test percentile normalization."""
        image = np.random.rand(100, 100) * 255

        normalized = normalize_intensity(image, method='percentile')

        # Check range
        assert normalized.min() >= 0
        assert normalized.max() <= 1


class TestResize:
    """Tests for image resizing."""

    def test_resize_basic(self):
        """Test basic resizing."""
        image = np.random.rand(512, 512)
        target_size = (224, 224)

        resized = resize_image(image, target_size)

        assert resized.shape == target_size

    def test_resize_with_aspect_ratio(self):
        """Test resizing with aspect ratio preservation."""
        image = np.random.rand(512, 256)  # Non-square
        target_size = (224, 224)

        resized = resize_image(image, target_size, maintain_aspect_ratio=True)

        assert resized.shape == target_size


class TestWindowing:
    """Tests for intensity windowing."""

    def test_windowing(self):
        """Test intensity windowing."""
        image = np.random.rand(100, 100) * 1000 - 500

        windowed = apply_windowing(image, window_center=40, window_width=80)

        # Check range
        assert windowed.min() >= 0
        assert windowed.max() <= 1


class TestOutlierRemoval:
    """Tests for outlier removal."""

    def test_remove_outliers(self):
        """Test outlier removal."""
        image = np.random.rand(100, 100) * 255

        # Add outliers
        image[0, 0] = 10000
        image[0, 1] = -1000

        cleaned = remove_outliers(image, percentile_range=(1, 99))

        # Outliers should be clipped
        assert cleaned.max() < 1000
        assert cleaned.min() > -100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
