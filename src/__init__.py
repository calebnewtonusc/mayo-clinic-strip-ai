"""Mayo Clinic STRIP AI - Deep Learning for Stroke Classification.

This package provides tools for training and deploying deep learning models
to classify stroke types (Cardioembolic vs Large Artery Atherosclerosis)
from medical imaging data.

Main modules:
- data: Dataset loading, preprocessing, and augmentation
- models: Neural network architectures (ResNet, EfficientNet, SimpleCNN)
- training: Training loops, hyperparameter search, and optimization
- evaluation: Metrics, interpretability, and uncertainty quantification
- utils: Utilities for visualization, logging, and configuration
"""

__version__ = "1.0.0"
__author__ = "Mayo Clinic STRIP AI Team"
__license__ = "MIT"

# Import key components for easier access
try:
    from src.data.dataset import StrokeDataset, PatientLevelDataset
    from src.data.preprocessing import preprocess_image
    from src.models.cnn import (
        SimpleCNN,
        ResNetClassifier,
        EfficientNetClassifier,
        DenseNetClassifier,
        VisionTransformerClassifier,
        SwinTransformerClassifier,
        MedicalCNN,
        SEBlock,
        get_model
    )
    from src.training.trainer import Trainer
except ImportError:
    # Handle case where dependencies aren't installed yet
    pass

__all__ = [
    "StrokeDataset",
    "PatientLevelDataset",
    "preprocess_image",
    "SimpleCNN",
    "ResNetClassifier",
    "EfficientNetClassifier",
    "DenseNetClassifier",
    "VisionTransformerClassifier",
    "SwinTransformerClassifier",
    "MedicalCNN",
    "SEBlock",
    "get_model",
    "Trainer",
]
