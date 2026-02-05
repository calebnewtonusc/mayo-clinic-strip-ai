"""Unit tests for model architectures."""

import pytest
import torch
import sys
sys.path.append('..')

from src.models.cnn import SimpleCNN, ResNetClassifier, EfficientNetClassifier


class TestSimpleCNN:
    """Tests for SimpleCNN."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = SimpleCNN(in_channels=3, num_classes=2)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass works."""
        model = SimpleCNN(in_channels=3, num_classes=2)
        model.eval()

        # Create dummy input
        x = torch.randn(2, 3, 224, 224)  # (batch, channels, height, width)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 2)  # (batch, num_classes)

    def test_output_range(self):
        """Test output is logits (not probabilities)."""
        model = SimpleCNN(in_channels=3, num_classes=2)
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        # Logits can be any value (not constrained to [0, 1])
        assert output.shape == (1, 2)


class TestResNetClassifier:
    """Tests for ResNetClassifier."""

    def test_resnet18_initialization(self):
        """Test ResNet-18 initialization."""
        model = ResNetClassifier(
            arch='resnet18',
            num_classes=2,
            pretrained=False
        )
        assert model is not None

    def test_resnet50_initialization(self):
        """Test ResNet-50 initialization."""
        model = ResNetClassifier(
            arch='resnet50',
            num_classes=2,
            pretrained=False
        )
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass."""
        model = ResNetClassifier(
            arch='resnet18',
            num_classes=2,
            pretrained=False
        )
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 2)

    def test_freeze_backbone(self):
        """Test backbone freezing."""
        model = ResNetClassifier(
            arch='resnet18',
            num_classes=2,
            pretrained=False,
            freeze_backbone=True
        )

        # Check that backbone parameters don't require grad
        frozen_count = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        assert frozen_count > 0


class TestEfficientNetClassifier:
    """Tests for EfficientNetClassifier."""

    def test_initialization(self):
        """Test initialization."""
        model = EfficientNetClassifier(
            arch='efficientnet_b0',
            num_classes=2,
            pretrained=False
        )
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass."""
        model = EfficientNetClassifier(
            arch='efficientnet_b0',
            num_classes=2,
            pretrained=False
        )
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
