"""Integration tests for the entire pipeline."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
sys.path.append('..')

from src.data.dataset import StrokeDataset
from src.data.augmentation import (
    get_train_augmentation,
    get_val_augmentation,
    mixup_data,
    cutmix_data,
    mixup_criterion
)
from src.models.cnn import SimpleCNN, ResNetClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import calculate_metrics
from src.visualization.gradcam import GradCAM
from src.evaluation.uncertainty import monte_carlo_dropout
from torch.utils.data import DataLoader
import torch.nn as nn


class TestEndToEndPipeline:
    """Test complete pipeline from data loading to prediction."""

    def test_data_pipeline(self):
        """Test data loading and preprocessing pipeline."""
        # Create temporary data directory
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / 'processed'
            data_dir.mkdir(parents=True)

            # Create dummy images
            ce_dir = data_dir / 'CE'
            laa_dir = data_dir / 'LAA'
            ce_dir.mkdir()
            laa_dir.mkdir()

            # Create sample images
            for i in range(5):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(str(ce_dir / f'ce_{i}.png'), img)
                cv2.imwrite(str(laa_dir / f'laa_{i}.png'), img)

            # Test dataset creation
            transform = get_val_augmentation(224)
            dataset = StrokeDataset(
                data_dir=str(data_dir),
                split='train',
                transform=transform
            )

            assert len(dataset) > 0

            # Test data loading
            image, label = dataset[0]
            assert isinstance(image, torch.Tensor)
            assert image.shape == (3, 224, 224)
            assert label in [0, 1]

    def test_model_training_pipeline(self):
        """Test model training pipeline."""
        # Create small dataset
        num_samples = 20
        images = torch.randn(num_samples, 3, 224, 224)
        labels = torch.randint(0, 2, (num_samples,))

        # Create simple dataset
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(images, labels)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Initialize small model
        model = SimpleCNN(in_channels=3, num_classes=2)

        # Setup trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                save_dir=temp_dir,
                num_epochs=2,
                early_stopping_patience=5
            )

            # Train for 2 epochs
            trainer.train()

            # Check that model was trained
            assert trainer.current_epoch == 2
            assert len(trainer.train_losses) == 2

    def test_inference_pipeline(self):
        """Test inference pipeline."""
        model = SimpleCNN(in_channels=3, num_classes=2)
        model.eval()

        # Test single image
        image = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)

        assert output.shape == (1, 2)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_evaluation_pipeline(self):
        """Test evaluation metrics pipeline."""
        # Generate predictions
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_prob = np.random.rand(8, 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_prob)

        # Verify metrics exist
        assert 'accuracy' in metrics
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
        assert 'auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1


class TestAugmentationPipeline:
    """Test augmentation pipeline."""

    def test_mixup_pipeline(self):
        """Test MixUp augmentation."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        y = torch.randint(0, 2, (batch_size,))

        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

        # Check shapes
        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert 0 <= lam <= 1

        # Test mixup criterion
        model = SimpleCNN(in_channels=3, num_classes=2)
        criterion = nn.CrossEntropyLoss()

        pred = model(mixed_x)
        loss = mixup_criterion(criterion, pred, y_a, y_b, lam)

        assert loss.item() >= 0

    def test_cutmix_pipeline(self):
        """Test CutMix augmentation."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        y = torch.randint(0, 2, (batch_size,))

        mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)

        # Check shapes
        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert 0 <= lam <= 1

    def test_augmentation_transforms(self):
        """Test augmentation transforms."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Test train augmentation
        train_transform = get_train_augmentation(224)
        augmented = train_transform(image=image)
        assert 'image' in augmented
        assert isinstance(augmented['image'], torch.Tensor)

        # Test val augmentation
        val_transform = get_val_augmentation(224)
        transformed = val_transform(image=image)
        assert 'image' in transformed
        assert isinstance(transformed['image'], torch.Tensor)


class TestInterpretabilityPipeline:
    """Test interpretability pipeline."""

    def test_gradcam_pipeline(self):
        """Test Grad-CAM pipeline."""
        model = SimpleCNN(in_channels=3, num_classes=2)
        model.eval()

        # Create Grad-CAM
        gradcam = GradCAM(model)

        # Test image
        image = torch.randn(1, 3, 224, 224)

        # Generate heatmap
        heatmap = gradcam.generate_heatmap(image, target_class=1)

        assert heatmap is not None
        assert heatmap.shape[0] == 224  # Height
        assert heatmap.shape[1] == 224  # Width

    def test_uncertainty_pipeline(self):
        """Test uncertainty quantification pipeline."""
        model = SimpleCNN(in_channels=3, num_classes=2)

        # Add dropout for MC dropout
        model.train()  # Enable dropout

        image = torch.randn(1, 3, 224, 224)

        # Run MC dropout
        mean_pred, std_pred, all_preds = monte_carlo_dropout(
            model, image, n_iterations=10
        )

        assert mean_pred.shape == (1, 2)
        assert std_pred.shape == (1, 2)
        assert all_preds.shape == (10, 1, 2)
        assert np.all(std_pred >= 0)  # Std should be non-negative


class TestModelArchitectures:
    """Test different model architectures."""

    def test_resnet_pipeline(self):
        """Test ResNet architecture."""
        model = ResNetClassifier(
            arch='resnet18',
            num_classes=2,
            pretrained=False
        )
        model.eval()

        image = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(image)

        assert output.shape == (2, 2)

    def test_model_save_load(self):
        """Test model save and load."""
        model = SimpleCNN(in_channels=3, num_classes=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            save_path = Path(temp_dir) / 'model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'arch': 'simple_cnn',
                'num_classes': 2
            }, save_path)

            # Load model
            checkpoint = torch.load(save_path)
            new_model = SimpleCNN(in_channels=3, num_classes=2)
            new_model.load_state_dict(checkpoint['model_state_dict'])

            # Test that models produce same output
            model.eval()
            new_model.eval()

            image = torch.randn(1, 3, 224, 224)

            with torch.no_grad():
                output1 = model(image)
                output2 = new_model(image)

            assert torch.allclose(output1, output2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
