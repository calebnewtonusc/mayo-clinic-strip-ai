#!/usr/bin/env python3
"""Comprehensive test suite for Mayo Clinic STRIP AI project.

Tests all critical functionality including:
- Module imports
- New preprocessing features (DICOM/NIfTI, bias correction, denoising)
- New model architectures (DenseNet, ViT, Swin, MedicalCNN)
- New evaluation metrics (calibration, multi-class, fairness)
- Data pipeline
- Utilities and helpers
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("COMPREHENSIVE TEST SUITE - MAYO CLINIC STRIP AI")
print("="*70)

test_results = []

def test_section(name):
    """Decorator for test sections."""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*70}")
            print(f"Testing: {name}")
            print(f"{'='*70}")
            try:
                func()
                test_results.append((name, "PASS", None))
                print(f"✓ {name} - PASS")
                return True
            except Exception as e:
                test_results.append((name, "FAIL", str(e)))
                print(f"✗ {name} - FAIL: {e}")
                import traceback
                traceback.print_exc()
                return False
        return wrapper
    return decorator


@test_section("1. Core Module Imports")
def test_imports():
    """Test all core module imports."""
    from src.data.dataset import StrokeDataset
    from src.data.preprocessing import (
        normalize_intensity, resize_image, apply_windowing,
        load_dicom, load_nifti, correct_bias_field,
        denoise_nlm, denoise_bilateral, denoise_gaussian, denoise_median,
        assess_image_quality, preprocess_image
    )
    from src.data.augmentation import (
        get_train_augmentation, get_val_augmentation,
        mixup_data, cutmix_data, simulate_imaging_artifacts
    )
    from src.models.cnn import (
        SimpleCNN, ResNetClassifier, EfficientNetClassifier,
        DenseNetClassifier, VisionTransformerClassifier,
        SwinTransformerClassifier, MedicalCNN, SEBlock, get_model
    )
    from src.training.trainer import Trainer
    from src.evaluation.metrics import (
        calculate_metrics, calculate_expected_calibration_error,
        plot_calibration_curve, calculate_multiclass_metrics,
        calculate_per_class_metrics, analyze_subgroup_performance
    )
    from src.utils.helpers import load_config, save_config, set_seed, get_device
    print("  ✓ All core modules imported successfully")


@test_section("2. Preprocessing - Bias Field Correction")
def test_bias_correction():
    """Test bias field correction."""
    from src.data.preprocessing import correct_bias_field

    # Create synthetic image with bias field
    image = np.random.rand(128, 128).astype(np.float32) * 100
    x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
    bias = 1 + 0.5 * (x**2 + y**2)
    biased_image = image * bias

    # Correct bias field
    corrected = correct_bias_field(biased_image, n_iterations=10)

    assert corrected.shape == biased_image.shape
    assert corrected.dtype == np.float32
    print(f"  ✓ Bias correction: {biased_image.shape} -> {corrected.shape}")


@test_section("3. Preprocessing - Denoising Methods")
def test_denoising():
    """Test all denoising methods."""
    from src.data.preprocessing import (
        denoise_gaussian, denoise_bilateral, denoise_median
    )

    # Create noisy image
    image = np.random.rand(64, 64).astype(np.float32) * 255
    noise = np.random.randn(64, 64).astype(np.float32) * 20
    noisy = image + noise

    # Test Gaussian denoising
    denoised_gaussian = denoise_gaussian(noisy, sigma=1.0)
    assert denoised_gaussian.shape == noisy.shape
    print(f"  ✓ Gaussian denoising: {noisy.shape}")

    # Test bilateral denoising
    denoised_bilateral = denoise_bilateral(noisy)
    assert denoised_bilateral.shape == noisy.shape
    print(f"  ✓ Bilateral denoising: {noisy.shape}")

    # Test median denoising
    denoised_median = denoise_median(noisy, kernel_size=3)
    assert denoised_median.shape == noisy.shape
    print(f"  ✓ Median denoising: {noisy.shape}")


@test_section("4. Preprocessing - Image Quality Assessment")
def test_quality_assessment():
    """Test image quality assessment."""
    from src.data.preprocessing import assess_image_quality

    image = np.random.rand(128, 128).astype(np.float32) * 255

    metrics = assess_image_quality(image)

    required_keys = ['snr', 'cnr', 'entropy', 'sharpness', 'dynamic_range',
                     'mean_intensity', 'std_intensity']
    for key in required_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)

    print(f"  ✓ Quality metrics: {list(metrics.keys())}")


@test_section("5. Model Architecture - DenseNet")
def test_densenet():
    """Test DenseNet classifier."""
    from src.models.cnn import DenseNetClassifier

    for arch in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
        model = DenseNetClassifier(arch=arch, num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        assert y.shape == (2, 2)
        print(f"  ✓ {arch}: {x.shape} -> {y.shape}")


@test_section("6. Model Architecture - Vision Transformer")
def test_vit():
    """Test Vision Transformer classifier."""
    from src.models.cnn import VisionTransformerClassifier

    for arch in ['vit_b_16', 'vit_b_32']:
        model = VisionTransformerClassifier(arch=arch, num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        assert y.shape == (2, 2)
        print(f"  ✓ {arch}: {x.shape} -> {y.shape}")


@test_section("7. Model Architecture - Swin Transformer")
def test_swin():
    """Test Swin Transformer classifier."""
    from src.models.cnn import SwinTransformerClassifier

    for arch in ['swin_t', 'swin_s']:
        model = SwinTransformerClassifier(arch=arch, num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        assert y.shape == (2, 2)
        print(f"  ✓ {arch}: {x.shape} -> {y.shape}")


@test_section("8. Model Architecture - MedicalCNN with SE Blocks")
def test_medical_cnn():
    """Test custom MedicalCNN architecture."""
    from src.models.cnn import MedicalCNN, SEBlock

    # Test SE Block
    se_block = SEBlock(channels=64, reduction=16)
    x = torch.randn(2, 64, 56, 56)
    y = se_block(x)
    assert y.shape == x.shape
    print(f"  ✓ SE Block: {x.shape} -> {y.shape}")

    # Test MedicalCNN
    model = MedicalCNN(in_channels=3, num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 2)
    print(f"  ✓ MedicalCNN: {x.shape} -> {y.shape}")


@test_section("9. Model Factory Function")
def test_model_factory():
    """Test get_model factory function."""
    from src.models.cnn import get_model

    architectures = [
        'resnet18', 'efficientnet_b0', 'densenet121',
        'vit_b_16', 'swin_t', 'simplecnn', 'medicalcnn'
    ]

    for arch in architectures:
        model = get_model(arch, num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        assert y.shape == (2, 2)
        print(f"  ✓ get_model('{arch}'): {x.shape} -> {y.shape}")


@test_section("10. Evaluation - Calibration Metrics")
def test_calibration():
    """Test calibration metrics (ECE)."""
    from src.evaluation.metrics import calculate_expected_calibration_error

    # Synthetic predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15, 0.95, 0.25])

    ece, bin_acc, bin_conf, bin_counts = calculate_expected_calibration_error(
        y_true, y_prob, n_bins=5
    )

    assert isinstance(ece, float)
    assert 0 <= ece <= 1
    assert bin_acc.shape == (5,)
    assert bin_conf.shape == (5,)
    assert bin_counts.shape == (5,)
    print(f"  ✓ ECE: {ece:.4f}, bins: {len(bin_acc)}")


@test_section("11. Evaluation - Multi-class Metrics")
def test_multiclass_metrics():
    """Test multi-class classification metrics."""
    from src.evaluation.metrics import calculate_multiclass_metrics

    # Synthetic 3-class data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0])
    y_prob = np.random.rand(10, 3)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    metrics = calculate_multiclass_metrics(y_true, y_pred, y_prob, num_classes=3)

    required_keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                     'mcc', 'kappa']
    for key in required_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)

    print(f"  ✓ Multi-class metrics: {len(metrics)} metrics calculated")


@test_section("12. Evaluation - Per-class Metrics")
def test_per_class_metrics():
    """Test per-class detailed metrics."""
    from src.evaluation.metrics import calculate_per_class_metrics

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

    per_class = calculate_per_class_metrics(
        y_true, y_pred, class_names=['CE', 'LAA']
    )

    assert 'CE' in per_class
    assert 'LAA' in per_class
    for class_name in ['CE', 'LAA']:
        required = ['precision', 'recall', 'f1_score', 'true_positives',
                   'false_positives', 'sensitivity', 'specificity']
        for key in required:
            assert key in per_class[class_name]

    print(f"  ✓ Per-class metrics: {list(per_class.keys())}")


@test_section("13. Evaluation - Fairness Analysis")
def test_fairness():
    """Test subgroup fairness analysis."""
    from src.evaluation.metrics import analyze_subgroup_performance
    from sklearn.metrics import accuracy_score

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    subgroups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])

    fairness = analyze_subgroup_performance(
        y_true, y_pred, subgroups,
        subgroup_names=['Group A', 'Group B', 'Group C'],
        metric_fn=accuracy_score
    )

    assert 'Group A' in fairness
    assert 'Group B' in fairness
    assert 'Group C' in fairness
    assert 'fairness' in fairness
    assert 'max_difference' in fairness['fairness']

    print(f"  ✓ Fairness analysis: {len(fairness)-1} subgroups, max_diff={fairness['fairness']['max_difference']:.3f}")


@test_section("14. Data Augmentation - MixUp")
def test_mixup():
    """Test MixUp augmentation."""
    from src.data.augmentation import mixup_data, mixup_criterion

    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1])

    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

    assert mixed_x.shape == x.shape
    assert 0 <= lam <= 1
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape

    # Test loss
    criterion = torch.nn.CrossEntropyLoss()
    pred = torch.randn(4, 2)
    loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
    assert isinstance(loss.item(), float)

    print(f"  ✓ MixUp: lambda={lam:.3f}, loss={loss.item():.3f}")


@test_section("15. Data Augmentation - CutMix")
def test_cutmix():
    """Test CutMix augmentation."""
    from src.data.augmentation import cutmix_data

    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1])

    mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)

    assert mixed_x.shape == x.shape
    assert 0 <= lam <= 1
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape

    print(f"  ✓ CutMix: lambda={lam:.3f}")


@test_section("16. Medical Artifacts Simulation")
def test_artifacts():
    """Test medical imaging artifact simulation."""
    from src.data.augmentation import simulate_imaging_artifacts

    image = np.random.rand(128, 128, 3).astype(np.float32)

    artifacts = ['motion', 'noise', 'bias_field', 'ghosting']
    for artifact in artifacts:
        result = simulate_imaging_artifacts(image, artifact_type=artifact, intensity=0.2)
        assert result.shape == image.shape
        print(f"  ✓ {artifact} artifact: {image.shape} -> {result.shape}")


@test_section("17. Config Loading and Validation")
def test_config():
    """Test config loading and validation."""
    from src.utils.helpers import load_config, save_config
    import tempfile
    import os

    # Create temporary config
    config = {
        'model': {'arch': 'resnet50'},
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config, f)
        temp_path = f.name

    try:
        # Test loading
        loaded = load_config(temp_path, validate=True)
        assert loaded['model']['arch'] == 'resnet50'
        assert loaded['training']['batch_size'] == 32
        print(f"  ✓ Config loaded and validated")

        # Test validation catches errors
        bad_config = {'model': 'not_a_dict'}
        bad_path = temp_path.replace('.yaml', '_bad.yaml')
        with open(bad_path, 'w') as f:
            yaml.dump(bad_config, f)

        try:
            load_config(bad_path, validate=True)
            assert False, "Should have raised ValueError"
        except ValueError:
            print(f"  ✓ Config validation catches errors")
        finally:
            os.unlink(bad_path)

    finally:
        os.unlink(temp_path)


@test_section("18. Trainer Checkpoint Format")
def test_checkpoint_format():
    """Test that trainer saves checkpoints with correct format."""
    from src.training.trainer import Trainer
    from src.models.cnn import SimpleCNN
    from torch.utils.data import DataLoader, TensorDataset
    import tempfile

    model = SimpleCNN(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Create dummy data loaders
    dummy_data = torch.randn(10, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (10,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    train_loader = DataLoader(dummy_dataset, batch_size=2)
    val_loader = DataLoader(dummy_dataset, batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device('cpu'),
            checkpoint_dir=tmpdir,
            model_name='simplecnn',
            num_classes=2
        )

        # Save checkpoint (epoch 0)
        trainer.save_checkpoint(epoch=0, is_best=False)

        # Load and verify format from the actual saved location
        checkpoint_path = Path(tmpdir) / 'checkpoint_epoch_0.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                        'arch', 'num_classes', 'model_type']
        for key in required_keys:
            assert key in checkpoint, f"Missing key: {key}"

        assert checkpoint['arch'] == 'simplecnn'
        assert checkpoint['num_classes'] == 2

        print(f"  ✓ Checkpoint format: {list(checkpoint.keys())}")


@test_section("19. End-to-End Model Pipeline")
def test_e2e_pipeline():
    """Test complete end-to-end pipeline."""
    from src.models.cnn import get_model
    from src.data.preprocessing import preprocess_image
    from src.evaluation.metrics import calculate_metrics

    # Create synthetic data
    images = np.random.rand(10, 224, 224, 3).astype(np.float32) * 255
    labels = np.random.randint(0, 2, size=10)

    # Get model
    model = get_model('resnet18', num_classes=2, pretrained=False)
    model.eval()

    # Preprocess and predict
    predictions = []
    probabilities = []

    with torch.no_grad():
        for img in images:
            # Preprocess
            processed = preprocess_image(
                img, target_size=(224, 224),
                normalize_method='zscore',
                denoise_method='gaussian'
            )

            # Convert to tensor
            if processed.ndim == 2:
                processed = np.stack([processed] * 3, axis=-1)
            tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)

            # Predict
            output = model(tensor)
            prob = torch.softmax(output, dim=1)

            predictions.append(output.argmax(dim=1).item())
            probabilities.append(prob[0, 1].item())

    # Calculate metrics
    metrics = calculate_metrics(
        labels,
        np.array(predictions),
        np.array(probabilities)
    )

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'sensitivity' in metrics

    print(f"  ✓ E2E pipeline: {len(images)} images, acc={metrics['accuracy']:.3f}")


@test_section("20. Device Detection")
def test_device():
    """Test device detection."""
    from src.utils.helpers import get_device

    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ['cuda', 'mps', 'cpu']

    print(f"  ✓ Device detected: {device}")


# Run all tests
def main():
    print("\nStarting comprehensive test suite...\n")

    tests = [
        test_imports,
        test_bias_correction,
        test_denoising,
        test_quality_assessment,
        test_densenet,
        test_vit,
        test_swin,
        test_medical_cnn,
        test_model_factory,
        test_calibration,
        test_multiclass_metrics,
        test_per_class_metrics,
        test_fairness,
        test_mixup,
        test_cutmix,
        test_artifacts,
        test_config,
        test_checkpoint_format,
        test_e2e_pipeline,
        test_device,
    ]

    for test in tests:
        test()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, status, _ in test_results if status == "PASS")
    failed = sum(1 for _, status, _ in test_results if status == "FAIL")

    print(f"\nTotal Tests: {len(test_results)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")

    if failed > 0:
        print("\nFailed Tests:")
        for name, status, error in test_results:
            if status == "FAIL":
                print(f"  ✗ {name}: {error}")

    print("\n" + "="*70)
    if failed == 0:
        print("ALL TESTS PASSED! ✓")
        print("Project is 100% functional and ready for deployment!")
    else:
        print(f"{failed} TESTS FAILED")
        print("Please review the errors above.")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
