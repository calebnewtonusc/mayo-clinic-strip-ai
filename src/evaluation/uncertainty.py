"""Uncertainty quantification for model predictions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from scipy.stats import entropy


def enable_dropout(model: nn.Module):
    """Enable dropout layers during inference for Monte Carlo dropout.

    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def monte_carlo_dropout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_iterations: int = 30,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform Monte Carlo dropout for uncertainty estimation.

    Args:
        model: Trained model with dropout layers
        input_tensor: Input image tensor (1, C, H, W) or (B, C, H, W)
        n_iterations: Number of forward passes
        device: Device to run on

    Returns:
        Tuple of (mean_prediction, std_prediction, all_predictions)
    """
    if device is None:
        device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)

    # Set model to eval mode but keep dropout enabled
    model.eval()
    enable_dropout(model)

    predictions = []

    with torch.no_grad():
        for _ in range(n_iterations):
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())

    predictions = np.array(predictions)  # (n_iterations, batch_size, n_classes)

    # Calculate statistics
    mean_pred = predictions.mean(axis=0)  # (batch_size, n_classes)
    std_pred = predictions.std(axis=0)    # (batch_size, n_classes)

    return mean_pred, std_pred, predictions


def test_time_augmentation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    augmentation_fn: callable,
    n_augmentations: int = 10,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform test-time augmentation for uncertainty estimation.

    Args:
        model: Trained model
        input_tensor: Input image tensor (1, C, H, W)
        augmentation_fn: Function to augment images
        n_augmentations: Number of augmented versions
        device: Device to run on

    Returns:
        Tuple of (mean_prediction, std_prediction)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    predictions = []

    # Original image
    with torch.no_grad():
        output = model(input_tensor.to(device))
        probs = F.softmax(output, dim=1)
        predictions.append(probs.cpu().numpy())

    # Augmented images
    img_numpy = input_tensor[0].cpu().numpy().transpose(1, 2, 0)  # CHW to HWC

    for _ in range(n_augmentations - 1):
        # Apply augmentation
        augmented = augmentation_fn(image=img_numpy)

        if isinstance(augmented, dict):
            aug_image = augmented['image']
        else:
            aug_image = augmented

        # Convert back to tensor
        if not isinstance(aug_image, torch.Tensor):
            if aug_image.ndim == 2:
                aug_image = torch.from_numpy(aug_image).unsqueeze(0)
            else:
                aug_image = torch.from_numpy(aug_image).permute(2, 0, 1)

        aug_image = aug_image.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(aug_image)
            probs = F.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())

    predictions = np.array(predictions)  # (n_augmentations, 1, n_classes)

    # Calculate statistics
    mean_pred = predictions.mean(axis=0)  # (1, n_classes)
    std_pred = predictions.std(axis=0)

    return mean_pred, std_pred


def predictive_entropy(probabilities: np.ndarray) -> float:
    """Calculate predictive entropy as a measure of uncertainty.

    Args:
        probabilities: Probability distribution (n_classes,)

    Returns:
        Entropy value (higher = more uncertain)
    """
    return entropy(probabilities, base=2)


def mutual_information(predictions: np.ndarray) -> float:
    """Calculate mutual information from MC dropout predictions.

    Args:
        predictions: Predictions from multiple forward passes (n_iterations, n_classes)

    Returns:
        Mutual information (higher = more uncertain)
    """
    # Expected entropy
    mean_probs = predictions.mean(axis=0)
    expected_entropy = entropy(mean_probs, base=2)

    # Entropy of expected
    entropies = np.array([entropy(p, base=2) for p in predictions])
    entropy_of_expected = entropies.mean()

    # Mutual information
    mi = expected_entropy - entropy_of_expected

    return mi


def calculate_confidence_metrics(
    probabilities: np.ndarray,
    predictions: Optional[np.ndarray] = None
) -> dict:
    """Calculate various confidence metrics.

    Args:
        probabilities: Mean probabilities (n_samples, n_classes)
        predictions: All predictions from MC dropout (n_iterations, n_samples, n_classes)

    Returns:
        Dictionary of confidence metrics
    """
    metrics = {}

    # Max probability (confidence)
    max_probs = probabilities.max(axis=1)
    metrics['mean_confidence'] = max_probs.mean()
    metrics['median_confidence'] = np.median(max_probs)
    metrics['min_confidence'] = max_probs.min()
    metrics['max_confidence'] = max_probs.max()

    # Predictive entropy
    entropies = np.array([predictive_entropy(p) for p in probabilities])
    metrics['mean_entropy'] = entropies.mean()
    metrics['median_entropy'] = np.median(entropies)

    # Prediction variance (if MC dropout used)
    if predictions is not None:
        variances = predictions.var(axis=0)  # (n_samples, n_classes)
        max_variance = variances.max(axis=1)
        metrics['mean_variance'] = max_variance.mean()
        metrics['median_variance'] = np.median(max_variance)

        # Mutual information
        mi_scores = []
        for i in range(predictions.shape[1]):  # For each sample
            mi = mutual_information(predictions[:, i, :])
            mi_scores.append(mi)
        metrics['mean_mutual_information'] = np.mean(mi_scores)

    return metrics


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve (reliability diagram).

    Args:
        y_true: True labels (n_samples,)
        y_prob: Predicted probabilities for positive class (n_samples,)
        n_bins: Number of bins

    Returns:
        Tuple of (bin_confidences, bin_accuracies)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_confidences = []
    bin_accuracies = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_conf = y_prob[mask].mean()
            bin_acc = y_true[mask].mean()
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
        else:
            bin_confidences.append(np.nan)
            bin_accuracies.append(np.nan)

    return np.array(bin_confidences), np.array(bin_accuracies)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error (ECE).

    Args:
        y_true: True labels (n_samples,)
        y_prob: Predicted probabilities for positive class (n_samples,)
        n_bins: Number of bins

    Returns:
        ECE value (lower is better)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_conf = y_prob[mask].mean()
            bin_acc = y_true[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return ece


def temperature_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: Optional[float] = None
) -> float:
    """Perform temperature scaling for calibration.

    Args:
        logits: Model logits (n_samples, n_classes)
        labels: True labels (n_samples,)
        temperature: Temperature value (if None, optimizes it)

    Returns:
        Optimal temperature
    """
    if temperature is None:
        # Optimize temperature
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def eval():
            loss = F.cross_entropy(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        temperature = temperature.item()

    return temperature


def identify_uncertain_samples(
    probabilities: np.ndarray,
    predictions_mc: Optional[np.ndarray] = None,
    threshold_confidence: float = 0.7,
    threshold_entropy: float = 0.5
) -> np.ndarray:
    """Identify samples with high uncertainty.

    Args:
        probabilities: Mean probabilities (n_samples, n_classes)
        predictions_mc: MC dropout predictions (n_iterations, n_samples, n_classes)
        threshold_confidence: Confidence threshold (below = uncertain)
        threshold_entropy: Entropy threshold (above = uncertain)

    Returns:
        Boolean array indicating uncertain samples
    """
    uncertain = np.zeros(len(probabilities), dtype=bool)

    # Low confidence
    max_probs = probabilities.max(axis=1)
    uncertain |= (max_probs < threshold_confidence)

    # High entropy
    entropies = np.array([predictive_entropy(p) for p in probabilities])
    uncertain |= (entropies > threshold_entropy)

    # High variance (if MC dropout available)
    if predictions_mc is not None:
        variances = predictions_mc.var(axis=0).max(axis=1)
        threshold_variance = np.percentile(variances, 75)
        uncertain |= (variances > threshold_variance)

    return uncertain
