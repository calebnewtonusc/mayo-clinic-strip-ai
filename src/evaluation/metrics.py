"""Evaluation metrics for medical image classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from typing import Dict, Tuple, List
import torch


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC-AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }

    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

    # Clinical metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive predictive value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value

    return metrics


def calculate_patient_level_metrics(
    patient_ids: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    aggregation: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate image-level predictions to patient-level.

    Args:
        patient_ids: List of patient IDs for each image
        y_true: Ground truth labels for each image
        y_prob: Predicted probabilities for each image
        aggregation: Aggregation method ('mean', 'max', 'majority')

    Returns:
        Tuple of (patient_labels, patient_predictions)
    """
    unique_patients = np.unique(patient_ids)
    patient_labels = []
    patient_probs = []

    for patient_id in unique_patients:
        # Get all predictions for this patient
        patient_mask = patient_ids == patient_id
        patient_true = y_true[patient_mask]
        patient_prob = y_prob[patient_mask]

        # Patient label (should be the same for all images)
        patient_label = patient_true[0]
        patient_labels.append(patient_label)

        # Aggregate predictions
        if aggregation == 'mean':
            agg_prob = np.mean(patient_prob)
        elif aggregation == 'max':
            agg_prob = np.max(patient_prob)
        elif aggregation == 'majority':
            # Majority vote of predicted classes
            patient_pred_classes = (patient_prob > 0.5).astype(int)
            agg_prob = np.mean(patient_pred_classes)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        patient_probs.append(agg_prob)

    return np.array(patient_labels), np.array(patient_probs)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None):
    """Print detailed classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['CE', 'LAA']

    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")


def calculate_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """Calculate confidence intervals using bootstrap.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_fn: Function to calculate metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)
    mean_score = np.mean(bootstrap_scores)

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    return mean_score, lower_bound, upper_bound


# TODO: Add more evaluation functions
# - Calibration plots and metrics
# - Multi-class support
# - Per-class metrics analysis
# - Subgroup analysis
