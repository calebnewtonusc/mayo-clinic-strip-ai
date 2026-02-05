"""Evaluation metrics for medical image classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from typing import Dict, Tuple, List, Optional
import torch
import matplotlib.pyplot as plt


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

    # Clinical metrics - handle edge case where confusion matrix isn't 2x2
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle degenerate case (all predictions same class)
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                if y_true[0] == 0:
                    tn = cm[0, 0]
                else:
                    tp = cm[0, 0]
    except ValueError:
        # Fallback for edge cases
        tn = fp = fn = tp = 0

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


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Expected Calibration Error (ECE).

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (ECE, bin_accuracies, bin_confidences, bin_counts)
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Initialize arrays
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Assign predictions to bins
    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find predictions in this bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if bin_idx == n_bins - 1:  # Include upper boundary for last bin
            in_bin = in_bin | (y_prob == 1.0)

        bin_count = in_bin.sum()

        if bin_count > 0:
            bin_confidences[bin_idx] = y_prob[in_bin].mean()
            bin_accuracies[bin_idx] = y_true[in_bin].mean()
            bin_counts[bin_idx] = bin_count

    # Calculate ECE
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / bin_counts.sum()

    return ece, bin_accuracies, bin_confidences, bin_counts


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = 'Calibration Curve'
):
    """Plot calibration curve (reliability diagram).

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save plot (if None, displays plot)
        title: Plot title
    """
    ece, bin_accuracies, bin_confidences, bin_counts = calculate_expected_calibration_error(
        y_true, y_prob, n_bins
    )

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(bin_confidences, bin_accuracies, 'o-', label=f'Model (ECE={ece:.4f})')

    # Add confidence bars
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ax1.plot([bin_confidences[i], bin_confidences[i]],
                    [0, bin_accuracies[i]], 'b-', alpha=0.3)

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot histogram of predictions
    ax2.hist(y_prob, bins=n_bins, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 3
) -> Dict[str, float]:
    """Calculate metrics for multi-class classification.

    Args:
        y_true: Ground truth labels (shape: [n_samples])
        y_pred: Predicted labels (shape: [n_samples])
        y_prob: Predicted probabilities (shape: [n_samples, num_classes])
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, matthews_corrcoef, cohen_kappa_score
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
    }

    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i in range(len(precision_per_class)):
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
        metrics[f'f1_class_{i}'] = f1_per_class[i]

    # Calculate multi-class AUC if probabilities provided
    if y_prob is not None:
        try:
            # One-vs-Rest AUC
            metrics['roc_auc_ovr_macro'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            )
            metrics['roc_auc_ovr_weighted'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='weighted'
            )
        except ValueError:
            # Handle case where not all classes are present
            pass

    return metrics


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Calculate detailed metrics for each class separately.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Dictionary with per-class metrics
    """
    from sklearn.metrics import precision_recall_fscore_support

    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if class_names is None:
        class_names = [f'Class {i}' for i in classes]

    # Calculate metrics for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class_metrics = {}
    for i, (cls, name) in enumerate(zip(classes, class_names)):
        # Get predictions for this class
        cls_mask = y_true == cls
        cls_pred_mask = y_pred == cls

        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))

        # Calculate metrics
        per_class_metrics[name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': int(support[i]),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'sensitivity': recall[i],  # Same as recall
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        }

    return per_class_metrics


def analyze_subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: np.ndarray,
    subgroup_names: Optional[List[str]] = None,
    metric_fn: callable = accuracy_score
) -> Dict[str, float]:
    """Analyze model performance across different subgroups.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        subgroup_labels: Subgroup identifier for each sample
        subgroup_names: Optional names for subgroups
        metric_fn: Metric function to use

    Returns:
        Dictionary with per-subgroup metrics
    """
    unique_subgroups = np.unique(subgroup_labels)

    if subgroup_names is None:
        subgroup_names = [f'Subgroup {i}' for i in unique_subgroups]

    subgroup_metrics = {}
    for subgroup, name in zip(unique_subgroups, subgroup_names):
        mask = subgroup_labels == subgroup
        if mask.sum() > 0:
            metric = metric_fn(y_true[mask], y_pred[mask])
            subgroup_metrics[name] = {
                'metric': float(metric),
                'n_samples': int(mask.sum())
            }

    # Calculate fairness metrics
    metric_values = [v['metric'] for v in subgroup_metrics.values()]
    subgroup_metrics['fairness'] = {
        'max_difference': float(np.max(metric_values) - np.min(metric_values)),
        'std': float(np.std(metric_values)),
        'min': float(np.min(metric_values)),
        'max': float(np.max(metric_values)),
        'mean': float(np.mean(metric_values)),
    }

    return subgroup_metrics


def plot_confusion_matrix_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    save_path: Optional[str] = None,
    title: str = 'Confusion Matrix'
):
    """Plot confusion matrix for multi-class classification.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize by true label counts
        save_path: Path to save plot
        title: Plot title
    """
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
