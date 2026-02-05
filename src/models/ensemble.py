"""Model ensemble methods for improved predictions.

Ensemble methods combine multiple models to achieve better performance
than any single model. Critical for medical AI applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Literal
from pathlib import Path


class EnsembleVoting(nn.Module):
    """Ensemble using voting (hard or soft) from multiple models.

    For medical applications, ensembling reduces individual model errors
    and provides more reliable predictions.

    Input shape: (batch_size, in_channels, height, width)
    Output shape: (batch_size, num_classes)

    Example:
        >>> model1 = ResNetClassifier(arch='resnet18', num_classes=2)
        >>> model2 = EfficientNetClassifier(arch='efficientnet_b0', num_classes=2)
        >>> ensemble = EnsembleVoting(models=[model1, model2], voting='soft')
        >>> x = torch.randn(8, 3, 224, 224)
        >>> y = ensemble(x)  # Output: (8, 2)
    """

    def __init__(
        self,
        models: List[nn.Module],
        voting: Literal['hard', 'soft'] = 'soft',
        weights: Optional[List[float]] = None
    ):
        """Initialize ensemble.

        Args:
            models: List of models to ensemble
            voting: 'hard' for majority voting, 'soft' for probability averaging
            weights: Optional weights for each model (must sum to 1.0)
        """
        super(EnsembleVoting, self).__init__()

        if len(models) == 0:
            raise ValueError("Must provide at least one model")

        self.models = nn.ModuleList(models)
        self.voting = voting

        # Validate and normalize weights
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")
            weights = np.array(weights)
            if not np.isclose(weights.sum(), 1.0):
                weights = weights / weights.sum()
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            # Equal weights
            self.weights = torch.ones(len(models)) / len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Ensemble predictions (batch_size, num_classes)
        """
        if self.voting == 'soft':
            # Soft voting: average probabilities
            all_probs = []

            for model, weight in zip(self.models, self.weights):
                with torch.no_grad():
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                    all_probs.append(probs * weight.to(x.device))

            # Weighted average of probabilities
            ensemble_probs = torch.stack(all_probs).sum(dim=0)

            # Convert back to logits for consistency
            ensemble_logits = torch.log(ensemble_probs + 1e-10)
            return ensemble_logits

        else:  # hard voting
            # Hard voting: majority vote on predictions
            all_preds = []

            for model in self.models:
                with torch.no_grad():
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    all_preds.append(preds)

            # Stack predictions and find mode
            all_preds = torch.stack(all_preds)  # (n_models, batch_size)

            # Get majority vote for each sample
            batch_size = x.size(0)
            num_classes = self.models[0](x[:1]).size(1)
            ensemble_preds = torch.zeros(batch_size, dtype=torch.long, device=x.device)

            for i in range(batch_size):
                # Count votes for each class
                votes = torch.bincount(all_preds[:, i], minlength=num_classes)
                ensemble_preds[i] = votes.argmax()

            # Convert to one-hot then to logits
            ensemble_logits = F.one_hot(ensemble_preds, num_classes).float()
            return ensemble_logits

    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation.

        Args:
            x: Input tensor

        Returns:
            Tuple of (predictions, uncertainty)
            - predictions: Ensemble predictions (batch_size, num_classes)
            - uncertainty: Standard deviation across models (batch_size, num_classes)
        """
        all_probs = []

        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # Stack probabilities: (n_models, batch_size, num_classes)
        all_probs = torch.stack(all_probs)

        # Mean and std across models
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)

        return mean_probs, std_probs


class EnsembleStacking(nn.Module):
    """Stacking ensemble with meta-learner.

    Stacking trains a meta-model to combine base model predictions.
    Often achieves better performance than simple voting.

    Example:
        >>> base_models = [model1, model2, model3]
        >>> meta_model = SimpleCNN(in_channels=6, num_classes=2)  # 3 models * 2 classes = 6
        >>> ensemble = EnsembleStacking(base_models=base_models, meta_model=meta_model)
    """

    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: nn.Module,
        use_original_features: bool = False
    ):
        """Initialize stacking ensemble.

        Args:
            base_models: List of base models
            meta_model: Meta-learner that combines base predictions
            use_original_features: Whether to concatenate original input with predictions
        """
        super(EnsembleStacking, self).__init__()

        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
        self.use_original_features = use_original_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacking ensemble.

        Args:
            x: Input tensor

        Returns:
            Stacked ensemble predictions
        """
        # Get predictions from all base models
        base_predictions = []

        for model in self.base_models:
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                base_predictions.append(probs)

        # Concatenate base predictions
        stacked_features = torch.cat(base_predictions, dim=1)

        # Optionally add original features
        if self.use_original_features:
            # Flatten and concatenate with original input
            x_flat = x.view(x.size(0), -1)
            stacked_features = torch.cat([stacked_features, x_flat], dim=1)

        # Meta-model prediction
        meta_logits = self.meta_model(stacked_features.unsqueeze(-1).unsqueeze(-1))

        return meta_logits


class EnsembleSnapshot(nn.Module):
    """Snapshot ensemble using cyclic learning rates.

    Trains a single model with cyclic learning rates and saves snapshots
    at local minima. Free ensemble without training multiple models!

    Reference: Snapshot Ensembles (Huang et al., 2017)
    """

    def __init__(self, model: nn.Module, snapshot_paths: List[str]):
        """Initialize snapshot ensemble.

        Args:
            model: Base model architecture
            snapshot_paths: Paths to snapshot checkpoints
        """
        super(EnsembleSnapshot, self).__init__()

        self.model_template = model
        self.snapshots = nn.ModuleList()

        # Load all snapshots
        for path in snapshot_paths:
            snapshot = type(model)(**self._get_model_config(model))
            checkpoint = torch.load(path, map_location='cpu')
            snapshot.load_state_dict(checkpoint['model_state_dict'])
            snapshot.eval()
            self.snapshots.append(snapshot)

    def _get_model_config(self, model: nn.Module) -> Dict:
        """Extract model configuration."""
        # This is a simplified version - adjust based on your models
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through snapshot ensemble.

        Args:
            x: Input tensor

        Returns:
            Averaged predictions from all snapshots
        """
        all_probs = []

        for snapshot in self.snapshots:
            with torch.no_grad():
                logits = snapshot(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # Average probabilities
        ensemble_probs = torch.stack(all_probs).mean(dim=0)

        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-10)
        return ensemble_logits


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    device: torch.device = torch.device('cpu'),
    voting: str = 'soft',
    weights: Optional[List[float]] = None
) -> EnsembleVoting:
    """Create ensemble from model checkpoints.

    Args:
        checkpoint_paths: List of paths to model checkpoints
        device: Device to load models on
        voting: Voting strategy ('soft' or 'hard')
        weights: Optional weights for each model

    Returns:
        Ensemble model ready for inference

    Example:
        >>> paths = ['model1.pth', 'model2.pth', 'model3.pth']
        >>> ensemble = create_ensemble_from_checkpoints(paths, voting='soft')
        >>> predictions = ensemble(images)
    """
    from src.models.cnn import get_model

    models = []

    for path in checkpoint_paths:
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)

        # Get model configuration
        arch = checkpoint.get('arch', 'resnet18')
        num_classes = checkpoint.get('num_classes', 2)

        # Create and load model
        model = get_model(arch, num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        models.append(model)

    # Create ensemble
    ensemble = EnsembleVoting(models=models, voting=voting, weights=weights)
    ensemble.to(device)
    ensemble.eval()

    return ensemble


def evaluate_ensemble_diversity(
    models: List[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate diversity of ensemble models.

    Higher diversity often leads to better ensemble performance.

    Args:
        models: List of models in ensemble
        dataloader: Data loader for evaluation
        device: Device to run on

    Returns:
        Dictionary with diversity metrics:
        - disagreement: Fraction of samples where models disagree
        - q_statistic: Pairwise Q-statistic (correlation of errors)
        - kappa: Kappa statistic (inter-rater agreement)
    """
    all_predictions = []
    all_correct = []
    true_labels = []

    # Collect predictions from all models
    for model in models:
        model.eval()
        predictions = []
        correct = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)
                correct.extend((preds == labels.numpy()).astype(int))

                if len(true_labels) == 0:
                    true_labels.extend(labels.numpy())

        all_predictions.append(np.array(predictions))
        all_correct.append(np.array(correct))

    all_predictions = np.array(all_predictions)  # (n_models, n_samples)
    all_correct = np.array(all_correct)

    # Calculate disagreement rate
    disagreement = 0
    for i in range(len(true_labels)):
        if len(np.unique(all_predictions[:, i])) > 1:
            disagreement += 1
    disagreement /= len(true_labels)

    # Calculate Q-statistic (pairwise correlation)
    n_models = len(models)
    q_stats = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Contingency table
            n11 = np.sum((all_correct[i] == 1) & (all_correct[j] == 1))
            n00 = np.sum((all_correct[i] == 0) & (all_correct[j] == 0))
            n10 = np.sum((all_correct[i] == 1) & (all_correct[j] == 0))
            n01 = np.sum((all_correct[i] == 0) & (all_correct[j] == 1))

            # Q-statistic
            if (n11 * n00 + n01 * n10) > 0:
                q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
                q_stats.append(q)

    avg_q = np.mean(q_stats) if q_stats else 0.0

    return {
        'disagreement_rate': disagreement,
        'avg_q_statistic': avg_q,
        'n_models': n_models,
        'n_samples': len(true_labels)
    }
