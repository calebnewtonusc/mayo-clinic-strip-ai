"""MLflow experiment tracking integration.

Comprehensive experiment tracking for ML workflows with MLflow.
Tracks parameters, metrics, artifacts, and models.
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class MLflowTracker:
    """MLflow experiment tracker for ML training runs.

    Features:
    - Automatic parameter logging
    - Real-time metric tracking
    - Model checkpointing and versioning
    - Artifact logging (plots, configs, etc.)
    - Tag management
    - Run comparison

    Example:
        >>> tracker = MLflowTracker(
        ...     experiment_name="mayo-strip-ai",
        ...     run_name="resnet50-baseline",
        ...     tracking_uri="./mlruns"
        ... )
        >>>
        >>> with tracker:
        ...     # Log parameters
        ...     tracker.log_params({
        ...         'architecture': 'resnet50',
        ...         'batch_size': 32,
        ...         'learning_rate': 0.001
        ...     })
        ...
        ...     # Training loop
        ...     for epoch in range(100):
        ...         train_loss, train_acc = train_epoch()
        ...         val_loss, val_acc = validate()
        ...
        ...         # Log metrics
        ...         tracker.log_metrics({
        ...             'train_loss': train_loss,
        ...             'train_acc': train_acc,
        ...             'val_loss': val_loss,
        ...             'val_acc': val_acc
        ...         }, step=epoch)
        ...
        ...         # Log model checkpoint
        ...         if val_acc > best_acc:
        ...             tracker.log_model(model, 'best_model')
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: str = './mlruns',
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_name: Name of MLflow experiment
            run_name: Name for this specific run (auto-generated if None)
            tracking_uri: MLflow tracking server URI or local directory
            tags: Dictionary of tags for this run
            nested: Whether this is a nested run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self.nested = nested
        self.run = None

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not set up MLflow experiment: {e}")
            self.experiment_id = None

        print(f"MLflow Tracker initialized:")
        print(f"  - Experiment: {experiment_name}")
        print(f"  - Run: {self.run_name}")
        print(f"  - Tracking URI: {tracking_uri}")

    def __enter__(self):
        """Start MLflow run context."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run context."""
        self.end_run()

    def start_run(self):
        """Start MLflow run."""
        if self.experiment_id is not None:
            self.run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                nested=self.nested
            )

            # Log tags
            if self.tags:
                mlflow.set_tags(self.tags)

            print(f"✓ Started MLflow run: {self.run.info.run_id}")
        else:
            print("Warning: MLflow run not started (experiment not initialized)")

    def end_run(self):
        """End MLflow run."""
        if self.run is not None:
            mlflow.end_run()
            print(f"✓ Ended MLflow run: {self.run.info.run_id}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters.

        Args:
            params: Dictionary of parameters to log
        """
        if self.run is None:
            return

        try:
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
            print(f"✓ Logged {len(flat_params)} parameters")
        except Exception as e:
            print(f"Warning: Could not log parameters: {e}")

    def log_param(self, key: str, value: Any):
        """Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        if self.run is None:
            return

        try:
            mlflow.log_param(key, value)
        except Exception as e:
            print(f"Warning: Could not log parameter {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics.

        Args:
            metrics: Dictionary of metric name -> value
            step: Step number (epoch, iteration, etc.)
        """
        if self.run is None:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Step number
        """
        if self.run is None:
            return

        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"Warning: Could not log metric {key}: {e}")

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ):
        """Log PyTorch model.

        Args:
            model: PyTorch model to log
            artifact_path: Path within run's artifact directory
            registered_model_name: Name for model registry (optional)
            signature: Model signature (optional)
            input_example: Example input for model (optional)
        """
        if self.run is None:
            return

        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
            print(f"✓ Logged model to {artifact_path}")
        except Exception as e:
            print(f"Warning: Could not log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact (file).

        Args:
            local_path: Local file path
            artifact_path: Path within run's artifact directory (optional)
        """
        if self.run is None:
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
            print(f"✓ Logged artifact: {local_path}")
        except Exception as e:
            print(f"Warning: Could not log artifact: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log directory of artifacts.

        Args:
            local_dir: Local directory path
            artifact_path: Path within run's artifact directory (optional)
        """
        if self.run is None:
            return

        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            print(f"✓ Logged artifacts from: {local_dir}")
        except Exception as e:
            print(f"Warning: Could not log artifacts: {e}")

    def log_dict(self, dictionary: Dict, filename: str):
        """Log dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            filename: Filename for artifact (e.g., 'config.json')
        """
        if self.run is None:
            return

        try:
            mlflow.log_dict(dictionary, filename)
            print(f"✓ Logged dictionary: {filename}")
        except Exception as e:
            print(f"Warning: Could not log dictionary: {e}")

    def log_figure(self, figure: plt.Figure, filename: str):
        """Log matplotlib figure.

        Args:
            figure: Matplotlib figure
            filename: Filename for artifact (e.g., 'plot.png')
        """
        if self.run is None:
            return

        try:
            mlflow.log_figure(figure, filename)
            print(f"✓ Logged figure: {filename}")
        except Exception as e:
            print(f"Warning: Could not log figure: {e}")

    def log_training_history(self, history: Dict[str, List[float]]):
        """Log training history plots.

        Args:
            history: Dictionary with keys like 'train_loss', 'val_loss', etc.
        """
        if self.run is None:
            return

        try:
            # Create loss plot
            if 'train_loss' in history and 'val_loss' in history:
                fig, ax = plt.subplots(figsize=(10, 6))
                epochs = range(1, len(history['train_loss']) + 1)
                ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
                ax.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                self.log_figure(fig, 'training_loss.png')
                plt.close(fig)

            # Create accuracy plot
            if 'train_acc' in history and 'val_acc' in history:
                fig, ax = plt.subplots(figsize=(10, 6))
                epochs = range(1, len(history['train_acc']) + 1)
                ax.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
                ax.plot(epochs, history['val_acc'], label='Val Accuracy', marker='o')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.set_title('Training and Validation Accuracy')
                ax.legend()
                ax.grid(True, alpha=0.3)
                self.log_figure(fig, 'training_accuracy.png')
                plt.close(fig)

            # Log history as JSON
            self.log_dict(history, 'training_history.json')

            print("✓ Logged training history plots")
        except Exception as e:
            print(f"Warning: Could not log training history: {e}")

    def log_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Log confusion matrix plot.

        Args:
            cm: Confusion matrix (numpy array)
            class_names: List of class names
        """
        if self.run is None:
            return

        try:
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            self.log_figure(fig, 'confusion_matrix.png')
            plt.close(fig)

            print("✓ Logged confusion matrix")
        except Exception as e:
            print(f"Warning: Could not log confusion matrix: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """Set tags for run.

        Args:
            tags: Dictionary of tag name -> value
        """
        if self.run is None:
            return

        try:
            mlflow.set_tags(tags)
            print(f"✓ Set {len(tags)} tags")
        except Exception as e:
            print(f"Warning: Could not set tags: {e}")

    def set_tag(self, key: str, value: str):
        """Set a single tag.

        Args:
            key: Tag name
            value: Tag value
        """
        if self.run is None:
            return

        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            print(f"Warning: Could not set tag {key}: {e}")

    def log_system_metrics(self):
        """Log system information (GPU, CPU, etc.)."""
        if self.run is None:
            return

        try:
            import psutil

            # CPU info
            self.log_param('cpu_count', psutil.cpu_count())
            self.log_param('cpu_percent', psutil.cpu_percent())

            # Memory info
            mem = psutil.virtual_memory()
            self.log_param('memory_total_gb', round(mem.total / 1e9, 2))
            self.log_param('memory_available_gb', round(mem.available / 1e9, 2))

            # GPU info
            if torch.cuda.is_available():
                self.log_param('gpu_count', torch.cuda.device_count())
                self.log_param('gpu_name', torch.cuda.get_device_name(0))
                self.log_param('cuda_version', torch.version.cuda)

            # PyTorch info
            self.log_param('pytorch_version', torch.__version__)

            print("✓ Logged system metrics")
        except Exception as e:
            print(f"Warning: Could not log system metrics: {e}")

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to string if not a basic type
                if not isinstance(v, (str, int, float, bool)):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)

    def get_run_id(self) -> Optional[str]:
        """Get current run ID.

        Returns:
            Run ID or None if no active run
        """
        if self.run is not None:
            return self.run.info.run_id
        return None

    def get_artifact_uri(self) -> Optional[str]:
        """Get artifact URI for current run.

        Returns:
            Artifact URI or None if no active run
        """
        if self.run is not None:
            return self.run.info.artifact_uri
        return None
