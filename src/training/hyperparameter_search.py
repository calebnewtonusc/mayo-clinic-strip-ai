"""Hyperparameter search utilities."""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Callable
import json
from pathlib import Path
import itertools
from datetime import datetime

logger = logging.getLogger(__name__)


class HyperparameterSearch:
    """Base class for hyperparameter search."""

    def __init__(
        self,
        train_fn: Callable,
        param_grid: Dict[str, List[Any]],
        metric: str = 'val_acc',
        maximize: bool = True
    ):
        """Initialize hyperparameter search.

        Args:
            train_fn: Training function that takes config dict and returns metrics
            param_grid: Dictionary of parameter names to lists of values
            metric: Metric to optimize
            maximize: Whether to maximize (True) or minimize (False) the metric
        """
        self.train_fn = train_fn
        self.param_grid = param_grid
        self.metric = metric
        self.maximize = maximize
        self.results = []

    def search(self):
        """Run the search. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from completed search."""
        if not self.results:
            raise ValueError("No results available. Run search() first.")

        if self.maximize:
            best = max(self.results, key=lambda x: x['metrics'][self.metric])
        else:
            best = min(self.results, key=lambda x: x['metrics'][self.metric])

        return best['params']

    def save_results(self, output_path: str):
        """Save search results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Saved search results to {output_path}")


class GridSearch(HyperparameterSearch):
    """Grid search over hyperparameter space."""

    def search(self) -> Dict[str, Any]:
        """Perform grid search."""
        # Generate all combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Starting grid search with {len(combinations)} combinations...")

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            logger.info(f"[{i+1}/{len(combinations)}] Testing: {params}")

            # Train with these parameters
            metrics = self.train_fn(params)

            # Store results
            self.results.append({
                'params': params,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"  {self.metric}: {metrics[self.metric]:.4f}")

        # Get best parameters
        best_params = self.get_best_params()
        logger.info(f"Best parameters: {best_params}")

        return best_params


class RandomSearch(HyperparameterSearch):
    """Random search over hyperparameter space."""

    def __init__(
        self,
        train_fn: Callable,
        param_distributions: Dict[str, Callable],
        n_iterations: int = 20,
        metric: str = 'val_acc',
        maximize: bool = True,
        seed: int = 42
    ):
        """Initialize random search.

        Args:
            train_fn: Training function
            param_distributions: Dict of parameter names to sampling functions
            n_iterations: Number of random samples
            metric: Metric to optimize
            maximize: Whether to maximize metric
            seed: Random seed
        """
        super().__init__(train_fn, {}, metric, maximize)
        self.param_distributions = param_distributions
        self.n_iterations = n_iterations
        self.rng = np.random.RandomState(seed)

    def search(self) -> Dict[str, Any]:
        """Perform random search."""
        logger.info(f"Starting random search with {self.n_iterations} iterations...")

        for i in range(self.n_iterations):
            # Sample parameters
            params = {}
            for param_name, sampling_fn in self.param_distributions.items():
                params[param_name] = sampling_fn(self.rng)

            logger.info(f"[{i+1}/{self.n_iterations}] Testing: {params}")

            # Train with these parameters
            metrics = self.train_fn(params)

            # Store results
            self.results.append({
                'params': params,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"  {self.metric}: {metrics[self.metric]:.4f}")

        # Get best parameters
        best_params = self.get_best_params()
        logger.info(f"Best parameters: {best_params}")

        return best_params


def sample_uniform(low: float, high: float):
    """Create uniform sampling function."""
    return lambda rng: rng.uniform(low, high)


def sample_loguniform(low: float, high: float):
    """Create log-uniform sampling function."""
    return lambda rng: np.exp(rng.uniform(np.log(low), np.log(high)))


def sample_choice(choices: List[Any]):
    """Create choice sampling function."""
    return lambda rng: rng.choice(choices)


def sample_int(low: int, high: int):
    """Create integer sampling function."""
    return lambda rng: rng.randint(low, high + 1)


class ExperimentTracker:
    """Track and compare multiple experiments."""

    def __init__(self, experiments_dir: str = 'experiments'):
        self.experiments_dir = Path(experiments_dir)
        self.experiments = []

    def add_experiment(
        self,
        name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        checkpoint_path: str = None
    ):
        """Add an experiment to tracker."""
        experiment = {
            'name': name,
            'params': params,
            'metrics': metrics,
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.now().isoformat()
        }

        self.experiments.append(experiment)

    def save(self, filename: str = 'experiment_tracker.json'):
        """Save all experiments to JSON."""
        output_path = self.experiments_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)

        logger.info(f"Saved {len(self.experiments)} experiments to {output_path}")

    def load(self, filename: str = 'experiment_tracker.json'):
        """Load experiments from JSON."""
        input_path = self.experiments_dir / filename

        if not input_path.exists():
            logger.warning(f"No tracker file found at {input_path}")
            return

        with open(input_path, 'r') as f:
            self.experiments = json.load(f)

        logger.info(f"Loaded {len(self.experiments)} experiments")

    def get_best(self, metric: str = 'val_acc', maximize: bool = True):
        """Get best experiment by metric."""
        if not self.experiments:
            return None

        if maximize:
            best = max(self.experiments, key=lambda x: x['metrics'][metric])
        else:
            best = min(self.experiments, key=lambda x: x['metrics'][metric])

        return best

    def compare(self, metric: str = 'val_acc'):
        """Compare all experiments by metric."""
        if not self.experiments:
            logger.warning("No experiments to compare")
            return

        logger.info(f"Experiment Comparison ({metric}):")
        logger.info("="*80)

        # Sort by metric
        sorted_exps = sorted(
            self.experiments,
            key=lambda x: x['metrics'][metric],
            reverse=True
        )

        for i, exp in enumerate(sorted_exps):
            logger.info(f"{i+1}. {exp['name']}: {exp['metrics'][metric]:.4f}")
            logger.info(f"   Params: {exp['params']}")

        logger.info("="*80)
