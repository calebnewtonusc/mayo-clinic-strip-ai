"""General utility functions."""

import torch
import numpy as np
import random
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, validate: bool = True) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file
        validate: Whether to validate required keys

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is unsupported or validation fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config format: {config_path.suffix}. "
                    f"Expected .yaml, .yml, or .json"
                )
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse config file {config_path}: {e}")

    if config is None:
        raise ValueError(f"Config file {config_path} is empty")

    if not isinstance(config, dict):
        raise ValueError(
            f"Config must be a dictionary, got {type(config).__name__}"
        )

    # Basic validation of common required keys
    if validate:
        _validate_config(config, config_path)

    return config


def _validate_config(config: Dict[str, Any], config_path: Path) -> None:
    """Validate that config has reasonable structure.

    Args:
        config: Configuration dictionary
        config_path: Path to config file (for error messages)

    Raises:
        ValueError: If required keys are missing or invalid
    """
    # Check for common training config keys
    if 'model' in config:
        if not isinstance(config['model'], dict):
            raise ValueError(
                f"Config 'model' must be a dict in {config_path}"
            )
        if 'architecture' not in config['model']:
            raise ValueError(
                f"Config 'model' missing required 'architecture' key in {config_path}"
            )

    if 'training' in config:
        if not isinstance(config['training'], dict):
            raise ValueError(
                f"Config 'training' must be a dict in {config_path}"
            )

        # Validate numeric parameters
        numeric_keys = ['batch_size', 'num_epochs', 'learning_rate']
        for key in numeric_keys:
            if key in config['training']:
                value = config['training'][key]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(
                        f"Config 'training.{key}' must be a positive number, "
                        f"got {value} in {config_path}"
                    )

    # Validate data config
    if 'data' in config:
        if not isinstance(config['data'], dict):
            raise ValueError(
                f"Config 'data' must be a dict in {config_path}"
            )
        if 'data_dir' not in config['data']:
            raise ValueError(
                f"Config 'data' missing required 'data_dir' key in {config_path}"
            )

    # Validate loss config
    if 'loss' in config:
        if not isinstance(config['loss'], dict):
            raise ValueError(
                f"Config 'loss' must be a dict in {config_path}"
            )
        if 'type' not in config['loss']:
            raise ValueError(
                f"Config 'loss' missing required 'type' key in {config_path}"
            )


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {save_path}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU).

    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info('Using MPS device (Apple Silicon)')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU device')
    return device


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create directory for experiment outputs.

    Args:
        base_dir: Base experiments directory
        experiment_name: Name of the experiment

    Returns:
        Path to experiment directory
    """
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)

    return exp_dir
