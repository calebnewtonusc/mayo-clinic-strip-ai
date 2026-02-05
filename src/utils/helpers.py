"""General utility functions."""

import torch
import numpy as np
import random
import json
import yaml
from pathlib import Path
from typing import Dict, Any


def set_seed(seed: int = 42):
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: str):
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
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS device (Apple Silicon)')
    else:
        device = torch.device('cpu')
        print('Using CPU device')
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
