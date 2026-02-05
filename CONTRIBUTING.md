# Contributing to Mayo Clinic STRIP AI

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## Code of Conduct

This is a research project with clinical implications. Please:

- **Be respectful** and professional in all interactions
- **Prioritize patient safety** - this code may influence medical decisions
- **Follow HIPAA guidelines** - never commit patient data or PII
- **Maintain scientific rigor** - validate claims and document assumptions
- **Collaborate openly** - share knowledge and help teammates

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- CUDA-capable GPU (recommended)
- Familiarity with PyTorch and medical imaging

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mayo-clinic-strip-ai.git
cd mayo-clinic-strip-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (including dev tools)
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy isort

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install

# Run tests to verify setup
python scripts/run_tests.py
```

### Understanding the Codebase

1. Read [PROJECT_STATUS.md](PROJECT_STATUS.md) for current status
2. Review [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for roadmap
3. Check [QUICKSTART.md](QUICKSTART.md) to understand basic usage
4. Run `python scripts/run_end_to_end_test.py` to see the pipeline in action

## Development Workflow

### Finding Tasks

1. Check [GitHub Issues](https://github.com/calebnewtonusc/mayo-clinic-strip-ai/issues)
2. Review [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for Phase 15 and 17 tasks
3. Look for `TODO` comments in the code
4. Propose new features by opening an issue first

### Branch Strategy

- `main` - Stable, production-ready code
- `develop` - Integration branch for features (if using GitFlow)
- `feature/feature-name` - Feature branches
- `bugfix/bug-name` - Bug fix branches
- `hotfix/critical-fix` - Critical fixes for main

### Making Changes

```bash
# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Run tests
python scripts/run_tests.py

# Run specific tests
pytest tests/test_models.py -v

# Commit changes
git add .
git commit -m "feat: add new feature"

# Push to your fork
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style

Follow [PEP 8](https://pep8.org/) with these additions:

- **Line length**: 100 characters (not 79)
- **Imports**: Organized with `isort`
- **Formatting**: Use `black` for consistent formatting
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Google-style docstrings for all functions/classes

### Example

```python
"""Module for preprocessing medical images."""

import numpy as np
import torch
from typing import Optional, Tuple


def normalize_intensity(
    image: np.ndarray,
    method: str = 'zscore',
    clip_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """Normalize image intensity values.

    Args:
        image: Input image as numpy array
        method: Normalization method ('zscore', 'minmax', 'percentile')
        clip_range: Optional tuple of (min, max) values to clip

    Returns:
        Normalized image

    Raises:
        ValueError: If method is not supported

    Example:
        >>> image = np.random.rand(100, 100) * 255
        >>> normalized = normalize_intensity(image, method='zscore')
        >>> assert abs(np.mean(normalized)) < 1e-5
    """
    if method not in ['zscore', 'minmax', 'percentile']:
        raise ValueError(f"Unsupported method: {method}")

    # Implementation...
    return image
```

### Code Organization

- **Keep functions focused** - Single responsibility principle
- **Avoid magic numbers** - Use constants or config values
- **Handle errors gracefully** - Use try/except with specific exceptions
- **Log appropriately** - Use logging module, not print statements
- **Document assumptions** - Especially for medical domain knowledge

### Medical Imaging Specific

- **Always preserve patient-level splits** - Never mix patients between train/val/test
- **Document preprocessing steps** - Medical images require careful handling
- **Validate clinical metrics** - Sensitivity, specificity, PPV, NPV must be correct
- **Consider class imbalance** - Medical datasets are often imbalanced
- **Interpretability required** - Medical models must be explainable

## Testing

### Test Requirements

All contributions must include tests:

- **Unit tests** for new functions/classes
- **Integration tests** for new features
- **Regression tests** for bug fixes
- **Coverage** - Aim for >80% coverage on new code

### Running Tests

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=src tests/ --cov-report=html

# Run specific test
pytest tests/test_models.py::TestResNetClassifier::test_forward_pass -v
```

### Writing Tests

```python
"""Tests for preprocessing functions."""

import pytest
import numpy as np
from src.data.preprocessing import normalize_intensity


class TestNormalization:
    """Tests for intensity normalization."""

    def test_zscore_normalization(self):
        """Test z-score normalization produces mean~0, std~1."""
        image = np.random.rand(100, 100) * 255

        normalized = normalize_intensity(image, method='zscore')

        assert abs(np.mean(normalized)) < 1e-5
        assert abs(np.std(normalized) - 1.0) < 1e-5

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        image = np.random.rand(100, 100)

        with pytest.raises(ValueError, match="Unsupported method"):
            normalize_intensity(image, method='invalid')
```

## Documentation

### Documentation Requirements

- **README updates** - Update README.md for user-facing changes
- **Docstrings** - All public functions/classes must have docstrings
- **Inline comments** - Complex logic should have explanatory comments
- **Usage examples** - Include examples in docstrings
- **Changelog** - Significant changes should be noted

### Writing Good Documentation

```python
def process_dicom_series(
    dicom_dir: str,
    output_dir: str,
    window_center: int = 40,
    window_width: int = 80
) -> List[np.ndarray]:
    """Process a series of DICOM files into normalized arrays.

    This function loads all DICOM files from a directory, applies
    intensity windowing (common in CT imaging), and normalizes
    the resulting images.

    Args:
        dicom_dir: Directory containing DICOM files (*.dcm)
        output_dir: Directory to save processed images
        window_center: Center value for intensity windowing (HU units)
        window_width: Width of intensity window (HU units)

    Returns:
        List of processed image arrays (H, W) with values in [0, 1]

    Raises:
        FileNotFoundError: If dicom_dir does not exist
        ValueError: If no DICOM files found in directory

    Example:
        >>> images = process_dicom_series(
        ...     'data/raw/patient_001',
        ...     'data/processed/patient_001',
        ...     window_center=40,
        ...     window_width=80
        ... )
        >>> print(f"Processed {len(images)} images")

    Note:
        Window center and width values are specific to the imaging
        modality and anatomical region. For stroke CT imaging:
        - Brain window: center=40, width=80
        - Bone window: center=700, width=3000
    """
    # Implementation...
```

## Pull Request Process

### Before Submitting

1. ✅ Tests pass: `python scripts/run_tests.py`
2. ✅ Code formatted: `black src/ tests/ scripts/`
3. ✅ Imports sorted: `isort src/ tests/ scripts/`
4. ✅ No linting errors: `flake8 src/ tests/ scripts/`
5. ✅ Documentation updated
6. ✅ CHANGELOG updated (for significant changes)

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Related Issue
Closes #123 (if applicable)

## Changes Made
- Added X feature
- Fixed Y bug
- Improved Z performance

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] All tests pass

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed my own code
- [ ] Commented complex code sections
- [ ] Updated documentation
- [ ] No new warnings generated
- [ ] Added tests that prove fix/feature works
- [ ] New and existing tests pass locally

## Screenshots (if applicable)
Add screenshots for UI changes or visualizations.

## Additional Notes
Any additional context or notes for reviewers.
```

### Review Process

1. **Automated checks** run on PR submission
2. **Code review** by at least one maintainer
3. **Address feedback** by pushing new commits
4. **Approval** required before merge
5. **Squash and merge** to keep history clean

## Project Structure

### Adding New Features

#### New Model Architecture

1. Add model class to `src/models/`
2. Register in architecture factory (if using one)
3. Add tests to `tests/test_models.py`
4. Document architecture and usage
5. Add example to README

#### New Preprocessing Function

1. Add function to `src/data/preprocessing.py`
2. Add tests to `tests/test_preprocessing.py`
3. Update preprocessing pipeline if needed
4. Document function and parameters

#### New Evaluation Metric

1. Add metric to `src/evaluation/metrics.py`
2. Add tests with known ground truth
3. Update evaluation script
4. Document clinical interpretation

#### New Script

1. Create script in `scripts/`
2. Add comprehensive argparse interface
3. Add usage example to README
4. Test with dummy data

## Common Tasks

### Running Experiments

```bash
# Train model
python train.py --config config/default_config.yaml

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pth

# Generate interpretability visualizations
python scripts/generate_interpretability.py --checkpoint checkpoints/best_model.pth

# Analyze uncertainty
python scripts/analyze_uncertainty.py --checkpoint checkpoints/best_model.pth
```

### Debugging

```bash
# Enable debug logging
export LOGLEVEL=DEBUG
python train.py --config config/default_config.yaml

# Test with small dataset
python scripts/generate_dummy_data.py --num-patients 10
python train.py --config config/default_config.yaml --max-epochs 2

# Profile code
python -m cProfile -o profile.stats train.py
python -m pstats profile.stats
```

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Feature requests**: Open a GitHub Issue with "Feature Request" label
- **Security concerns**: Email maintainers directly (do not open public issue)

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in papers/presentations using this code
- Credited in commit history

Thank you for contributing to advancing stroke care through AI! 🧠❤️
