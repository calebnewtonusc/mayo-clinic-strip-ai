# Mayo Clinic - STRIP AI: Stroke Blood Clot Origin Classification

## Project Overview

This project focuses on classifying stroke blood clot origin using deep learning techniques on medical imaging data. The goal is to distinguish between:
- **Cardioembolic (CE)**: Blood clots originating from the heart
- **Large Artery Atherosclerosis (LAA)**: Blood clots from atherosclerotic plaques in large arteries

## Why This Project Matters

- **Clinical Relevance**: Accurate classification of stroke etiology is critical for treatment planning and secondary prevention strategies
- **Real-World Impact**: Helps clinicians make faster, more informed decisions about patient care
- **Technical Challenge**: Addresses unique challenges in medical imaging ML including limited labeled data, data privacy, and model interpretability requirements

## Project Goals

1. Develop a robust deep learning classifier for CE vs LAA stroke classification
2. Achieve high generalization performance despite limited labeled training data
3. Implement interpretability methods to understand model predictions
4. Create a reproducible, well-documented ML pipeline for medical imaging

## Key Learning Objectives

- **Data Engineering**: Collection strategies, augmentation techniques, preprocessing pipelines
- **Advanced Architectures**: Transfer learning with CNNs and Vision Transformers
- **Medical Imaging**: Patch-based modeling, handling variable image sizes
- **Evaluation**: Patient-level prediction aggregation, clinical metrics
- **Reliability**: Model interpretability, uncertainty quantification, bias detection

## Project Structure

```
mayo-clinic-strip-ai/
├── data/                      # Data storage (not committed to git)
│   ├── raw/                   # Original images and labels
│   ├── processed/             # Preprocessed images
│   ├── augmented/             # Augmented dataset
│   └── splits/                # Train/val/test splits
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model architectures
│   ├── training/              # Training loops and optimization
│   ├── evaluation/            # Evaluation metrics and analysis
│   ├── visualization/         # Visualization utilities
│   └── utils/                 # Helper functions
├── notebooks/                 # Jupyter notebooks for exploration
├── experiments/               # Experiment outputs
│   ├── logs/                  # Training logs
│   ├── checkpoints/           # Model checkpoints
│   └── results/               # Evaluation results
├── docs/                      # Documentation
├── tests/                     # Unit and integration tests
├── config/                    # Configuration files
└── requirements.txt           # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Access to Mayo Clinic STRIP dataset (requires IRB approval)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mayo-clinic-strip-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place your data in the `data/raw/` directory following the structure:
```
data/raw/
├── CE/
│   ├── patient_001/
│   └── patient_002/
└── LAA/
    ├── patient_003/
    └── patient_004/
```

See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for detailed instructions.

## Development Roadmap

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the detailed, phase-by-phase implementation plan.

## Team

This is a collaborative project. See commit history for individual contributions.

## Ethics & Compliance

- This project uses de-identified medical imaging data
- All data handling follows HIPAA compliance requirements
- Model predictions are for research purposes only and not for clinical use without validation
- See [docs/ETHICS.md](docs/ETHICS.md) for detailed ethical considerations

## License

TBD - Ensure compliance with Mayo Clinic data usage agreements

## Acknowledgments

- Mayo Clinic for providing the STRIP dataset
- Research supervisors and clinical collaborators
