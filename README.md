# Mayo Clinic - STRIP AI: Stroke Blood Clot Origin Classification

## 🚀 Quick Start

**New to this project?** Get started in 5 minutes:

```bash
# Test with dummy data (recommended first!)
python scripts/run_end_to_end_test.py
```

📖 **Full guide**: See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

---

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

### 📚 Essential Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[PHASES_1_5_COMPLETE.md](PHASES_1_5_COMPLETE.md)** - Complete guide for Phases 1-5 (data pipeline)
- **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** - Full 17-phase roadmap (250+ tasks)
- **[docs/MEDICAL_IMAGING_BEST_PRACTICES.md](docs/MEDICAL_IMAGING_BEST_PRACTICES.md)** - ML best practices
- **[docs/MEDICAL_DOMAIN.md](docs/MEDICAL_DOMAIN.md)** - Medical background knowledge

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Access to Mayo Clinic STRIP dataset (requires IRB approval, or use dummy data for testing)

### Quick Installation

```bash
# Clone and setup
git clone https://github.com/calebnewtonusc/mayo-clinic-strip-ai.git
cd mayo-clinic-strip-ai

# Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test with dummy data
python scripts/run_end_to_end_test.py
```

### Data Setup

Place your data in `data/raw/` following this structure:
```
data/raw/
├── CE/patient_001/*.dcm
├── CE/patient_002/*.dcm
└── LAA/patient_003/*.dcm
```

Or generate test data: `python scripts/generate_dummy_data.py`

See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for detailed instructions.

## Development Status

✅ **Phases 1-5 Complete** - Full data pipeline implemented!
- Environment setup
- Data exploration and validation
- Preprocessing pipeline
- Augmentation system
- Dataset & DataLoader

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the full 17-phase roadmap.

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
