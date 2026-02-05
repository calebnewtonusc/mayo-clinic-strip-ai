# Quick Start Guide

Get started with the Mayo Clinic STRIP AI project in 5 minutes!

## Option 1: Test with Dummy Data (Recommended First)

Perfect for testing the pipeline before you have real medical images.

```bash
# 1. Clone and setup
git clone https://github.com/calebnewtonusc/mayo-clinic-strip-ai.git
cd mayo-clinic-strip-ai

# 2. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run end-to-end test with dummy data
python scripts/run_end_to_end_test.py
```

**That's it!** This will:
- Generate synthetic medical-looking images
- Validate the data
- Preprocess images
- Create train/val/test splits
- Test the DataLoader

If everything passes ✅, your environment is ready!

## Option 2: Use Real Medical Images

### Prerequisites
- IRB approval for Mayo Clinic STRIP dataset
- Data in DICOM, NIfTI, or PNG format

### Steps

```bash
# 1. Setup environment (same as above)
source venv/bin/activate

# 2. Organize your data
# Place images in this structure:
#   data/raw/CE/patient_001/*.dcm
#   data/raw/CE/patient_002/*.dcm
#   data/raw/LAA/patient_003/*.dcm
#   ...

# 3. Validate data
python scripts/validate_data.py --data_dir data/raw

# 4. Explore data (creates visualizations)
python scripts/explore_data.py --data_dir data/raw --output_dir experiments/exploration

# 5. Preprocess images
python scripts/preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --target_size 224 224 \
    --normalize zscore

# 6. Create patient-level splits (prevents data leakage!)
python scripts/create_splits.py \
    --data_dir data/processed \
    --output_dir data/splits \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15

# 7. Test DataLoader
python scripts/test_dataloader.py \
    --data_dir data/processed \
    --split_file data/splits/train.json

# 8. Start training!
python train.py --config config/default_config.yaml
```

## Training Your First Model

### Quick Training Run

```bash
# Train a baseline ResNet-50 model
python train.py \
    --config config/default_config.yaml \
    --data_dir data/processed \
    --experiment_name my_first_experiment
```

### Monitor Training

```bash
# In another terminal, start TensorBoard
tensorboard --logdir experiments/logs
# Open http://localhost:6006 in your browser
```

### Evaluate Trained Model

```bash
# Evaluate on test set
python evaluate.py \
    --checkpoint experiments/checkpoints/my_first_experiment/best_model.pth \
    --data_dir data/processed \
    --split test
```

## Explore with Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open these notebooks:
# 1. notebooks/01_exploratory_data_analysis.ipynb - Explore your data
# 2. notebooks/02_augmentation_visualization.ipynb - Visualize augmentations
```

## Common Configurations

### Train Simple CNN (Fast Baseline)

```yaml
# config/simple_baseline.yaml
model:
  architecture: simple_cnn
  pretrained: false

training:
  num_epochs: 50
  learning_rate: 0.001
  batch_size: 32
```

```bash
python train.py --config config/simple_baseline.yaml
```

### Train with Transfer Learning

```yaml
# config/transfer_learning.yaml
model:
  architecture: resnet50
  pretrained: true
  freeze_backbone: false  # Fine-tune all layers

training:
  num_epochs: 100
  learning_rate: 0.0001  # Lower LR for fine-tuning
```

### Train with Strong Augmentation (Limited Data)

Edit `src/data/augmentation.py` to use `get_strong_augmentation()` in your training script.

## Project Structure Overview

```
mayo-clinic-strip-ai/
├── data/                    # Data (not in git)
│   ├── raw/                # Original images
│   ├── processed/          # Preprocessed images
│   └── splits/             # Train/val/test splits
├── src/                     # Source code
│   ├── data/               # Dataset, preprocessing, augmentation
│   ├── models/             # Model architectures
│   ├── training/           # Training loops
│   ├── evaluation/         # Metrics and evaluation
│   └── utils/              # Helper functions
├── scripts/                 # Utility scripts
├── notebooks/               # Jupyter notebooks
├── experiments/             # Training outputs
│   ├── logs/               # TensorBoard logs
│   ├── checkpoints/        # Saved models
│   └── results/            # Evaluation results
├── docs/                    # Documentation
├── config/                  # Configuration files
├── train.py                # Main training script
└── evaluate.py             # Evaluation script
```

## Key Scripts Reference

| Script | Purpose |
|--------|---------|
| `generate_dummy_data.py` | Create synthetic test data |
| `run_end_to_end_test.py` | Test entire pipeline |
| `validate_data.py` | Check data structure and quality |
| `explore_data.py` | Analyze and visualize dataset |
| `preprocess_data.py` | Preprocess medical images |
| `create_splits.py` | Create patient-level splits |
| `test_dataloader.py` | Test data loading |
| `train.py` | Train models |
| `evaluate.py` | Evaluate trained models |

## Configuration Options

Edit `config/default_config.yaml` to customize:

- **Data**: Image size, batch size, train/val/test ratios
- **Model**: Architecture (ResNet, EfficientNet, etc.)
- **Training**: Learning rate, optimizer, epochs
- **Augmentation**: Types and probabilities
- **Evaluation**: Metrics, patient-level aggregation

## Troubleshooting

### "No module named..."
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Reduce batch size in config:
```yaml
data:
  batch_size: 16  # or 8
```

### "Dataset is empty!"
- Check data is in correct directory structure
- Run `validate_data.py` to diagnose issues

### Slow data loading
- Increase `num_workers` in config
- Use SSD storage for data
- Preprocess to PNG format

## Next Steps

1. ✅ **Review Implementation Plan**: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
2. ✅ **Read Best Practices**: [docs/MEDICAL_IMAGING_BEST_PRACTICES.md](docs/MEDICAL_IMAGING_BEST_PRACTICES.md)
3. ✅ **Understand Medical Context**: [docs/MEDICAL_DOMAIN.md](docs/MEDICAL_DOMAIN.md)
4. ✅ **Follow Phases 1-5 Guide**: [PHASES_1_5_COMPLETE.md](PHASES_1_5_COMPLETE.md)

## Getting Help

- 📖 Check the [README.md](README.md) for project overview
- 📋 Review the 17-phase [implementation plan](docs/IMPLEMENTATION_PLAN.md)
- 🐛 Open an issue on GitHub for bugs
- 💡 See [docs/](docs/) for detailed documentation

## Team Collaboration

### Adding Team Members

1. Go to GitHub repo settings → Collaborators
2. Add team members by GitHub username
3. They can clone and start working:

```bash
git clone https://github.com/calebnewtonusc/mayo-clinic-strip-ai.git
cd mayo-clinic-strip-ai
source venv/bin/activate
pip install -r requirements.txt
```

### Workflow Suggestions

- Create feature branches for new work
- Use pull requests for code review
- Track tasks using GitHub Issues
- Document experiments in notebooks
- Keep model checkpoints organized

---

**Ready to start?** Run the end-to-end test:
```bash
python scripts/run_end_to_end_test.py
```

🚀 **Let's build something amazing!**
