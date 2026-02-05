# Mayo Clinic STRIP AI: User Manual

**Version 1.0.0** | **February 2026**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Training Models](#3-training-models)
4. [Evaluating Models](#4-evaluating-models)
5. [Using the API](#5-using-the-api)
6. [Interpreting Results](#6-interpreting-results)
7. [Troubleshooting](#7-troubleshooting)
8. [Best Practices](#8-best-practices)
9. [FAQs](#9-faqs)

---

## 1. Introduction

### 1.1 What is Mayo Clinic STRIP AI?

Mayo Clinic STRIP AI is an automated system for classifying stroke blood clot origin from medical images. It uses deep learning to distinguish between:

- **Cardioembolic (CE)**: Clots from the heart
- **Large Artery Atherosclerosis (LAA)**: Clots from artery plaques

### 1.2 Who Should Use This Manual?

This manual is for:
- **Clinical Researchers** conducting stroke classification studies
- **Data Scientists** developing medical imaging models
- **Radiologists** using AI as decision support
- **Medical Students** learning about AI in healthcare
- **IT Staff** deploying medical AI systems

### 1.3 What You Need

**Minimum Requirements:**
- Computer with Python 3.8+
- 8 GB RAM (16 GB recommended)
- 10 GB free disk space
- Internet connection for installation

**Recommended:**
- NVIDIA GPU with 6+ GB VRAM
- Ubuntu 20.04 / macOS 11+ / Windows 10+
- 32 GB RAM for large datasets

**Data Requirements:**
- De-identified medical images
- DICOM, NIfTI, PNG, or NPY format
- Labels (CE or LAA) for training

---

## 2. Getting Started

### 2.1 Installation

#### Option A: Quick Install (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/calebnewtonusc/mayo-clinic-strip-ai.git
cd mayo-clinic-strip-ai

# 2. Create environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test installation
python scripts/run_end_to_end_test.py
```

#### Option B: Install as Package

```bash
pip install git+https://github.com/calebnewtonusc/mayo-clinic-strip-ai.git
```

### 2.2 Quick Test

Test the system with dummy data (no real medical data needed):

```bash
# Generate test data
python scripts/generate_dummy_data.py

# Run complete pipeline
python scripts/run_end_to_end_test.py
```

**Expected Output:**
```
✓ Generated 50 dummy patients
✓ Preprocessed images
✓ Created data splits
✓ Trained model (2 epochs)
✓ Evaluated model
✓ Generated visualizations
All tests passed!
```

### 2.3 Understanding the File Structure

```
mayo-clinic-strip-ai/
├── data/              # Your medical images (not in git)
│   ├── raw/          # Original images
│   ├── processed/    # Preprocessed images
│   └── splits/       # Train/val/test splits
├── src/              # Source code
├── scripts/          # Utility scripts
├── config/           # Configuration files
├── checkpoints/      # Trained models
└── results/          # Evaluation results
```

---

## 3. Training Models

### 3.1 Preparing Your Data

#### Step 1: Organize Your Images

```
data/raw/
├── CE/
│   ├── patient_001/
│   │   ├── image_001.dcm
│   │   ├── image_002.dcm
│   │   └── ...
│   └── patient_002/
│       └── ...
└── LAA/
    ├── patient_003/
    └── ...
```

**Important**: Each patient should be in their own folder!

#### Step 2: Validate Data

```bash
python scripts/validate_data.py --data_dir data/raw
```

This checks:
- ✓ Correct directory structure
- ✓ Valid image formats
- ✓ No corrupted files
- ✓ Balanced classes

#### Step 3: Preprocess Images

```bash
python scripts/preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --image_size 224 \
    --normalize zscore
```

**Options:**
- `--image_size`: Target size (default: 224)
- `--normalize`: Normalization method (zscore, minmax, percentile)
- `--format`: Output format (png, npy)

#### Step 4: Create Data Splits

⚠️ **Critical**: Use patient-level splits to prevent data leakage!

```bash
python scripts/create_splits.py \
    --data_dir data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --stratify
```

This ensures:
- No patient appears in multiple splits
- Class balance maintained
- Reproducible splits (fixed seed)

### 3.2 Training Your First Model

#### Basic Training

```bash
python train.py --config config/default_config.yaml
```

**What Happens:**
1. Loads configuration
2. Creates data loaders
3. Initializes model (ResNet-18 by default)
4. Trains for specified epochs
5. Saves checkpoints
6. Logs to TensorBoard

#### Monitor Training

```bash
# In another terminal
tensorboard --logdir logs/
```

Open http://localhost:6006 to see:
- Training/validation loss
- Accuracy curves
- Learning rate schedule
- Example predictions

#### Customize Training

Edit `config/default_config.yaml`:

```yaml
model:
  architecture: "resnet50"  # resnet18, resnet50, efficientnet_b0
  pretrained: true
  num_classes: 2

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  optimizer: "adam"

data:
  image_size: 224
  augmentation: "strong"  # none, normal, strong
```

### 3.3 Advanced Training

#### Using MixUp/CutMix

In your config:

```yaml
training:
  use_mixup: true
  mixup_alpha: 1.0
  use_cutmix: false
```

#### Hyperparameter Search

```bash
python scripts/run_hyperparameter_search.py \
    --config config/default_config.yaml \
    --search-type grid \
    --output-dir results/hp_search
```

Searches over:
- Learning rates
- Batch sizes
- Augmentation strategies
- Model architectures

#### Resume Training

```bash
python train.py \
    --config config/default_config.yaml \
    --resume checkpoints/latest.pth
```

---

## 4. Evaluating Models

### 4.1 Basic Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data-dir data/processed \
    --output-dir results/evaluation
```

**Outputs:**
- `metrics.json`: All metrics
- `confusion_matrix.png`: Visual confusion matrix
- `roc_curve.png`: ROC curve
- `predictions.csv`: Per-image predictions

### 4.2 Understanding Metrics

#### Classification Metrics

**Accuracy**: Overall correctness
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Sensitivity (Recall)**: True positive rate
```
Sensitivity = TP / (TP + FN)
```
*How many actual CE cases did we correctly identify?*

**Specificity**: True negative rate
```
Specificity = TN / (TN + FP)
```
*How many actual LAA cases did we correctly identify?*

**PPV (Precision)**: Positive predictive value
```
PPV = TP / (TP + FP)
```
*When we predict CE, how often are we right?*

**ROC-AUC**: Overall discrimination ability (0.5-1.0)
- 1.0 = Perfect classifier
- 0.9-1.0 = Excellent
- 0.8-0.9 = Good
- 0.7-0.8 = Fair
- 0.5-0.7 = Poor

#### Patient-Level Metrics

Since patients have multiple images, we aggregate predictions:

**Majority Voting**:
```
Final prediction = most common prediction across images
```

**Mean Probability**:
```
Final probability = average probability across images
```

**Max Confidence**:
```
Final prediction = prediction with highest confidence
```

Patient-level metrics are typically 2-3% higher than image-level.

### 4.3 Interpretability

#### Generate Grad-CAM Visualizations

```bash
python scripts/generate_interpretability.py \
    --checkpoint checkpoints/best_model.pth \
    --output-dir results/interpretability \
    --num-samples 50
```

**Outputs:**
- Heatmaps showing model attention
- Overlaid visualizations
- Feature space plots (t-SNE, UMAP)

**Interpreting Grad-CAM**:
- 🔴 Red/yellow = High importance regions
- 🔵 Blue = Low importance regions
- Check if model focuses on clinically relevant areas

#### Analyze Uncertainty

```bash
python scripts/analyze_uncertainty.py \
    --checkpoint checkpoints/best_model.pth \
    --output-dir results/uncertainty \
    --n-iterations 30
```

**Outputs:**
- Uncertainty estimates per prediction
- Calibration curves
- Uncertain samples flagged for review

**Using Uncertainty**:
```python
if uncertainty > 0.1:
    # Flag for expert review
    recommend_human_review()
```

### 4.4 Robustness Testing

```bash
python scripts/analyze_robustness.py \
    --checkpoint checkpoints/best_model.pth \
    --output-dir results/robustness
```

Tests model against:
- Gaussian noise
- Image blur
- Brightness changes
- Contrast changes

**Good Robustness**: <5% accuracy drop on any corruption

### 4.5 Bias Analysis

```bash
python scripts/analyze_bias.py \
    --checkpoint checkpoints/best_model.pth \
    --metadata-file data/metadata.json \
    --subgroup-key age_group
```

Analyzes performance across:
- Age groups
- Sex
- Scanner types
- Image quality

**Fair Model**: <5% performance difference across subgroups

---

## 5. Using the API

### 5.1 Starting the API

#### Option A: Local

```bash
python deploy/api.py --checkpoint checkpoints/best_model.pth
```

API runs at http://localhost:5000

#### Option B: Docker

```bash
cd deploy
docker-compose up --build
```

### 5.2 Making Predictions

#### Using Python Client

```python
from deploy.api_client import predict_image

# Single prediction
result = predict_image(
    api_url="http://localhost:5000",
    image_path="patient_scan.png",
    uncertainty=True
)

print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
if 'uncertainty' in result:
    print(f"Uncertainty: {result['uncertainty']}")
```

#### Using cURL

```bash
# Single prediction
curl -X POST \
  -F "file=@image.png" \
  http://localhost:5000/predict

# With uncertainty
curl -X POST \
  -F "file=@image.png" \
  -F "uncertainty=true" \
  http://localhost:5000/predict

# Batch prediction
curl -X POST \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  http://localhost:5000/batch_predict
```

### 5.3 API Response Format

```json
{
  "predicted_class": 0,
  "class_name": "Cardioembolic (CE)",
  "confidence": 0.92,
  "probabilities": {
    "Cardioembolic (CE)": 0.92,
    "Large Artery Atherosclerosis (LAA)": 0.08
  },
  "uncertainty": {
    "Cardioembolic (CE)": 0.03,
    "Large Artery Atherosclerosis (LAA)": 0.05
  }
}
```

### 5.4 Production Deployment

See [deploy/README.md](../deploy/README.md) for:
- Cloud deployment (AWS, GCP, Azure)
- Load balancing
- Monitoring
- Security
- HIPAA compliance

---

## 6. Interpreting Results

### 6.1 Clinical Decision Support

**AI Provides:**
1. **Primary Prediction**: CE or LAA
2. **Confidence Score**: How certain (0-100%)
3. **Attention Map**: Where model looked
4. **Uncertainty**: Model's doubt level

**Clinical Workflow:**

```
1. AI analyzes image → Predicts LAA (confidence: 95%)
2. Show Grad-CAM → Confirms focus on artery plaque
3. Check uncertainty → Low (0.02) = confident prediction
4. Radiologist reviews → Agrees with AI
5. Final diagnosis → LAA (AI-assisted)
```

### 6.2 When to Trust AI

✅ **High Confidence Cases** (>90% confidence, low uncertainty):
- AI likely correct
- Can reduce radiologist review time
- Still requires human verification

⚠️ **Medium Confidence** (70-90% confidence):
- Careful review recommended
- Consider additional clinical data
- AI as second opinion

🛑 **Low Confidence** (<70% confidence or high uncertainty):
- Mandatory expert review
- May indicate rare case
- Consider additional imaging

### 6.3 Common Prediction Patterns

**Pattern 1: Clear CE**
```
Confidence: 95%
Uncertainty: 0.02
Grad-CAM: Focused on heart-related structures
Action: High confidence prediction
```

**Pattern 2: Clear LAA**
```
Confidence: 93%
Uncertainty: 0.03
Grad-CAM: Focused on arterial plaques
Action: High confidence prediction
```

**Pattern 3: Uncertain**
```
Confidence: 65%
Uncertainty: 0.12
Grad-CAM: Diffuse attention
Action: Flag for expert review
```

**Pattern 4: Borderline**
```
Confidence: 52%
Uncertainty: 0.18
Grad-CAM: Multiple focus areas
Action: Requires human judgment
```

---

## 7. Troubleshooting

### 7.1 Installation Issues

**Problem**: `pip install` fails
```bash
# Solution: Upgrade pip
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Problem**: PyTorch installation fails
```bash
# Solution: Install PyTorch separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: Out of memory during installation
```bash
# Solution: Install without cache
pip install --no-cache-dir -r requirements.txt
```

### 7.2 Training Issues

**Problem**: CUDA out of memory
```yaml
# Solution: Reduce batch size in config
training:
  batch_size: 16  # or 8
```

**Problem**: Training very slow
```python
# Check if GPU is being used
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be >= 1
```

**Problem**: Model not improving
```yaml
# Solutions:
# 1. Check learning rate
training:
  learning_rate: 0.0001  # Try 0.001 or 0.00001

# 2. Increase epochs
training:
  epochs: 100

# 3. Use data augmentation
data:
  augmentation: "strong"

# 4. Try different architecture
model:
  architecture: "efficientnet_b0"
```

**Problem**: Overfitting (val loss increases)
```yaml
# Solutions:
# 1. Increase dropout
model:
  dropout: 0.5  # or 0.6

# 2. Add weight decay
training:
  weight_decay: 0.0001

# 3. Use early stopping (already enabled)

# 4. Get more data or use stronger augmentation
```

### 7.3 Evaluation Issues

**Problem**: Checkpoint not found
```bash
# Check checkpoints directory
ls checkpoints/

# Use full path
python evaluate.py --checkpoint /full/path/to/best_model.pth
```

**Problem**: Different results each time
```python
# Set seed for reproducibility
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### 7.4 API Issues

**Problem**: API won't start
```bash
# Check if port is already in use
lsof -i :5000

# Use different port
python deploy/api.py --checkpoint model.pth --port 5001
```

**Problem**: Predictions too slow
```bash
# Use model optimization
python scripts/optimize_model.py \
    --checkpoint model.pth \
    --method quantize

# Then use optimized model
python deploy/api.py --checkpoint model_quantized.pth
```

### 7.5 Data Issues

**Problem**: Class imbalance
```yaml
# Solution 1: Use weighted loss
training:
  use_weighted_loss: true

# Solution 2: Oversample minority class
data:
  oversampling: true
  oversampling_ratio: 1.0
```

**Problem**: Poor performance on test set
```
# Possible causes:
1. Data leakage - Check patient-level splits
2. Domain shift - Test data from different source
3. Overfitting - Need more data or regularization
4. Poor labels - Verify ground truth
```

---

## 8. Best Practices

### 8.1 Data Management

✅ **DO:**
- Use patient-level data splits
- De-identify all images before processing
- Validate data quality before training
- Keep raw data separate from processed
- Document preprocessing steps
- Version your datasets

❌ **DON'T:**
- Mix patients between train/val/test
- Use image-level splits (causes leakage)
- Commit medical data to git
- Process without validation
- Ignore class imbalance

### 8.2 Model Development

✅ **DO:**
- Start with baseline (SimpleCNN)
- Use transfer learning
- Monitor both training and validation metrics
- Save multiple checkpoints
- Document hyperparameters
- Test on held-out data
- Use cross-validation

❌ **DON'T:**
- Jump to complex models first
- Only check training accuracy
- Overfit to validation set
- Skip baseline comparison
- Change architecture without reason

### 8.3 Clinical Use

✅ **DO:**
- Always use human oversight
- Set appropriate confidence thresholds
- Use uncertainty quantification
- Review Grad-CAM visualizations
- Monitor performance continuously
- Validate on local data
- Document AI-assisted decisions
- Follow institutional guidelines

❌ **DON'T:**
- Use AI as sole decision maker
- Ignore uncertainty estimates
- Skip expert review on edge cases
- Deploy without validation
- Use on unseen data types
- Bypass ethical approval

### 8.4 Security & Privacy

✅ **DO:**
- Encrypt data at rest and in transit
- Use authentication on API
- Log all access (HIPAA requirement)
- Regular security audits
- Follow institutional policies
- De-identify all data
- Use secure connections (HTTPS)

❌ **DON'T:**
- Expose API without authentication
- Log patient identifiable information
- Share models trained on identified data
- Skip security updates
- Use default passwords

---

## 9. FAQs

### General Questions

**Q: Can I use this for clinical diagnosis?**
A: Not without proper validation and regulatory approval (FDA/CE marking). This system is for research purposes only.

**Q: What image formats are supported?**
A: DICOM, NIfTI, PNG, NPY. Most medical imaging formats are supported.

**Q: How much data do I need?**
A: Minimum 100 patients per class, 300+ recommended. More data = better performance.

**Q: How long does training take?**
A: 2-6 hours on GPU for typical datasets (1000 images, 50 epochs). CPU training takes much longer.

### Technical Questions

**Q: Which model should I use?**
A: Start with EfficientNet-B0 for best balance. Use ResNet-50 for maximum accuracy. See [Model Cards](MODEL_CARDS.md) for details.

**Q: What's patient-level splitting?**
A: Ensures all images from a patient go into the same split (train/val/test). Prevents data leakage and optimistic performance.

**Q: How do I handle class imbalance?**
A: Use weighted loss, oversampling, or under sampling. See config options.

**Q: Can I use 3D images?**
A: Currently 2D only. 3D support is planned for future versions.

**Q: How do I deploy in production?**
A: Use Docker deployment. See [deploy/README.md](../deploy/README.md) for detailed instructions.

### Performance Questions

**Q: Why is accuracy low?**
A: Check data quality, ensure proper splits, try stronger augmentation, increase model capacity, or get more data.

**Q: How do I improve performance?**
A: 1) More data, 2) Better labels, 3) Stronger augmentation, 4) Larger model, 5) Hyperparameter tuning.

**Q: What's a good ROC-AUC?**
A: >0.9 is excellent, 0.8-0.9 is good, 0.7-0.8 is fair. Depends on task difficulty.

**Q: Should I use transfer learning?**
A: Yes! Almost always beneficial, especially with limited data.

### Clinical Questions

**Q: Can AI replace radiologists?**
A: No. AI is a decision support tool, not a replacement. Human expertise is essential.

**Q: How do I trust AI predictions?**
A: Use uncertainty quantification, check Grad-CAM visualizations, validate on local data, and always have human oversight.

**Q: What if AI disagrees with radiologist?**
A: Radiologist decision takes precedence. AI provides second opinion only. Investigate discrepancies to improve model.

**Q: How often should I retrain?**
A: Every 6-12 months, or when performance degrades, or when new data patterns emerge.

---

## 10. Getting Help

### Documentation
- **Quick Start**: [QUICKSTART.md](../QUICKSTART.md)
- **Technical Report**: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- **Model Cards**: [MODEL_CARDS.md](MODEL_CARDS.md)
- **API Documentation**: [deploy/README.md](../deploy/README.md)

### Support
- **GitHub Issues**: https://github.com/calebnewtonusc/mayo-clinic-strip-ai/issues
- **Discussions**: https://github.com/calebnewtonusc/mayo-clinic-strip-ai/discussions

### Contributing
See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to contribute.

---

## 11. Glossary

**Augmentation**: Artificially increasing dataset size by transforming images

**Cardioembolic (CE)**: Stroke caused by clots from the heart

**Calibration**: Alignment between predicted confidence and actual accuracy

**Grad-CAM**: Visualization technique showing model attention

**Large Artery Atherosclerosis (LAA)**: Stroke caused by artery plaques

**Patient-Level Split**: Data splitting that keeps all of a patient's images together

**ROC-AUC**: Area Under the Receiver Operating Characteristic curve

**Sensitivity**: True positive rate (recall)

**Specificity**: True negative rate

**Transfer Learning**: Using pre-trained model as starting point

**Uncertainty Quantification**: Measuring model confidence/doubt

---

**Manual Version**: 1.0.0
**Last Updated**: February 4, 2026
**Feedback**: Please report issues or suggestions on GitHub

---

*Happy classifying! Remember: AI assists, humans decide.* 🧠❤️
