# Phases 9-10: Interpretability & Uncertainty - Complete Guide

This document covers **Phase 9 (Model Interpretability)** and **Phase 10 (Uncertainty Quantification)** implementations.

---

## Phase 9: Model Interpretability ✅

### What's Been Implemented

#### 1. Grad-CAM Visualization
**File**: [src/visualization/gradcam.py](../src/visualization/gradcam.py)

- **GradCAM**: Standard Gradient-weighted Class Activation Mapping
- **GradCAMPlusPlus**: Improved version with better localization
- **Guided Backpropagation**: Saliency map generation
- **Overlay Functions**: Heatmap visualization on original images
- **Target Layer Detection**: Automatic layer selection for different architectures

#### 2. Feature Visualization
**File**: [src/visualization/features.py](../src/visualization/features.py)

- **Feature Extraction**: Extract features from any layer
- **t-SNE Embedding**: 2D visualization of feature space
- **PCA Embedding**: Linear dimensionality reduction
- **UMAP Embedding**: Non-linear manifold learning (optional)
- **Feature Separability Analysis**: Quantify class separation
- **Feature Distribution Plots**: Visualize feature distributions by class

#### 3. Scripts & Tools

**[scripts/generate_interpretability.py](../scripts/generate_interpretability.py)**
- Generate Grad-CAM/Grad-CAM++ for multiple samples
- Create feature space visualizations
- Analyze feature separability
- Save all visualizations automatically

**[notebooks/03_model_interpretability.ipynb](../notebooks/03_model_interpretability.ipynb)**
- Interactive interpretability analysis
- Step-by-step Grad-CAM generation
- Feature space exploration
- Multiple sample visualization

### How to Use

#### Generate Grad-CAM Visualizations

```bash
python scripts/generate_interpretability.py \
    --checkpoint experiments/checkpoints/best_model.pth \
    --data_dir data/processed \
    --split val \
    --output_dir experiments/interpretability \
    --num_samples 20 \
    --gradcam_plus  # Use Grad-CAM++ instead of regular Grad-CAM
```

**Output**:
- `experiments/interpretability/gradcam/` - Grad-CAM visualizations
- `experiments/interpretability/features/` - Feature space plots

#### Interactive Analysis

```bash
jupyter notebook notebooks/03_model_interpretability.ipynb
```

### Key Features

**Grad-CAM**:
- Visualizes which image regions influence predictions
- Helps validate clinical relevance
- Identifies potential spurious correlations
- Supports multiple architectures (ResNet, EfficientNet, etc.)

**Feature Analysis**:
- t-SNE: Non-linear visualization
- PCA: Linear visualization (faster)
- UMAP: Best visualization (requires `umap-learn`)
- Separability metrics: Silhouette score, inter/intra-class distances

### Clinical Validation

**Questions to Answer**:
1. Does the model focus on clinically relevant regions?
2. Are attention patterns consistent across similar cases?
3. Do misclassifications show confounding features?
4. Is there a pattern in uncertain predictions?

**Best Practices**:
- Review Grad-CAM with clinical experts
- Check for dataset artifacts (text, markers)
- Validate on diverse samples
- Document unexpected attention patterns

---

## Phase 10: Uncertainty Quantification ✅

### What's Been Implemented

#### 1. Monte Carlo Dropout
**File**: [src/evaluation/uncertainty.py](../src/evaluation/uncertainty.py)

- Runs multiple forward passes with dropout enabled
- Estimates epistemic uncertainty (model uncertainty)
- Calculates prediction mean and standard deviation
- Computes mutual information

#### 2. Test-Time Augmentation (TTA)
- Applies multiple augmentations during inference
- Estimates aleatoric uncertainty (data uncertainty)
- Provides robustness to input variations

#### 3. Calibration Metrics
- **Calibration Curve**: Reliability diagram
- **Expected Calibration Error (ECE)**: Quantifies miscalibration
- **Temperature Scaling**: Post-hoc calibration method

#### 4. Uncertainty Analysis
- **Predictive Entropy**: Measures prediction uncertainty
- **Confidence Metrics**: Mean, median, min, max confidence
- **Uncertain Sample Identification**: Flags low-confidence predictions
- **Confidence vs Correctness**: Analyzes relationship

#### 5. Scripts & Tools

**[scripts/analyze_uncertainty.py](../scripts/analyze_uncertainty.py)**
- Perform MC dropout analysis
- Run test-time augmentation
- Generate uncertainty visualizations
- Calculate calibration metrics

**[notebooks/04_uncertainty_quantification.ipynb](../notebooks/04_uncertainty_quantification.ipynb)**
- Interactive uncertainty exploration
- MC dropout demonstration
- Calibration analysis
- Uncertain sample identification

### How to Use

#### Analyze Uncertainty

```bash
python scripts/analyze_uncertainty.py \
    --checkpoint experiments/checkpoints/best_model.pth \
    --data_dir data/processed \
    --split val \
    --output_dir experiments/uncertainty \
    --mc_iterations 30 \
    --tta_augmentations 10
```

**Optional Flags**:
- `--skip_tta`: Skip TTA (faster but less complete)
- `--mc_iterations N`: Number of MC dropout forward passes
- `--tta_augmentations N`: Number of TTA augmentations

**Output**:
- `experiments/uncertainty/mc_dropout/` - MC dropout analysis
- `experiments/uncertainty/calibration_curve.png` - Model calibration
- `experiments/uncertainty/confidence_analysis.png` - Confidence distribution

#### Interactive Analysis

```bash
jupyter notebook notebooks/04_uncertainty_quantification.ipynb
```

### Key Metrics

**Confidence Metrics**:
- `mean_confidence`: Average prediction confidence
- `mean_entropy`: Average predictive entropy (lower = more certain)
- `mean_variance`: Average prediction variance from MC dropout
- `mean_mutual_information`: Epistemic uncertainty measure

**Calibration Metrics**:
- `ECE`: Expected Calibration Error (lower = better calibrated)
- Lower ECE means predicted confidence matches actual accuracy

### Clinical Decision Support

**Confidence Thresholds** (recommendations):

| Confidence | Action | Reasoning |
|------------|--------|-----------|
| > 0.9 | Accept automatically | Very high confidence, low uncertainty |
| 0.7 - 0.9 | Review recommended | Medium confidence |
| < 0.7 | Manual diagnosis required | Low confidence |
| High variance | Flag for expert | High model uncertainty |

**Identifying Uncertain Cases**:
```python
uncertain = identify_uncertain_samples(
    probabilities,
    predictions_mc,
    threshold_confidence=0.7,
    threshold_entropy=0.5
)
```

### Advanced Usage

#### In Custom Scripts

```python
from src.evaluation.uncertainty import (
    monte_carlo_dropout,
    calculate_confidence_metrics,
    identify_uncertain_samples
)

# MC Dropout
mean_probs, std_probs, all_preds = monte_carlo_dropout(
    model, input_tensor, n_iterations=30
)

# Calculate metrics
metrics = calculate_confidence_metrics(mean_probs, all_preds)

# Find uncertain samples
uncertain = identify_uncertain_samples(mean_probs, all_preds)
```

#### Calibration

```python
from src.evaluation.uncertainty import (
    calibration_curve,
    expected_calibration_error,
    temperature_scaling
)

# Compute calibration
bin_confs, bin_accs = calibration_curve(y_true, y_prob)
ece = expected_calibration_error(y_true, y_prob)

# Temperature scaling (post-hoc calibration)
optimal_temp = temperature_scaling(logits, labels)
```

---

## Integration with Evaluation

### Complete Evaluation Pipeline

```bash
# 1. Standard evaluation
python evaluate.py \
    --checkpoint experiments/checkpoints/best_model.pth \
    --split test

# 2. Generate interpretability visualizations
python scripts/generate_interpretability.py \
    --checkpoint experiments/checkpoints/best_model.pth \
    --split test

# 3. Analyze uncertainty
python scripts/analyze_uncertainty.py \
    --checkpoint experiments/checkpoints/best_model.pth \
    --split test

# 4. Visualize predictions with uncertainty
python scripts/visualize_predictions.py \
    --checkpoint experiments/checkpoints/best_model.pth \
    --split test
```

---

## Key Takeaways

### Phase 9: Interpretability

✅ **Grad-CAM** shows what the model sees
✅ **Feature visualization** reveals learned representations
✅ **Clinical validation** ensures medical relevance
✅ **Debug tool** for identifying model issues

### Phase 10: Uncertainty

✅ **MC Dropout** estimates model uncertainty
✅ **TTA** estimates data uncertainty
✅ **Calibration** ensures reliable confidence
✅ **Decision support** guides clinical workflow

---

## Best Practices

### Interpretability

1. **Always validate with clinicians**: Ensure attention makes medical sense
2. **Check for artifacts**: Look for spurious correlations (markers, text)
3. **Document patterns**: Record consistent attention regions
4. **Compare to baseline**: How does attention differ from random?

### Uncertainty

1. **Use appropriate thresholds**: Tune for your clinical context
2. **Combine metrics**: Use both confidence and variance
3. **Validate calibration**: Check ECE regularly
4. **Human oversight**: Always allow expert review for uncertain cases

---

## Troubleshooting

### Grad-CAM Issues

**Problem**: Grad-CAM shows random patterns
- Check target layer (should be last conv layer)
- Verify model is in eval mode
- Ensure gradients are being computed

**Problem**: Heatmap doesn't align with image
- Check image preprocessing matches training
- Verify resize/normalization steps

### Uncertainty Issues

**Problem**: All predictions have high variance
- Model may need more training
- Check if dropout is too aggressive
- Increase number of MC iterations

**Problem**: Poor calibration (high ECE)
- Use temperature scaling
- Re-train with label smoothing
- Check class imbalance

---

## File Summary

### New Files Created

**Source Code**:
- `src/visualization/__init__.py`
- `src/visualization/gradcam.py` - Grad-CAM implementations
- `src/visualization/features.py` - Feature visualization
- `src/evaluation/uncertainty.py` - Uncertainty quantification

**Scripts**:
- `scripts/generate_interpretability.py` - Generate visualizations
- `scripts/analyze_uncertainty.py` - Analyze uncertainty

**Notebooks**:
- `notebooks/03_model_interpretability.ipynb` - Interactive interpretability
- `notebooks/04_uncertainty_quantification.ipynb` - Interactive uncertainty

**Documentation**:
- `docs/PHASES_9_10_COMPLETE.md` - This file

---

## Next Steps

With Phases 9-10 complete, you can now:

1. **Validate model decisions** with Grad-CAM
2. **Identify uncertain predictions** for expert review
3. **Improve calibration** with temperature scaling
4. **Build clinical decision support** using confidence thresholds

**Ready for Phase 11**: Hyperparameter Optimization! 🚀

---

**Your model is now interpretable and uncertainty-aware!** 🎉
