# Medical Imaging Best Practices

## Key Principles

### 1. Patient-Level Data Splitting
**CRITICAL**: Always split data by patient, not by image to prevent data leakage.

### 2. Domain-Specific Preprocessing
- CT: Use Hounsfield Units, apply windowing
- MRI: Z-score normalization, bias field correction
- Handle medical image formats (DICOM, NIfTI)

### 3. Class Imbalance Handling
- Use weighted loss functions
- Focal loss for hard examples
- Balanced evaluation metrics (F1, ROC-AUC)

### 4. Transfer Learning
- ImageNet pretraining for RGB images
- Medical imaging pretrained models when available
- Fine-tune with lower learning rates

### 5. Model Interpretability
- Use Grad-CAM for visualization
- Validate clinical relevance with experts
- Check for spurious correlations

### 6. Evaluation
- Report sensitivity, specificity, PPV, NPV
- Use patient-level aggregation
- Calculate confidence intervals
- Perform cross-validation

### 7. Handling Limited Data
- Aggressive data augmentation
- Transfer learning
- Semi-supervised learning techniques

### 8. Common Pitfalls to Avoid
- Data leakage from improper splitting
- Learning scanner artifacts instead of disease
- Overfitting to small datasets
- Not validating on external data

### 9. Clinical Validation
- External validation on different scanners/institutions
- Compare to clinician performance
- Document failure modes

### 10. Ethics & Privacy
- Ensure proper de-identification
- Test for bias across demographics
- Maintain HIPAA compliance
- Document limitations clearly
