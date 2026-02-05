# Changelog

All notable changes to the Mayo Clinic STRIP AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD workflow for automated testing
- CONTRIBUTING.md guide for contributors
- MIT LICENSE with medical research disclaimers
- Flask and deployment dependencies in requirements.txt

## [0.9.0] - 2026-02-04

### Added
- **Phase 11: Hyperparameter Optimization**
  - Grid search implementation
  - Random search with sampling strategies
  - Experiment tracking and comparison
- **Phase 12: Limited Data Handling**
  - MixUp augmentation (Zhang et al., 2017)
  - CutMix augmentation (Yun et al., 2019)
  - Mixed loss computation
- **Phase 13: Robustness & Validation**
  - Robustness testing against corruptions (noise, blur, brightness, contrast)
  - Bias analysis with fairness metrics
  - Subgroup performance comparison
- **Phase 14: Deployment**
  - Model optimization (quantization, pruning, ONNX export)
  - Flask REST API with CORS support
  - Docker containerization with docker-compose
  - Python API client for testing
  - Comprehensive deployment documentation
- **Phase 16: Testing & QA**
  - Unit tests for datasets, models, and preprocessing
  - Integration tests for end-to-end pipeline
  - Test runner with coverage support
- **Documentation**
  - FINAL_SUMMARY.md with complete project overview
  - Updated README.md with badges and comprehensive guide
  - Updated PROJECT_STATUS.md reflecting 82% completion

### Changed
- Enhanced augmentation.py with MixUp/CutMix implementations
- Updated PROJECT_STATUS.md with completion status

## [0.5.0] - 2026-01-28

### Added
- **Phase 9: Model Interpretability**
  - Grad-CAM and Grad-CAM++ implementations
  - Guided backpropagation
  - Feature visualization (t-SNE, PCA, UMAP)
  - Feature separability analysis
- **Phase 10: Uncertainty Quantification**
  - Monte Carlo Dropout
  - Test-Time Augmentation
  - Calibration metrics (ECE, calibration curves)
  - Uncertain sample identification
- Interactive Jupyter notebooks for interpretability and uncertainty

### Changed
- Updated documentation with Phase 9-10 completion guide
- Enhanced evaluation framework with uncertainty quantification

## [0.3.0] - 2026-01-20

### Added
- **Phase 1-5: Core Infrastructure**
  - Complete Python environment setup
  - Data validation and exploration tools
  - Medical image preprocessing pipeline (DICOM, NIfTI, PNG, NPY)
  - Patient-level data splitting (prevents leakage)
  - Augmentation pipeline with Albumentations
  - PyTorch datasets (StrokeDataset, PatientLevelDataset)
- **Phase 6-8: Training & Evaluation**
  - Multiple CNN architectures (SimpleCNN, ResNet, EfficientNet)
  - Transfer learning support
  - Training pipeline with early stopping
  - Evaluation framework with clinical metrics
  - TensorBoard integration
- **Documentation**
  - Implementation plan (17 phases, 250+ tasks)
  - Medical domain knowledge documentation
  - Medical imaging best practices guide
  - Data preparation guide
  - Ethics and HIPAA compliance guide
  - QUICKSTART guide
- **Testing Infrastructure**
  - Dummy data generator
  - End-to-end pipeline test
  - Prediction visualization tools
  - Model comparison utilities

### Technical Details
- Patient-level data splitting implemented
- Clinical metrics: Sensitivity, specificity, PPV, NPV, ROC-AUC
- Support for multiple medical image formats
- Comprehensive data augmentation for medical images

## [0.1.0] - 2026-01-15

### Added
- Initial project structure
- Git repository setup
- Basic documentation
- Requirements file with core dependencies

---

## Release Notes

### Version 0.9.0 - Production-Ready Release

This release brings the Mayo Clinic STRIP AI project to production-ready status (82% complete, 14/17 phases). Key highlights:

**New Capabilities:**
- Complete hyperparameter optimization suite
- Advanced data augmentation for limited datasets
- Comprehensive robustness and fairness testing
- Production deployment infrastructure with Docker
- Model optimization for faster inference

**Production Features:**
- REST API for inference
- Model quantization (3-4x size reduction)
- Docker containerization
- Comprehensive test suite
- CI/CD ready with GitHub Actions

**For Researchers:**
- Hyperparameter search tools
- Robustness benchmarking
- Bias and fairness analysis
- Complete testing framework

**Total Additions:**
- ~4,000 lines of production code
- 19 new files including tests, deployment, and scripts
- 10+ comprehensive documentation guides
- 20+ utility scripts

This version is suitable for:
- Research experiments
- Clinical validation studies
- Production deployment (with proper validation)
- Educational purposes
- Collaborative development

### Version 0.5.0 - Interpretability & Uncertainty

Added complete interpretability and uncertainty quantification capabilities:
- Grad-CAM visualizations for model explanations
- Feature space analysis with dimensionality reduction
- Uncertainty estimation with MC Dropout
- Model calibration assessment

These tools are essential for clinical validation and building trust in AI predictions.

### Version 0.3.0 - Initial Complete Pipeline

First complete end-to-end pipeline from raw medical images to trained models:
- Full data preprocessing pipeline
- Multiple model architectures
- Training with early stopping
- Clinical evaluation metrics
- Patient-level data splitting (critical for medical ML)

This version provides a solid foundation for medical imaging research.

---

## Migration Guides

### Migrating to 0.9.0

**New Dependencies:**
Update your environment:
```bash
pip install -r requirements.txt
```

**New Features:**
- Use `scripts/run_hyperparameter_search.py` for HP optimization
- Use `scripts/optimize_model.py` before deployment
- Use `deploy/api.py` for production serving
- Run tests with `scripts/run_tests.py`

**Breaking Changes:**
None - all changes are backward compatible.

---

## Future Roadmap

### Version 1.0.0 (Planned)
- [ ] Complete Phase 15: Documentation & Reporting
- [ ] Research paper/technical report
- [ ] Presentation materials
- [ ] Complete code documentation
- [ ] User manual

### Version 1.1.0 (Future)
- [ ] Multi-class classification (additional stroke types)
- [ ] Multi-modal learning (combine imaging modalities)
- [ ] Online learning capabilities
- [ ] Active learning for efficient annotation
- [ ] Attention-based models
- [ ] Vision transformer architectures

---

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of contributors to this project.

---

## Links

- **Repository**: https://github.com/calebnewtonusc/mayo-clinic-strip-ai
- **Documentation**: [docs/](docs/)
- **Issues**: https://github.com/calebnewtonusc/mayo-clinic-strip-ai/issues
- **Project Status**: [PROJECT_STATUS.md](PROJECT_STATUS.md)
