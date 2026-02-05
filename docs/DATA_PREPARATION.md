# Data Preparation Guide

This guide explains how to prepare and organize the Mayo Clinic STRIP dataset for training.

## Data Directory Structure

The data should be organized as follows:

```
data/
├── raw/                    # Original, unprocessed data
│   ├── CE/                # Cardioembolic samples
│   │   ├── patient_001/
│   │   │   ├── image_001.dcm
│   │   │   ├── image_002.dcm
│   │   │   └── ...
│   │   └── patient_002/
│   └── LAA/               # Large Artery Atherosclerosis samples
│       ├── patient_003/
│       └── patient_004/
├── processed/             # Preprocessed images
│   ├── CE/
│   └── LAA/
├── augmented/             # Augmented dataset (optional)
│   ├── CE/
│   └── LAA/
└── splits/                # Train/val/test split information
    ├── train.json
    ├── val.json
    └── test.json
```

## Step-by-Step Data Preparation

### Step 1: Obtain Dataset
1. Ensure you have IRB approval and data use agreements in place
2. Download the dataset from Mayo Clinic's secure transfer system
3. Verify data integrity (checksums if provided)

### Step 2: Organize Raw Data
Place patient data in appropriate CE or LAA directories, maintaining patient-level organization.

### Step 3: Run Preprocessing
```bash
python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed
```

### Step 4: Create Data Splits
```bash
python scripts/create_splits.py --data_dir data/processed --patient_level
```

See docs/MEDICAL_IMAGING_BEST_PRACTICES.md for detailed preprocessing guidelines.
