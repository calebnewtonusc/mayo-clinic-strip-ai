#!/usr/bin/env python3
"""
Prepare Mayo Clinic STRIP AI dataset:
1. Reorganize into patient-level directories
2. Create train/val/test splits at patient level
3. Generate statistics
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def reorganize_data(raw_data_dir, processed_dir):
    """Reorganize data into patient-level directory structure."""

    # Read training CSV
    train_csv = os.path.join(raw_data_dir, 'train.csv')
    df = pd.read_csv(train_csv)

    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total images: {len(df)}")
    print(f"CE images: {len(df[df['label'] == 'CE'])} ({len(df[df['label'] == 'CE'])/len(df)*100:.1f}%)")
    print(f"LAA images: {len(df[df['label'] == 'LAA'])} ({len(df[df['label'] == 'LAA'])/len(df)*100:.1f}%)")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Images per patient: {df.groupby('patient_id').size().describe()}")

    # Create processed directory structure
    os.makedirs(processed_dir, exist_ok=True)

    # Process each label
    for label in ['CE', 'LAA']:
        label_dir = os.path.join(processed_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        label_df = df[df['label'] == label]
        patients = label_df['patient_id'].unique()

        print(f"\nProcessing {label}: {len(patients)} patients, {len(label_df)} images")

        for patient_id in patients:
            # Create patient directory
            patient_dir = os.path.join(label_dir, f"patient_{patient_id}")
            os.makedirs(patient_dir, exist_ok=True)

            # Copy all images for this patient
            patient_images = label_df[label_df['patient_id'] == patient_id]
            for _, row in patient_images.iterrows():
                src = os.path.join(raw_data_dir, 'train', f"{row['image_id']}.jpg")
                dst = os.path.join(patient_dir, f"{row['image_id']}.jpg")

                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    print(f"Warning: {src} not found")

    print(f"\n✓ Data reorganized into {processed_dir}")
    return df


def create_patient_splits(df, processed_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Create patient-level train/val/test splits."""

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Create splits directory
    splits_dir = os.path.join(processed_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # Split by label to ensure balanced splits
    train_patients, val_patients, test_patients = [], [], []

    for label in ['CE', 'LAA']:
        label_df = df[df['label'] == label]
        patients = label_df['patient_id'].unique()

        # First split: train + val vs test
        train_val_patients, test_pts = train_test_split(
            patients,
            test_size=test_ratio,
            random_state=random_state
        )

        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_pts, val_pts = train_test_split(
            train_val_patients,
            test_size=val_ratio_adjusted,
            random_state=random_state
        )

        train_patients.extend(train_pts)
        val_patients.extend(val_pts)
        test_patients.extend(test_pts)

        print(f"\n{label} split:")
        print(f"  Train: {len(train_pts)} patients")
        print(f"  Val:   {len(val_pts)} patients")
        print(f"  Test:  {len(test_pts)} patients")

    # Save split files
    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        for patient_id in sorted(train_patients):
            f.write(f"patient_{patient_id}\n")

    with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
        for patient_id in sorted(val_patients):
            f.write(f"patient_{patient_id}\n")

    with open(os.path.join(splits_dir, 'test.txt'), 'w') as f:
        for patient_id in sorted(test_patients):
            f.write(f"patient_{patient_id}\n")

    print(f"\n{'='*60}")
    print(f"Patient-Level Splits Created")
    print(f"{'='*60}")
    print(f"Train: {len(train_patients)} patients ({len(train_patients)/(len(train_patients)+len(val_patients)+len(test_patients))*100:.1f}%)")
    print(f"Val:   {len(val_patients)} patients ({len(val_patients)/(len(train_patients)+len(val_patients)+len(test_patients))*100:.1f}%)")
    print(f"Test:  {len(test_patients)} patients ({len(test_patients)/(len(train_patients)+len(val_patients)+len(test_patients))*100:.1f}%)")

    # Calculate image counts per split
    train_images = df[df['patient_id'].isin([p for p in train_patients])].shape[0]
    val_images = df[df['patient_id'].isin([p for p in val_patients])].shape[0]
    test_images = df[df['patient_id'].isin([p for p in test_patients])].shape[0]

    print(f"\nImage counts:")
    print(f"Train: {train_images} images")
    print(f"Val:   {val_images} images")
    print(f"Test:  {test_images} images")

    print(f"\n✓ Split files saved to {splits_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Mayo Clinic STRIP AI dataset')
    parser.add_argument('--raw-dir', type=str, default='data/raw/data',
                        help='Directory containing raw downloaded data')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Mayo Clinic STRIP AI Data Preparation")
    print(f"{'='*60}")
    print(f"Raw data: {args.raw_dir}")
    print(f"Processed data: {args.processed_dir}")
    print(f"Split ratios: {args.train_ratio:.2f} / {args.val_ratio:.2f} / {args.test_ratio:.2f}")

    # Step 1: Reorganize data into patient-level structure
    print(f"\n[1/2] Reorganizing data into patient-level structure...")
    df = reorganize_data(args.raw_dir, args.processed_dir)

    # Step 2: Create patient-level splits
    print(f"\n[2/2] Creating patient-level train/val/test splits...")
    create_patient_splits(
        df,
        args.processed_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )

    print(f"\n{'='*60}")
    print(f"✓ Data preparation complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Train model: python train.py --config config/mayo_config.yaml")
    print(f"2. Evaluate: python evaluate.py --checkpoint checkpoints/best_model.pth")
    print(f"3. Deploy: Copy model to models/ and push to Railway")


if __name__ == '__main__':
    main()
