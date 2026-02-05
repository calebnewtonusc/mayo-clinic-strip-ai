# Raw Data Directory

Place your raw, unprocessed medical imaging data here.

## Expected Structure

```
raw/
├── CE/                    # Cardioembolic samples
│   ├── patient_001/
│   │   ├── image_001.dcm
│   │   └── image_002.dcm
│   └── patient_002/
└── LAA/                   # Large Artery Atherosclerosis samples
    ├── patient_003/
    └── patient_004/
```

## Important Notes

- Data files are NOT committed to git (see .gitignore)
- Ensure all data is properly de-identified
- Maintain patient-level directory organization
- See docs/DATA_PREPARATION.md for detailed instructions
