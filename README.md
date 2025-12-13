# Stress Detection System

A comprehensive stress detection system combining WESAD, DREAMER, and ASCERTAIN datasets with subject-independent machine learning models.

## Quick Start

### 1. Dataset Preparation

```bash
# Process raw datasets and create merged dataset
python dataset_processor.py
```

This creates:

- `data/stress_merged_all_pseudowindows.csv` - Merged dataset with 29,800 rows
- `data/stress_ascertain_clip_level.csv` - ASCERTAIN clip-level data

### 2. Model Training

```bash
# Train Random Forest model
python train_.py --data data/stress_merged_all_pseudowindows.csv --model rf --out models/stress_model_rf.pkl

# Train CNN-LSTM model
python train_.py --data data/stress_merged_all_pseudowindows.csv --model cnn --out models/stress_model_cnn.pkl
```

### 3. Additional Analyses

```bash
# Learning curve analysis
python train_.py --curve --data data/stress_merged_all_pseudowindows.csv --model rf

# Early warning analysis
python train_.py --early_warning --data data/stress_merged_all_pseudowindows.csv --early_horizons "1,3,5"
```

## Documentation

- **[Training Documentation](TRAINING_DOCUMENTATION.md)** - Complete guide to the training pipeline, model architectures, and evaluation methods
- **Dataset Statistics**: See `dataset_processor.py` for dataset composition and statistics

## Dataset Overview

- **Total Rows**: 29,800 windows
- **Total Columns**: 346 (24 features + metadata)
- **Subjects**: 74 unique subjects
- **Datasets**: WESAD (53.3%), ASCERTAIN (35.0%), DREAMER (11.7%)
- **Stress Distribution**: 60.61% non-stress, 39.39% stress

## Features

- **24 Physiological Features**: HRV (19), EDA (2), Temperature (2), Accelerometer (2)
- **Subject-Independent Evaluation**: Leave-One-Subject-Out (LOSO) cross-validation
- **Baseline Normalization**: Per-subject baseline correction for improved generalization
- **Model Architectures**: Random Forest and CNN-LSTM
- **Advanced Techniques**: Probability calibration, threshold optimization, hyperparameter search

## File Structure

```
├── dataset_processor.py      # Dataset merging and preprocessing
├── train_.py                 # Main training script
├── cnn_lstm.py              # CNN-LSTM model architecture
├── TRAINING_DOCUMENTATION.md # Complete training documentation
├── data/
│   ├── stress_merged_all_pseudowindows.csv
│   ├── WESAD/
│   ├── ASCERTAIN_Features/
│   └── processed/
└── models/                   # Trained model artifacts
```

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn
- torch (for CNN-LSTM)
- scipy (for ASCERTAIN .mat files)
