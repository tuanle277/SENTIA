# Stress Detection Model Training Documentation

## Overview

The training pipeline (`train_.py`) implements a comprehensive subject-independent stress detection system using Leave-One-Subject-Out (LOSO) cross-validation. The system supports two model architectures (Random Forest and CNN-LSTM), includes advanced preprocessing techniques like subject baseline normalization, and provides additional analysis modes including learning curves and early-warning prediction.

---

## Table of Contents

1. [Feature Configuration](#feature-configuration)
2. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
3. [Subject Baseline Normalization](#subject-baseline-normalization)
4. [Cross-Validation Strategy](#cross-validation-strategy)
5. [Model Architectures](#model-architectures)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [Model Calibration and Threshold Tuning](#model-calibration-and-threshold-tuning)
8. [Training Modes](#training-modes)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Model Artifacts](#model-artifacts)

---

## Feature Configuration

The training pipeline uses **24 physiological features** extracted from wearable sensor data, organized into four categories:

### Heart Rate Variability (HRV) Features (19 features)

- **Temporal Features**: `HRV_MeanNN`, `HRV_SDNN`, `HRV_RMSSD`, `HRV_SDSD`, `HRV_CVNN`, `HRV_CVSD`
- **Distribution Features**: `HRV_MedianNN`, `HRV_MadNN`, `HRV_MCVNN`, `HRV_IQRNN`, `HRV_SDRMSSD`
- **Percentile Features**: `HRV_Prc20NN`, `HRV_Prc80NN`, `HRV_pNN50`, `HRV_pNN20`
- **Range Features**: `HRV_MinNN`, `HRV_MaxNN`
- **Geometric Features**: `HRV_HTI`, `HRV_TINN`
- **Segmented Features**: `HRV_SDANN1`, `HRV_SDNNI1`, `HRV_SDANN2`, `HRV_SDNNI2`, `HRV_SDANN5`, `HRV_SDNNI5`

### Electrodermal Activity (EDA) Features (2 features)

- `EDA_Mean`: Mean skin conductance level
- `SCR_Peaks_N`: Number of skin conductance response peaks

### Temperature Features (2 features)

- `TEMP_Mean`: Mean skin temperature
- `TEMP_Std`: Standard deviation of skin temperature

### Accelerometer Features (2 features)

- `ACC_Mag_Mean`: Mean magnitude of acceleration
- `ACC_Mag_Std`: Standard deviation of acceleration magnitude

**Total Feature Count**: 24 features (`FEATURES_ALL`)

---

## Data Preprocessing Pipeline

### 1. Data Loading

- Loads the merged dataset CSV (`stress_merged_all_pseudowindows.csv`)
- Uses `low_memory=False` to ensure consistent dtype inference
- Validates presence of required columns: `stress_label`, `subject`, and all feature columns

### 2. Data Cleaning

- **Type Coercion**: Converts all feature columns to numeric using `pd.to_numeric()` with `errors='coerce'` (invalid values become NaN)
- **Missing Value Handling**: Drops rows containing any NaN values in features or labels
- **Label Validation**: Ensures `stress_label` is binary (0 = non-stress, 1 = stress)

### 3. Feature Selection

- Extracts only the 24 features defined in `FEATURES_ALL`
- Ensures consistent feature ordering across all folds

### 4. Preprocessing Pipeline (Applied During Hyperparameter Search)

The pipeline includes three sequential steps:

1. **Imputation**: `SimpleImputer(strategy='median')` - Fills missing values with median (though rows with NaNs are typically dropped earlier)
2. **Variance Threshold**: `VarianceThreshold(threshold=1e-6)` - Removes features with near-zero variance
3. **Standardization**: `StandardScaler(with_mean=True, with_std=True)` - Z-score normalization (mean=0, std=1)

---

## Subject Baseline Normalization

### Purpose

Subject baseline normalization addresses inter-subject variability in physiological signals. Each individual has different baseline levels (e.g., resting heart rate), so absolute feature values are less informative than deviations from personal baselines.

### Implementation (`apply_subject_baseline_shift`)

**Step 1: Baseline Computation**

- Computes baseline means per subject using **only neutral (label=0) windows from the TRAINING set**
- For each feature, calculates: `baseline[subject][feature] = mean(feature_values where label=0)`
- **Critical**: Baselines are computed ONLY from training data to prevent data leakage

**Step 2: Baseline Subtraction**

- For each row (both train and test), subtracts the subject's baseline: `feature_normalized = feature_original - baseline[subject][feature]`
- This creates deviation-from-baseline features that are more comparable across subjects

**Step 3: Fallback Handling**

- If a subject has no neutral windows in training: uses global neutral mean from training set
- If still missing: fills with zero

**Step 4: Application**

- Applied separately to each LOSO fold's train/test splits
- Ensures test subject's baseline is never used during training

### Example

```
Subject A baseline HRV_SDNN (from neutral windows): 50 ms
Subject A test window HRV_SDNN: 80 ms
Normalized value: 80 - 50 = 30 ms (deviation from baseline)

Subject B baseline HRV_SDNN: 100 ms
Subject B test window HRV_SDNN: 130 ms
Normalized value: 130 - 100 = 30 ms (same deviation, comparable!)
```

---

## Cross-Validation Strategy

### Leave-One-Subject-Out (LOSO) Cross-Validation

**Rationale**: Subject-independent evaluation ensures the model generalizes to unseen individuals, which is critical for real-world deployment.

**Process**:

1. **Subject Identification**: Extracts unique subject IDs from the dataset (74 total subjects)
2. **Fold Creation**: For each subject:
   - **Test Set**: All windows belonging to that subject
   - **Training Set**: All windows from all other subjects
3. **Training**: Trains a separate model for each fold using only training subjects
4. **Evaluation**: Evaluates on the held-out test subject
5. **Aggregation**: Collects predictions from all folds and computes overall metrics

**Key Properties**:

- **No Data Leakage**: Test subject's data never influences training
- **Subject Diversity**: Each subject appears exactly once as test data
- **Realistic Performance**: Mimics deployment scenario where model encounters new individuals

**Example**:

```
Fold 1: Train on Subjects 2-74, Test on Subject 1
Fold 2: Train on Subjects 1,3-74, Test on Subject 2
...
Fold 74: Train on Subjects 1-73, Test on Subject 74
```

---

## Model Architectures

### 1. Random Forest Classifier

**Configuration**:

- **Base Estimators**: 300 decision trees (`n_estimators=300`)
- **Class Weighting**: `'balanced'` - Automatically adjusts weights to handle class imbalance
- **Parallelization**: `n_jobs=-1` - Uses all available CPU cores
- **Random State**: `42` - Ensures reproducibility

**Hyperparameter Search Space**:

- `n_estimators`: 200-800 (uniform random)
- `max_depth`: 4-32 (uniform random)
- `min_samples_split`: 2-20 (uniform random)
- `min_samples_leaf`: 1-10 (uniform random)
- `max_features`: ['sqrt', 'log2', None]

**Search Method**: `RandomizedSearchCV` with 30 iterations, 5-fold GroupKFold CV

**Advantages**:

- Fast training and inference
- Handles non-linear relationships
- Feature importance interpretation
- Robust to outliers

### 2. CNN-LSTM Hybrid Model

**Architecture** (`CNNLSTM` class):

**Input**: `(batch_size, seq_len=1, num_features=24)`

**Stage 1: 1D Convolutional Layers**

- **Conv1D Layer 1**:
  - Input channels: 24 (features)
  - Output channels: 32
  - Kernel size: 3
  - Padding: 1 (maintains sequence length)
  - Activation: ReLU
  - Batch Normalization
- **Conv1D Layer 2**:
  - Input channels: 32
  - Output channels: 32
  - Kernel size: 3
  - Padding: 1
  - Activation: ReLU
  - Batch Normalization
  - Dropout: 0.3

**Stage 2: LSTM Layer**

- **Input size**: 32 (from CNN output)
- **Hidden size**: 64
- **Number of layers**: 1
- **Bidirectional**: False
- **Batch first**: True

**Stage 3: Fully Connected Classifier**

- **Layer 1**: Linear(64 → 64), ReLU, Dropout(0.3)
- **Layer 2**: Linear(64 → 2) - Binary classification output

**Training Configuration**:

- **Optimizer**: Adam (`lr=1e-3`)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64 (training), 256 (validation)
- **Epochs**: 15 (final model), 8 (LOSO folds)
- **Device**: CUDA if available, else CPU

**Advantages**:

- Captures complex feature interactions
- Can learn hierarchical representations
- Potential for sequence modeling (though currently seq_len=1)

**Note**: Currently configured with `seq_len=1`, treating each window independently. Can be extended for temporal sequences.

---

## Hyperparameter Optimization

### Search Strategy

**Method**: `RandomizedSearchCV` with GroupKFold cross-validation

**Configuration**:

- **Iterations**: 30 random parameter combinations
- **CV Folds**: 5-fold GroupKFold (groups = subjects)
- **Scoring Metric**: Weighted F1-score (`make_scorer(f1_score, average='weighted')`)
- **Refit**: True (retrains best model on full training set)

### Search Process

1. **GroupKFold CV**: Splits training subjects into 5 folds (ensures no subject appears in both train and validation)
2. **Parameter Sampling**: Randomly samples from defined distributions
3. **Model Training**: Trains pipeline (imputation → variance threshold → scaling → classifier) for each parameter set
4. **Evaluation**: Computes weighted F1-score on validation folds
5. **Best Model Selection**: Selects parameters with highest mean validation F1-score

### Leakage Prevention

- **Subject Grouping**: GroupKFold ensures subjects don't leak between train/validation
- **Baseline Computation**: Subject baselines computed only from training fold's neutral windows
- **No Test Data Usage**: Test subject data never used in hyperparameter search

---

## Model Calibration and Threshold Tuning

### Probability Calibration

**Method**: `CalibratedClassifierCV` with isotonic regression

**Process**:

1. **3-Fold Stratified CV**: Splits training data into 3 folds
2. **Calibration**: Fits isotonic regression to map raw probabilities to calibrated probabilities
3. **Ensemble**: Averages calibrated probabilities across folds

**Purpose**: Raw classifier probabilities may not be well-calibrated (e.g., predicted 0.7 doesn't mean 70% chance). Calibration ensures predicted probabilities reflect true likelihoods.

### Threshold Optimization

**Method**: Grid search over threshold values

**Process**:

1. **Threshold Range**: Tests thresholds from 0.2 to 0.9 in 36 steps (0.02 increments)
2. **Metric**: Maximizes weighted F1-score on training predictions
3. **Selection**: Chooses threshold that yields highest F1-score

**Rationale**: Default threshold of 0.5 may not be optimal for imbalanced datasets. Optimizing threshold balances precision and recall.

**Application**:

- During LOSO: Each fold gets its own optimal threshold
- Final predictions: `y_pred = (calibrated_proba >= optimal_threshold).astype(int)`

---

## Training Modes

### 1. Standard LOSO Training (Default)

**Command**: `python train_.py --data <path> --model <rf|cnn> --out <model_path>`

**Process**:

1. Loads dataset and validates structure
2. Performs LOSO cross-validation:
   - For each subject: train model, calibrate, optimize threshold, evaluate
3. Aggregates results across all folds
4. Trains final production model on ALL data
5. Saves model artifact to specified path

**Outputs**:

- Per-subject performance metrics
- Overall LOSO performance (accuracy, F1, AUROC, AUPRC)
- Classification report and confusion matrix
- Saved production model (.pkl)

### 2. Learning Curve Analysis

**Command**: `python train_.py --curve --data <path> --model <rf|cnn> --curve_out <plot_path>`

**Purpose**: Analyzes how model performance scales with training data size

**Process**:

1. Fixed 80/20 train/validation split (stratified)
2. Trains models using 10%, 20%, ..., 90% of training data
3. Evaluates on fixed validation set
4. Plots F1-score vs. training fraction

**Output**: PNG plot showing learning curve

**Use Case**: Determines if more data would improve performance

### 3. Early-Warning Analysis

**Command**: `python train_.py --early_warning --data <path> --early_horizons "1,3,5" --early_out <csv_path>`

**Purpose**: Evaluates ability to predict stress H steps ahead

**Process**:

1. **Label Shifting**: Creates future labels (`stress_label_t+1`, `stress_label_t+3`, `stress_label_t+5`)
   - For each subject, shifts labels forward by H steps
   - Only shifts within subject boundaries
2. **LOSO Training**: For each horizon:
   - Trains LOSO models predicting future stress state
   - Baseline normalization uses CURRENT label (not future)
   - Target is future-shifted label
3. **Metrics**: Computes accuracy, F1, precision, recall for each horizon

**Output**: CSV with metrics per horizon

**Use Case**: Determines how far ahead stress can be predicted (e.g., 1 window = 30 seconds ahead)

---

## Evaluation Metrics

### Primary Metrics

1. **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`

   - Overall correctness rate

2. **Weighted F1-Score**: Harmonic mean of precision and recall, weighted by class support

   - Balances precision and recall
   - Accounts for class imbalance

3. **AUROC** (Area Under ROC Curve): Measures separability between classes

   - Range: 0-1 (higher is better)
   - 0.5 = random, 1.0 = perfect

4. **AUPRC** (Area Under Precision-Recall Curve): Better for imbalanced datasets
   - Focuses on positive class performance

### Per-Subject Metrics

For each LOSO fold, computes:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Optimal threshold (τ)

### Classification Report

Provides per-class metrics:

- Precision, Recall, F1-Score for each class
- Support (number of instances)
- Macro and weighted averages

### Confusion Matrix

```
                Predicted
              Neutral  Stress
True Neutral    TN      FP
True Stress     FN      TP
```

---

## Model Artifacts

### Saved Model Structure

**Random Forest Model** (`stress_model_rf.pkl`):

```python
{
    'scaler': StandardScaler,      # Fitted StandardScaler
    'model': RandomForestClassifier, # Trained RF model
    'features': list                # Feature names (FEATURES_ALL)
}
```

**CNN-LSTM Model** (`stress_model_cnn.pkl`):

```python
{
    'scaler': StandardScaler,           # Fitted StandardScaler
    'model_state_dict': dict,           # PyTorch model weights
    'model_params': {
        'input_channels': int,
        'seq_len': int,
        'num_features': int,
        'num_classes': int
    },
    'features': list                     # Feature names (FEATURES_ALL)
}
```

### Model Loading and Inference

**Random Forest**:

```python
import pickle
with open('models/stress_model_rf.pkl', 'rb') as f:
    artifact = pickle.load(f)
    scaler = artifact['scaler']
    model = artifact['model']
    features = artifact['features']

# Preprocess new data
X_scaled = scaler.transform(X_new[features])
proba = model.predict_proba(X_scaled)[:, 1]
prediction = (proba >= 0.5).astype(int)
```

**CNN-LSTM**:

```python
import torch
import pickle
from cnn_lstm import CNNLSTM

with open('models/stress_model_cnn.pkl', 'rb') as f:
    artifact = pickle.load(f)
    scaler = artifact['scaler']
    params = artifact['model_params']
    state_dict = artifact['model_state_dict']
    features = artifact['features']

# Reconstruct model
model = CNNLSTM(**params)
model.load_state_dict(state_dict)
model.eval()

# Preprocess and predict
X_scaled = scaler.transform(X_new[features])
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    logits = model(X_tensor)
    proba = torch.softmax(logits, dim=1)[:, 1].numpy()
    prediction = (proba >= 0.5).astype(int)
```

---

## Training Workflow Summary

### Complete Training Pipeline

1. **Data Loading** → Load merged dataset CSV
2. **Data Cleaning** → Coerce types, drop NaNs, validate labels
3. **Feature Selection** → Extract 24 physiological features
4. **LOSO Cross-Validation**:
   - For each subject:
     - Split: Test = subject, Train = all others
     - Baseline normalization (train-only)
     - Hyperparameter search (GroupKFold CV on train)
     - Model calibration (3-fold CV)
     - Threshold optimization
     - Evaluation on test subject
   - Aggregate metrics across all folds
5. **Final Model Training** → Train on ALL data (no CV)
6. **Model Saving** → Serialize model + scaler + metadata

### Key Design Principles

1. **Subject Independence**: No subject appears in both train and test
2. **No Data Leakage**: Baselines computed only from training data
3. **Reproducibility**: Fixed random seeds (42)
4. **Class Imbalance Handling**: Balanced class weights, threshold optimization
5. **Robust Evaluation**: Multiple metrics, per-subject analysis
6. **Production Ready**: Final model trained on full dataset

---

## Performance Expectations

Based on the dataset statistics:

- **Overall Stress Distribution**: 60.61% non-stress, 39.39% stress
- **Dataset-Specific Distributions**:
  - WESAD: 45.7% non-stress, 54.3% stress (most balanced)
  - ASCERTAIN: 73.0% non-stress, 27.0% stress
  - DREAMER: 91.5% non-stress, 8.5% stress (highly imbalanced)

**Expected Challenges**:

- Class imbalance (especially DREAMER)
- Inter-subject variability (addressed by baseline normalization)
- Dataset heterogeneity (different sensors, protocols)

**Typical Performance Range**:

- Accuracy: 0.70-0.85 (subject-independent)
- Weighted F1: 0.65-0.80
- AUROC: 0.75-0.90

---

## Usage Examples

### Standard Training

```bash
# Train Random Forest model
python train_.py --data data/stress_merged_all_pseudowindows.csv --model rf --out models/stress_model_rf.pkl

# Train CNN-LSTM model
python train_.py --data data/stress_merged_all_pseudowindows.csv --model cnn --out models/stress_model_cnn.pkl
```

### Learning Curve

```bash
python train_.py --curve --data data/stress_merged_all_pseudowindows.csv --model rf --curve_out f1_vs_fraction.png
```

### Early Warning Analysis

```bash
python train_.py --early_warning --data data/stress_merged_all_pseudowindows.csv --early_horizons "1,3,5,10" --early_out early_warning_results.csv
```

---

## Troubleshooting

### Common Issues

1. **Missing Features**: Ensure dataset contains all 24 features in `FEATURES_ALL`
2. **Subject Column Missing**: Dataset must have `subject` column for LOSO CV
3. **Memory Issues**: Reduce `n_jobs` or use smaller `n_estimators` for RF
4. **CUDA Out of Memory**: Reduce batch size or use CPU for CNN-LSTM
5. **No Valid Samples**: Check that subjects have sufficient data (both classes)

### Validation Checks

- Verify dataset shape matches expected (29,800 rows, 346 columns)
- Check stress label distribution (should be ~60/40 split)
- Ensure feature columns are numeric and non-null
- Validate subject IDs are present and unique

---

## References

- **Dataset**: Merged WESAD, DREAMER, and ASCERTAIN datasets
- **Model Architectures**: Random Forest (scikit-learn), CNN-LSTM (PyTorch)
- **Evaluation**: Subject-independent LOSO cross-validation
- **Preprocessing**: Subject baseline normalization, standardization

---

_Last Updated: Based on `train_.py`and`cnn*lstm.py` implementation*
