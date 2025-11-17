# -*- coding: utf-8 -*-
"""
aura/prediction/train.py

This script trains and evaluates the stress prediction model.
It performs a subject-independent cross-validation to get a robust
measure of performance and then saves a final model trained on all data.

It also supports:
- Learning curves: F1 vs training fraction (--curve)
- Early-warning stress prediction: predict stress t+H steps ahead (--early_warning)
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns  # may be unused; kept for backwards compatibility
from cnn_lstm import CNNLSTM
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import randint, uniform
import random

# =============================================================================
# Shared feature configuration
# =============================================================================

FEATURES_HRV = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD",
    "HRV_MedianNN", "HRV_MadNN", "HRV_MCVNN", "HRV_IQRNN", "HRV_SDRMSSD",
    "HRV_Prc20NN", "HRV_Prc80NN", "HRV_pNN50", "HRV_pNN20", "HRV_MinNN",
    "HRV_MaxNN", "HRV_HTI", "HRV_TINN",
    "HRV_SDANN1", "HRV_SDNNI1", "HRV_SDANN2", "HRV_SDNNI2", "HRV_SDANN5",
    "HRV_SDNNI5",
]

FEATURES_EDA = ["EDA_Mean", "SCR_Peaks_N"]
FEATURES_TEMP = ["TEMP_Mean", "TEMP_Std"]
FEATURES_ACC = ["ACC_Mag_Mean", "ACC_Mag_Std"]

FEATURES_ALL = FEATURES_HRV + FEATURES_EDA + FEATURES_TEMP + FEATURES_ACC


# =============================================================================
# Learning curve: F1 vs training fraction
# =============================================================================

def learning_curve_f1_vs_fraction(
    data_path: str,
    model_type: str = 'rf',
    output_plot: str = 'f1_vs_fraction.png'
):
    """
    Train using increasing fractions of the available training data (10%-90%)
    and plot the weighted F1 score achieved via a fixed validation split.

    Note: This uses a single random 80/20 split of the full dataset (subject info ignored),
    then gradually increases the fraction of the training subset used to fit the model.
    """
    print("--- Loading dataset for learning curve ---")
    df = pd.read_csv(data_path, low_memory=False)
    if 'stress_label' not in df.columns:
        raise ValueError("'stress_label' column not found in dataset.")

    features_to_keep = FEATURES_ALL

    # Coerce numerics and build base frame (no subject baseline shift here to keep routine simple)
    for col in features_to_keep:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Expected feature column '{col}' not found in dataset.")
    df['stress_label'] = pd.to_numeric(df['stress_label'], errors='coerce')

    base = df[features_to_keep + ['stress_label']]
    base = base.dropna()

    X_full = base[features_to_keep].astype('float64').values
    y_full = base['stress_label'].astype(int).values

    # Fixed train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)

    fractions = [i / 100 for i in range(10, 100, 10)]  # 0.1 to 0.9
    f1_scores = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for frac in fractions:
        n = max(1, int(len(X_train_std) * frac))
        X_sub = X_train_std[:n]
        y_sub = y_train[:n]

        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_sub, y_sub)
            y_pred = model.predict(X_val_std)
        else:
            # CNN-LSTM on tabular features as seq_len=1
            model = CNNLSTM(
                input_channels=X_sub.shape[1],
                seq_len=1,
                num_features=X_sub.shape[1],
                num_classes=2
            )
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_sub, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(y_sub, dtype=torch.long)
                ),
                batch_size=64,
                shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val_std, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(y_val, dtype=torch.long)
                ),
                batch_size=256,
                shuffle=False
            )
            model.train()
            for _ in range(8):
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
            # Evaluate on validation
            model.eval()
            preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    preds.append(pred)
            y_pred = np.concatenate(preds) if len(preds) else np.array([])

        f1w = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        f1_scores.append(f1w)
        print(f"Fraction {int(frac*100)}% -> F1 (weighted): {f1w:.3f} (n={n})")

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot([f * 100 for f in fractions], f1_scores, marker='o')
    plt.xlabel('Training fraction (%)')
    plt.ylabel('F1 score (weighted)')
    plt.title(f'F1 vs Training Fraction ({"RF" if model_type=="rf" else "CNN-LSTM"})')
    plt.grid(True, linestyle='--', alpha=0.4)
    os.makedirs(os.path.dirname(output_plot) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Saved learning curve plot to: {output_plot}")


# =============================================================================
# Subject baseline shift
# =============================================================================

def apply_subject_baseline_shift(train_df, test_df, features, label_col='stress_label', subj_col='subject'):
    """
    Compute baseline means per subject using only neutral (label==0) windows from the TRAIN split,
    then subtract for BOTH train and test rows belonging to that subject. Subjects missing neutral
    windows fall back to global train neutral mean or zero.

    NOTE: label_col is used only to identify "neutral" windows. For early-warning,
    you typically want this to be the *current* label (stress_label), not a future-shifted one.
    """
    assert subj_col in train_df.columns
    # Compute baseline from train neutrals
    neutral_train = train_df[train_df[label_col] == 0]
    baselines = neutral_train.groupby(subj_col)[features].mean()

    # Fallback: global neutral mean if a subject has no neutral in train
    global_neutral = neutral_train[features].mean()

    def _shift(df):
        df = df.copy()
        # join may create NaNs for subjects missing baseline -> fill with global -> fill 0
        df = df.join(baselines, on=subj_col, rsuffix='_baseline')
        for feat in features:
            base_col = f"{feat}_baseline"
            if base_col in df.columns:
                df[feat] = df[feat] - df[base_col]
        drop_cols = [f"{feat}_baseline" for feat in features if f"{feat}_baseline" in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
        # fill remaining NaNs from global neutral; if still NaN, fill 0
        df[features] = df[features].fillna(global_neutral).fillna(0.0)
        return df

    return _shift(train_df), _shift(test_df)


# =============================================================================
# Simple aggregate performance analysis
# =============================================================================

def statistical_performance_analysis(full_dataset, y_true, y_pred):
    """
    Performs basic statistical analysis of model performance ignoring subject IDs.
    """
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE ANALYSIS (No Subject Grouping)")
    print("=" * 80)

    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {overall_accuracy:.3f}")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1-Score: {overall_f1:.3f}")
    print("\n" + "=" * 80)

    return {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1
    }


# =============================================================================
# Hyperparameter search model factory (RF or HistGradientBoosting)
# =============================================================================

def make_model_and_search(model_type='rf'):
    """
    Returns a Pipeline wrapped in a RandomizedSearchCV with GroupKFold CV over TRAIN subjects.
    """
    if model_type == 'rf':
        base = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        param_dist = {
            'clf__n_estimators': randint(200, 800),
            'clf__max_depth': randint(4, 32),
            'clf__min_samples_split': randint(2, 20),
            'clf__min_samples_leaf': randint(1, 10),
            'clf__max_features': ['sqrt', 'log2', None],
        }
    else:
        # Strong tabular baseline when not using CNN; HistGradientBoosting is great for tabular
        from sklearn.ensemble import HistGradientBoostingClassifier
        base = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=None,
            max_iter=300,
            early_stopping=True,
            class_weight='balanced',
            random_state=42
        )
        param_dist = {
            'clf__learning_rate': uniform(0.01, 0.19),
            'clf__max_depth': randint(2, 24),
            'clf__max_iter': randint(150, 600),
            'clf__l2_regularization': uniform(0.0, 0.2),
            'clf__min_samples_leaf': randint(10, 200),
        }

    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('varth', VarianceThreshold(threshold=1e-6)),
        ('scale', StandardScaler(with_mean=True, with_std=True)),
        ('clf', base),
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=30,
        scoring=make_scorer(f1_score, average='weighted'),
        n_jobs=-1,
        cv=GroupKFold(n_splits=5),
        verbose=1,
        random_state=42,
        refit=True
    )
    return search


# =============================================================================
# Calibration + threshold tuning
# =============================================================================

def calibrate_and_pick_threshold(model, X_train, y_train):
    """
    Calibrate predicted probabilities via 3-fold CV and pick the decision threshold
    that maximizes weighted F1 on the (calibrated) train predictions.
    """
    calib = CalibratedClassifierCV(
        model,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        method='isotonic'
    )
    calib.fit(X_train, y_train)
    probs = calib.predict_proba(X_train)[:, 1]
    taus = np.linspace(0.2, 0.9, 36)
    best_tau, best_f1 = 0.5, -1
    for t in taus:
        preds = (probs >= t).astype(int)
        f1w = f1_score(y_train, preds, average='weighted', zero_division=0)
        if f1w > best_f1:
            best_f1, best_tau = f1w, t
    return calib, float(best_tau)


# =============================================================================
# Early-warning helpers
# =============================================================================

def add_shifted_labels(df: pd.DataFrame, horizons: list, label_col='stress_label', subj_col='subject'):
    """
    Adds columns like 'stress_label_t+1', 'stress_label_t+3' for each horizon.

    Assumes data is already sorted in temporal order within each subject.
    For each subject, shift only within their segment.
    """
    df = df.copy()
    for h in horizons:
        shifted = df.groupby(subj_col)[label_col].shift(-h)
        df[f"{label_col}_t+{h}"] = shifted
    return df


def loso_early_warning(
    df: pd.DataFrame,
    feature_cols: list,
    horizons: list,
    subj_col='subject',
    label_col='stress_label'
):
    """
    Trains LOSO models for each early-warning horizon.

    For each horizon h, we train to predict 'stress_label_t+h':
        - Baseline normalization is computed using the *current* label (stress_label).
        - Target is the future label column stress_label_t+h.

    Returns a metrics DataFrame across horizons.
    """
    results = []

    for h in horizons:
        shifted_label = f"{label_col}_t+{h}"
        if shifted_label not in df.columns:
            print(f"Warning: {shifted_label} not in dataframe, skipping.")
            continue

        print(f"\n=== Early Warning: Predicting {h} steps ahead ===")
        # Keep both current label and shifted label so baseline uses current state
        cols_needed = [subj_col] + feature_cols + [label_col, shifted_label]
        df_h = df[cols_needed].dropna()

        y_true_all, y_pred_all = [], []

        subjects = df_h[subj_col].unique()
        for sid in subjects:
            test_mask = df_h[subj_col] == sid
            train_mask = ~test_mask

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            train_df = df_h.loc[train_mask].copy()
            test_df = df_h.loc[test_mask].copy()

            # Baseline shift using *current* label_col, not future-shifted label
            train_df, test_df = apply_subject_baseline_shift(
                train_df,
                test_df,
                features=feature_cols,
                label_col=label_col,
                subj_col=subj_col
            )

            X_train = train_df[feature_cols].values
            y_train = train_df[shifted_label].values  # target is future label
            X_test = test_df[feature_cols].values
            y_test = test_df[shifted_label].values

            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(X_train)
            X_test_std = scaler.transform(X_test)

            clf = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            clf.fit(X_train_std, y_train)
            y_pred = clf.predict(X_test_std)

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())

        if len(y_true_all) == 0:
            print(f"No valid samples for horizon {h}")
            continue

        acc = accuracy_score(y_true_all, y_pred_all)
        f1w = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        prec = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        rec = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)

        print(f"Horizon={h} | Acc={acc:.3f} F1w={f1w:.3f} Prec={prec:.3f} Rec={rec:.3f}")
        results.append({
            "horizon": h,
            "accuracy": acc,
            "f1_weighted": f1w,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "n_samples": len(y_true_all),
        })

    return pd.DataFrame(results)


def run_early_warning(
    data_path: str,
    horizons: list = None,
    output_csv: str = "early_warning_results.csv"
):
    """
    Runs early warning LOSO evaluation across given horizons.

    horizons: list of integer steps ahead (e.g. [1, 3, 5]).
              The meaning of "step" depends on your windowing (e.g., 30s, 60s).
    """
    if horizons is None:
        horizons = [1, 3, 5]

    print(f"Loading dataset for early warning: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    if 'subject' not in df.columns or 'stress_label' not in df.columns:
        print("Error: dataset must contain 'subject' and 'stress_label' for early warning analysis.")
        return

    # Add shifted labels
    df = add_shifted_labels(df, horizons=horizons, label_col='stress_label', subj_col='subject')

    feature_cols = FEATURES_ALL

    # Run early-warning LOSO
    results_df = loso_early_warning(df, feature_cols, horizons=horizons, subj_col='subject', label_col='stress_label')

    # Save results
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved early warning results to: {output_csv}")
    print(results_df)


# =============================================================================
# Main LOSO training for stress detection
# =============================================================================

def train_stress_model(data_path, model_save_path, model_type: str = 'cnn'):
    """
    Trains a stress prediction model using subject-independent cross-validation
    and saves the final model trained on all data.

    Args:
        data_path (str): Path to the processed features_dataset.csv.
        model_save_path (str): Path to save the final trained model (.pkl).
    """
    print("--- Loading Processed Feature Dataset ---")
    try:
        # Use low_memory=False to ensure consistent dtype inference across chunks
        full_dataset = pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The file {data_path} was not found.")
        print("Please ensure you have successfully run the feature_extractor.py script first.")
        return

    if full_dataset.empty:
        print("Error: The feature dataset is empty. Cannot train the model.")
        return

    print(f"Dataset loaded successfully with shape: {full_dataset.shape}")

    features_to_keep = FEATURES_ALL

    # Coerce selected feature columns to numeric (invalid values become NaN)
    for col in features_to_keep:
        if col in full_dataset.columns:
            full_dataset[col] = pd.to_numeric(full_dataset[col], errors='coerce')
        else:
            print(f"Warning: Expected feature column '{col}' not found in dataset.")

    # Coerce label to numeric
    if 'stress_label' not in full_dataset.columns:
        print("Error: 'stress_label' column not found in dataset. Cannot train the model.")
        return
    full_dataset['stress_label'] = pd.to_numeric(full_dataset['stress_label'], errors='coerce')

    # Build base frame with features, label, and (if present) subject
    columns_to_collect = features_to_keep + ['stress_label']
    if 'subject' in full_dataset.columns:
        columns_to_collect.append('subject')
    base = full_dataset[columns_to_collect]

    before_drop_shape = base.shape[0]
    valid = base.dropna()
    dropped = before_drop_shape - valid.shape[0]
    if dropped > 0:
        print(f"Dropped {dropped} rows with non-numeric or missing values.")

    # Final X and y after baseline shift
    X = valid[features_to_keep].astype('float64')
    y = valid['stress_label'].astype(int)

    # --- Leave-One-Subject-Out Cross-Validation ---
    if 'subject' not in valid.columns and 'subject' not in full_dataset.columns:
        print("Error: 'subject' column not found. Required for Leave-One-Subject-Out CV.")
        return

    subjects_series = (
        valid['subject']
        if 'subject' in valid.columns
        else full_dataset.loc[valid.index, 'subject']
    ).astype(str)
    unique_subjects = subjects_series.unique()

    print(f"\n--- Starting Leave-One-Subject-Out CV across {len(unique_subjects)} subjects (model={model_type}) ---")
    y_true_all, y_pred_all, y_prob_all = [], [], []
    per_subject_results = []

    for subject_id in unique_subjects:
        test_idx = subjects_series == subject_id
        train_idx = ~test_idx

        # Build fold dataframes (only the columns we need)
        fold_train = valid.loc[train_idx, features_to_keep + ['stress_label', 'subject']].copy()
        fold_test = valid.loc[test_idx, features_to_keep + ['stress_label', 'subject']].copy()

        if fold_test.shape[0] == 0 or fold_train.shape[0] == 0:
            print(f"Skipping subject {subject_id}: insufficient samples.")
            continue

        # Leakage-safe baseline shift (train-only)
        fold_train, fold_test = apply_subject_baseline_shift(
            fold_train, fold_test, features_to_keep, label_col='stress_label', subj_col='subject'
        )

        X_train = fold_train[features_to_keep].astype('float64').values
        y_train = fold_train['stress_label'].astype(int).values
        X_test = fold_test[features_to_keep].astype('float64').values
        y_test = fold_test['stress_label'].astype(int).values

        # Group-aware HP search on TRAIN subjects only
        search = make_model_and_search('rf' if model_type == 'rf' else 'hgb')
        groups_train = fold_train['subject'].astype(str).values
        search.fit(X_train, y_train, groups=groups_train)
        best_clf = search.best_estimator_

        # Calibrate + threshold tune on TRAIN
        calib_model, tau = calibrate_and_pick_threshold(best_clf, X_train, y_train)

        # Predict on TEST
        y_prob = calib_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= tau).astype(int)

        # Collect
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_prob_all.extend(y_prob.tolist())

        acc = accuracy_score(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        per_subject_results.append({
            'subject': subject_id,
            'samples': int(X_test.shape[0]),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1w,
            'tau': tau
        })
        print(f"Subject {subject_id}: n={X_test.shape[0]} | Acc={acc:.3f} F1w={f1w:.3f} τ={tau:.2f}")

    # Overall evaluation across concatenated folds
    if len(y_true_all) == 0:
        print("No evaluation data collected. Aborting.")
        return

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_f1w = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    overall_auroc = roc_auc_score(y_true_all, y_prob_all)
    overall_auprc = average_precision_score(y_true_all, y_prob_all)
    print(f"\n--- LOSO Overall Performance ---")
    print(f"Accuracy: {overall_acc:.3f}")
    print(f"F1-Score (weighted): {overall_f1w:.3f}")
    print(f"AUROC: {overall_auroc:.3f} | AUPRC: {overall_auprc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=['Neutral', 'Stress']))
    print("Confusion Matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_true_all, y_pred_all),
        index=['True Neutral', 'True Stress'],
        columns=['Pred Neutral', 'Pred Stress']
    ))

    # --- Basic Statistical Performance Analysis ---
    statistical_performance_analysis(full_dataset, np.array(y_true_all), np.array(y_pred_all))

    # --- Final Model Training on ALL Data ---
    print(f"\n--- Training Final {('RandomForest' if model_type=='rf' else 'CNN-LSTM')} on ALL Data for Production ---")
    # Standardize globally for final model artifact to match evaluation preprocessing
    final_scaler = StandardScaler()
    X_std = final_scaler.fit_transform(X)
    if model_type == 'rf':
        final_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        final_model.fit(X_std, y)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        final_model = CNNLSTM(
            input_channels=X.shape[1],
            seq_len=1,
            num_features=X.shape[1],
            num_classes=2
        )
        final_model.to(device)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        # Small holdout for sanity
        X_tr_all, X_val_all, y_tr_all, y_val_all = train_test_split(
            X_std, y, test_size=0.1, random_state=42, stratify=y
        )
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_tr_all, dtype=torch.float32).unsqueeze(1),
                torch.tensor(
                    y_tr_all.values if hasattr(y_tr_all, 'values') else y_tr_all,
                    dtype=torch.long
                )
            ),
            batch_size=64,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val_all, dtype=torch.float32).unsqueeze(1),
                torch.tensor(
                    y_val_all.values if hasattr(y_val_all, 'values') else y_val_all,
                    dtype=torch.long
                )
            ),
            batch_size=256,
            shuffle=False
        )
        final_model.train()
        for _ in range(15):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = final_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    # Save the model artifact to the specified path
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        if model_type == 'rf':
            pickle.dump({'scaler': final_scaler, 'model': final_model, 'features': FEATURES_ALL}, f)
        else:
            pickle.dump({
                'scaler': final_scaler,
                'model_state_dict': final_model.state_dict(),
                'model_params': {
                    'input_channels': X.shape[1],
                    'seq_len': 1,
                    'num_features': X.shape[1],
                    'num_classes': 2
                },
                'features': FEATURES_ALL
            }, f)

    print(f"\nProduction model trained and saved to: {model_save_path}")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train stress prediction model with LOSO CV / learning curves / early warning')
    parser.add_argument('--data', type=str, default="data/processed/merged_features_dataset.csv",
                        help='Path to processed features CSV')
    parser.add_argument('--out', type=str, default=None,
                        help='Path to save trained model .pkl (ignored for early_warning or curve unless needed)')
    parser.add_argument('--model', type=str, choices=['rf', 'cnn'], default='cnn',
                        help='Model type to use')
    parser.add_argument('--curve', action='store_true',
                        help='Run F1 vs training fraction curve instead of LOSO training')
    parser.add_argument('--curve_out', type=str, default='f1_vs_fraction.png',
                        help='Output path for learning curve plot')
    parser.add_argument('--early_warning', action='store_true',
                        help='Run early-warning LOSO analysis instead of standard LOSO training')
    parser.add_argument('--early_horizons', type=str, default='1,3,5',
                        help='Comma-separated integer horizons (in steps) for early-warning prediction, e.g. "1,3,5"')
    parser.add_argument('--early_out', type=str, default='early_warning_results.csv',
                        help='Output CSV for early-warning metrics')

    args = parser.parse_args()

    # Resolve paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(current_dir, 'data', 'processed', 'merged_features_dataset.csv')
    default_out = 'stress_model_rf.pkl' if args.model == 'rf' else 'stress_model_cnn.pkl'
    model_out = args.out or os.path.join(current_dir, 'models', default_out)

    # Dispatch
    if args.curve:
        # Learning curve mode
        learning_curve_f1_vs_fraction(
            data_path=data_path,
            model_type=args.model,
            output_plot=args.curve_out
        )
    elif args.early_warning:
        # Early-warning LOSO mode
        horizons = [int(h.strip()) for h in args.early_horizons.split(',') if h.strip()]
        run_early_warning(
            data_path=data_path,
            horizons=horizons,
            output_csv=args.early_out
        )
    else:
        # Standard LOSO training + final production model
        train_stress_model(data_path, model_out, model_type=args.model)
