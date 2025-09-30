# -*- coding: utf-8 -*-
"""
aura/prediction/train.py

This script trains and evaluates the stress prediction model.
It performs a subject-independent cross-validation to get a robust
measure of performance and then saves a final model trained on all data.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from cnn_lstm import CNNLSTM

def learning_curve_f1_vs_fraction(data_path: str, model_type: str = 'rf', output_plot: str = 'f1_vs_fraction.png'):
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

    # Define features consistent with main training
    features_to_keep = [
        "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD",
        "HRV_MedianNN", "HRV_MadNN", "HRV_MCVNN", "HRV_IQRNN", "HRV_SDRMSSD", "HRV_Prc20NN",
        "HRV_Prc80NN", "HRV_pNN50", "HRV_pNN20", "HRV_MinNN", "HRV_MaxNN", "HRV_HTI", "HRV_TINN",
        "HRV_SDANN1", "HRV_SDNNI1", "HRV_SDANN2", "HRV_SDNNI2", "HRV_SDANN5", "HRV_SDNNI5",
        "EDA_Mean", "SCR_Peaks_N", "TEMP_Mean", "TEMP_Std", "ACC_Mag_Mean", "ACC_Mag_Std"
    ]

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

    fractions = [i/100 for i in range(10, 100, 10)]  # 0.1 to 0.9
    f1_scores = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for frac in fractions:
        n = max(1, int(len(X_train_std) * frac))
        X_sub = X_train_std[:n]
        y_sub = y_train[:n]

        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
            model.fit(X_sub, y_sub)
            y_pred = model.predict(X_val_std)
        else:
            # CNN-LSTM on tabular features as seq_len=1
            model = CNNLSTM(input_channels=X_sub.shape[1], seq_len=1, num_features=X_sub.shape[1], num_classes=2)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_sub, dtype=torch.float32).unsqueeze(1),
                              torch.tensor(y_sub, dtype=torch.long)),
                batch_size=64, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.tensor(X_val_std, dtype=torch.float32).unsqueeze(1),
                              torch.tensor(y_val, dtype=torch.long)),
                batch_size=256, shuffle=False
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
    plt.figure(figsize=(7,4))
    plt.plot([f*100 for f in fractions], f1_scores, marker='o')
    plt.xlabel('Training fraction (%)')
    plt.ylabel('F1 score (weighted)')
    plt.title(f'F1 vs Training Fraction ({"RF" if model_type=="rf" else "CNN-LSTM"})')
    plt.grid(True, linestyle='--', alpha=0.4)
    os.makedirs(os.path.dirname(output_plot) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Saved learning curve plot to: {output_plot}")

def statistical_performance_analysis(full_dataset, y_true, y_pred):
    """
    Performs basic statistical analysis of model performance ignoring subject IDs.
    """
    print("\n" + "="*80)
    print("📊 PERFORMANCE ANALYSIS (No Subject Grouping)")
    print("="*80)
    
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy: {overall_accuracy:.3f}")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1-Score: {overall_f1:.3f}")
    print("\n" + "="*80)
    
    return {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1
    }

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

    # Define features (X), target (y), and groups for cross-validation
    # We drop 'subject' and 'label' to create our feature set X, drop columns that are not float
    # We only get wearable features
    features_to_keep = [
        "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD", 
        "HRV_MedianNN", "HRV_MadNN", "HRV_MCVNN", "HRV_IQRNN", "HRV_SDRMSSD", "HRV_Prc20NN", 
        "HRV_Prc80NN", "HRV_pNN50", "HRV_pNN20", "HRV_MinNN", "HRV_MaxNN", "HRV_HTI", "HRV_TINN", 
        "HRV_SDANN1", "HRV_SDNNI1", "HRV_SDANN2", "HRV_SDNNI2", "HRV_SDANN5", "HRV_SDNNI5", 
        "EDA_Mean", "SCR_Peaks_N", "TEMP_Mean", "TEMP_Std", "ACC_Mag_Mean", "ACC_Mag_Std"
    ]

    wearable_features = []
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

    # --- Subject-wise Baseline Shift using Neutral (label==0) windows ---
    if 'subject' in valid.columns:
        print("Applying subject-wise baseline shift using neutral windows (label=0)...")
        neutral_mask = valid['stress_label'] == 0
        if neutral_mask.any():
            baseline_means = valid.loc[neutral_mask].groupby('subject')[features_to_keep].mean()
            # Join baseline means back to rows by subject
            valid = valid.join(baseline_means, on='subject', rsuffix='_baseline')
            # Subtract baseline per feature
            for feat in features_to_keep:
                base_col = f"{feat}_baseline"
                if base_col in valid.columns:
                    valid[feat] = valid[feat] - valid[base_col]
            # Drop helper baseline columns
            drop_cols = [f"{feat}_baseline" for feat in features_to_keep if f"{feat}_baseline" in valid.columns]
            if drop_cols:
                valid = valid.drop(columns=drop_cols)
        else:
            print("Warning: No neutral (label=0) windows found; skipping baseline shift.")
    else:
        print("Warning: 'subject' column not found; skipping baseline shift.")

    # Final X and y after baseline shift
    X = valid[features_to_keep].astype('float64')
    y = valid['stress_label'].astype(int)

    # --- Leave-One-Subject-Out Cross-Validation ---
    if 'subject' not in valid.columns and 'subject' not in full_dataset.columns:
        print("Error: 'subject' column not found. Required for Leave-One-Subject-Out CV.")
        return

    subjects_series = (valid['subject'] if 'subject' in valid.columns else full_dataset.loc[valid.index, 'subject']).astype(str)
    unique_subjects = subjects_series.unique()

    print(f"\n--- Starting Leave-One-Subject-Out CV across {len(unique_subjects)} subjects (model={model_type}) ---")
    y_true_all = []
    y_pred_all = []
    per_subject_results = []

    for subject_id in unique_subjects:
        test_idx = subjects_series == subject_id
        train_idx = ~test_idx

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fold-wise standardization (fit on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Skip subjects with no samples or single-class y in train (stratification not used here)
        if X_test.shape[0] == 0 or X_train.shape[0] == 0:
            print(f"Skipping subject {subject_id}: insufficient samples.")
            continue

        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            # Train CNN-LSTM on standardized features (seq_len=1)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = CNNLSTM(input_channels=X.shape[1], seq_len=1, num_features=X.shape[1], num_classes=2)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            # Dataloaders
            X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
            y_train_t = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
            y_test_t = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.long)
            train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False)

            model.train()
            for _ in range(10):
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

            # Evaluate fold
            model.eval()
            preds = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    preds.append(pred)
            y_pred = np.concatenate(preds) if len(preds) else np.array([])

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

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
            'f1_score': f1w
        })

        print(f"Subject {subject_id}: n={X_test.shape[0]} | Acc={acc:.3f} F1w={f1w:.3f}")

    # Overall evaluation across concatenated folds
    if len(y_true_all) == 0:
        print("No evaluation data collected. Aborting.")
        return

    print("\n--- LOSO Overall Performance ---")
    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_f1w = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    print(f"Accuracy: {overall_acc:.3f}")
    print(f"F1-Score (weighted): {overall_f1w:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=['Neutral', 'Stress']))
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_true_all, y_pred_all)
    print(pd.DataFrame(conf_matrix, index=['True Neutral', 'True Stress'], columns=['Pred Neutral', 'Pred Stress']))
    
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
        final_model = CNNLSTM(input_channels=X.shape[1], seq_len=1, num_features=X.shape[1], num_classes=2)
        final_model.to(device)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        # Small holdout for sanity
        X_tr_all, X_val_all, y_tr_all, y_val_all = train_test_split(X_std, y, test_size=0.1, random_state=42, stratify=y)
        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_all, dtype=torch.float32).unsqueeze(1), torch.tensor(y_tr_all.values if hasattr(y_tr_all, 'values') else y_tr_all, dtype=torch.long)), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_all, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val_all.values if hasattr(y_val_all, 'values') else y_val_all, dtype=torch.long)), batch_size=256, shuffle=False)
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
            pickle.dump({'scaler': final_scaler, 'model': final_model, 'features': features_to_keep}, f)
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
                'features': features_to_keep
            }, f)
        
    print(f"\nProduction model trained and saved to: {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train stress prediction model with LOSO CV')
    parser.add_argument('--data', type=str, default="data/processed/merged_features_dataset.csv", help='Path to processed features CSV')
    parser.add_argument('--out', type=str, default="models/stress_model_cnn.pkl", help='Path to save trained model .pkl')
    parser.add_argument('--model', type=str, choices=['rf', 'cnn'], default='cnn', help='Model type to use')
    parser.add_argument('--curve', action='store_true', help='Run F1 vs training fraction curve instead of LOSO training')
    parser.add_argument('--curve_out', type=str, default='f1_vs_fraction.png', help='Output path for learning curve plot')
    args = parser.parse_args()

    # Define paths based on our project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(current_dir, 'data', 'processed', 'merged_features_dataset.csv')
    default_out = 'stress_model_rf.pkl' if args.model == 'rf' else 'stress_model_cnn.pkl'
    model_out = args.out or os.path.join(current_dir, 'models', default_out)
    
    # Either run curve or full LOSO training
    if args.curve:
        learning_curve_f1_vs_fraction(data_path=data_path, model_type=args.model, output_plot=args.curve_out)
    else:
        train_stress_model(data_path, model_out, model_type=args.model)
