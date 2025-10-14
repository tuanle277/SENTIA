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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from cnn_lstm import CNNLSTM
from datetime import datetime

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
    # features_to_keep = [
    #     "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD",
    #     "HRV_MedianNN", "HRV_MadNN", "HRV_MCVNN", "HRV_IQRNN", "HRV_SDRMSSD", "HRV_Prc20NN",
    #     "HRV_Prc80NN", "HRV_pNN50", "HRV_pNN20", "HRV_MinNN", "HRV_MaxNN", "HRV_HTI", "HRV_TINN",
    #     "HRV_SDANN1", "HRV_SDNNI1", "HRV_SDANN2", "HRV_SDNNI2", "HRV_SDANN5", "HRV_SDNNI5",
    #     "EDA_Mean", "SCR_Peaks_N", "TEMP_Mean", "TEMP_Std", "ACC_Mag_Mean", "ACC_Mag_Std"
    # ]
    features_to_keep = [
        "HRV_MeanNN", 
        "EDA_Mean",
        "TEMP_Mean",
        "ACC_Mag_Mean"
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

        if model_type in ['rf', 'lr', 'svm', 'knn', 'dt']:
            if model_type == 'rf':
                model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
            elif model_type == 'lr':
                model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            elif model_type == 'svm':
                model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=False, random_state=42)
            elif model_type == 'knn':
                model = KNeighborsClassifier(n_neighbors=15, weights='distance')
            elif model_type == 'dt':
                model = DecisionTreeClassifier(random_state=42)
            model.fit(X_sub, y_sub)
            y_pred = model.predict(X_val_std)
        else:
            # CNN-LSTM on tabular features as seq_len=1
            model = CNNLSTM(input_channels=X_sub.shape[1], seq_len=5, num_features=X_sub.shape[1], num_classes=2)
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
    title_map = {
        'rf': 'Random Forest',
        'lr': 'Logistic Regression',
        'svm': 'SVM (RBF)',
        'knn': 'KNN',
        'cnn': 'CNN-LSTM'
    }
    plt.title(f'F1 vs Training Fraction ({title_map.get(model_type, model_type)})')
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

def save_results_to_csv(per_subject_results, overall_metrics, model_type, output_dir="results"):
    """
    Save per-subject and overall results to CSV files with model name in filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save per-subject results
    per_subject_df = pd.DataFrame(per_subject_results)
    per_subject_file = os.path.join(output_dir, f"per_subject_results_{model_type}_{timestamp}.csv")
    per_subject_df.to_csv(per_subject_file, index=False)
    print(f"Per-subject results saved to: {per_subject_file}")
    
    # Save overall metrics
    # Allow optional dispersion metrics if provided in overall_metrics
    overall_entry = {
        'model_type': model_type,
        'timestamp': timestamp,
        'total_subjects': len(per_subject_results),
        'accuracy': overall_metrics['accuracy'],
        'precision': overall_metrics['precision'],
        'recall': overall_metrics['recall'],
        'f1_score': overall_metrics['f1_score']
    }
    # Extend with per-subject variance/std if present
    for key in [
        'per_subject_accuracy_variance', 'per_subject_accuracy_std',
        'per_subject_f1_variance', 'per_subject_f1_std'
    ]:
        if key in overall_metrics:
            overall_entry[key] = overall_metrics[key]
    overall_df = pd.DataFrame([overall_entry])
    overall_file = os.path.join(output_dir, f"overall_results_{model_type}_{timestamp}.csv")
    overall_df.to_csv(overall_file, index=False)
    print(f"Overall results saved to: {overall_file}")
    
    return per_subject_file, overall_file

def train_stress_model(data_path, model_save_path, model_type: str = 'cnn'):
    """
    Trains a stress prediction model using subject-independent cross-validation
    and saves the final model trained on all data.

    Args:
        data_path (str): Path to the processed features_dataset.csv.
        model_save_path (str): Path to save the final trained model (.pkl).
        model_type (str): Type of model to train ('rf', 'lr', 'svm', 'knn', 'dt', 'cnn').
    """
    print("--- Loading Processed Feature Dataset ---")
    try:
        full_dataset = pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The file {data_path} was not found.")
        return
    print(data_path)
    print(full_dataset)
    if full_dataset.empty:
        print("Error: The feature dataset is empty. Cannot train the model.")
        return

    print(f"Dataset loaded successfully with shape: {full_dataset.shape}")

    # Define features and target
    features_to_keep = ["HRV_MeanNN", "EDA_Mean", "TEMP_Mean", "ACC_Mag_Mean"]
    for col in features_to_keep:
        if col in full_dataset.columns:
            full_dataset[col] = pd.to_numeric(full_dataset[col], errors='coerce')
        else:
            print(f"Warning: Expected feature column '{col}' not found in dataset.")

    if 'stress_label' not in full_dataset.columns:
        print("Error: 'stress_label' column not found in dataset. Cannot train the model.")
        return
    full_dataset['stress_label'] = pd.to_numeric(full_dataset['stress_label'], errors='coerce')
    
    base = full_dataset[features_to_keep + ['stress_label']].dropna()
    X = base[features_to_keep].astype('float64')
    y = base['stress_label'].astype(int)

    # Train the final model on all data
    print(f"\n--- Training Final Model ({model_type}) on ALL Data ---")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    if model_type == 'dt':  # Decision Tree
        final_model = DecisionTreeClassifier(random_state=42)
        final_model.fit(X_std, y)

        # Extract decision rules
        print("\n--- Extracting Decision Rules ---")
        rules = export_text(final_model, feature_names=features_to_keep)
        print(rules)

        # Save decision rules to a text file
        rules_file = model_save_path.replace('.pkl', '_decision_rules.txt')
        with open(rules_file, 'w') as f:
            f.write(rules)
        print(f"Decision rules saved to: {rules_file}")
    else:
        # Other models
        if model_type == 'rf':
            final_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        elif model_type == 'lr':
            final_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        elif model_type == 'svm':
            final_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=False, random_state=42)
        elif model_type == 'knn':
            final_model = KNeighborsClassifier(n_neighbors=15, weights='distance')
        elif model_type == 'cnn':
            # CNN-LSTM logic here (omitted for brevity)
            pass
        final_model.fit(X_std, y)

    # Save the model artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'model': final_model, 'features': features_to_keep}, f)
    print(f"Model saved to: {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train stress prediction model with LOSO CV')
    parser.add_argument('--data', type=str, default="data/processed/merged_features_dataset.csv", help='Path to processed features CSV')
    parser.add_argument('--out', type=str, default="models/stress_model_features.pkl", help='Path to save trained model .pkl')
    parser.add_argument('--model', type=str, choices=['rf', 'lr', 'svm', 'knn', 'cnn', 'all', 'dt'], default='cnn', help='Model type to use')
    parser.add_argument('--curve', action='store_true', help='Run F1 vs training fraction curve instead of LOSO training')
    parser.add_argument('--curve_out', type=str, default='f1_vs_fraction.png', help='Output path for learning curve plot')
    args = parser.parse_args()

    # Define paths based on our project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(current_dir, 'data', 'processed', 'merged_features_dataset.csv')
    default_out = f"stress_model_{args.model}_few_features.pkl"
    model_out = os.path.join(current_dir, 'models', default_out)
    
    # Either run curve or full LOSO training
    if args.model == 'all' and not args.curve:
        for m in ['rf', 'lr', 'svm', 'knn', 'cnn', 'dt']:
            print(f"\n\n=== Training and Evaluating Model Type: {m} ===")
            out_path = model_out.replace('.pkl', f'_{m}_4_features.pkl')
            train_stress_model(data_path, out_path, model_type=m)
    else:
        if args.curve:
            learning_curve_f1_vs_fraction(data_path=data_path, model_type=args.model, output_plot=args.curve_out)
        else:
            train_stress_model(data_path, model_out, model_type=args.model)
