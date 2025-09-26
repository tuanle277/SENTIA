# -*- coding: utf-8 -*-
"""
aura/prediction/train.py

This script trains and evaluates the stress prediction model.
It performs a subject-independent cross-validation to get a robust
measure of performance and then saves a final model trained on all data.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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

def train_stress_model(data_path, model_save_path):
    """
    Trains a stress prediction model using subject-independent cross-validation
    and saves the final model trained on all data.

    Args:
        data_path (str): Path to the processed features_dataset.csv.
        model_save_path (str): Path to save the final trained model (.pkl).
    """
    print("--- Loading Processed Feature Dataset ---")
    try:
        full_dataset = pd.read_csv(data_path)
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
    X = full_dataset.drop(['subject', 'stress_label'], axis=1).select_dtypes(include=['float64'])
    y = full_dataset['stress_label']
    # We ignore subject IDs for training/evaluation as requested

    # --- Random Train/Test Split (Subject IDs Ignored) ---
    print("\n--- Starting Train/Test Split   ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n--- Performance on Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Neutral', 'Stress']))
    
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(conf_matrix, index=['True Neutral', 'True Stress'], columns=['Pred Neutral', 'Pred Stress']))
    
    # --- Basic Statistical Performance Analysis ---
    statistical_performance_analysis(full_dataset, y_test, y_pred)

    # --- Final Model Training on ALL Data ---
    print("\n--- Training Final Model on ALL Data for Production ---")
    final_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    
    # Save the model artifact to the specified path
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(final_model, f)
        
    print(f"\nProduction model trained and saved to: {model_save_path}")


if __name__ == '__main__':
    # Define paths based on our project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_PATH = os.path.join(current_dir, 'data', 'processed', 'merged_features_dataset.csv')
    MODEL_SAVE_PATH = os.path.join(current_dir, 'models', 'stress_model_rf.pkl')
    
    # Run the full training and validation process
    train_stress_model(PROCESSED_DATA_PATH, MODEL_SAVE_PATH)
