# -*- coding: utf-8 -*-
"""
aura/data_processing/feature_extractor.py

This is the final, production-ready script to process the raw WESAD dataset 
for all subjects and generate a clean feature dataset for model training.
"""

import os
import pickle
import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
import warnings

# Suppress the expected NeuroKitWarning for low-frequency EDA data
from neurokit2.misc import NeuroKitWarning

warnings.filterwarnings("ignore", category=NeuroKitWarning)

def process_subject_data(subject_id, base_path, window_size_sec=15, window_shift_sec=1):
    """
    Loads raw data for a single subject, slides a window over the signals,
    and extracts a feature vector for each window.
    """
    file_path = os.path.join(base_path, f"S{subject_id}", f"S{subject_id}.pkl")
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    wrist_signals = data['signal']['wrist']
    labels = data['label']
    
    fs = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700}
    master_fs = fs['label']
    window_size = master_fs * window_size_sec
    window_shift = master_fs * window_shift_sec

    subject_features_list = []
    
    print(f"Subject {subject_id}: Total labels: {len(labels)}")
    print(f"Window size: {window_size}, Window shift: {window_shift}")
    print(f"Number of possible windows: {len(range(0, len(labels) - window_size, window_shift))}")

    windows_processed = 0
    windows_with_valid_labels = 0
    windows_with_valid_bvp = 0
    
    for i in tqdm(range(0, len(labels) - window_size, window_shift), desc=f"Processing S{subject_id}", leave=False):
        window_start = i
        window_end = i + window_size
        windows_processed += 1
        
        window_labels = labels[window_start:window_end]
        dominant_label = np.bincount(window_labels.flatten()).argmax()

        if dominant_label not in [1, 2, 3]: # Include amusement (3) as neutral
            continue
        
        windows_with_valid_labels += 1
        
        # Group baseline (1) and amusement (3) as 'neutral' (0)
        target_label = 0 if dominant_label in [1, 3] else 1
        
        features = {}
        
        # --- BVP / HRV Features ---
        try:
            bvp_start = (window_start * fs['BVP']) // master_fs
            bvp_end = (window_end * fs['BVP']) // master_fs
            bvp_window = wrist_signals['BVP'][bvp_start:bvp_end].flatten()
            
            # Basic quality check
            if np.std(bvp_window) < 1: raise ValueError("BVP signal is flat")
            
            ppg_peaks, _ = nk.ppg_peaks(bvp_window, sampling_rate=fs['BVP'])
            
            if len(ppg_peaks['PPG_Peaks']) < 10: raise ValueError("Not enough BVP peaks")
                
            hrv_indices = nk.hrv_time(ppg_peaks, sampling_rate=fs['BVP'], show=False)
            hrv_features = {f"BVP_{k}": v for k, v in hrv_indices.to_dict('records')[0].items()}
            # Only keep HRV features that are not NaN
            hrv_features_clean = {k: v for k, v in hrv_features.items() if not pd.isna(v)}
            features.update(hrv_features_clean)
            windows_with_valid_bvp += 1
        except Exception as e:
            # If BVP fails, we can't get HRV, so we must skip the window.
            continue

        # --- EDA Features ---
        try:
            eda_start = (window_start * fs['EDA']) // master_fs
            eda_end = (window_end * fs['EDA']) // master_fs
            eda_window = wrist_signals['EDA'][eda_start:eda_end].flatten()
            
            if np.std(eda_window) < 0.01: raise ValueError("EDA signal is flat")

            eda_signals, _ = nk.eda_process(eda_window, sampling_rate=fs['EDA'])
            features['EDA_Mean'] = eda_signals['EDA_Clean'].mean()
            features['SCR_Peaks_N'] = eda_signals['SCR_Peaks'].sum()
        except Exception as e:
            # If EDA fails, fill with NaN. We can decide to drop these later.
            features['EDA_Mean'] = np.nan
            features['SCR_Peaks_N'] = np.nan

        # --- ACC & TEMP Features ---
        acc_start = (window_start * fs['ACC']) // master_fs
        acc_end = (window_end * fs['ACC']) // master_fs
        acc_window_3d = wrist_signals['ACC'][acc_start:acc_end]
        acc_mag = np.sqrt(np.sum(acc_window_3d**2, axis=1))
        features['ACC_Mag_Mean'] = acc_mag.mean()
        features['ACC_Mag_Std'] = acc_mag.std()

        temp_start = (window_start * fs['TEMP']) // master_fs
        temp_end = (window_end * fs['TEMP']) // master_fs
        temp_window = wrist_signals['TEMP'][temp_start:temp_end].flatten()
        features['TEMP_Mean'] = temp_window.mean()
        features['TEMP_Std'] = temp_window.std()

        features['subject'] = subject_id
        features['label'] = target_label
        subject_features_list.append(features)

    print(f"Subject {subject_id} Summary:")
    print(f"  Windows processed: {windows_processed}")
    print(f"  Windows with valid labels: {windows_with_valid_labels}")
    print(f"  Windows with valid BVP: {windows_with_valid_bvp}")
    print(f"  Final features extracted: {len(subject_features_list)}")
    
    return subject_features_list

# --- Main script execution block ---
if __name__ == '__main__':
    WESAD_BASE_PATH = "./WESAD"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "features_dataset.csv")

    SUBJECT_IDS = [x for x in range(2, 18) if x != 12]
    all_subjects_features_list = []

    print("Starting final feature extraction for all subjects...")
    for sid in SUBJECT_IDS:
        subject_features = process_subject_data(sid, WESAD_BASE_PATH)
        all_subjects_features_list.extend(subject_features)
        
    final_dataset = pd.DataFrame(all_subjects_features_list)
    
    print("\n--- Feature Extraction Complete! ---")
    print("\nFinal Feature Dataset Head:")
    print(final_dataset.head())
    print(f"\nDataset Shape: {final_dataset.shape}")
    print(f"\nLabel Distribution (0=Neutral, 1=Stress):\n{final_dataset['label'].value_counts(normalize=True)}")
    print(f"\nNaN values per column:")
    print(final_dataset.isnull().sum())
    
    # Now, we drop any rows that have missing values from failed EDA processing
    # Only drop rows where EDA features are NaN (the ones that actually failed)
    final_dataset_clean = final_dataset.dropna(subset=['EDA_Mean', 'SCR_Peaks_N'])
    print(f"\nDataset Shape after dropping EDA NaN: {final_dataset_clean.shape}")
    print(f"\nLabel Distribution after cleaning (0=Neutral, 1=Stress):\n{final_dataset_clean['label'].value_counts(normalize=True)}")
    
    final_dataset_clean.to_csv(output_filename, index=False)
    print(f"\nFeature dataset saved successfully to: {output_filename}")

