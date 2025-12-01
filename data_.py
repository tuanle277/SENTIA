import os
import pickle
import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
import warnings
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Suppress expected NeuroKitWarnings for short-signal processing or low-frequency data
from neurokit2.misc import NeuroKitWarning
warnings.filterwarnings("ignore", category=NeuroKitWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
WINDOW_SIZE_SEC = 30
WINDOW_SHIFT_SEC = 1

# --- PART 1: WESAD DATA PROCESSING ---
# This function is the "worker" for a single WESAD subject. It remains mostly unchanged.

def process_wesad_subject(subject_id, base_path):
    """
    Loads WESAD raw data for a single subject, slides a window over the signals,
    and extracts a feature vector for each window.
    """
    file_path = os.path.join(base_path, f"S{subject_id}", f"S{subject_id}.pkl")
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
    except FileNotFoundError:
        print(f"Warning: WESAD file for subject {subject_id} not found at {file_path}")
        return []

    wrist_signals = data['signal']['wrist']
    labels = data['label']
    
    fs = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700}
    master_fs = fs['label']
    window_size = master_fs * WINDOW_SIZE_SEC
    window_shift = master_fs * WINDOW_SHIFT_SEC

    subject_features_list = []
    
    for i in range(0, len(labels) - window_size, window_shift):
        window_start = i
        window_end = i + window_size
        
        window_labels = labels[window_start:window_end]
        dominant_label = np.bincount(window_labels.flatten()).argmax()

        if dominant_label not in [1, 2, 3]:
            continue
        
        target_label = 0 if dominant_label in [1, 3] else 1
        features = {}
        
        try:
            bvp_start, bvp_end = (window_start * fs['BVP']) // master_fs, (window_end * fs['BVP']) // master_fs
            bvp_window = wrist_signals['BVP'][bvp_start:bvp_end].flatten()
            if np.std(bvp_window) < 1: raise ValueError("BVP signal flat")
            
            peaks_info = nk.ppg_findpeaks(bvp_window, sampling_rate=fs['BVP'])
            if len(peaks_info['PPG_Peaks']) < 15: raise ValueError("Not enough PPG peaks")
            
            hrv_indices = nk.hrv(peaks_info, sampling_rate=fs['BVP'], show=False)
            features.update({f"HRV_{k}": v for k, v in hrv_indices.to_dict('records')[0].items()})
        except Exception:
            continue

        try:
            eda_start, eda_end = (window_start * fs['EDA']) // master_fs, (window_end * fs['EDA']) // master_fs
            eda_window = wrist_signals['EDA'][eda_start:eda_end].flatten()
            if np.std(eda_window) < 0.01: raise ValueError("EDA signal flat")
            eda_signals, _ = nk.eda_process(eda_window, sampling_rate=fs['EDA'])
            features['EDA_Mean'] = eda_signals['EDA_Clean'].mean()
            features['SCR_Peaks_N'] = eda_signals['SCR_Peaks'].sum()
        except Exception:
            features['EDA_Mean'], features['SCR_Peaks_N'] = np.nan, np.nan

        acc_start, acc_end = (window_start * fs['ACC']) // master_fs, (window_end * fs['ACC']) // master_fs
        acc_mag = np.sqrt(np.sum(wrist_signals['ACC'][acc_start:acc_end]**2, axis=1))
        features['ACC_Mag_Mean'], features['ACC_Mag_Std'] = acc_mag.mean(), acc_mag.std()

        temp_start, temp_end = (window_start * fs['TEMP']) // master_fs, (window_end * fs['TEMP']) // master_fs
        features['TEMP_Mean'] = wrist_signals['TEMP'][temp_start:temp_end].mean()

        features['subject'] = f"wesad_s{subject_id}"
        features['label'] = target_label
        features['dataset'] = 'WESAD'
        subject_features_list.append(features)
        
    return subject_features_list

# --- PART 2: DREAMER DATA PROCESSING ---
# These functions are "workers" for a single DREAMER trial.

def _extract_dreamer_features_from_df(trial_df):
    """Core logic to extract features from a DREAMER trial DataFrame."""
    fs = 128
    window_size = fs * WINDOW_SIZE_SEC
    window_shift = fs * WINDOW_SHIFT_SEC
    
    trial_features = []
    valence, arousal, dominance = trial_df[['Valence', 'Arousal', 'Dominance']].iloc[0]
    is_stress = 1 if arousal >= 3 and valence <= 2 else 0
    
    for i in range(0, len(trial_df) - window_size, window_shift):
        window_df = trial_df.iloc[i:i+window_size]
        features = {}

        try:
            ecg_window = window_df['ECG_channel_1'].values
            if np.std(ecg_window) < 1: raise ValueError("ECG signal flat")
            
            ecg_signals, info = nk.ecg_process(ecg_window, sampling_rate=fs)
            if len(info['ECG_R_Peaks']) < 15: raise ValueError("Not enough R-peaks")

            hrv_indices = nk.hrv(ecg_signals, sampling_rate=fs, show=False)
            features.update({f"HRV_{k}": v for k, v in hrv_indices.to_dict('records')[0].items()})
        except Exception:
            continue

        try:
            eeg_channels = [f'EEG_channel_{i+1}' for i in range(14)]
            power_bands = nk.eeg_power(window_df[eeg_channels], sampling_rate=fs, frequency_bands='common')
            for band in power_bands.columns:
                features[f"EEG_{band.replace(' (', '_').replace(')', '')}"] = power_bands[band].mean()
        except Exception:
            continue

        features.update({'Valence': valence, 'Arousal': arousal, 'Dominance': dominance, 'label': is_stress, 'dataset': 'DREAMER'})
        trial_features.append(features)
        
    return trial_features

def process_dreamer_file(file_path):
    """Wrapper function to load a DREAMER CSV and process it."""
    try:
        filename = os.path.basename(file_path)
        sid = int(filename.split('_')[1][1:])
        tid = int(filename.split('_')[2][1:].replace('.csv', ''))

        trial_df = pd.read_csv(file_path)
        features_list = _extract_dreamer_features_from_df(trial_df)
        
        for feat in features_list:
            feat['subject'] = f"dreamer_s{sid}"
            feat['trial'] = tid
        return features_list
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# --- PART 3: MAIN EXECUTION AND COMBINATION ---

if __name__ == '__main__':
    WESAD_BASE_PATH = "./data/WESAD"
    DREAMER_BASE_PATH = "./data/processed"
    
    # Initialize empty DataFrames
    wesad_df = pd.DataFrame()
    dreamer_df = pd.DataFrame()

    # --- Process WESAD Dataset in Parallel ---
    print("--- Starting WESAD Feature Extraction ---")
    WESAD_SUBJECT_IDS = [x for x in range(2, 18) if x != 12]
    # Use partial to pre-fill the 'base_path' argument for our worker function
    wesad_worker = partial(process_wesad_subject, base_path=WESAD_BASE_PATH)
    
    with ProcessPoolExecutor() as executor:
        # executor.map runs the worker function on each subject ID in parallel
        results = list(tqdm(executor.map(wesad_worker, WESAD_SUBJECT_IDS), total=len(WESAD_SUBJECT_IDS), desc="Processing WESAD Subjects"))
    
    # Flatten the list of lists returned by the parallel processes
    wesad_features = [item for sublist in results for item in sublist]
    if wesad_features:
        wesad_df = pd.DataFrame(wesad_features)
    print(f"Extracted {len(wesad_df)} feature windows from WESAD.")

    # --- Process DREAMER Dataset in Parallel ---
    print("\n--- Starting DREAMER Feature Extraction ---")
    # Use glob to find all trial files automatically
    dreamer_files = glob.glob(os.path.join(DREAMER_BASE_PATH, "dreamer_S*_T*.csv"))
    
    if dreamer_files:
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_dreamer_file, dreamer_files), total=len(dreamer_files), desc="Processing DREAMER Trials"))
        
        dreamer_features = [item for sublist in results for item in sublist]
        if dreamer_features:
            dreamer_df = pd.DataFrame(dreamer_features)
    print(f"Extracted {len(dreamer_df)} feature windows from DREAMER.")

    # --- Combine Datasets ---
    print("\n--- Combining Datasets ---")
    if wesad_df.empty and dreamer_df.empty:
        print("Both datasets are empty. No features were extracted. Exiting.")
    else:
        combined_df = pd.concat([wesad_df, dreamer_df], ignore_index=True)
        
        print("\n--- Unified Dataset Report ---")
        print(f"Final Dataset Shape: {combined_df.shape}")
        
        # Correctly find exclusive columns
        wesad_cols = set(wesad_df.columns) if not wesad_df.empty else set()
        dreamer_cols = set(dreamer_df.columns) if not dreamer_df.empty else set()
        print("\nColumns exclusive to WESAD:", wesad_cols - dreamer_cols)
        print("Columns exclusive to DREAMER:", dreamer_cols - wesad_cols)
        
        if 'label' in combined_df.columns:
            print("\nLabel Distribution in Combined Dataset (0=Neutral, 1=Stress):")
            print(combined_df['label'].value_counts(normalize=True))
        
        print("\nSum of NaN values per column:")
        print(combined_df.isnull().sum().sort_values(ascending=False))

        output_path = "./data/combined_features_dataset.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\nUnified feature dataset saved successfully to: {output_path}")