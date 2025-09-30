"""
General Dataset Processor for Multiple Affective Computing Datasets

This module processes the WESAD, SWELL, AffectiveROAD, StudentLife, and DREAMER datasets.
"""

import os
import pickle
import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import scipy.io
from pathlib import Path
from scipy.signal import resample, welch

# Suppress the expected NeuroKitWarning for low-frequency EDA data
from neurokit2.misc import NeuroKitWarning
warnings.filterwarnings("ignore", category=NeuroKitWarning)


class BaseDatasetProcessor(ABC):
    """
    Abstract base class for processing different affective computing datasets.
    Each dataset type should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, base_path: str, window_size_sec: int = 15, window_shift_sec: int = 1):
        """
        Initialize the dataset processor.
        
        Args:
            base_path: Path to the dataset directory
            window_size_sec: Size of the sliding window in seconds
            window_shift_sec: Shift of the sliding window in seconds
        """
        self.base_path = base_path
        self.window_size_sec = window_size_sec
        self.window_shift_sec = window_shift_sec
        self.features_list = []
        
    @abstractmethod
    def get_available_subjects(self) -> List[str]:
        """Return list of available subject IDs for this dataset."""
        pass
    
    @abstractmethod
    def load_subject_data(self, subject_id: str) -> Dict:
        """Load raw data for a specific subject."""
        pass
    
    @abstractmethod
    def extract_features_from_window(self, window_data: Dict, subject_id: str) -> Dict:
        """Extract features from a single window of data."""
        pass
    
    @abstractmethod
    def get_sampling_rates(self) -> Dict[str, int]:
        """Return sampling rates for different signal types."""
        pass
    
    @abstractmethod
    def get_master_sampling_rate(self) -> int:
        """Return the master sampling rate for label synchronization."""
        pass
    
    def process_subject(self, subject_id: str) -> List[Dict]:
        """
        Process a single subject and extract features from all valid windows.
        
        Args:
            subject_id: ID of the subject to process
            
        Returns:
            List of feature dictionaries for all valid windows
        """
        print(f"Processing subject {subject_id}...")
        
        # Load subject data
        subject_data = self.load_subject_data(subject_id)
        
        # Get sampling rates
        fs = self.get_sampling_rates()
        master_fs = self.get_master_sampling_rate()
        
        # Calculate window parameters
        window_size = master_fs * self.window_size_sec
        window_shift = master_fs * self.window_shift_sec
        
        # Get labels
        labels = subject_data.get('labels', [])
        if len(labels) == 0:
            print(f"No labels found for subject {subject_id}")
            return []
        
        subject_features_list = []
        windows_processed = 0
        windows_with_valid_labels = 0
        
        print(f"Subject {subject_id}: Total labels: {len(labels)}")
        print(f"Window size: {window_size}, Window shift: {window_shift}")
        print(f"Number of possible windows: {len(range(0, len(labels) - window_size, window_shift))}")
        
        for i in tqdm(range(0, len(labels) - window_size, window_shift), 
                     desc=f"Processing {subject_id}", leave=False):
            window_start = i
            window_end = i + window_size
            windows_processed += 1
            
            # Extract window data
            window_data = self._extract_window_data(subject_data, window_start, window_end, fs, master_fs)
            
            # Check if window has valid labels
            if not self._is_valid_window(window_data, labels[window_start:window_end]):
                continue
                
            windows_with_valid_labels += 1
            
            # Extract features from window
            try:
                features = self.extract_features_from_window(window_data, subject_id)
                if features:  # Only add if features were successfully extracted
                    subject_features_list.append(features)
            except Exception as e:
                print(f"Warning: Failed to extract features for window {i}: {e}")
                continue
        
        print(f"Subject {subject_id} Summary:")
        print(f"  Windows processed: {windows_processed}")
        print(f"  Windows with valid labels: {windows_with_valid_labels}")
        print(f"  Final features extracted: {len(subject_features_list)}")
        
        return subject_features_list
    
    def _extract_window_data(self, subject_data: Dict, window_start: int, window_end: int, 
                           fs: Dict[str, int], master_fs: int) -> Dict:
        """Extract data for a specific window from all signal types."""
        window_data = {}
        
        for signal_type, sampling_rate in fs.items():
            if signal_type == 'label':
                continue
                
            if signal_type in subject_data:
                signal_data = subject_data[signal_type]
                start_idx = (window_start * sampling_rate) // master_fs
                end_idx = (window_end * sampling_rate) // master_fs
                
                if isinstance(signal_data, np.ndarray):
                    if signal_data.ndim == 1:
                        window_data[signal_type] = signal_data[start_idx:end_idx]
                    else:
                        window_data[signal_type] = signal_data[start_idx:end_idx]
                else:
                    window_data[signal_type] = signal_data[start_idx:end_idx]
        
        return window_data
    
    def _is_valid_window(self, window_data: Dict, window_labels: np.ndarray) -> bool:
        """Check if a window has valid labels and data quality."""
        # Check label validity
        if len(window_labels) == 0:
            return False
            
        dominant_label = np.bincount(window_labels.flatten()).argmax()
        
        # Dataset-specific label validation (override in subclasses)
        return self._validate_labels(dominant_label)
    
    def _validate_labels(self, dominant_label: int) -> bool:
        """Validate if the dominant label is acceptable for this dataset."""
        # Default implementation - override in subclasses
        return True
    
    def process_all_subjects(self, subject_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process all subjects and return a combined feature dataset.
        
        Args:
            subject_ids: List of subject IDs to process. If None, processes all available subjects.
            
        Returns:
            DataFrame containing features from all subjects
        """
        if subject_ids is None:
            subject_ids = self.get_available_subjects()
        
        all_features = []
        
        print(f"Starting feature extraction for {len(subject_ids)} subjects...")
        for subject_id in subject_ids:
            try:
                subject_features = self.process_subject(subject_id)
                all_features.extend(subject_features)
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
                continue
        
        if not all_features:
            print("No features extracted from any subjects!")
            return pd.DataFrame()
        
        # Create DataFrame
        dataset = pd.DataFrame(all_features)
        
        print(f"\n--- Feature Extraction Complete! ---")
        print(f"Total features extracted: {len(dataset)}")
        print(f"Dataset shape: {dataset.shape}")
        
        if 'label' in dataset.columns:
            print(f"Label distribution:\n{dataset['label'].value_counts(normalize=True)}")
        
        # Check for missing values
        if dataset.isnull().any().any():
            print(f"\nMissing values per column:")
            print(dataset.isnull().sum())
        
        return dataset
    
    def save_dataset(self, dataset: pd.DataFrame, output_path: str) -> None:
        """Save the processed dataset to a CSV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")


class WESADProcessor(BaseDatasetProcessor):
    """Processor for WESAD dataset."""
    
    def get_available_subjects(self) -> List[str]:
        """Get available WESAD subject IDs."""
        subjects = []
        for i in range(2, 18):
            if i != 12:  # Subject 12 is missing
                subject_path = os.path.join(self.base_path, f"S{i}", f"S{i}.pkl")
                if os.path.exists(subject_path):
                    subjects.append(f"S{i}")
        return subjects
    
    def load_subject_data(self, subject_id: str) -> Dict:
        """Load WESAD subject data from pickle file."""
        file_path = os.path.join(self.base_path, subject_id, f"{subject_id}.pkl")
        
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        
        # Extract wrist signals and labels
        wrist_signals = data['signal']['wrist']
        labels = data['label']
        
        return {
            'ACC': wrist_signals['ACC'],
            'BVP': wrist_signals['BVP'],
            'EDA': wrist_signals['EDA'],
            'TEMP': wrist_signals['TEMP'],
            'labels': labels
        }
    
    def get_sampling_rates(self) -> Dict[str, int]:
        """Return WESAD sampling rates."""
        return {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700}
    
    def get_master_sampling_rate(self) -> int:
        """Return WESAD master sampling rate."""
        return 700
    
    def _validate_labels(self, dominant_label: int) -> bool:
        """Validate WESAD labels (1=baseline, 2=stress, 3=amusement)."""
        return dominant_label in [1, 2, 3]
    
    def extract_features_from_window(self, window_data: Dict, subject_id: str) -> Dict:
        """Extract features from a WESAD window."""
        features = {}
        
        # Get window labels for label assignment
        # Note: This is a simplified approach - in practice, you'd need to pass labels
        # through the window data or handle them differently
        target_label = 1  # Default to stress - this should be calculated from actual labels
        
        # --- BVP / HRV Features ---
        try:
            bvp_window = window_data['BVP'].flatten()
            
            # Basic quality check
            if np.std(bvp_window) < 1:
                raise ValueError("BVP signal is flat")
            
            ppg_peaks, _ = nk.ppg_peaks(bvp_window, sampling_rate=64)
            
            if len(ppg_peaks['PPG_Peaks']) < 10:
                raise ValueError("Not enough BVP peaks")
            
            hrv_indices = nk.hrv_time(ppg_peaks, sampling_rate=64, show=False)
            hrv_features = {f"BVP_{k}": v for k, v in hrv_indices.to_dict('records')[0].items()}
            # Only keep HRV features that are not NaN
            hrv_features_clean = {k: v for k, v in hrv_features.items() if not pd.isna(v)}
            features.update(hrv_features_clean)
        except Exception as e:
            # If BVP fails, we can't get HRV, so we must skip the window.
            return None
        
        # --- EDA Features ---
        try:
            eda_window = window_data['EDA'].flatten()
            
            if np.std(eda_window) < 0.01:
                raise ValueError("EDA signal is flat")
            
            eda_signals, _ = nk.eda_process(eda_window, sampling_rate=4)
            features['EDA_Mean'] = eda_signals['EDA_Clean'].mean()
            features['SCR_Peaks_N'] = eda_signals['SCR_Peaks'].sum()
        except Exception as e:
            # If EDA fails, fill with NaN
            features['EDA_Mean'] = np.nan
            features['SCR_Peaks_N'] = np.nan
        
        # --- ACC & TEMP Features ---
        acc_window_3d = window_data['ACC']
        acc_mag = np.sqrt(np.sum(acc_window_3d**2, axis=1))
        features['ACC_Mag_Mean'] = acc_mag.mean()
        features['ACC_Mag_Std'] = acc_mag.std()
        
        temp_window = window_data['TEMP'].flatten()
        features['TEMP_Mean'] = temp_window.mean()
        features['TEMP_Std'] = temp_window.std()
        
        features['subject'] = subject_id
        features['label'] = target_label
        
        return features


class SWELLProcessor(BaseDatasetProcessor):
    """Processor for SWELL dataset."""
    
    def get_available_subjects(self) -> List[str]:
        """Get available SWELL subject IDs."""
        # SWELL uses participant numbers (pp1, pp2, etc.)
        subjects = []
        physiology_path = os.path.join(self.base_path, "0 - Raw data", "D - Physiology - raw data")
        if os.path.exists(physiology_path):
            for file in os.listdir(physiology_path):
                if file.endswith('.S00') and file.startswith('pp'):
                    subject_id = file.split('_')[0]
                    if subject_id not in subjects:
                        subjects.append(subject_id)
        return sorted(subjects)
    
    def load_subject_data(self, subject_id: str) -> Dict:
        """Load SWELL subject data."""
        # SWELL data is more complex with multiple modalities
        # This is a simplified implementation
        return {
            'subject_id': subject_id,
            'labels': []  # Would need to load from questionnaire data
        }
    
    def get_sampling_rates(self) -> Dict[str, int]:
        """Return SWELL sampling rates."""
        # These would need to be determined from the actual data files
        return {'ECG': 1000, 'EMG': 1000, 'EDA': 4, 'label': 1}
    
    def get_master_sampling_rate(self) -> int:
        """Return SWELL master sampling rate."""
        return 1  # Labels are typically at 1Hz or lower
    
    def extract_features_from_window(self, window_data: Dict, subject_id: str) -> Dict:
        """Extract features from a SWELL window."""
        # Implementation would depend on the specific SWELL data structure
        features = {
            'subject': subject_id,
            'label': 0  # Placeholder
        }
        return features


class AffectiveROADProcessor(BaseDatasetProcessor):
    """Processor for AffectiveROAD dataset."""
    
    def get_available_subjects(self) -> List[str]:
        """Get available AffectiveROAD drive IDs."""
        records_file = os.path.join(self.base_path, "Records.txt")
        subjects = []
        if os.path.exists(records_file):
            with open(records_file, 'r') as f:
                for line in f:
                    if line.strip():
                        subjects.append(line.strip())
        return subjects
    
    def load_subject_data(self, subject_id: str) -> Dict:
        """Load AffectiveROAD subject data."""
        # Load from Bioharness, E4, and subjective metrics
        data = {'subject_id': subject_id}
        
        # Load Bioharness data
        bio_file = os.path.join(self.base_path, "Bioharness", f"Bio_{subject_id}.csv")
        if os.path.exists(bio_file):
            data['bioharness'] = pd.read_csv(bio_file)
        
        # Load E4 data (left and right wrist)
        e4_left_path = os.path.join(self.base_path, "E4", subject_id, "left")
        e4_right_path = os.path.join(self.base_path, "E4", subject_id, "right")
        
        if os.path.exists(e4_left_path):
            data['e4_left'] = self._load_e4_data(e4_left_path)
        if os.path.exists(e4_right_path):
            data['e4_right'] = self._load_e4_data(e4_right_path)
        
        # Load subjective metrics
        sm_file = os.path.join(self.base_path, "Subj_metric", f"SM_{subject_id}.csv")
        if os.path.exists(sm_file):
            data['subjective_metrics'] = pd.read_csv(sm_file)
        
        return data
    
    def _load_e4_data(self, e4_path: str) -> Dict:
        """Load E4 wristband data."""
        e4_data = {}
        for file in os.listdir(e4_path):
            if file.endswith('.csv'):
                signal_type = file.split('.')[0]
                e4_data[signal_type] = pd.read_csv(os.path.join(e4_path, file))
        return e4_data
    
    def get_sampling_rates(self) -> Dict[str, int]:
        """Return AffectiveROAD sampling rates."""
        return {'BVP': 64, 'EDA': 4, 'TEMP': 4, 'ACC': 32, 'label': 1}
    
    def get_master_sampling_rate(self) -> int:
        """Return AffectiveROAD master sampling rate."""
        return 1
    
    def extract_features_from_window(self, window_data: Dict, subject_id: str) -> Dict:
        """Extract features from an AffectiveROAD window."""
        features = {
            'subject': subject_id,
            'label': 0  # Placeholder
        }
        return features


class StudentLifeProcessor(BaseDatasetProcessor):
    """Processor for StudentLife dataset."""
    
    def get_available_subjects(self) -> List[str]:
        """Get available StudentLife user IDs."""
        subjects = []
        for data_type in ['app_usage', 'call_log', 'sms']:
            data_path = os.path.join(self.base_path, "dataset", data_type)
            if os.path.exists(data_path):
                for file in os.listdir(data_path):
                    if file.endswith('.csv'):
                        user_id = file.split('_')[-1].split('.')[0]
                        if user_id not in subjects:
                            subjects.append(user_id)
        return sorted(subjects)
    
    def load_subject_data(self, subject_id: str) -> Dict:
        """Load StudentLife subject data."""
        data = {'subject_id': subject_id}
        
        # Load different data types
        data_types = ['app_usage', 'call_log', 'sms', 'calendar']
        for data_type in data_types:
            file_path = os.path.join(self.base_path, "dataset", data_type, f"{data_type}_u{subject_id}.csv")
            if os.path.exists(file_path):
                data[data_type] = pd.read_csv(file_path)
        
        return data
    
    def get_sampling_rates(self) -> Dict[str, int]:
        """Return StudentLife sampling rates."""
        return {'app_usage': 1, 'call_log': 1, 'sms': 1, 'calendar': 1, 'label': 1}
    
    def get_master_sampling_rate(self) -> int:
        """Return StudentLife master sampling rate."""
        return 1
    
    def extract_features_from_window(self, window_data: Dict, subject_id: str) -> Dict:
        """Extract features from a StudentLife window."""
        features = {
            'subject': subject_id,
            'label': 0  # Placeholder
        }
        return features

class DREAMERProcessor(BaseDatasetProcessor):
    """Processor for DREAMER dataset."""
    
    def get_available_subjects(self) -> List[str]:
        """Get available DREAMER subject IDs."""
        # DREAMER has 23 subjects
        return [f"subject_{i:02d}" for i in range(1, 24)]
    
    def load_subject_data(self, subject_id: str) -> Dict:
        """Load DREAMER subject data from MATLAB file."""
        mat_file = os.path.join(self.base_path, "DREAMER.mat")
        
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"DREAMER.mat not found at {mat_file}")
        
        # Load MATLAB file
        mat_data = scipy.io.loadmat(mat_file)
        print(len(mat_data['DREAMER']))
        
        # Extract data for specific subject
        subject_idx = int(subject_id.split('_')[1]) - 1
        
        data = {
            'ECG': mat_data['DREAMER'][0, subject_idx]['ECG'][0, 0],
            'EEG': mat_data['DREAMER'][0, subject_idx]['EEG'][0, 0],
            'labels': mat_data['DREAMER'][0, subject_idx]['ScoreValence'][0, 0],
            'arousal': mat_data['DREAMER'][0, subject_idx]['ScoreArousal'][0, 0],
            'dominance': mat_data['DREAMER'][0, subject_idx]['ScoreDominance'][0, 0]
        }
        
        return data
    
    def get_sampling_rates(self) -> Dict[str, int]:
        """Return DREAMER sampling rates."""
        return {'ECG': 128, 'EEG': 128, 'label': 1}
    
    def get_master_sampling_rate(self) -> int:
        """Return DREAMER master sampling rate."""
        return 1
    
    def extract_features_from_window(self, window_data: Dict, subject_id: str) -> Dict:
        """Extract features from a DREAMER window."""
        features = {
            'subject': subject_id,
            'label': 0  # Placeholder
        }
        return features

class DatasetProcessorFactory:
    """Factory class to create appropriate dataset processors."""
    
    @staticmethod
    def create_processor(dataset_type: str, base_path: str, **kwargs) -> BaseDatasetProcessor:
        """
        Create a dataset processor for the specified dataset type.
        
        Args:
            dataset_type: Type of dataset ('WESAD', 'SWELL', 'AffectiveROAD', 'StudentLife', 'DREAMER')
            base_path: Path to the dataset directory
            **kwargs: Additional arguments for the processor
            
        Returns:
            Appropriate dataset processor instance
        """
        processors = {
            'WESAD': WESADProcessor,
            'SWELL': SWELLProcessor,
            'AffectiveROAD': AffectiveROADProcessor,
            'StudentLife': StudentLifeProcessor,
            'DREAMER': DREAMERProcessor
        }
        
        if dataset_type not in processors:
            raise ValueError(f"Unknown dataset type: {dataset_type}. "
                           f"Available types: {list(processors.keys())}")
        
        return processors[dataset_type](base_path, **kwargs)


WINDOW_SIZE_SECONDS = 15
STEP_SIZE_SECONDS = 1
SAMPLING_RATE = 128
EEG_BANDS = {
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45)
}

def create_stress_label(valence, arousal):
    """Creates a binary stress label from Valence and Arousal scores."""
    # This rule is based on the circumplex model of affect.
    # We are looking for unpleasant, high-energy emotions.
    if arousal > 3 and valence < 3:
        return 1  # Stress
    else:
        return 0  # Neutral/Calm

def extract_eeg_features(eeg_window):
    """Extracts EEG band power features for all 14 channels."""
    features = {}
    for channel in range(eeg_window.shape[1]):
        channel_data = eeg_window[:, channel]
        freqs, psd = welch(channel_data, fs=SAMPLING_RATE)
        for band, (low, high) in EEG_BANDS.items():
            band_power = np.mean(psd[(freqs >= low) & (freqs <= high)])
            features[f'eeg_{band}_ch{channel+1}'] = band_power
    return features

def extract_ecg_features(ecg_window):
    """Extracts HR and HRV features from the first ECG channel."""
    try:
        # Use the first channel of the ECG data
        ecg_signal = ecg_window[:, 0]
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=SAMPLING_RATE)

        if len(rpeaks['ECG_R_Peaks']) < 2:
            return {}
        hrv_metrics = nk.hrv_time(rpeaks, sampling_rate=SAMPLING_RATE, show=False)
        return hrv_metrics
    except Exception:
        return {}

def create_dataframe_for_trial(ecg_data, eeg_data, valence_data, arousal_data, dominance_data):
    """
    Main function to process the aligned NumPy arrays for one trial into a DataFrame.
    
    Args:
        ecg_data (np.array): Shape (25472, 2)
        eeg_data (np.array): Shape (25472, 14)
        valence_data (np.array): Shape (25472,)
        arousal_data (np.array): Shape (25472,)
        dominance_data (np.array): Shape (25472,)

    Returns:
        pd.DataFrame: A feature-rich DataFrame where each row is a window.
    """
    all_window_features = []
    
    window_len_samples = WINDOW_SIZE_SECONDS * SAMPLING_RATE
    step_len_samples = STEP_SIZE_SECONDS * SAMPLING_RATE
    
    total_samples = ecg_data.shape[0]

    print(f"Processing {total_samples} samples into {WINDOW_SIZE_SECONDS}s windows...")

    # Slide the window across the data
    for i in range(0, total_samples - window_len_samples + 1, step_len_samples):
        window_id = i // step_len_samples
        
        # --- 1. Get the data for the current window ---
        eeg_window = eeg_data[i : i + window_len_samples]
        ecg_window = ecg_data[i : i + window_len_samples]
        
        # Get the label from the start of the window
        valence = valence_data[i]
        arousal = arousal_data[i]
        dominance = dominance_data[i]

        # --- 2. Engineer the stress label ---
        stress_label = create_stress_label(valence, arousal)
        
        # --- 3. Extract features for the window ---
        eeg_features = extract_eeg_features(eeg_window)
        ecg_features = extract_ecg_features(ecg_window)

            
        # --- 4. Assemble the row for the DataFrame ---
        current_window_data = {
            'window_id': window_id,
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'stress_label': stress_label
        }
        current_window_data.update(ecg_features)
        current_window_data.update(eeg_features)
        
        all_window_features.append(current_window_data)

    # --- 5. Create the final DataFrame ---
    final_df = pd.DataFrame(all_window_features)
    
    # Filter out the rows that are not clearly "Stress" or "Neutral"
    # final_df = final_df.dropna(subset=['stress_label'])
    final_df['stress_label'] = final_df['stress_label'].astype(int)
    
    print(f"Successfully created DataFrame with shape: {final_df.shape}")
    
    return final_df

def access_all_dreamer_data(subject_id: int, trial_id: int):
    """
    Demonstrates how to access every nested piece of data in the DREAMER dataset
    for a specific subject and trial.
    """
    mat_file = os.path.join("./data", "DREAMER.mat") # Assume file is in the current directory
    
    # --- Step 1: Load the main file ---
    mat_data = scipy.io.loadmat(mat_file)
    
    # --- Step 2: Select the subject ---
    # The 'DREAMER' object has a shape of (1, 23). We access the first row (0)
    # and the column for our subject (subject_id - 1 for zero-indexing).

    try:
        # Access the main 'dreamer' struct
        dreamer_struct = mat_data['DREAMER'][0, 0]
        
        # Access the 'Data' field which contains the array of subjects
        all_subject_data = dreamer_struct['Data'][0, 0]

        # Select the specific subject from the 'Data' array
        print(all_subject_data.shape)
        subject_struct = all_subject_data[0, subject_id - 1]
        
    except (KeyError, IndexError) as e:
        print(f"FATAL ERROR: Could not navigate the expected MATLAB struct. Error: {e}")
        print("Please ensure the DREAMER.mat file structure is correct.")
        return

    # --- Step 3: Extract Data Fields with Error Handling ---
    # Now we are inside the correct subject struct, we can safely check for fields.
    field_names = subject_struct.dtype.names
    print(f"\nAvailable data fields for Subject {subject_id}: {field_names}")

    # Safely get ECG data
    if 'ECG' in field_names:
        ecg_data = subject_struct['ECG'][0, 0]
        print("stimuli shape: ", ecg_data['stimuli'].shape)
        ecg_stimuli = ecg_data['stimuli']
    else:
        ecg_stimuli = None
        print(f"WARNING: No 'ECG' data found for Subject {subject_id}.")

    # Safely get EEG data
    if 'EEG' in field_names:
        eeg_data = subject_struct['EEG'][0, 0]
        eeg_stimuli = eeg_data['stimuli']
    else:
        eeg_stimuli = None
        print(f"WARNING: No 'EEG' data found for Subject {subject_id}.")
    # Get self-report scores
    valence_scores = subject_struct['ScoreValence']
    arousal_scores = subject_struct['ScoreArousal']
    dominance_scores = subject_struct['ScoreDominance'] 

    final_df = []
    for i in tqdm(range(ecg_stimuli.shape[0]), desc=f"Processing Subject {subject_id} Trial {trial_id}"):
        num_samples_eeg = eeg_stimuli[i, 0].shape[0]
        num_samples_ecg = ecg_stimuli[i, 0].shape[0]
        if num_samples_eeg > num_samples_ecg:
            eeg_stimuli_resampled = resample(eeg_stimuli[i, 0], num_samples_ecg)
        elif num_samples_eeg < num_samples_ecg:
            ecg_stimuli_resampled = resample(ecg_stimuli[i, 0], num_samples_eeg)

        vid = {
            'ECG': ecg_stimuli_resampled,
            'EEG': eeg_stimuli[i, 0],
            'valence': np.array([valence_scores[i, 0]] * ecg_stimuli_resampled.shape[0]),
            'arousal': np.array([arousal_scores[i, 0]] * ecg_stimuli_resampled.shape[0]),
            'dominance': np.array([dominance_scores[i, 0]] * ecg_stimuli_resampled.shape[0])
        }

        df = create_dataframe_for_trial(vid['ECG'], vid['EEG'], vid['valence'], vid['arousal'], vid['dominance'])
        final_df.append(df)

    final_df = pd.concat(final_df)
    final_df.to_csv(f"./data/processed/dreamer_features_{subject_id}_{trial_id}.csv", index=False)
    if ecg_stimuli is not None:
        print(f"\nShape of ECG signal data: {ecg_stimuli.shape} (timesteps, channels)")
    else:
        print("\nECG signal data is not available for this subject.")
        
    if eeg_stimuli is not None:
        print(f"\nShape of EEG signal data: {eeg_stimuli.shape} (timesteps, channels)")
    else:
        print("\nEEG signal data is not available for this subject.")

# Example usagerin
if __name__ == '__main__':
    # access_all_dreamer_data(1, 1)

    # wesad_df = pd.read_csv("./data/processed/features_dataset.csv")
    # dreamer_df = pd.read_csv("./data/processed/dreamer_features_1_1.csv")
    # wesad_cols = {col: col.replace('BVP_', '') for col in wesad_df.columns if col.startswith('BVP_')}
    # wesad_df = wesad_df.rename(columns=wesad_cols)
    # print(wesad_df.shape)
    # print(dreamer_df.shape)
    # # Compare the columns of wesad_df and dreamer_df
    # # Concatenate the two DataFrames
    # # The `sort=False` argument is used to maintain column order
    # merged_df = pd.concat([wesad_df, dreamer_df], ignore_index=True, sort=False)

    # # Fill all NaN values with 0
    # merged_df = merged_df.fillna(0)
    # merged_df.to_csv("./data/processed/merged_features_dataset.csv", index=False)
    pass