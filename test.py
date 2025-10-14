import os
import scipy.io
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

def create_dataframe_for_trial(ecg_data, eeg_data, valence, arousal, dominance):
    """
    Creates a Pandas DataFrame for a single trial's synchronized data.
    """
    eeg_cols = [f'EEG_channel_{i+1}' for i in range(eeg_data.shape[1])]
    df = pd.DataFrame(data=eeg_data, columns=eeg_cols)
    df['ECG_channel_1'] = ecg_data[:, 0]
    df['ECG_channel_2'] = ecg_data[:, 1]
    df['Valence'] = valence
    df['Arousal'] = arousal
    df['Dominance'] = dominance
    return df

def process_dreamer_trial(subject_id, trial_id, dreamer_struct):
    """
    Loads, processes, synchronizes, and saves the data for a single subject 
    and a single trial from the pre-loaded DREAMER structure.
    """
    subject_idx = subject_id - 1
    trial_idx = trial_id - 1

    try:
        # Correctly navigate the nested subject array structure.
        all_subject_containers = dreamer_struct['Data']
        subject_container = all_subject_containers[0, subject_idx]
        participant_data = subject_container[0, 0]

        # --- Extract EEG & ECG Data ---
        eeg_struct = participant_data['EEG'][0, 0]
        # FINAL BUG FIX: The 'stimuli' field directly holds the (18, 1) object array.
        # The extra [0, 0] was causing it to resolve to a scalar.
        eeg_stimuli = eeg_struct['stimuli']
        eeg_trial_data = eeg_stimuli[trial_idx, 0]

        ecg_struct = participant_data['ECG'][0, 0]
        # Apply the same fix for the ECG data.
        ecg_stimuli = ecg_struct['stimuli']
        ecg_trial_data = ecg_stimuli[trial_idx, 0]

        # Correctly access the scores from the (18, 1) array.
        valence_score = participant_data['ScoreValence'][trial_idx, 0]
        arousal_score = participant_data['ScoreArousal'][trial_idx, 0]
        dominance_score = participant_data['ScoreDominance'][trial_idx, 0]

    except (KeyError, IndexError) as e:
        # This will catch any remaining structural navigation errors.
        print(f"Error accessing data for Subject {subject_id}, Trial {trial_id}. Skipping. Error: {e}")
        return

    # --- Resample signals to match lengths ---
    num_samples_eeg = eeg_trial_data.shape[0]
    ecg_resampled = resample(ecg_trial_data, num_samples_eeg)

    # --- Create DataFrame and Save ---
    df = create_dataframe_for_trial(
        ecg_data=ecg_resampled, 
        eeg_data=eeg_trial_data, 
        valence=valence_score, 
        arousal=arousal_score, 
        dominance=dominance_score
    )

    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"dreamer_S{subject_id:02d}_T{trial_id:02d}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    df.to_csv(output_path, index=False)


def main():
    """
    Main function to orchestrate the processing of the entire DREAMER dataset.
    """
    data_path = os.path.join("data", "DREAMER.mat")

    if not os.path.exists(data_path):
        print(f"FATAL ERROR: The file was not found at {data_path}")
        return
        
    print("Loading DREAMER.mat file... This may take a moment.")
    try:
        mat_data = scipy.io.loadmat(data_path)
        dreamer_struct = mat_data['DREAMER'][0, 0]
    except Exception as e:
        print(f"FATAL ERROR: Could not load or parse the .mat file. Error: {e}")
        return
        
    print("File loaded successfully. Starting data processing...")
    
    num_subjects = dreamer_struct['noOfSubjects'][0, 0]
    num_trials = dreamer_struct['noOfVideoSequences'][0, 0]

    for subject_id in tqdm(range(1, num_subjects + 1), desc="Processing Subjects"):
        for trial_id in tqdm(range(1, num_trials + 1), desc=f"Subject {subject_id} Trials", leave=False):
            process_dreamer_trial(subject_id, trial_id, dreamer_struct)

    print("\n-------------------------------------------------")
    print("All data has been processed successfully!")
    print(f"Processed data for {num_subjects} subjects and {num_trials} trials each.")
    print(f"Output files are located in: {os.path.join('data', 'processed')}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()

