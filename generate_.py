import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_model(model_path):
    """
    Load the trained model and scaler from a .pkl file.
    """
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['features']

def generate_synthetic_samples(real_data, model, scaler, target_class, confidence_threshold=0.95, num_samples=100):
    """
    Generate synthetic samples by perturbing real data and filtering based on model confidence.

    Args:
        real_data (pd.DataFrame): Real data samples.
        model: Trained classification model.
        scaler: Scaler used to standardize the data.
        target_class (int): Target class label (e.g., 1 for "Stress").
        confidence_threshold (float): Minimum confidence to keep a synthetic sample.
        num_samples (int): Number of synthetic samples to generate.

    Returns:
        pd.DataFrame: Synthetic samples that meet the confidence threshold.
    """
    synthetic_samples = []
    # Use existing scaler to avoid data leakage; if None, fit a new one
    if scaler is None:
        from sklearn.preprocessing import StandardScaler as _SS
        scaler = _SS().fit(real_data.values)
    real_data_std = scaler.transform(real_data.values)

    for _ in tqdm(range(num_samples)):
        # Select a random sample from the real data
        idx = np.random.randint(0, len(real_data_std))
        sample = real_data_std[idx]

        # Add noise scaled to the standard deviation of each feature
        noise = np.random.normal(0, 0.1, size=sample.shape)  # Adjust noise scale as needed
        synthetic_sample = sample + noise

        # Classify the synthetic sample
        if isinstance(model, RandomForestClassifier):
            probabilities = model.predict_proba([synthetic_sample])[0]
        elif isinstance(model, SVC) and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([synthetic_sample])[0]
        else:
            raise ValueError("Model must support probability predictions (e.g., Random Forest or SVM with probability=True).")

        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

        # Keep the sample if it is classified as the target class with high confidence
        if predicted_class == target_class and confidence >= confidence_threshold:
            synthetic_samples.append(synthetic_sample)

    # Inverse transform the synthetic samples back to the original scale
    synthetic_samples = scaler.inverse_transform(synthetic_samples)
    synthetic_samples_df = pd.DataFrame(synthetic_samples, columns=real_data.columns)
    synthetic_samples_df['stress_label'] = target_class

    return synthetic_samples_df

if __name__ == '__main__':
    # Paths
    model_path = "models/stress_model_rf_few_features.pkl"  # Path to the trained model
    data_path = "data/processed/merged_features_dataset.csv"  # Path to the real dataset
    output_path = "data/synthetic_samples.csv"  # Path to save synthetic samples

    # Load the trained model and scaler
    model, scaler, _ = load_model(model_path)
    features = [
        "HRV_MeanNN", 
        "EDA_Mean",
        "TEMP_Mean",
        "ACC_Mag_Mean"
    ]

    features.append('stress_label')  # Assuming the label column is named 'stress_label'

    # Load the real dataset
    real_data = pd.read_csv(data_path, low_memory=False)
    real_data = real_data[features]  # Keep only the features used for training

    # Ensure all columns are numeric
    real_data = real_data.apply(pd.to_numeric, errors='coerce')
    real_data = real_data.dropna()  # Drop rows with missing or non-numeric values

    # Filter real data for the target class (e.g., "Stress")
    target_class = 1  # Assuming "Stress" is labeled as 1
    real_data_target = real_data[real_data['stress_label'] == target_class].drop(columns=['stress_label'])

    # Generate synthetic samples
    synthetic_samples = generate_synthetic_samples(
        real_data=real_data_target,
        model=model,
        scaler=scaler,
        target_class=target_class,
        confidence_threshold=0.95,
        num_samples=10000
    )

    # Save synthetic samples to a CSV file
    synthetic_samples.to_csv(output_path, index=False)
    print(f"Synthetic samples saved to: {output_path}")