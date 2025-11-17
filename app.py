from flask import Flask, request, jsonify
from flask_cors import CORS
from generate_ import generate_synthetic_samples
import joblib
import pandas as pd
import traceback # Import traceback for detailed error logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
# Initialize the Flask application
app = Flask(__name__)
CORS(app)

model = None
scaler = None
features = None

try:
    # Load the entire object from the .pkl file
    loaded_object = joblib.load('models/stress_model_rf_few_features.pkl')
    print("Pickle file loaded successfully.")

    # Check if the loaded object is a dictionary
    if isinstance(loaded_object, dict):
        print("Loaded object is a dictionary. Looking for 'model' key...")
        # Assuming the model is stored with the key 'model'
        if 'model' in loaded_object:
            model = loaded_object['model']
            scaler = loaded_object.get('scaler', None)
            features = loaded_object.get('features', None)
            print("✅ Model extracted successfully from dictionary!")
            if scaler is not None:
                print("✅ Scaler loaded from artifact.")
            else:
                print("ℹ️  No scaler found in artifact; will fit within generator.")
            if features is not None:
                print("✅ Feature list loaded from artifact.")
            else:
                print("ℹ️  No feature list found; falling back to defaults.")
        else:
            print("❌ Error: Dictionary loaded, but key 'model' was not found.")
            print("Available keys:", loaded_object.keys())
    else:
        # If it's not a dictionary, assume it's the model itself
        print("Loaded object is not a dictionary. Assuming it's the model.")
        model = loaded_object
        print("✅ Model loaded directly!")

except FileNotFoundError:
    print("❌ Error: Model file not found! Make sure 'models/stress_model.pkl' is in the correct directory.")
except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded or failed to load correctly.'}), 500

    try:
        # Get the JSON data from the request body
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON input'}), 400

        # Build feature frame in the exact training order
        # Try to use saved training feature list if present; otherwise fallback to common defaults
        feature_order = features or [
            'HRV_MeanNN',
            'EDA_Mean',
            'ACC_Mag_Mean',
            'TEMP_Mean',
        ]

        # Accept both exact keys and relaxed casing keys by mapping
        incoming = pd.DataFrame([data])
        # If incoming keys are different casing, try to align
        lower_map = {k.lower(): k for k in incoming.columns}
        ordered_cols = []
        for col in feature_order:
            if col in incoming.columns:
                ordered_cols.append(col)
            elif col.lower() in lower_map:
                # rename a copy for prediction
                incoming[col] = incoming[lower_map[col.lower()]]
                ordered_cols.append(col)
            else:
                return jsonify({'error': f"Missing required feature: {col}"}), 400

        features_df = incoming[ordered_cols]

        # Apply scaler if available
        if scaler is not None:
            X = scaler.transform(features_df.values)
        else:
            X = features_df.values

        # Predict label and probability, normalize mapping so 1=stressed, 0=not_stressed
        # Many sklearn classifiers expose classes_ ordering
        classes = getattr(model, 'classes_', None)
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
        pred_label = int(model.predict(X)[0])

        prob_stressed = None
        if proba is not None and classes is not None:
            # If classes_ is e.g., array([0,1]), index of 1 is prob stressed
            try:
                idx = int(np.where(classes == 1)[0][0])
                prob_stressed = float(proba[idx])
            except Exception:
                prob_stressed = float(np.max(proba))

        # Normalize label: if classes_ encodes stressed as 0, flip for consistency
        if classes is not None and len(classes) == 2:
            # expected classes {0,1}; if model predicted label not in {0,1}, keep as-is
            if 0 in classes and 1 in classes:
                normalized_label = pred_label  # already aligned
            else:
                # fallback
                normalized_label = pred_label
        else:
            normalized_label = pred_label

        return jsonify({'prediction': normalized_label, 'prob_stressed': prob_stressed})

    except Exception as e:
        # Return a more detailed error message for debugging
        print(traceback.format_exc()) # Print full error stack to server console
        return jsonify({'error': str(e)}), 400

@app.route('/check', methods=['POST'])
def check():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid input data'}), 400

        # Build input using the same feature order and optional scaler
        feature_order = features or [
            'HRV_MeanNN',
            'EDA_Mean',
            'ACC_Mag_Mean',
            'TEMP_Mean',
        ]
        incoming = pd.DataFrame([data])
        lower_map = {k.lower(): k for k in incoming.columns}
        for col in feature_order:
            if col not in incoming.columns:
                if col.lower() in lower_map:
                    incoming[col] = incoming[lower_map[col.lower()]]
                else:
                    return jsonify({'error': f"Missing required feature: {col}"}), 400
        incoming = incoming[feature_order]
        if scaler is not None:
            X = scaler.transform(incoming.values)
        else:
            X = incoming.values

        if model is None:
            return jsonify({'error': 'Model is not loaded'}), 500

        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities)  # Get the highest confidence score

        return jsonify({'confidence': confidence}), 200

    except Exception as e:
        print(f"Error in /check endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Define the generate endpoint
@app.route('/generate', methods=['POST'])
def generate():
    if model is None or scaler is None or features is None:
        return jsonify({'error': 'Model or scaler is not loaded or failed to load correctly.'}), 500

    try:
        # Get the JSON data from the request body
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON input'}), 400

        # Extract parameters from the request
        target_class = data.get('target_class', 1)  # Default to 1 (Stress)
        confidence_threshold = data.get('confidence_threshold', 0.95)  # Default to 95% confidence
        num_samples = data.get('num_samples', 100)  # Default to 100 samples

        # Load the real dataset
        data_path = "data/processed/merged_features_dataset.csv"
        real_data = pd.read_csv(data_path, low_memory=False)
        real_data = real_data[features]  # Keep only the features used for training

        # Ensure all columns are numeric
        real_data = real_data.apply(pd.to_numeric, errors='coerce')
        real_data = real_data.dropna()  # Drop rows with missing or non-numeric values

        # Filter real data for the target class
        if 'stress_label' not in real_data.columns:
            return jsonify({'error': "'stress_label' column not found in the dataset."}), 400
        real_data_target = real_data[real_data['stress_label'] == target_class].drop(columns=['stress_label'])

        # Generate synthetic samples using the imported function
        synthetic_samples = generate_synthetic_samples(
            real_data=real_data_target,
            model=model,
            scaler=scaler,
            target_class=target_class,
            confidence_threshold=confidence_threshold,
            num_samples=num_samples
        )

        # Convert the synthetic samples to JSON and return
        return jsonify(synthetic_samples.to_dict(orient='records'))

    except Exception as e:
        # Return a more detailed error message for debugging
        print(traceback.format_exc())  # Print full error stack to server console
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    # Using host='0.0.0.0' makes it accessible on your network
    app.run(host='0.0.0.0', debug=True, port=5000)