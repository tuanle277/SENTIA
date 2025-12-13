import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Paths and loading model artifact
# -------------------------------------------------------------------
MODEL_PATH = "models/stress_model_rf_few_features.pkl"  # adjust if different
DATA_PATH = "data/processed/merged_features_dataset.csv"  # adjust if different

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

scaler = artifact["scaler"]
rf_model = artifact["model"]
features = artifact["features"]  # list of feature names used in training

print("Loaded model with", len(features), "features.")

# -------------------------------------------------------------------
# 2. Load dataset and prepare feature matrix
# -------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

# Ensure features exist
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected feature columns in dataset: {missing}")

# Coerce to numeric (in case of stray strings)
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with NaNs in those features
X_raw = df[features].dropna()

# Optional: if you have stress_label, you can subset to a smaller sample for speed
if "stress_label" in df.columns:
    y = df.loc[X_raw.index, "stress_label"].astype(int)
else:
    y = None

# Scale with the same scaler used in training
X_scaled = scaler.transform(X_raw.values)

# For SHAP, we can take a subset to keep plots readable and fast
N_SAMPLES = 1000
if X_scaled.shape[0] > N_SAMPLES:
    # stratified subsample if labels available
    if y is not None:
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=N_SAMPLES, random_state=42
        )
        idx_sub, _ = next(splitter.split(X_scaled, y))
    else:
        idx_sub = np.random.RandomState(42).choice(
            X_scaled.shape[0], size=N_SAMPLES, replace=False
        )
    X_shap = X_scaled[idx_sub]
    y_shap = y.iloc[idx_sub] if y is not None else None
else:
    X_shap = X_scaled
    y_shap = y

print("Using", X_shap.shape[0], "samples for SHAP analysis.")

# -------------------------------------------------------------------
# 3. Build SHAP explainer for Random Forest
# -------------------------------------------------------------------
# For tree-based models, TreeExplainer is efficient
X_shap_df = pd.DataFrame(X_shap, columns=features)
explainer = shap.TreeExplainer(rf_model)

# Compute SHAP values
shap_values = explainer.shap_values(X_shap_df)

# Handle different SHAP output formats (depends on SHAP version)
# For binary classification: can be list of 2 arrays, or 3D array (n_samples, n_features, 2)
if isinstance(shap_values, list):
    # Old SHAP format: list of [class_0_values, class_1_values]
    shap_values_pos = shap_values[1]  # Stress class (1)
    print(f"SHAP values (list format): class 1 shape = {shap_values_pos.shape}")
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    # Newer SHAP format: 3D array (n_samples, n_features, n_classes)
    shap_values_pos = shap_values[:, :, 1]  # Stress class (1)
    print(f"SHAP values (3D format): class 1 shape = {shap_values_pos.shape}")
else:
    # Single output (e.g., regressor or single-class)
    shap_values_pos = shap_values
    print(f"SHAP values shape = {shap_values_pos.shape}")

# Verify shapes match
print(f"X_shap_df shape: {X_shap_df.shape}")
print(f"shap_values_pos shape: {shap_values_pos.shape}")

# Summary bar plot
plt.figure()
shap.summary_plot(shap_values_pos, X_shap_df, plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance (Stress Class)")
plt.tight_layout()
plt.savefig("shap_rf_global_bar.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: shap_rf_global_bar.png")

# Beeswarm plot
plt.figure()
shap.summary_plot(shap_values_pos, X_shap_df, show=False)
plt.title("SHAP Feature Impact on Stress Prediction")
plt.tight_layout()
plt.savefig("shap_rf_beeswarm.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: shap_rf_beeswarm.png")