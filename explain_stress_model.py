# -*- coding: utf-8 -*-
"""
Real-Time SHAP Explanations for Stress Prediction Models

This module provides continuous, real-time feature attributions that explain 
what physiological signals are contributing to each stress prediction.

Key Features:
- Time-series aware SHAP explanations
- Real-time explanation streaming
- Stability and consistency evaluation
- Integration with existing model pipeline
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
import json

# SHAP imports
import shap
from shap import TreeExplainer, LinearExplainer, KernelExplainer
from shap.utils import hclust

# Model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from cnn_lstm import CNNLSTM

warnings.filterwarnings("ignore", category=UserWarning)


class StressModelExplainer:
    """
    Real-time SHAP explainer for stress prediction models.
    Handles both tree-based and neural network models with time-series awareness.
    """
    
    def __init__(self, model_path: str, model_type: str = 'rf'):
        """
        Initialize the explainer with a trained model.
        
        Args:
            model_path: Path to the saved model pickle file
            model_type: Type of model ('rf', 'lr', 'svm', 'knn', 'cnn')
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.features = None
        self.explainer = None
        self.explanation_history = []
        
        # Load the model
        self._load_model()
        self._initialize_explainer()
        
    def _load_model(self):
        """Load the trained model and preprocessing components."""
        print(f"Loading {self.model_type} model from {self.model_path}...")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.scaler = model_data['scaler']
        self.features = model_data['features']
        
        if self.model_type in ['rf', 'lr', 'svm', 'knn']:
            self.model = model_data['model']
        elif self.model_type == 'cnn':
            # Reconstruct CNN-LSTM model
            model_params = model_data['model_params']
            self.model = CNNLSTM(**model_params)
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.eval()
            
        print(f"Model loaded successfully. Features: {len(self.features)}")
        
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        print("Initializing SHAP explainer...")
        
        if self.model_type == 'rf':
            # TreeExplainer is most efficient for tree-based models
            self.explainer = TreeExplainer(self.model)
        elif self.model_type == 'lr':
            # LinearExplainer for linear models
            self.explainer = LinearExplainer(self.model)
        elif self.model_type in ['svm', 'knn']:
            # KernelExplainer for non-tree models
            # We'll use a subset of training data as background
            self.explainer = None  # Will be initialized with background data
        elif self.model_type == 'cnn':
            # Custom explainer for neural networks
            self.explainer = None  # Will be initialized with background data
            
        print("SHAP explainer initialized.")
        
    def set_background_data(self, background_data: np.ndarray):
        """
        Set background data for KernelExplainer (needed for SVM, KNN, CNN).
        
        Args:
            background_data: Array of shape (n_samples, n_features) for background
        """
        if self.model_type in ['svm', 'knn', 'cnn']:
            if self.model_type == 'cnn':
                # For CNN, we need to handle the sequence dimension
                background_tensor = torch.tensor(background_data, dtype=torch.float32).unsqueeze(1)
                self.explainer = self._create_cnn_explainer(background_tensor)
            else:
                # For SVM and KNN
                self.explainer = KernelExplainer(self._model_predict, background_data)
        print(f"Background data set for {self.model_type} explainer.")
        
    def _create_cnn_explainer(self, background_data: torch.Tensor):
        """Create a custom explainer for CNN-LSTM models."""
        def model_predict(x):
            """Wrapper for CNN model prediction."""
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            with torch.no_grad():
                logits = self.model(x)
                probabilities = torch.softmax(logits, dim=1)
                return probabilities.cpu().numpy()
        
        return KernelExplainer(model_predict, background_data.numpy())
        
    def _model_predict(self, x):
        """Wrapper for model prediction (for KernelExplainer)."""
        if self.model_type == 'svm':
            return self.model.predict_proba(x)
        elif self.model_type == 'knn':
            return self.model.predict_proba(x)
        else:
            return self.model.predict_proba(x)
    
    def explain_prediction(self, window_data: np.ndarray, 
                          subject_id: str = None, 
                          timestamp: str = None) -> Dict:
        """
        Generate SHAP explanations for a single prediction window.
        
        Args:
            window_data: Array of shape (n_features,) - single window of features
            subject_id: Optional subject identifier
            timestamp: Optional timestamp for the prediction
            
        Returns:
            Dictionary containing prediction and SHAP values
        """
        # Ensure data is properly shaped
        if window_data.ndim == 1:
            window_data = window_data.reshape(1, -1)
            
        # Standardize the data
        window_std = self.scaler.transform(window_data)
        
        # Get prediction
        if self.model_type == 'cnn':
            window_tensor = torch.tensor(window_std, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                logits = self.model(window_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0, prediction].item()
        else:
            probabilities = self.model.predict_proba(window_std)
            prediction = self.model.predict(window_std)[0]
            confidence = probabilities[0, prediction]
        
        # Generate SHAP values
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call set_background_data() first.")
            
        if self.model_type == 'cnn':
            shap_values = self.explainer.shap_values(window_std)
        else:
            shap_values = self.explainer.shap_values(window_std)
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # Multi-class case - use the class that was predicted
            shap_values = shap_values[prediction]
        
        # Ensure SHAP values are properly formatted
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Take first sample if batch
        
        # Convert to list and ensure scalar values
        shap_list = shap_values.tolist() if hasattr(shap_values, 'tolist') else list(shap_values)
        
        # Create explanation dictionary
        explanation = {
            'subject_id': subject_id,
            'timestamp': timestamp or datetime.now().isoformat(),
            'prediction': int(prediction),
            'confidence': float(confidence),
            'stress_probability': float(probabilities[0, 1]),
            'shap_values': shap_list,
            'feature_names': self.features,
            'feature_contributions': dict(zip(self.features, shap_list))
        }
        
        # Store in history for stability analysis
        self.explanation_history.append(explanation)
        
        return explanation
    
    def explain_time_series(self, time_series_data: np.ndarray, 
                           subject_id: str = None) -> List[Dict]:
        """
        Generate explanations for a time series of windows.
        
        Args:
            time_series_data: Array of shape (n_windows, n_features)
            subject_id: Optional subject identifier
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i, window in enumerate(time_series_data):
            timestamp = f"window_{i}"
            explanation = self.explain_prediction(
                window, subject_id=subject_id, timestamp=timestamp
            )
            explanations.append(explanation)
            
        return explanations
    
    def get_top_contributing_features(self, explanation: Dict, 
                                    top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get the top K features contributing to the prediction.
        
        Args:
            explanation: Explanation dictionary from explain_prediction
            top_k: Number of top features to return
            
        Returns:
            List of (feature_name, contribution) tuples
        """
        contributions = explanation['feature_contributions']
        
        # Handle case where contributions might be arrays
        clean_contributions = {}
        for feature, value in contributions.items():
            if isinstance(value, (list, np.ndarray)):
                # Take the first element if it's an array
                clean_contributions[feature] = float(value[0]) if len(value) > 0 else 0.0
            else:
                clean_contributions[feature] = float(value)
        
        sorted_features = sorted(clean_contributions.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        return sorted_features[:top_k]
    
    def analyze_explanation_stability(self, window_size: int = 10) -> Dict:
        """
        Analyze the stability of explanations across consecutive windows.
        
        Args:
            window_size: Number of recent windows to analyze
            
        Returns:
            Dictionary with stability metrics
        """
        if len(self.explanation_history) < window_size:
            # Use all available explanations if we don't have enough
            if len(self.explanation_history) < 2:
                return {"error": "Insufficient explanation history (need at least 2 windows)"}
            recent_explanations = self.explanation_history
            actual_window_size = len(recent_explanations)
        else:
            recent_explanations = self.explanation_history[-window_size:]
            actual_window_size = window_size
        
        # Extract SHAP values for each feature across windows
        feature_stability = {}
        for feature in self.features:
            values = []
            for exp in recent_explanations:
                val = exp['feature_contributions'][feature]
                # Handle array values
                if isinstance(val, (list, np.ndarray)):
                    val = float(val[0]) if len(val) > 0 else 0.0
                values.append(float(val))
            
            feature_stability[feature] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'coefficient_of_variation': np.std(values) / (abs(np.mean(values)) + 1e-8)
            }
        
        # Calculate overall stability metrics
        all_shap_values = np.array([exp['shap_values'] for exp in recent_explanations])
        stability_metrics = {
            'feature_stability': feature_stability,
            'overall_std': np.std(all_shap_values, axis=0).tolist(),
            'prediction_consistency': len(set(exp['prediction'] for exp in recent_explanations)),
            'confidence_stability': np.std([exp['confidence'] for exp in recent_explanations]),
            'window_size_analyzed': actual_window_size
        }
        
        return stability_metrics
    
    def visualize_explanation(self, explanation: Dict, 
                            save_path: str = None, 
                            top_k: int = 10) -> None:
        """
        Create visualization of SHAP explanation.
        
        Args:
            explanation: Explanation dictionary
            save_path: Optional path to save the plot
            top_k: Number of top features to show
        """
        # Get top contributing features
        top_features = self.get_top_contributing_features(explanation, top_k)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # SHAP values bar plot
        features, values = zip(*top_features)
        colors = ['red' if v < 0 else 'blue' for v in values]
        
        ax1.barh(range(len(features)), values, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('SHAP Value')
        ax1.set_title(f'Top {top_k} Feature Contributions\n'
                     f'Prediction: {"Stress" if explanation["prediction"] else "Neutral"} '
                     f'(Confidence: {explanation["confidence"]:.3f})')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Feature importance by category
        hrv_features = [f for f in features if 'HRV' in f or 'BVP' in f]
        eda_features = [f for f in features if 'EDA' in f or 'SCR' in f]
        acc_features = [f for f in features if 'ACC' in f]
        temp_features = [f for f in features if 'TEMP' in f]
        
        categories = ['HRV', 'EDA', 'ACC', 'TEMP']
        category_contributions = [
            sum(explanation['feature_contributions'][f] for f in hrv_features),
            sum(explanation['feature_contributions'][f] for f in eda_features),
            sum(explanation['feature_contributions'][f] for f in acc_features),
            sum(explanation['feature_contributions'][f] for f in temp_features)
        ]
        
        ax2.bar(categories, category_contributions, 
               color=['green', 'orange', 'purple', 'brown'], alpha=0.7)
        ax2.set_ylabel('Total SHAP Contribution')
        ax2.set_title('Contributions by Signal Type')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation visualization saved to: {save_path}")
        
        plt.show()
    
    def save_explanations(self, explanations: List[Dict], 
                         output_path: str) -> None:
        """
        Save explanations to JSON file.
        
        Args:
            explanations: List of explanation dictionaries
            output_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(explanations, f, indent=2)
        
        print(f"Explanations saved to: {output_path}")
    
    def load_background_data_from_csv(self, csv_path: str, 
                                    n_samples: int = 100) -> np.ndarray:
        """
        Load background data from the processed dataset CSV.
        
        Args:
            csv_path: Path to the processed features CSV
            n_samples: Number of samples to use as background
            
        Returns:
            Background data array
        """
        print(f"Loading background data from {csv_path}...")
        
        df = pd.read_csv(csv_path)
        
        # Use the same features as the model
        feature_cols = [col for col in df.columns if col in self.features]
        background_data = df[feature_cols].sample(n=min(n_samples, len(df)), 
                                                random_state=42).values
        
        print(f"Background data shape: {background_data.shape}")
        return background_data


def demonstrate_explanation_system():
    """
    Demonstrate the explanation system with example data.
    """
    print("=== Stress Model Explanation System Demo ===\n")
    
    # Example model paths (update these to your actual model paths)
    model_paths = {
        'rf': 'models/stress_model_rf.pkl',
        'lr': 'models/stress_model_lr.pkl',
        'svm': 'models/stress_model_svm.pkl',
        'knn': 'models/stress_model_knn.pkl',
        'cnn': 'models/stress_model_cnn.pkl'
    }
    
    # Try to load the best performing model
    best_model_type = 'rf'  # Update based on your results
    model_path = model_paths.get(best_model_type)
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train models first using train_.py")
        return
    
    try:
        # Initialize explainer
        explainer = StressModelExplainer(model_path, best_model_type)
        
        # Load background data
        background_data = explainer.load_background_data_from_csv(
            'data/processed/merged_features_dataset.csv', n_samples=50
        )
        explainer.set_background_data(background_data)
        
        # Generate example explanations
        print("\nGenerating example explanations...")
        
        # Simulate a few windows of data
        n_features = len(explainer.features)
        example_windows = np.random.randn(5, n_features)
        
        explanations = explainer.explain_time_series(
            example_windows, subject_id="demo_subject"
        )
        
        # Display results
        for i, explanation in enumerate(explanations):
            print(f"\n--- Window {i+1} ---")
            print(f"Prediction: {'Stress' if explanation['prediction'] else 'Neutral'}")
            print(f"Confidence: {explanation['confidence']:.3f}")
            print(f"Stress Probability: {explanation['stress_probability']:.3f}")
            
            # Top contributing features
            top_features = explainer.get_top_contributing_features(explanation, top_k=5)
            print("Top 5 Contributing Features:")
            for feature, contribution in top_features:
                print(f"  {feature}: {contribution:.4f}")
        
        # Analyze stability
        print("\n--- Stability Analysis ---")
        stability = explainer.analyze_explanation_stability()
        
        if 'error' in stability:
            print(f"Stability analysis error: {stability['error']}")
        else:
            print(f"Prediction Consistency: {stability['prediction_consistency']}/5")
            print(f"Confidence Stability (std): {stability['confidence_stability']:.4f}")
            
            # Show most stable features
            feature_stability = stability['feature_stability']
            most_stable = sorted(feature_stability.items(), 
                               key=lambda x: x[1]['coefficient_of_variation'])[:3]
            print("Most Stable Features:")
            for feature, metrics in most_stable:
                print(f"  {feature}: CV={metrics['coefficient_of_variation']:.4f}")
        
        # Save explanations
        explainer.save_explanations(
            explanations, 
            f"results/explanations_{best_model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        print(f"\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    demonstrate_explanation_system()
