"""

Basically sample of what happens during a single
live prediction in the system. It simulates receiving sensor data,
running it through the prediction and explanation pipeline, and generating a
final, user-facing suggestion.
"""

import os
import json
import pickle
import pandas as pd

# --- Replicating Core Classes (for a self-contained script) ---
# In the full application, these would be imported from their respective files.

class StressPredictor:
    """Loads a pre-trained model to make predictions."""
    def __init__(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ StressPredictor: Model loaded successfully.")
        except FileNotFoundError:
            print(f"❌ FATAL: Model file not found at {model_path}")
            self.model = None

    def predict(self, feature_df):
        if not self.model: return None
        model_cols = self.model.feature_names_in_
        feature_df = feature_df.reindex(columns=model_cols, fill_value=0)
        prediction = self.model.predict(feature_df)[0]
        probabilities = self.model.predict_proba(feature_df)[0]
        state = "stress" if prediction == 1 else "neutral"
        return {"state": state, "confidence": float(probabilities[prediction])}

class DictAsObject:
    """Simple class to allow dictionary access via dot notation for eval() compatibility."""
    def __init__(self, d):
        for key, value in d.items():
            setattr(self, key, value)

class ExplainableActionEngine:
    """Generates suggestions by combining predictions, features, and a knowledge base."""
    def __init__(self, actions_path, explanations_path):
        self.actions = self._load_json(actions_path, "Actions")
        self.explanations = self._load_json(explanations_path, "Explanations")
        print("✅ ExplainableActionEngine: Knowledge bases loaded.")

    def _load_json(self, file_path, name):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ FATAL: {name} knowledge base not found at {file_path}")
            return {}

    def _construct_explanation(self, rules, features):
        phrases = []
        for rule_key, text in self.explanations.items():
            if rule_key in rules:
                feature, op, val_str = rule_key.split()
                value = float(val_str)
                feature_val = features.get(feature, 0)
                if (op == '<' and feature_val < value) or \
                   (op == '>' and feature_val > value):
                    phrases.append(text)
        
        if not phrases: return ""
        if len(phrases) == 1: return f"We're noticing {phrases[0]}."
        return f"We're noticing {', '.join(phrases[:-1])}, and {phrases[-1]}."

    def get_suggestion(self, prediction, features, context):
        state = prediction['state']
        if state not in self.actions:
            return {"error": f"No actions for state: {state}"}
        
        # Convert context dict to object for dot notation access in eval()
        context_obj = DictAsObject(context)
        
        for action in sorted(self.actions[state], key=lambda x: x['priority']):
            condition = action.get('condition', 'True')
            if eval(condition, {"context": context_obj}):
                core_suggestion = action['suggestion_text']
                explanation_rules = action.get('explanation_rules', [])
                explanation_text = self._construct_explanation(explanation_rules, features)
                
                if explanation_text:
                    return f"It looks like you might be {state}. {explanation_text} {core_suggestion}"
                return f"It looks like you might be {state}. {core_suggestion}"
        
        return "No suitable action found for the current context."

# --- Simulation Runner ---

def run_simulation(name, features, context, predictor, engine):
    """Runs and prints a single end-to-end scenario."""
    print("\n" + "="*50)
    print(f"🎬 SCENARIO: {name}")
    print("="*50)

    # Step 1 & 2: Feature Extraction (Simulated)
    print(f"1.  INPUT FEATURES (from wearable sensors):\n    {features}")
    print(f"2.  INPUT CONTEXT (from phone/app):\n    {context}\n")
    
    feature_df = pd.DataFrame([features])

    # Step 3: Prediction
    prediction = predictor.predict(feature_df)
    if not prediction:
        print("Could not generate a prediction.")
        return
    
    print(f"3.  MODEL PREDICTION (from stress_model_rf.pkl):\n    {prediction}\n")

    # Step 4 & 5: Explainable Action & Final Output
    final_suggestion = engine.get_suggestion(prediction, features, context)
    print(f"4.  FINAL EXPLAINABLE SUGGESTION (to user's watch):")
    print(f"    ➡️  \"{final_suggestion}\"")
    print("="*50 + "\n")


def main():
    """Main function to run the simulations."""
    # --- Setup Paths ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(current_dir, 'models', 'stress_model_rf.pkl')
    ACTIONS_PATH = os.path.join(current_dir, 'data', 'knowledge_base', 'actions.json')
    EXPLANATIONS_PATH = os.path.join(current_dir, 'data', 'knowledge_base', 'explanations.json')

    # --- Initialize System ---
    print("--- Initializing Project Aura Simulation ---")
    predictor = StressPredictor(MODEL_PATH)
    engine = ExplainableActionEngine(ACTIONS_PATH, EXPLANATIONS_PATH)

    if not predictor.model or not engine.actions or not engine.explanations:
        print("\n❌ Simulation aborted due to missing files. Please ensure:")
        print("   1. You have run `feature_extractor.py` and `train.py`.")
        print("   2. The paths in this script are correct.")
        return

    # --- Define Scenarios ---

    # SCENARIO 1: The Stressed Developer (Alex)
    alex_features = {
      "BVP_HRV_SDNN": 28.1, "BVP_HRV_RMSSD": 19.5,
      "EDA_Mean": 1.8, "SCR_Peaks_N": 5,
      "ACC_Mag_Mean": 0.09, "TEMP_Mean": 33.8
    }
    alex_context = {
        "time_of_day": "afternoon", 
        "recent_activity": "sedentary",
        "day_of_week": "Tuesday"
    }

    # SCENARIO 2: A Relaxed Weekend Morning (Casey)
    casey_features = {
        "BVP_HRV_SDNN": 75.2, "BVP_HRV_RMSSD": 55.1,
        "EDA_Mean": 0.6, "SCR_Peaks_N": 1,
        "ACC_Mag_Mean": 0.25, "TEMP_Mean": 34.5
    }
    casey_context = {
        "time_of_day": "morning", 
        "recent_activity": "light_activity",
        "day_of_week": "Saturday"
    }

    # --- Run Simulations ---
    run_simulation("Stressed Developer at Work", alex_features, alex_context, predictor, engine)
    run_simulation("Relaxed Weekend Morning", casey_features, casey_context, predictor, engine)


if __name__ == '__main__':
    main()
