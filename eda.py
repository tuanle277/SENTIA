import os
import pandas as pd
import json
import requests
import dotenv
from tqdm import tqdm
dotenv.load_dotenv()

def extract_overall_results():
    # Directory containing results
    results_dir = "results"
    models = ["rf", "lr", "svm", "knn", "cnn"]

    # Prepare lists to collect data
    overall_results = []
    per_subject_results = []

    # List all files in the results directory
    for fname in os.listdir(results_dir):
        fpath = os.path.join(results_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith(".csv"):
            parts = fname.split("_")    
            if len(parts) > 3:
                second_part, third_part = parts[2], parts[3]
                if second_part in models:
                    df = pd.read_csv(fpath)
                    if fname.startswith("overall_results"):
                        df["model"] = second_part
                        overall_results.append(df)
                    elif fname.startswith("per_subject_results"):
                        df["model"] = second_part
                        per_subject_results.append(df)
                elif third_part in models:
                    df = pd.read_csv(fpath)
                    if fname.startswith("overall_results"):
                        df["model"] = third_part
                        overall_results.append(df)
                    elif fname.startswith("per_subject_results"):
                        df["model"] = third_part
                        per_subject_results.append(df)
                else:
                    print(f"Unknown model name: {fname}")

    # Concatenate dataframes
    if overall_results:
        overall_df = pd.concat(overall_results, ignore_index=True)
    else:
        overall_df = pd.DataFrame()

    if per_subject_results:
        per_subject_df = pd.concat(per_subject_results, ignore_index=True)
    else:
        per_subject_df = pd.DataFrame()

    overall_df.to_csv("results/overall_results.csv", index=False)

    # Save per-subject results as CSV with models as columns and subjects as rows (F1 scores only)
    if not per_subject_df.empty:
        # Pivot the data so models are columns and subjects are rows
        f1_pivot = per_subject_df.pivot(index='subject', columns='model', values='f1_score')
        
        # Reset index to make subject a column
        f1_pivot = f1_pivot.reset_index()
        
        # Fill NaN values with 0 (subjects that don't have results for certain models)
        f1_pivot = f1_pivot.fillna(0.0)
        
        f1_pivot.to_csv("results/per_subject_results.csv", index=False)
        print("Per-subject F1 scores saved to: results/per_subject_results.csv")
        print("Columns are models, rows are subjects")
    else:
        print("No per-subject results to save")

def extract_ema_questions():
    path = "data/suggestive_actions_dataset.json"

    with open(path, 'r') as f:
        data = json.load(f)

    n = len(data)
    print(f"Number of actions: {n}")

    gemini_api_key = os.getenv("GEMINI_API")
    llm_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
    full_url = f"{llm_api_url}{gemini_api_key}"

    results = []
    for i in tqdm(range(n)):
        action = data[i]
        prompt = (
            "You are an expert in behavioral health and digital phenotyping. "
            "Given the following action description, extract all potential Ecological Momentary Assessment (EMA) components. "
            "For each component, identify:\n"
            "1. The key behavior, event, or state being targeted or measured.\n"
            "2. The type of EMA question that could be asked (e.g., binary, Likert scale, open-ended).\n"
            "3. A sample EMA question phrased clearly and concisely for a participant.\n"
            "If there are multiple components, list each separately. "
            "If no EMA component is present, state 'None.'\n\n"
            f"Action Description: {action['action_description']}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(full_url, json=payload)
        results.append(response.json())
    
    with open("data/suggestive_actions_dataset_with_ema_questions.json", 'w') as f:
        json.dump(results, f, indent=4)

def graph_raw_data():
    

if __name__ == "__main__":
    extract_ema_questions()
