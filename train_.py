"""
This script trains and evaluates the stress prediction model.
It performs a subject-independent cross-validation to get a robust
measure of performance and then saves a final model trained on all data.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def statistical_performance_analysis(full_dataset, subject_results):
    """
    Performs comprehensive statistical analysis of model performance across subjects.
    
    Args:
        full_dataset (pd.DataFrame): The complete dataset with subject information
        subject_results (list): List of dictionaries containing per-subject results
    """
    print("\n" + "="*80)
    print("📊 STATISTICAL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Convert subject results to DataFrame for easier analysis
    results_df = pd.DataFrame(subject_results)
    
    # --- 1. Descriptive Statistics ---
    print("\n--- 1. DESCRIPTIVE STATISTICS ---")
    print(f"Number of subjects tested: {len(results_df)}")
    print(f"Total samples across all subjects: {len(full_dataset)}")
    print(f"Average samples per subject: {len(full_dataset) / len(results_df):.1f}")
    
    # Performance metrics summary
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    print(f"\nPerformance Metrics Summary (across {len(results_df)} subjects):")
    for metric in metrics:
        if metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            min_val = results_df[metric].min()
            max_val = results_df[metric].max()
            print(f"  {metric.capitalize()}: {mean_val:.3f} ± {std_val:.3f} (range: {min_val:.3f}-{max_val:.3f})")
    
    # --- 2. Subject-wise Performance Analysis ---
    print(f"\n--- 2. SUBJECT-WISE PERFORMANCE BREAKDOWN ---")
    print("Subject | Samples | Accuracy | Precision | Recall | F1-Score | Label Balance")
    print("-" * 75)
    
    for _, row in results_df.iterrows():
        subject_id = int(row['subject'])
        subject_data = full_dataset[full_dataset['subject'] == subject_id]
        label_balance = subject_data['label'].mean()  # Proportion of stress samples
        
        print(f"   {subject_id:2.0f}   |   {row['samples']:3.0f}   |   {row['accuracy']:.3f}   |   {row['precision']:.3f}   |  {row['recall']:.3f}  |   {row['f1_score']:.3f}   |     {label_balance:.3f}")
    
    # --- 3. Statistical Tests for Performance Differences ---
    print(f"\n--- 3. STATISTICAL TESTS FOR PERFORMANCE DIFFERENCES ---")
    
    # Test if performance is significantly different from random (50% accuracy)
    accuracy_values = results_df['accuracy'].values
    t_stat, p_value = stats.ttest_1samp(accuracy_values, 0.5)
    print(f"One-sample t-test (vs. random chance 50%):")
    print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.6f}")
    print(f"  Result: {'Significantly better than random' if p_value < 0.05 else 'Not significantly different from random'}")
    
    # Test for normality of performance distribution
    shapiro_stat, shapiro_p = stats.shapiro(accuracy_values)
    print(f"\nShapiro-Wilk test for normality of accuracy distribution:")
    print(f"  W-statistic: {shapiro_stat:.3f}, p-value: {shapiro_p:.6f}")
    print(f"  Result: {'Normally distributed' if shapiro_p > 0.05 else 'Not normally distributed'}")
    
    # --- 4. Performance vs. Subject Characteristics ---
    print(f"\n--- 4. PERFORMANCE VS. SUBJECT CHARACTERISTICS ---")
    
    # Analyze relationship between sample size and performance
    sample_sizes = results_df['samples'].values
    accuracies = results_df['accuracy'].values
    
    # Correlation between sample size and accuracy
    corr_coef, corr_p = stats.pearsonr(sample_sizes, accuracies)
    print(f"Correlation between sample size and accuracy:")
    print(f"  Pearson r: {corr_coef:.3f}, p-value: {corr_p:.6f}")
    print(f"  Interpretation: {'Strong positive' if corr_coef > 0.5 else 'Moderate positive' if corr_coef > 0.3 else 'Weak positive' if corr_coef > 0.1 else 'Weak negative' if corr_coef > -0.1 else 'Moderate negative' if corr_coef > -0.3 else 'Strong negative'} correlation")
    
    # Analyze relationship between label balance and performance
    label_balances = []
    for _, row in results_df.iterrows():
        subject_id = int(row['subject'])
        subject_data = full_dataset[full_dataset['subject'] == subject_id]
        label_balance = subject_data['label'].mean()
        label_balances.append(label_balance)
    
    label_balances = np.array(label_balances)
    corr_coef_balance, corr_p_balance = stats.pearsonr(label_balances, accuracies)
    print(f"\nCorrelation between label balance (stress proportion) and accuracy:")
    print(f"  Pearson r: {corr_coef_balance:.3f}, p-value: {corr_p_balance:.6f}")
    print(f"  Interpretation: {'Balanced datasets perform better' if corr_coef_balance < -0.3 else 'Imbalanced datasets perform better' if corr_coef_balance > 0.3 else 'No clear relationship'}")
    
    # --- 5. Generalization Analysis ---
    print(f"\n--- 5. GENERALIZATION ANALYSIS ---")
    
    # Calculate coefficient of variation (CV) to measure consistency
    cv_accuracy = stats.variation(accuracy_values) * 100
    cv_f1 = stats.variation(results_df['f1_score'].values) * 100
    
    print(f"Coefficient of Variation (lower = more consistent):")
    print(f"  Accuracy CV: {cv_accuracy:.1f}%")
    print(f"  F1-Score CV: {cv_f1:.1f}%")
    print(f"  Interpretation: {'Good generalization' if cv_accuracy < 15 else 'Moderate generalization' if cv_accuracy < 25 else 'Poor generalization'}")
    
    # Identify outliers (subjects with significantly different performance)
    q1, q3 = np.percentile(accuracy_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = results_df[(results_df['accuracy'] < lower_bound) | (results_df['accuracy'] > upper_bound)]
    print(f"\nOutlier Analysis (IQR method):")
    print(f"  Normal range: {lower_bound:.3f} - {upper_bound:.3f}")
    print(f"  Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outlier subjects: {outliers['subject'].tolist()}")
        print(f"  These subjects may have unique characteristics affecting model performance")
    
    # --- 6. Confidence Intervals ---
    print(f"\n--- 6. CONFIDENCE INTERVALS ---")
    
    # 95% confidence interval for mean accuracy
    mean_acc = np.mean(accuracy_values)
    std_acc = np.std(accuracy_values, ddof=1)
    n = len(accuracy_values)
    se_acc = std_acc / np.sqrt(n)
    
    # t-distribution critical value for 95% CI
    t_critical = stats.t.ppf(0.975, n-1)
    ci_lower = mean_acc - t_critical * se_acc
    ci_upper = mean_acc + t_critical * se_acc
    
    print(f"95% Confidence Interval for Mean Accuracy:")
    print(f"  Mean: {mean_acc:.3f}")
    print(f"  CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  Interpretation: We are 95% confident that the true mean accuracy across subjects lies in this range")
    
    # --- 7. Effect Size Analysis ---
    print(f"\n--- 7. EFFECT SIZE ANALYSIS ---")
    
    # Cohen's d for effect size (compared to random chance)
    effect_size = (mean_acc - 0.5) / std_acc
    print(f"Cohen's d (effect size vs. random chance): {effect_size:.3f}")
    if abs(effect_size) < 0.2:
        effect_interpretation = "negligible"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "small"
    elif abs(effect_size) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    print(f"  Interpretation: {effect_interpretation} effect size")
    
    # --- 8. Recommendations ---
    print(f"\n--- 8. RECOMMENDATIONS ---")
    
    if mean_acc > 0.7:
        print("✅ Model shows good overall performance (>70% accuracy)")
    elif mean_acc > 0.6:
        print("⚠️  Model shows moderate performance (60-70% accuracy)")
    else:
        print("❌ Model shows poor performance (<60% accuracy)")
    
    if cv_accuracy < 15:
        print("✅ Model generalizes well across subjects (low variance)")
    elif cv_accuracy < 25:
        print("⚠️  Model shows moderate generalization (moderate variance)")
    else:
        print("❌ Model shows poor generalization (high variance)")
    
    if len(outliers) == 0:
        print("✅ No significant outliers detected")
    else:
        print(f"⚠️  {len(outliers)} outlier(s) detected - consider investigating these subjects")
    
    if corr_coef > 0.3:
        print("💡 Consider collecting more data from subjects with fewer samples")
    
    if corr_coef_balance > 0.3 or corr_coef_balance < -0.3:
        print("💡 Consider balancing the dataset or using stratified sampling")
    
    print("\n" + "="*80)
    
    return results_df

def train_stress_model(data_path, model_save_path):
    """
    Trains a stress prediction model using subject-independent cross-validation
    and saves the final model trained on all data.

    Args:
        data_path (str): Path to the processed features_dataset.csv.
        model_save_path (str): Path to save the final trained model (.pkl).
    """
    print("--- Loading Processed Feature Dataset ---")
    try:
        full_dataset = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file {data_path} was not found.")
        print("Please ensure you have successfully run the feature_extractor.py script first.")
        return

    if full_dataset.empty:
        print("Error: The feature dataset is empty. Cannot train the model.")
        return

    print(f"Dataset loaded successfully with shape: {full_dataset.shape}")

    # Define features (X), target (y), and groups for cross-validation
    # We drop 'subject' and 'label' to create our feature set X.
    X = full_dataset.drop(['subject', 'label'], axis=1)
    y = full_dataset['label']
    groups = full_dataset['subject']

    # --- Subject-Independent Cross-Validation ---
    # This method ensures that all data from a single subject is either in the
    # training set or the test set, but never both. This is crucial for
    # building a model that can generalize to new people.
    logo = LeaveOneGroupOut()
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    
    all_preds = []
    all_true = []
    subject_results = []  # Store per-subject results for statistical analysis

    print("\n--- Starting Leave-One-Subject-Out Cross-Validation ---")
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        subject_being_tested = groups.iloc[test_index].unique()[0]
        
        # Train the model on all other subjects
        model.fit(X_train, y_train)
        
        # Make predictions on the held-out subject
        y_pred = model.predict(X_test)
        
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        
        # Calculate per-subject metrics
        subject_accuracy = accuracy_score(y_test, y_pred)
        subject_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        subject_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        subject_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Store results for statistical analysis
        subject_results.append({
            'subject': subject_being_tested,
            'samples': len(y_test),
            'accuracy': subject_accuracy,
            'precision': subject_precision,
            'recall': subject_recall,
            'f1_score': subject_f1
        })
        
        print(f"Tested on Subject {int(subject_being_tested)}: Accuracy = {subject_accuracy:.3f}, F1 = {subject_f1:.3f}")

    # Save the subject results to a csv file
    subject_results_df = pd.DataFrame(subject_results)
    subject_results_df.to_csv('subject_results.csv', index=False)

    # --- Final Performance Evaluation ---
    print("\n--- Overall Cross-Validation Performance Summary ---")
    
    # Calculate overall metrics from all folds
    overall_accuracy = accuracy_score(all_true, all_preds)
    overall_f1 = f1_score(all_true, all_preds, average='weighted')
    
    print(f"Average Accuracy: {overall_accuracy:.3f}")
    print(f"Average F1-Score: {overall_f1:.3f}")

    print("\nOverall Classification Report:")
    print(classification_report(all_true, all_preds, target_names=['Neutral', 'Stress']))
    
    print("Overall Confusion Matrix:")
    # Rows are True Labels, Columns are Predicted Labels
    conf_matrix = confusion_matrix(all_true, all_preds)
    print(pd.DataFrame(conf_matrix, index=['True Neutral', 'True Stress'], columns=['Pred Neutral', 'Pred Stress']))

    # --- Statistical Performance Analysis ---
    statistical_performance_analysis(full_dataset, subject_results)

    # --- Final Model Training on ALL Data ---
    print("\n--- Training Final Model on ALL Data for Production ---")
    final_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    
    # Save the model artifact to the specified path
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(final_model, f)
        
    print(f"\nProduction model trained and saved to: {model_save_path}")

if __name__ == '__main__':
    # Define paths based on our project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_PATH = os.path.join(current_dir, '..', '..', 'data', 'processed', 'features_dataset.csv')
    MODEL_SAVE_PATH = os.path.join(current_dir, 'models', 'stress_model_rf.pkl')
    
    # Run the full training and validation process
    train_stress_model(PROCESSED_DATA_PATH, MODEL_SAVE_PATH)
