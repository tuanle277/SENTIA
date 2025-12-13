"""
Exploratory Data Analysis (EDA) for Stress Detection Dataset

This script performs comprehensive EDA on the merged stress detection dataset,
including visualizations, statistics, and data quality checks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Color palette
COLORS = {
    'WESAD': '#2E86AB',
    'ASCERTAIN': '#A23B72',
    'DREAMER': '#F18F01'
}

def load_dataset(data_path='data/stress_merged_all_pseudowindows.csv'):
    """Load the merged dataset."""
    print("=" * 80)
    print("Loading Dataset...")
    print("=" * 80)
    
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    return df


def dataset_overview(df):
    """Basic dataset overview statistics."""
    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    
    # Dataset composition
    dataset_counts = df['dataset'].value_counts()
    dataset_pct = df['dataset'].value_counts(normalize=True) * 100
    
    print("\nRows per Dataset:")
    for dataset in dataset_counts.index:
        print(f"  {dataset:12s}: {dataset_counts[dataset]:6,} rows ({dataset_pct[dataset]:5.2f}%)")
    
    # Subject counts
    print("\nSubjects per Dataset:")
    subjects_per_dataset = df.groupby('dataset')['subject'].nunique()
    for dataset in subjects_per_dataset.index:
        print(f"  {dataset:12s}: {subjects_per_dataset[dataset]:3d} unique subjects")
    
    print(f"\nTotal unique subjects: {df['subject'].nunique()}")
    
    # Stress label distribution
    print("\nStress Label Distribution (Overall):")
    stress_dist = df['stress_label'].value_counts(normalize=True) * 100
    stress_counts = df['stress_label'].value_counts()
    print(f"  Non-stress (0): {stress_counts.get(0, 0):6,} rows ({stress_dist.get(0, 0):5.2f}%)")
    print(f"  Stress (1):     {stress_counts.get(1, 0):6,} rows ({stress_dist.get(1, 0):5.2f}%)")
    
    # Stress by dataset
    print("\nStress Label Distribution by Dataset:")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        dist = subset['stress_label'].value_counts(normalize=True) * 100
        counts = subset['stress_label'].value_counts()
        print(f"\n  {dataset}:")
        print(f"    Non-stress (0): {counts.get(0, 0):6,} rows ({dist.get(0, 0):5.2f}%)")
        print(f"    Stress (1):     {counts.get(1, 0):6,} rows ({dist.get(1, 0):5.2f}%)")
    
    print()


def create_bar_charts(df, output_dir='plots'):
    """Create bar charts for rows and subjects per dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    dataset_counts = df['dataset'].value_counts()
    subjects_per_dataset = df.groupby('dataset')['subject'].nunique()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart 1: Rows per dataset
    colors1 = [COLORS.get(d, '#808080') for d in dataset_counts.index]
    bars1 = ax1.bar(dataset_counts.index, dataset_counts.values, color=colors1, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Rows', fontsize=12, fontweight='bold')
    ax1.set_title('Number of Rows per Dataset', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Bar chart 2: Subjects per dataset
    colors2 = [COLORS.get(d, '#808080') for d in subjects_per_dataset.index]
    bars2 = ax2.bar(subjects_per_dataset.index, subjects_per_dataset.values, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
    ax2.set_title('Number of Subjects per Dataset', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dataset_composition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar charts to: {output_path}")
    plt.close()


def stress_distribution_analysis(df, output_dir='plots'):
    """Analyze and visualize stress label distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall stress distribution
    stress_counts = df['stress_label'].value_counts().sort_index()
    colors_overall = ['#4CAF50', '#F44336']
    axes[0].bar(['Non-stress (0)', 'Stress (1)'], stress_counts.values, 
                color=colors_overall, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Number of Rows', fontsize=12, fontweight='bold')
    axes[0].set_title('Overall Stress Label Distribution', fontsize=14, fontweight='bold', pad=15)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels
    total = stress_counts.sum()
    for i, (label, count) in enumerate(stress_counts.items()):
        pct = count / total * 100
        axes[0].text(i, count, f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Stress distribution by dataset
    stress_by_dataset = pd.crosstab(df['dataset'], df['stress_label'], normalize='index') * 100
    stress_by_dataset.plot(kind='bar', ax=axes[1], color=['#4CAF50', '#F44336'], 
                          alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Dataset', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Stress Distribution by Dataset', fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(['Non-stress (0)', 'Stress (1)'], fontsize=10)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'stress_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved stress distribution plot to: {output_path}")
    plt.close()


def missing_values_analysis(df, output_dir='plots'):
    """Analyze missing values in the dataset."""
    print("=" * 80)
    print("MISSING VALUES ANALYSIS")
    print("=" * 80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    # Columns with missing values
    cols_with_missing = missing_df[missing_df['Missing Count'] > 0]
    
    if len(cols_with_missing) > 0:
        print(f"\nColumns with missing values: {len(cols_with_missing)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"\nTop 20 columns with most missing values:")
        print(cols_with_missing.head(20).to_string())
        
        # Visualize missing values
        fig, ax = plt.subplots(figsize=(12, 8))
        top_missing = cols_with_missing.head(30)
        ax.barh(range(len(top_missing)), top_missing['Missing Percentage'].values, 
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_yticks(range(len(top_missing)))
        ax.set_yticklabels(top_missing.index, fontsize=9)
        ax.set_xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Top 30 Columns with Missing Values', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'missing_values.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved missing values plot to: {output_path}")
        plt.close()
    else:
        print("\nNo missing values found in the dataset!")
    
    print()


def feature_statistics(df, output_dir='plots'):
    """Analyze feature statistics and distributions."""
    print("=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)
    
    # Define feature categories
    feature_cols = {
        'HRV': [c for c in df.columns if 'HRV' in c.upper()],
        'EDA': [c for c in df.columns if 'EDA' in c.upper()],
        'TEMP': [c for c in df.columns if 'TEMP' in c.upper()],
        'ACC': [c for c in df.columns if 'ACC' in c.upper()],
        'ASCERTAIN': [c for c in df.columns if c.startswith('asc_')]
    }
    
    # Count features by category
    print("\nFeature Categories:")
    for category, cols in feature_cols.items():
        if cols:
            print(f"  {category:12s}: {len(cols):3d} features")
    
    # Get main features (WESAD/DREAMER compatible)
    main_features = [c for c in df.columns if any(x in c.upper() for x in ['HRV', 'EDA', 'TEMP', 'ACC']) 
                     and not c.startswith('asc_')]
    
    if main_features:
        print(f"\nMain physiological features (WESAD/DREAMER): {len(main_features)}")
        
        # Statistics for main features
        numeric_df = df[main_features].select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 0:
            stats_df = numeric_df.describe().T
            stats_df = stats_df[['count', 'mean', 'std', 'min', 'max']]
            print("\nSummary Statistics (Main Features):")
            print(stats_df.head(10).to_string())
            
            # Distribution plots for key features
            key_features = main_features[:6]  # Plot first 6 features
            n_features = len(key_features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for idx, feat in enumerate(key_features):
                if feat in df.columns:
                    data = df[feat].dropna()
                    if len(data) > 0:
                        axes[idx].hist(data, bins=50, color=COLORS.get('WESAD', '#2E86AB'), 
                                      alpha=0.7, edgecolor='black', linewidth=0.5)
                        axes[idx].set_xlabel(feat, fontsize=10, fontweight='bold')
                        axes[idx].set_ylabel('Frequency', fontsize=10)
                        axes[idx].set_title(f'{feat}\n(mean={data.mean():.2f}, std={data.std():.2f})', 
                                          fontsize=9)
                        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
            
            # Hide unused subplots
            for idx in range(len(key_features), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'feature_distributions.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved feature distribution plots to: {output_path}")
            plt.close()
    
    print()


def correlation_analysis(df, output_dir='plots'):
    """Analyze feature correlations."""
    print("=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Get main features
    main_features = [c for c in df.columns if any(x in c.upper() for x in ['HRV', 'EDA', 'TEMP', 'ACC']) 
                     and not c.startswith('asc_')]
    
    if len(main_features) >= 2:
        # Compute correlation matrix
        numeric_df = df[main_features].select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('Feature Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to: {output_path}")
        plt.close()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            print(f"\nHighly correlated feature pairs (|r| > 0.8): {len(high_corr_pairs)}")
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            print(high_corr_df.head(10).to_string(index=False))
        else:
            print("\nNo highly correlated feature pairs found (|r| > 0.8)")
    
    print()


def window_analysis(df, output_dir='plots'):
    """Analyze window-level statistics."""
    print("=" * 80)
    print("WINDOW-LEVEL ANALYSIS")
    print("=" * 80)
    
    if 'window_id' in df.columns:
        window_stats = df.groupby('dataset')['window_id'].agg(['min', 'max', 'mean', 'std', 'count'])
        print("\nWindow Statistics by Dataset:")
        print(window_stats.to_string())
        
        # Visualize window distributions
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, dataset in enumerate(df['dataset'].unique()):
            subset = df[df['dataset'] == dataset]
            axes[idx].hist(subset['window_id'], bins=min(50, subset['window_id'].nunique()), 
                          color=COLORS.get(dataset, '#808080'), alpha=0.7, 
                          edgecolor='black', linewidth=0.5)
            axes[idx].set_xlabel('Window ID', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].set_title(f'{dataset}\n(mean={subset["window_id"].mean():.1f}, std={subset["window_id"].std():.1f})', 
                              fontsize=12, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'window_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved window distribution plots to: {output_path}")
        plt.close()
    
    print()


def subject_analysis(df, output_dir='plots'):
    """Analyze subject-level statistics."""
    print("=" * 80)
    print("SUBJECT-LEVEL ANALYSIS")
    print("=" * 80)
    
    # Samples per subject
    samples_per_subject = df.groupby('subject').size().sort_values(ascending=False)
    
    print(f"\nSamples per Subject Statistics:")
    print(f"  Mean:   {samples_per_subject.mean():.1f}")
    print(f"  Median: {samples_per_subject.median():.1f}")
    print(f"  Min:    {samples_per_subject.min()}")
    print(f"  Max:    {samples_per_subject.max()}")
    
    # Samples per subject by dataset
    print("\nSamples per Subject by Dataset:")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        samples = subset.groupby('subject').size()
        print(f"\n  {dataset}:")
        print(f"    Mean:   {samples.mean():.1f}")
        print(f"    Median: {samples.median():.1f}")
        print(f"    Min:    {samples.min()}")
        print(f"    Max:    {samples.max()}")
    
    # Visualize samples per subject
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram of samples per subject
    axes[0].hist(samples_per_subject.values, bins=30, color='#2E86AB', alpha=0.7, 
                edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Samples per Subject', fontsize=14, fontweight='bold', pad=15)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Box plot by dataset
    samples_by_dataset = []
    labels_by_dataset = []
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        samples = subset.groupby('subject').size().values
        samples_by_dataset.append(samples)
        labels_by_dataset.append(f"{dataset}\n(n={len(samples)} subjects)")
    
    bp = axes[1].boxplot(samples_by_dataset, tick_labels=labels_by_dataset, patch_artist=True)
    for patch, dataset in zip(bp['boxes'], df['dataset'].unique()):
        patch.set_facecolor(COLORS.get(dataset, '#808080'))
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[1].set_title('Samples per Subject by Dataset', fontsize=14, fontweight='bold', pad=15)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'subject_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved subject analysis plots to: {output_path}")
    plt.close()
    
    print()


def generate_summary_report(df, output_dir='plots'):
    """Generate a summary report."""
    report_path = os.path.join(output_dir, 'eda_summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
        f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        f.write("Dataset Composition:\n")
        dataset_counts = df['dataset'].value_counts()
        for dataset in dataset_counts.index:
            f.write(f"  {dataset}: {dataset_counts[dataset]:,} rows\n")
        
        f.write(f"\nTotal Unique Subjects: {df['subject'].nunique()}\n")
        
        f.write("\nStress Label Distribution:\n")
        stress_counts = df['stress_label'].value_counts()
        stress_pct = df['stress_label'].value_counts(normalize=True) * 100
        f.write(f"  Non-stress (0): {stress_counts.get(0, 0):,} ({stress_pct.get(0, 0):.2f}%)\n")
        f.write(f"  Stress (1):     {stress_counts.get(1, 0):,} ({stress_pct.get(1, 0):.2f}%)\n")
        
        f.write("\nMissing Values:\n")
        missing_count = df.isnull().sum().sum()
        f.write(f"  Total missing values: {missing_count:,}\n")
        f.write(f"  Columns with missing values: {(df.isnull().sum() > 0).sum()}\n")
        
        f.write("\nFeature Categories:\n")
        feature_cols = {
            'HRV': [c for c in df.columns if 'HRV' in c.upper()],
            'EDA': [c for c in df.columns if 'EDA' in c.upper()],
            'TEMP': [c for c in df.columns if 'TEMP' in c.upper()],
            'ACC': [c for c in df.columns if 'ACC' in c.upper()],
            'ASCERTAIN': [c for c in df.columns if c.startswith('asc_')]
        }
        for category, cols in feature_cols.items():
            if cols:
                f.write(f"  {category}: {len(cols)} features\n")
    
    print(f"Saved summary report to: {report_path}")


def main():
    """Main EDA pipeline."""
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA) FOR STRESS DETECTION DATASET")
    print("=" * 80 + "\n")
    
    # Load dataset
    df = load_dataset()
    
    # Create output directory
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analyses
    dataset_overview(df)
    create_bar_charts(df, output_dir)
    stress_distribution_analysis(df, output_dir)
    missing_values_analysis(df, output_dir)
    feature_statistics(df, output_dir)
    correlation_analysis(df, output_dir)
    window_analysis(df, output_dir)
    subject_analysis(df, output_dir)
    generate_summary_report(df, output_dir)
    
    print("=" * 80)
    print("EDA COMPLETE!")
    print("=" * 80)
    print(f"\nAll plots saved to: {output_dir}/")
    print("Summary report saved to: plots/eda_summary_report.txt")


if __name__ == '__main__':
    main()

