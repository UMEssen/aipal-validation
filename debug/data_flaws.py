import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
import yaml

# Add the parent directory to system path to import data_loader
sys.path.append(str(Path(__file__).parent.parent / "jupyter" / "publish" / "utils"))
from data_loader import load_data

# Load the standardized feature names from config
CONFIG_PATH = str(Path(__file__).parent.parent / "aipal_validation" / "config" / "config_outlier.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extract standardized feature names from config
STANDARDIZED_FEATURES = {
    'MCV_fL': {'unit': 'fl'},
    'PT_percent': {'unit': '%'},
    'LDH_UI_L': {'unit': 'U/l'},
    'MCHC_g_L': {'unit': 'g/l'},
    'WBC_G_L': {'unit': '/nl'},
    'Fibrinogen_g_L': {'unit': 'g/l'},
    'Monocytes_G_L': {'unit': '/nl'},
    'Platelets_G_L': {'unit': '/nl'},
    'Lymphocytes_G_L': {'unit': '/nl'},
    'age': {'unit': 'years'}
}

# Reference values from the supplementary table (using standardized names)
REFERENCE_VALUES = {
    'ALL': {
        'age': {'mean': 37.74, 'sd': 24.95, 'median': 36.48, 'iqr': [17, 59.98]},
        'WBC_G_L': {'mean': 39.24, 'sd': 81.18, 'median': 9.74, 'iqr': [3.8, 32.26]},
        'MCV_fL': {'mean': 86.64, 'sd': 8.61, 'median': 86.4, 'iqr': [81.57, 91.5]},
        'MCHC_g_L': {'mean': 288.34, 'sd': 113.32, 'median': 336, 'iqr': [319, 346.25]},
        'Monocytes_G_L': {'mean': 0.49, 'sd': 1.22, 'median': 0.1, 'iqr': [0, 0.4]},
        'Lymphocytes_G_L': {'mean': 3.72, 'sd': 5.19, 'median': 2.49, 'iqr': [1.23, 4.11]},
        'Platelets_G_L': {'mean': 91.63, 'sd': 92.59, 'median': 52.5, 'iqr': [30, 131]},
        'Fibrinogen_g_L': {'mean': 4.13, 'sd': 1.66, 'median': 3.85, 'iqr': [3.08, 5]},
        'LDH_UI_L': {'mean': 1068.5, 'sd': 1593.61, 'median': 566, 'iqr': [351, 1112]},
        'PT_percent': {'mean': 84.32, 'sd': 13.66, 'median': 86, 'iqr': [77, 94]}
    },
    'AML': {
        'age': {'mean': 60.99, 'sd': 17.2, 'median': 64, 'iqr': [51.1, 73]},
        'WBC_G_L': {'mean': 37.75, 'sd': 65.18, 'median': 9.46, 'iqr': [2.5, 42.3]},
        'MCV_fL': {'mean': 95.93, 'sd': 8.77, 'median': 95.7, 'iqr': [90.1, 101.3]},
        'MCHC_g_L': {'mean': 278.89, 'sd': 122.22, 'median': 336, 'iqr': [318, 346]},
        'Monocytes_G_L': {'mean': 5.34, 'sd': 17.95, 'median': 0.24, 'iqr': [0.03, 2.4]},
        'Lymphocytes_G_L': {'mean': 3.37, 'sd': 4.96, 'median': 1.89, 'iqr': [1.05, 3.8]},
        'Platelets_G_L': {'mean': 88.82, 'sd': 77.09, 'median': 66, 'iqr': [36, 120]},
        'Fibrinogen_g_L': {'mean': 4.18, 'sd': 1.67, 'median': 3.92, 'iqr': [3.1, 5.1]},
        'LDH_UI_L': {'mean': 742.53, 'sd': 1084.42, 'median': 396.5, 'iqr': [235.75, 763.75]},
        'PT_percent': {'mean': 76.03, 'sd': 16.58, 'median': 78, 'iqr': [66, 88]}
    },
    'APL': {
        'age': {'mean': 49.13, 'sd': 20.74, 'median': 51, 'iqr': [33.08, 63.39]},
        'WBC_G_L': {'mean': 16.1, 'sd': 32.14, 'median': 3.2, 'iqr': [1.18, 17.68]},
        'MCV_fL': {'mean': 89.36, 'sd': 7.23, 'median': 89.7, 'iqr': [84.05, 93.85]},
        'MCHC_g_L': {'mean': 317.78, 'sd': 98.68, 'median': 350, 'iqr': [342, 357.75]},
        'Monocytes_G_L': {'mean': 0.14, 'sd': 0.4, 'median': 0, 'iqr': [0, 0.1]},
        'Lymphocytes_G_L': {'mean': 1.33, 'sd': 1.36, 'median': 0.9, 'iqr': [0.5, 1.65]},
        'Platelets_G_L': {'mean': 41.8, 'sd': 36.64, 'median': 30.5, 'iqr': [17, 52.25]},
        'Fibrinogen_g_L': {'mean': 2.51, 'sd': 8.42, 'median': 1.57, 'iqr': [1, 2.2]},
        'LDH_UI_L': {'mean': 474.3, 'sd': 387.43, 'median': 363, 'iqr': [239, 553]},
        'PT_percent': {'mean': 61.78, 'sd': 16.15, 'median': 62, 'iqr': [53, 71]}
    }
}

def analyze_feature_distributions(df, features, cohorts):
    """Analyze the distribution of features across different cohorts."""
    stats_dict = {}
    for cohort in cohorts:
        cohort_data = df[df['city_country'] == cohort]
        cohort_stats = {}
        for feature in features:
            feature_stats = {
                'mean': cohort_data[feature].mean(),
                'std': cohort_data[feature].std(),
                'median': cohort_data[feature].median(),
                'skew': cohort_data[feature].skew(),
                'kurtosis': cohort_data[feature].kurtosis(),
                'missing': cohort_data[feature].isnull().sum() / len(cohort_data)
            }
            cohort_stats[feature] = feature_stats
        stats_dict[cohort] = cohort_stats
    return stats_dict

def compare_with_reference(df, feature, class_type):
    """Compare a feature's distribution with reference values for a specific class."""
    class_data = df[df['class'] == class_type][feature]
    if len(class_data) == 0 or feature not in REFERENCE_VALUES[class_type]:
        return None

    ref_values = REFERENCE_VALUES[class_type][feature]
    current_stats = {
        'mean': class_data.mean(),
        'sd': class_data.std(),
        'median': class_data.median(),
        'iqr': [class_data.quantile(0.25), class_data.quantile(0.75)]
    }

    # Calculate relative differences
    diff_stats = {
        'mean_diff_percent': ((current_stats['mean'] - ref_values['mean']) / ref_values['mean']) * 100,
        'sd_diff_percent': ((current_stats['sd'] - ref_values['sd']) / ref_values['sd']) * 100,
        'median_diff_percent': ((current_stats['median'] - ref_values['median']) / ref_values['median']) * 100,
    }

    return current_stats, ref_values, diff_stats

def analyze_cohort_differences(df, feature, cohorts):
    """Analyze differences between cohorts for a specific feature."""
    cohort_stats = {}
    for cohort in cohorts:
        cohort_data = df[df['city_country'] == cohort][feature]
        if len(cohort_data) > 0:
            cohort_stats[cohort] = {
                'mean': cohort_data.mean(),
                'median': cohort_data.median(),
                'std': cohort_data.std(),
                'unit': STANDARDIZED_FEATURES[feature]['unit']
            }
    return cohort_stats

def plot_distribution_comparison(df, feature, save_dir=None):
    """Create violin plots comparing distributions across classes with reference values."""
    plt.figure(figsize=(12, 6))

    # Reset index to avoid duplicate label issues and create plot
    plot_df = df.reset_index(drop=True)
    sns.violinplot(data=plot_df, x='class', y=feature)

    # Add reference means as points for classes that have reference values
    classes = ['ALL', 'AML', 'APL']
    ref_means = []
    valid_classes = []
    for c in classes:
        if feature in REFERENCE_VALUES[c]:
            ref_means.append(REFERENCE_VALUES[c][feature]['mean'])
            valid_classes.append(c)

    if ref_means:
        plt.plot(range(len(valid_classes)), ref_means, 'r*', markersize=15, label='Reference Mean')

    plt.title(f'{feature} Distribution Comparison')
    plt.ylabel(f"{feature} ({STANDARDIZED_FEATURES[feature]['unit']})")
    plt.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{feature}_comparison.png'))
    plt.close()

def main():
    # Load the data
    print("Loading data...")
    config_path = str(Path(__file__).parent.parent / "aipal_validation" / "config" / "config_outlier.yaml")
    df, config, _ = load_data(config_path=config_path)


    # Get unique cohorts
    cohorts = df['city_country'].unique()

    # Use standardized features from config
    analysis_features = list(STANDARDIZED_FEATURES.keys())

    print("\n=== Feature Distribution Analysis by Cohort and Disease Class ===\n")

    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(__file__), 'distribution_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Analyze each feature
    for feature in analysis_features:
        print(f"\n{feature} Analysis ({STANDARDIZED_FEATURES[feature]['unit']}):")
        print("-" * 60)

        # Compare with reference values for each class
        for class_type in ['ALL', 'AML', 'APL']:
            comparison = compare_with_reference(df, feature, class_type)
            if comparison:
                current, reference, diffs = comparison
                print(f"\n{class_type}:")
                print(f"Current:   Mean={current['mean']:.2f} (±{current['sd']:.2f}), "
                      f"Median={current['median']:.2f}, IQR=[{current['iqr'][0]:.2f}, {current['iqr'][1]:.2f}]")
                print(f"Reference: Mean={reference['mean']:.2f} (±{reference['sd']:.2f}), "
                      f"Median={reference['median']:.2f}, IQR=[{reference['iqr'][0]:.2f}, {reference['iqr'][1]:.2f}]")
                print(f"Differences: Mean={diffs['mean_diff_percent']:.1f}%, "
                      f"SD={diffs['sd_diff_percent']:.1f}%, "
                      f"Median={diffs['median_diff_percent']:.1f}%")

                # Flag significant differences
                if abs(diffs['mean_diff_percent']) > 50:
                    print("⚠️ WARNING: Large mean difference from reference values")
                if abs(diffs['sd_diff_percent']) > 100:
                    print("⚠️ WARNING: Much higher variability than reference")

        # Analyze cohort differences
        cohort_stats = analyze_cohort_differences(df, feature, cohorts)
        print("\nCohort Comparison:")
        for cohort, stats in cohort_stats.items():
            print(f"{cohort:15} Mean={stats['mean']:.2f} (±{stats['std']:.2f}), Median={stats['median']:.2f}")

        # Create distribution plot
        plot_distribution_comparison(df, feature, plots_dir)

    print("\n=== Summary of Major Discrepancies ===")
    print("\nFeatures with significant differences from reference values:")
    for feature in analysis_features:
        for class_type in ['ALL', 'AML', 'APL']:
            comparison = compare_with_reference(df, feature, class_type)
            if comparison:
                _, _, diffs = comparison
                if abs(diffs['mean_diff_percent']) > 50 or abs(diffs['sd_diff_percent']) > 100:
                    print(f"\n- {feature} ({STANDARDIZED_FEATURES[feature]['unit']}) in {class_type}:")
                    print(f"  Mean difference: {diffs['mean_diff_percent']:.1f}%")
                    print(f"  SD difference: {diffs['sd_diff_percent']:.1f}%")

if __name__ == "__main__":
    main()
