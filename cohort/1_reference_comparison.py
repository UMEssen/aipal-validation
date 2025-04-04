import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
import yaml
import numpy as np
import warnings
import pandas as pd
from util import load_data, save_to_excel

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

# Add the parent directory to system path to import data_loader
sys.path.append(str(Path(__file__).parent.parent / "jupyter" / "publish" / "utils"))

# Load the standardized feature names from config
CONFIG_PATH = str(Path(__file__).parent.parent / "aipal_validation" / "config" / "config_analysis.yaml")
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

def analyze_class_distribution(df):
    """Analyze the distribution of disease classes across different cohorts."""
    # Get class distribution for each cohort
    class_dist = {}
    total_patients = {}

    for cohort in df['city_country'].unique():
        cohort_data = df[df['city_country'] == cohort]
        total_patients[cohort] = len(cohort_data)
        class_counts = cohort_data['class'].value_counts()
        class_percentages = (class_counts / len(cohort_data) * 100).round(1)
        class_dist[cohort] = {
            'counts': class_counts.to_dict(),
            'percentages': class_percentages.to_dict()
        }

    return class_dist, total_patients

def analyze_cohort_differences(df, feature, cohorts):
    """Analyze differences between cohorts for a specific feature."""
    cohort_stats = {}
    overall_stats = {
        'mean': df[feature].mean(),
        'median': df[feature].median(),
        'std': df[feature].std(),
        'unit': STANDARDIZED_FEATURES[feature]['unit']
    }

    for cohort in cohorts:
        cohort_data = df[df['city_country'] == cohort][feature]
        if len(cohort_data) > 0:
            # Calculate relative differences from overall
            mean_diff = ((cohort_data.mean() - overall_stats['mean']) / overall_stats['mean'] * 100)
            median_diff = ((cohort_data.median() - overall_stats['median']) / overall_stats['median'] * 100)

            cohort_stats[cohort] = {
                'mean': cohort_data.mean(),
                'median': cohort_data.median(),
                'std': cohort_data.std(),
                'unit': STANDARDIZED_FEATURES[feature]['unit'],
                'mean_diff_percent': mean_diff,
                'median_diff_percent': median_diff
            }
    return cohort_stats, overall_stats

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

def calculate_percentage_diff(current, reference):
    """Safely calculate percentage difference handling zero and invalid cases."""
    try:
        if reference == 0:
            return float('inf') if current > 0 else float('-inf') if current < 0 else 0
        return ((current - reference) / reference) * 100
    except (TypeError, ValueError):
        return float('nan')

def analyze_cohort_class_differences(df, feature, cohorts, class_type):
    """Analyze differences between cohorts for a specific feature and class."""
    cohort_stats = {}
    class_data = df[df['class'] == class_type]

    # Skip if no data for this class or feature not in reference values
    if len(class_data) == 0 or feature not in REFERENCE_VALUES[class_type]:
        return None

    ref_values = REFERENCE_VALUES[class_type][feature]
    overall_stats = {
        'mean': class_data[feature].mean(),
        'median': class_data[feature].median(),
        'std': class_data[feature].std(),
        'unit': STANDARDIZED_FEATURES[feature]['unit'],
        'total_count': len(class_data)
    }

    # Only process cohorts that have data for this class
    valid_cohorts = []
    for cohort in cohorts:
        cohort_class_data = class_data[class_data['city_country'] == cohort][feature]
        if len(cohort_class_data) >= 5:  # Only include if at least 5 patients
            valid_cohorts.append(cohort)

            # Calculate differences from reference values using safe function
            mean_diff_ref = calculate_percentage_diff(cohort_class_data.mean(), ref_values['mean'])
            median_diff_ref = calculate_percentage_diff(cohort_class_data.median(), ref_values['median'])

            cohort_stats[cohort] = {
                'mean': cohort_class_data.mean(),
                'median': cohort_class_data.median(),
                'std': cohort_class_data.std(),
                'count': len(cohort_class_data),
                'unit': STANDARDIZED_FEATURES[feature]['unit'],
                'mean_diff_ref_percent': mean_diff_ref,
                'median_diff_ref_percent': median_diff_ref
            }

    # Return None if no valid cohorts found
    if not valid_cohorts:
        return None

    return cohort_stats, overall_stats, ref_values

def analyze_combined_vs_reference(df, feature, class_type):
    """Compare combined data across all countries with reference values for a specific class."""
    class_data = df[df['class'] == class_type][feature]
    if len(class_data) == 0 or feature not in REFERENCE_VALUES[class_type]:
        return None

    ref_values = REFERENCE_VALUES[class_type][feature]
    combined_stats = {
        'mean': class_data.mean(),
        'sd': class_data.std(),
        'median': class_data.median(),
        'iqr': [class_data.quantile(0.25), class_data.quantile(0.75)],
        'count': len(class_data)
    }

    # Calculate relative differences
    diff_stats = {
        'mean_diff_percent': ((combined_stats['mean'] - ref_values['mean']) / ref_values['mean']) * 100,
        'sd_diff_percent': ((combined_stats['sd'] - ref_values['sd']) / ref_values['sd']) * 100,
        'median_diff_percent': ((combined_stats['median'] - ref_values['median']) / ref_values['median']) * 100,
    }

    return combined_stats, ref_values, diff_stats

def create_combined_analysis_table(df, features):
    """Create a table of combined analysis results for all features and classes with a layout optimized for A4 paper."""
    # Initialize dictionary to hold data for each feature
    feature_data = {}

    # Calculate class counts first
    class_counts = {}
    for class_type in ['ALL', 'AML', 'APL']:
        class_counts[class_type] = len(df[df['class'] == class_type])

    # Process each feature
    for feature in features:
        feature_rows = {}
        # First get data for all classes
        for class_type in ['ALL', 'AML', 'APL']:
            result = analyze_combined_vs_reference(df, feature, class_type)
            if result:
                combined_stats, ref_values, diff_stats = result

                # Store data for this class
                feature_rows[class_type] = {
                    'n': combined_stats['count'],
                    'ref_mean': ref_values['mean'],
                    'ref_sd': ref_values['sd'],
                    'ref_median': ref_values['median'],
                    'ref_iqr_low': ref_values['iqr'][0],
                    'ref_iqr_high': ref_values['iqr'][1],
                    'comb_mean': combined_stats['mean'],
                    'comb_sd': combined_stats['sd'],
                    'comb_median': combined_stats['median'],
                    'comb_iqr_low': combined_stats['iqr'][0],
                    'comb_iqr_high': combined_stats['iqr'][1],
                    'mean_diff': diff_stats['mean_diff_percent'],
                }

        # Only include feature if we have data for at least one class
        if feature_rows:
            feature_data[feature] = {
                'unit': STANDARDIZED_FEATURES[feature]['unit'],
                'classes': feature_rows
            }

    # Now create the restructured DataFrame
    rows = []

    # Define the columns with sample sizes in the headers
    columns = {'Variable': 'Variable', 'Type': 'Type'}
    for class_type in ['ALL', 'AML', 'APL']:
        columns[class_type] = f"{class_type} (n={class_counts[class_type]})"

    # Add rows for each feature
    for feature, data in feature_data.items():
        # Row for feature name
        feature_row = {'Variable': f"{feature} ({data['unit']})", 'Type': '', 'ALL': '', 'AML': '', 'APL': ''}
        rows.append(feature_row)

        # Add Reference Mean (sd) row
        ref_mean_row = {'Variable': '', 'Type': 'Reference Mean (sd)'}
        for class_type in ['ALL', 'AML', 'APL']:
            if class_type in data['classes']:
                class_data = data['classes'][class_type]
                ref_mean_row[class_type] = f"{class_data['ref_mean']:.2f} ({class_data['ref_sd']:.2f})"
            else:
                ref_mean_row[class_type] = ''
        rows.append(ref_mean_row)

        # Add Reference Median [IQR] row
        ref_median_row = {'Variable': '', 'Type': 'Reference Median [IQR]'}
        for class_type in ['ALL', 'AML', 'APL']:
            if class_type in data['classes']:
                class_data = data['classes'][class_type]
                ref_median_row[class_type] = f"{class_data['ref_median']:.2f} [{class_data['ref_iqr_low']:.2f}-{class_data['ref_iqr_high']:.2f}]"
            else:
                ref_median_row[class_type] = ''
        rows.append(ref_median_row)

        # Add Combined Mean (sd) row
        mean_row = {'Variable': '', 'Type': 'Mean (sd)'}
        for class_type in ['ALL', 'AML', 'APL']:
            if class_type in data['classes']:
                class_data = data['classes'][class_type]
                mean_row[class_type] = f"{class_data['comb_mean']:.2f} ({class_data['comb_sd']:.2f})"
            else:
                mean_row[class_type] = ''
        rows.append(mean_row)

        # Add Combined Median [IQR] row
        median_row = {'Variable': '', 'Type': 'Median [IQR]'}
        for class_type in ['ALL', 'AML', 'APL']:
            if class_type in data['classes']:
                class_data = data['classes'][class_type]
                median_row[class_type] = f"{class_data['comb_median']:.2f} [{class_data['comb_iqr_low']:.2f}-{class_data['comb_iqr_high']:.2f}]"
            else:
                median_row[class_type] = ''
        rows.append(median_row)

        # Add NA n (%) row
        na_row = {'Variable': '', 'Type': 'NA n (%)'}
        for class_type in ['ALL', 'AML', 'APL']:
            if class_type in data['classes']:
                # Calculate missing value percentage
                class_data = df[df['class'] == class_type]
                if len(class_data) > 0:
                    missing_count = class_data[feature].isnull().sum()
                    missing_pct = (missing_count / len(class_data)) * 100
                    na_row[class_type] = f"{missing_count} ({missing_pct:.2f})"
                else:
                    na_row[class_type] = "0 (0)"
            else:
                na_row[class_type] = "0 (0)"
        rows.append(na_row)

    # Create the DataFrame with the proper columns and return
    result_df = pd.DataFrame(rows)

    # Explicitly set the column names with the class sizes
    result_df.columns = list(columns.values())

    return result_df

def create_city_comparison_table(df, features, class_type):
    """Create a table comparing each city/cohort to reference values for a specific leukemia class."""
    # Get unique cohorts
    cohorts = sorted(df['city_country'].unique())

    # Initialize data dictionary
    feature_data = {}

    # Process each feature
    for feature in features:
        # Only include features that have reference values for this class
        if feature not in REFERENCE_VALUES[class_type]:
            continue

        feature_rows = {}
        # Get reference values for this feature and class
        ref_values = REFERENCE_VALUES[class_type][feature]

        # Get data for the combined cohort first
        class_data = df[df['class'] == class_type][feature]
        if len(class_data) >= 5:  # Only include if we have enough patients
            # Calculate statistics
            combined_stats = {
                'count': len(class_data),
                'mean': class_data.mean(),
                'sd': class_data.std(),
                'median': class_data.median(),
                'iqr_low': class_data.quantile(0.25),
                'iqr_high': class_data.quantile(0.75),
                'mean_diff': ((class_data.mean() - ref_values['mean']) / ref_values['mean']) * 100
            }
            feature_rows['Combined'] = combined_stats

        # Now get data for each cohort
        for cohort in cohorts:
            cohort_class_data = df[(df['class'] == class_type) & (df['city_country'] == cohort)][feature]
            if len(cohort_class_data) >= 5:  # Only include if we have enough patients in this cohort
                # Calculate statistics
                cohort_stats = {
                    'count': len(cohort_class_data),
                    'mean': cohort_class_data.mean(),
                    'sd': cohort_class_data.std(),
                    'median': cohort_class_data.median(),
                    'iqr_low': cohort_class_data.quantile(0.25),
                    'iqr_high': cohort_class_data.quantile(0.75),
                    'mean_diff': ((cohort_class_data.mean() - ref_values['mean']) / ref_values['mean']) * 100
                }
                feature_rows[cohort] = cohort_stats

        # Only include feature if we have data for at least one cohort
        if feature_rows:
            feature_data[feature] = {
                'unit': STANDARDIZED_FEATURES[feature]['unit'],
                'ref': ref_values,
                'cohorts': feature_rows
            }

    # Now create the table
    rows = []

    # Create columns with cohort counts in the header
    columns = {'Feature': 'Feature', 'Metric': 'Metric', 'Reference': 'Reference'}

    # Add cohorts as columns (including combined)
    valid_cohorts = ['Combined']
    for cohort in cohorts:
        # Only include cohorts that have at least one feature with data
        for feature in feature_data:
            if cohort in feature_data[feature]['cohorts']:
                if cohort not in valid_cohorts:
                    valid_cohorts.append(cohort)
                    count = feature_data[feature]['cohorts'][cohort]['count']
                    columns[cohort] = f"{cohort} (n={count})"
                break

    # Add combined column if it has data
    if 'Combined' in valid_cohorts:
        # Find an example feature that has combined data
        for feature in feature_data:
            if 'Combined' in feature_data[feature]['cohorts']:
                count = feature_data[feature]['cohorts']['Combined']['count']
                columns['Combined'] = f"Combined (n={count})"
                break

    # Add rows for each feature
    for feature, data in feature_data.items():
        # Row for feature name
        feature_row = {'Feature': f"{feature} ({data['unit']})", 'Metric': '', 'Reference': ''}
        for cohort in valid_cohorts:
            feature_row[cohort] = ''
        rows.append(feature_row)

        # Add Mean (SD) row
        mean_row = {'Feature': '', 'Metric': 'Mean (SD)'}
        mean_row['Reference'] = f"{data['ref']['mean']:.2f} (±{data['ref']['sd']:.2f})"
        for cohort in valid_cohorts:
            if cohort in data['cohorts']:
                cohort_data = data['cohorts'][cohort]
                mean_row[cohort] = f"{cohort_data['mean']:.2f} (±{cohort_data['sd']:.2f})"
            else:
                mean_row[cohort] = '-'
        rows.append(mean_row)

        # Add Median [IQR] row
        median_row = {'Feature': '', 'Metric': 'Median [IQR]'}
        median_row['Reference'] = f"{data['ref']['median']:.2f} [{data['ref']['iqr'][0]:.2f}-{data['ref']['iqr'][1]:.2f}]"
        for cohort in valid_cohorts:
            if cohort in data['cohorts']:
                cohort_data = data['cohorts'][cohort]
                median_row[cohort] = f"{cohort_data['median']:.2f} [{cohort_data['iqr_low']:.2f}-{cohort_data['iqr_high']:.2f}]"
            else:
                median_row[cohort] = '-'
        rows.append(median_row)

        # Add Difference (%) row
        diff_row = {'Feature': '', 'Metric': 'Mean Difference (%)'}
        diff_row['Reference'] = 'Reference'
        for cohort in valid_cohorts:
            if cohort in data['cohorts']:
                cohort_data = data['cohorts'][cohort]
                diff_row[cohort] = f"{cohort_data['mean_diff']:.1f}%"
            else:
                diff_row[cohort] = '-'
        rows.append(diff_row)

    # Create the DataFrame with the proper columns
    result_df = pd.DataFrame(rows)

    # Set the column order
    result_df = result_df[list(columns.keys())]

    # Rename the columns to their display names
    result_df.columns = list(columns.values())

    return result_df

def main():
    # Load the data
    print("Loading data...")
    config_path = str(Path(__file__).parent.parent / "aipal_validation" / "config" / "config_analysis.yaml")
    df, config, features = load_data(config_path=config_path, is_adult=True)

    # Get unique cohorts
    cohorts = df['city_country'].unique()

    # Use standardized features from config
    analysis_features = list(STANDARDIZED_FEATURES.keys())

    # Create combined analysis table
    print("\nCreating combined analysis table...")
    combined_table = create_combined_analysis_table(df, analysis_features)
    output_dir = os.path.join(os.path.dirname(__file__), 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV (original format)
    title_suffix = 'adult' if config['is_adult'] else 'pediatric'
    combined_table.to_csv(os.path.join(output_dir, f'0_combined_analysis_{title_suffix}.csv'), index=False)

    # Save to formatted Excel
    excel_path = os.path.join(output_dir, f'0_combined_analysis_{title_suffix}.xlsx')
    save_to_excel(combined_table, excel_path)
    print(f"Combined analysis table saved to CSV: {output_dir}/0_combined_analysis_{title_suffix}.csv")
    print(f"Formatted Excel file saved to: {excel_path}")

    # Create and save city-by-city comparison tables for each class
    print("\nCreating city-by-city comparison tables...")
    for class_type in ['ALL', 'AML', 'APL']:
        print(f"Processing {class_type}...")
        city_table = create_city_comparison_table(df, analysis_features, class_type)

        # Save to CSV
        city_table.to_csv(os.path.join(output_dir, f'1_city_comparison_{class_type}_{title_suffix}.csv'), index=False)

        # Save to formatted Excel
        excel_path = os.path.join(output_dir, f'1_city_comparison_{class_type}_{title_suffix}.xlsx')
        save_to_excel(city_table, excel_path)
        print(f"City comparison table for {class_type} saved to: {excel_path}")

    # Analyze class distribution
    print("\n=== Class Distribution Analysis by Cohort ===\n")
    class_dist, total_patients = analyze_class_distribution(df)

    # Print class distribution
    for cohort in sorted(class_dist.keys()):
        print(f"\n{cohort} (Total patients: {total_patients[cohort]}):")
        for class_name, count in class_dist[cohort]['counts'].items():
            percentage = class_dist[cohort]['percentages'][class_name]
            print(f"{class_name}: {count} patients ({percentage:.1f}%)")

    print("\n=== Feature Distribution Analysis by Cohort and Disease Class ===\n")

    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(__file__), 'distribution_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Analyze each feature
    for feature in analysis_features:
        print(f"\n{feature} Analysis ({STANDARDIZED_FEATURES[feature]['unit']}):")
        print("-" * 60)

        # First show combined analysis
        print("\nCombined Analysis (All Countries):")
        for class_type in ['ALL', 'AML', 'APL']:
            result = analyze_combined_vs_reference(df, feature, class_type)
            if result:
                combined_stats, ref_values, diff_stats = result
                print(f"\n{class_type} (Total patients: {combined_stats['count']}):")
                print(f"Reference: Mean={ref_values['mean']:.2f} (±{ref_values['sd']:.2f}), "
                      f"Median={ref_values['median']:.2f}, IQR=[{ref_values['iqr'][0]:.2f}, {ref_values['iqr'][1]:.2f}]")
                print(f"Combined: Mean={combined_stats['mean']:.2f} (±{combined_stats['sd']:.2f}), "
                      f"Median={combined_stats['median']:.2f}, IQR=[{combined_stats['iqr'][0]:.2f}, {combined_stats['iqr'][1]:.2f}]")
                print(f"Differences: Mean={diff_stats['mean_diff_percent']:.1f}%, "
                      f"SD={diff_stats['sd_diff_percent']:.1f}%, "
                      f"Median={diff_stats['median_diff_percent']:.1f}%")

                # Flag significant differences in combined analysis
                if abs(diff_stats['mean_diff_percent']) > 50:
                    print("⚠️ WARNING: Combined data shows large mean difference from reference values")
                if abs(diff_stats['sd_diff_percent']) > 100:  # More than double thehe reference SD
                    print("⚠️ WARNING: Combined data shows much higher variability than reference")

        print("\nIndividual Cohort Analysis:")
        # Analyze by class and cohort (existing code)
        for class_type in ['ALL', 'AML', 'APL']:
            # Get cohort-specific statistics for this class
            result = analyze_cohort_class_differences(df, feature, cohorts, class_type)
            if result:
                cohort_stats, overall_stats, ref_values = result

                print(f"\n{class_type} Analysis (Total patients: {overall_stats['total_count']}):")

                # Print reference values
                print("\nReference values:")
                print(f"Mean={ref_values['mean']:.2f} (±{ref_values['sd']:.2f}), "
                      f"Median={ref_values['median']:.2f}, IQR=[{ref_values['iqr'][0]:.2f}, {ref_values['iqr'][1]:.2f}]")

                # Print overall statistics for this class
                print(f"\nOverall {class_type} statistics:")
                print(f"Mean={overall_stats['mean']:.2f} (±{overall_stats['std']:.2f}), "
                      f"Median={overall_stats['median']:.2f}")

                if cohort_stats:  # Only print if we have valid cohorts
                    print("\nCohort-specific statistics (cohorts with ≥5 patients):")
                    for cohort, stats in sorted(cohort_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                        mean_diff = stats['mean_diff_ref_percent']
                        median_diff = stats['median_diff_ref_percent']

                        # Format the difference strings
                        mean_diff_str = f"{mean_diff:.1f}%" if not np.isinf(mean_diff) else "∞%" if mean_diff > 0 else "-∞%"
                        median_diff_str = f"{median_diff:.1f}%" if not np.isinf(median_diff) else "∞%" if median_diff > 0 else "-∞%"

                        print(f"{cohort:15} (n={stats['count']:3d}): "
                              f"Mean={stats['mean']:.2f} (±{stats['std']:.2f}), "
                              f"Median={stats['median']:.2f} "
                              f"[Diff from ref: Mean={mean_diff_str}, "
                              f"Median={median_diff_str}]")

                        # Flag significant differences
                        if abs(mean_diff) > 50 and not np.isinf(mean_diff):
                            print(f"⚠️ WARNING: {cohort} shows large mean difference from reference values")
                        if abs(stats['std']) > ref_values['sd'] * 2:
                            print(f"⚠️ WARNING: {cohort} shows much higher variability than reference")

        # Create distribution plot
        plot_distribution_comparison(df, feature, plots_dir)

    print("\n=== Summary of Major Discrepancies ===")
    print("\nFeatures with significant differences from reference values:")

    # First show combined analysis discrepancies
    print("\nCombined Data Discrepancies:")
    for feature in analysis_features:
        for class_type in ['ALL', 'AML', 'APL']:
            result = analyze_combined_vs_reference(df, feature, class_type)
            if result:
                combined_stats, ref_values, diff_stats = result
                if abs(diff_stats['mean_diff_percent']) > 50 or abs(diff_stats['sd_diff_percent']) > 100:
                    print(f"\n- {feature} ({STANDARDIZED_FEATURES[feature]['unit']}) in {class_type} for combined data:")
                    print(f"  Mean difference: {diff_stats['mean_diff_percent']:.1f}%")
                    print(f"  SD ratio: {(combined_stats['sd']/ref_values['sd']):.1f}x reference")

    print("\nIndividual Cohort Discrepancies:")
    for feature in analysis_features:
        for class_type in ['ALL', 'AML', 'APL']:
            result = analyze_cohort_class_differences(df, feature, cohorts, class_type)
            if result:
                cohort_stats, _, ref_values = result
                for cohort, stats in cohort_stats.items():
                    mean_diff = stats['mean_diff_ref_percent']
                    if (abs(mean_diff) > 50 and not np.isinf(mean_diff)) or abs(stats['std']) > ref_values['sd'] * 2:
                        mean_diff_str = f"{mean_diff:.1f}%" if not np.isinf(mean_diff) else "∞%" if mean_diff > 0 else "-∞%"
                        print(f"\n- {feature} ({STANDARDIZED_FEATURES[feature]['unit']}) in {class_type} for {cohort}:")
                        print(f"  Mean difference: {mean_diff_str}")
                        print(f"  SD ratio: {(stats['std']/ref_values['sd']):.1f}x reference")

if __name__ == "__main__":
    main()
