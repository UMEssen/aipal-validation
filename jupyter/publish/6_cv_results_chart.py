# %%
import pandas as pd
import yaml
import ast
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '../aipal_validation/config/config_training.yaml')

root_path = '/local/work/merengelke/aipal/'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

cities_countries = config['cities_countries']
paths = [f"{root_path}{city_country}/aipal/results.csv" for city_country in cities_countries]
results_paths = [f"{root_path}{city_country}/aipal/predict.csv" for city_country in cities_countries]

# Load and combine data with proper handling of empty cells
df = pd.DataFrame()
for path in paths:
    df_small = pd.read_csv(path)
    df_small['city_country'] = path.split('/')[-3]
    # Fill empty cells with NaN
    df_small = df_small.replace('', np.nan)
    df = pd.concat([df, df_small])

df_results = pd.DataFrame()
for path in results_paths:
    df_small = pd.read_csv(path)
    df_small['city_country'] = path.split('/')[-3]
    df_results = pd.concat([df_results, df_small])

# Define cities to exclude for adult analysis
excluded_cities = ['Vessen', 'newcastle', 'turkey']

# Filter based on age group
if config['is_adult']:
    df = df.loc[:, ~df.columns.str.contains('kids')]
    # Remove specific cities for adult analysis
    df = df[~df['city_country'].isin(excluded_cities)]
else:
    df = df.loc[:, ~df.columns.str.contains('adults')]
    # No city exclusions for kids analysis

df.rename(columns={'Unnamed: 0': 'class'}, inplace=True)

def safe_extract_metric(row, metric_name):
    """Extract metric values from the dictionary-like string in the CSV"""
    try:
        # Convert row to string if it's not already
        row_str = str(row)

        # Use regex to extract the values for the specific metric
        # The pattern matches: 'MetricName': [value1, value2, ...]
        pattern = f"'{metric_name}':\s*\[(.*?)\]"
        match = re.search(pattern, row_str)

        if match:
            values_str = match.group(1)
            # Split by comma and convert to floats, excluding nan values
            values = [float(v.strip()) for v in values_str.split(',')
                     if v.strip().lower() != 'nan']

            if not values:
                return np.nan

            # Return the first value for the chart (using mean would be an option too)
            return float(values[0])
        return np.nan
    except Exception as e:
        return np.nan

# List of metrics to plot
metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

# Make cutoff types dynamic based on is_adult setting
if config['is_adult']:
    cutoff_types = ['no cutoff - adults', 'overall cutoff - adults', 'confident cutoff - adults']
    age_group = 'Adults'
else:
    cutoff_types = ['no cutoff - kids', 'overall cutoff - kids', 'confident cutoff - kids']
    age_group = 'Kids'

titles = ['No Cutoff', 'Overall Cutoff', 'Confident Cutoff']

# Ensure plots directory exists
plots_dir = os.path.join(script_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Create a figure for each metric
for metric in metrics:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, (cutoff, title) in enumerate(zip(cutoff_types, titles)):
        # Calculate metric scores for current cutoff
        df[f'{metric}_{idx}'] = df[cutoff].apply(lambda x: safe_extract_metric(x, metric) if pd.notna(x) else np.nan)

        # Get AML scores and sort cohorts
        aml_data = df[df['class'] == 'AML'].groupby('city_country')[f'{metric}_{idx}'].mean()

        # Check if we have valid data
        if aml_data.empty or aml_data.isna().all():
            axes[idx].text(0.5, 0.5, f'No data available for {title}',
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[idx].transAxes, fontsize=14)
            continue

        sorted_cohorts = aml_data.sort_values(ascending=False).index

        # Plot each class
        for class_name, group in df.groupby('class'):
            # Calculate mean scores for each city_country
            mean_scores = group.groupby('city_country')[f'{metric}_{idx}'].mean()

            # Check if we have any data for this class
            if mean_scores.empty or mean_scores.isna().all():
                continue

            # Reorder based on sorted_cohorts
            plot_data = mean_scores.reindex(sorted_cohorts)

            # Filter out NaN values for plotting
            valid_indices = ~plot_data.isna()
            if valid_indices.any():
                # Convert to numpy arrays for proper indexing
                indices = np.arange(len(sorted_cohorts))
                values = plot_data.values

                # Plot only valid points
                axes[idx].plot(indices[valid_indices],
                              values[valid_indices],
                              marker='o', linestyle='None', label=class_name)

        # Customize subplot
        if not sorted_cohorts.empty:
            axes[idx].set_xticks(range(len(sorted_cohorts)))
            axes[idx].set_xticklabels(sorted_cohorts, rotation=45, ha='right')
        axes[idx].set_title(f'{title}', fontsize=14)
        axes[idx].set_xlabel('Cohort', fontsize=12)
        if idx == 0:  # Only add y-label to the first subplot
            axes[idx].set_ylabel(metric, fontsize=12)
        axes[idx].legend(title="Class")

        # Set y-axis limits between 0 and 1 since these are all normalized metrics
        axes[idx].set_ylim(0, 1.1)

    plt.suptitle(f'{metric} by Cohort and Cutoff Type for {age_group} (Sorted by AML {metric})', fontsize=16, y=1.05)
    plt.tight_layout()

    # Save the figure as SVG
    filename = os.path.join(plots_dir, f'{metric.lower().replace(" ", "_")}_{age_group.lower()}_comparison.svg')
    plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
    print(f"Saved plot: {filename}")
    plt.close()  # Close the figure to free memory

print(f"All plots have been saved as SVG files in the '{plots_dir}' directory.")
