import pandas as pd
import os
from pathlib import Path
import ast
import numpy as np
from util import save_to_excel
import yaml
import json

def load_config(file_path: str = "config/config_training.yaml"):
    path = Path.cwd() / file_path
    if path.suffix == ".yaml":
        return yaml.safe_load(path.open())
    elif path.suffix == ".json":
        return json.load(path.open())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

def safe_eval(x):
    """Safely evaluate string representation of dictionary."""
    if not isinstance(x, str) or not x.strip():
        return None
    try:
        # First try json.loads for cleaner parsing
        try:
            return json.loads(x.replace("'", '"').replace("nan", "null"))
        except Exception:
            pass

        # If that fails, try ast.literal_eval with cleaning
        x = x.strip()
        if x.startswith('{') and x.endswith('}'):
            # Replace NaN values with None before evaluation
            x = x.replace("nan", "None")
            return ast.literal_eval(x)
        return None
    except Exception as e:
        print(f"Failed to parse: {x} - Error: {e}")
        return None

def load_results(csv_path):
    """Load and parse the results CSV file."""
    print(f"Reading results from: {csv_path}")

    # First check if file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Results file not found at: {csv_path}")

    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded data shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # Fix the first column - rename it to Leukemia_Type
    df.rename(columns={df.columns[0]: 'Leukemia_Type'}, inplace=True)

    # Convert string representations of dictionaries to actual dictionaries
    for col in df.columns:
        if col != 'Leukemia_Type':  # Skip the leukemia type column
            df[col] = df[col].apply(safe_eval)
            # Print first non-null value for debugging
            first_val = df[col].dropna().iloc[0] if not df[col].isna().all() else None
            print(f"Column {col} first value type: {type(first_val)}, value: {first_val}")

    return df

def create_performance_table(results_df, age_group):
    """Create a formatted table for either kids or adults results."""

    # Define the metrics we want to display
    metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Define the cutoff types based on age group
    cutoff_cols = [f'no cutoff - {age_group}',
                   f'overall cutoff - {age_group}',
                   f'confident cutoff - {age_group}']

    # Initialize rows for our table
    rows = []

    # Process each leukemia type
    for leukemia_type in ['ALL', 'AML', 'APL']:
        # Add a header row for the leukemia type
        rows.append({
            'Leukemia Type': leukemia_type,
            'Metric': '',
            'No Cutoff': '',
            'Overall Cutoff': '',
            'Confident Cutoff': ''
        })

        # For each cutoff type and metric, extract and format values
        for metric in metrics:
            row = {
                'Leukemia Type': '',
                'Metric': metric,
                'No Cutoff': 'N/A',
                'Overall Cutoff': 'N/A',
                'Confident Cutoff': 'N/A'
            }

            # Process each cutoff column
            for i, cutoff_col in enumerate(cutoff_cols):
                # Find rows where the leukemia type matches and the cutoff column has data
                for idx, data_row in results_df.iterrows():
                    if data_row['Leukemia_Type'] == leukemia_type and pd.notna(data_row[cutoff_col]):
                        cell_data = data_row[cutoff_col]
                        if isinstance(cell_data, dict) and metric in cell_data:
                            values = cell_data[metric]

                            # Skip if values is None or contains NaN values
                            if values is None or any(isinstance(v, float) and np.isnan(v) for v in values) or len(values) < 3:
                                continue

                            try:
                                mean = values[0]  # First value is the mean
                                ci_low = values[1]  # Second value is CI low
                                ci_high = values[2]  # Third value is CI high

                                # Skip if any value is None
                                if mean is None or ci_low is None or ci_high is None:
                                    continue

                                # Format the cell with mean and CI
                                if metric in ['AUC', 'Accuracy', 'F1 Score']:
                                    formatted_value = f"{mean:.2f} [{ci_low:.2f}-{ci_high:.2f}]"
                                else:  # For Precision and Recall, show as percentages
                                    formatted_value = f"{mean*100:.1f}% [{ci_low*100:.1f}-{ci_high*100:.1f}]"

                                # Update the appropriate column
                                if i == 0:
                                    row['No Cutoff'] = formatted_value
                                elif i == 1:
                                    row['Overall Cutoff'] = formatted_value
                                elif i == 2:
                                    row['Confident Cutoff'] = formatted_value

                                break
                            except (TypeError, IndexError) as e:
                                print(f"Error processing {leukemia_type} - {metric} - {cutoff_col}: {e}")
                                continue

            rows.append(row)

        # Add a blank row between leukemia types
        rows.append({
            'Leukemia Type': '',
            'Metric': '',
            'No Cutoff': '',
            'Overall Cutoff': '',
            'Confident Cutoff': ''
        })

    # Create DataFrame from rows
    table_df = pd.DataFrame(rows)

    return table_df

def main():
    # Set up paths according to project structure
    config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    results_path = Path("/local/work/merengelke/aipal/all_cohorts/aipal/results.csv")
    output_dir = os.path.join(config['root_results'], '3_bs_results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Looking for results file at: {results_path}")

    # Load results
    results_df = load_results(results_path)

    # Print the first few rows of the dataframe for debugging
    print("\nFirst few rows of the loaded data:")
    print(results_df.head())
    print("\nColumns:", results_df.columns.tolist())

    # Create tables for both age groups
    kids_table = create_performance_table(results_df, 'kids')
    adults_table = create_performance_table(results_df, 'adults')

    # Save tables to Excel with formatting
    kids_excel_path = os.path.join(output_dir, '3_performance_metrics_kids.xlsx')
    adults_excel_path = os.path.join(output_dir, '3_performance_metrics_adults.xlsx')

    save_to_excel(kids_table, str(kids_excel_path))
    save_to_excel(adults_table, str(adults_excel_path))

    # Also save as CSV for easier viewing
    kids_table.to_csv(os.path.join(output_dir, '3_performance_metrics_kids.csv'), index=False)
    adults_table.to_csv(os.path.join(output_dir, '3_performance_metrics_adults.csv'), index=False)

    print("\nPerformance metric tables have been saved to:")
    print(f"Kids: {kids_excel_path}")
    print(f"Adults: {adults_excel_path}")

if __name__ == "__main__":
    main()
