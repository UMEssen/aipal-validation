import pandas as pd
import os
from pathlib import Path
import ast
import numpy as np
from util import save_to_excel
import yaml
import json
import glob

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

def load_city_results(city_path):
    """Load and parse a city's results CSV file."""
    print(f"Reading results from: {city_path}")

    # First check if file exists
    if not Path(city_path).exists():
        print(f"Results file not found at: {city_path}")
        return None, None

    # Read CSV file
    df = pd.read_csv(city_path)

    # Fix the first column - rename it to Leukemia_Type
    df.rename(columns={df.columns[0]: 'Leukemia_Type'}, inplace=True)

    # Convert string representations of dictionaries to actual dictionaries
    for col in df.columns:
        if col != 'Leukemia_Type':  # Skip the leukemia type column
            df[col] = df[col].apply(safe_eval)

    # Extract city name from path
    city = city_path.split('/')[-3]

    return city, df

def format_city_name(city, age_group='kids'):
    """Format the city name - capitalize and handle special cases."""
    # Replace all_cohorts with All Centers
    if city.lower() == 'all_cohorts':
        return 'All Centers'

    # Replace underscores with spaces and capitalize each word
    formatted_name = ' '.join(word.capitalize() for word in city.split('_'))

    # Special case for Vessen in adults table
    if age_group == 'adults' and formatted_name == 'Vessen':
        return 'Essen'

    return formatted_name

def create_performance_table_per_city(results_dict, age_group):
    """Create a formatted table for either kids or adults results from all cities."""

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
            'City': '',
            'No Cutoff': '',
            'Overall Cutoff': '',
            'Confident Cutoff': ''
        })

        # For each metric, add rows for each city
        for metric in metrics:
            # First add a row with just the metric name
            rows.append({
                'Leukemia Type': '',
                'Metric': metric,
                'City': '',
                'No Cutoff': '',
                'Overall Cutoff': '',
                'Confident Cutoff': ''
            })

            # Add a row for each city
            for city, results_df in results_dict.items():
                row = {
                    'Leukemia Type': '',
                    'Metric': '',
                    'City': format_city_name(city, age_group),
                    'No Cutoff': 'N/A',
                    'Overall Cutoff': 'N/A',
                    'Confident Cutoff': 'N/A'
                }

                # Process each cutoff column
                for i, cutoff_col in enumerate(cutoff_cols):
                    # Skip if this column doesn't exist in this city's data
                    if cutoff_col not in results_df.columns:
                        continue

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
                                    print(f"Error processing {city} - {leukemia_type} - {metric} - {cutoff_col}: {e}")
                                    continue

                rows.append(row)

            # Add a blank row between metrics
            rows.append({
                'Leukemia Type': '',
                'Metric': '',
                'City': '',
                'No Cutoff': '',
                'Overall Cutoff': '',
                'Confident Cutoff': ''
            })

        # Add a blank row between leukemia types
        rows.append({
            'Leukemia Type': '',
            'Metric': '',
            'City': '',
            'No Cutoff': '',
            'Overall Cutoff': '',
            'Confident Cutoff': ''
        })

    # Create DataFrame from rows
    table_df = pd.DataFrame(rows)

    return table_df

def main():
    # Set up paths to look for city results
    config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    base_path = Path("/local/work/merengelke/aipal")
    cities_pattern = f"{base_path}/*/aipal/results.csv"
    output_dir = os.path.join(config['root_results'], '4_bs_results_per_center')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Looking for city results in: {cities_pattern}")
    city_results = glob.glob(cities_pattern)
    print(f"Found {len(city_results)} city result files")

    # Load results from each city
    kids_results = {}
    adults_results = {}

    for city_path in city_results:
        city, df = load_city_results(city_path)
        if df is not None:
            # Check which age groups this city has data for
            kids_cols = [col for col in df.columns if 'kids' in col]
            adults_cols = [col for col in df.columns if 'adults' in col]

            if kids_cols:
                print(f"City {city} has kids data columns: {kids_cols}")
                kids_results[city] = df
            if adults_cols:
                print(f"City {city} has adults data columns: {adults_cols}")
                adults_results[city] = df

    print(f"\nCities with kids data: {list(kids_results.keys())}")
    print(f"Cities with adults data: {list(adults_results.keys())}")

    # Create tables for both age groups
    if kids_results:
        kids_table = create_performance_table_per_city(kids_results, 'kids')
        kids_excel_path = os.path.join(output_dir, '4_performance_metrics_kids_by_city.xlsx')
        kids_csv_path = os.path.join(output_dir, '4_performance_metrics_kids_by_city.csv')
        save_to_excel(kids_table, str(kids_excel_path))
        kids_table.to_csv(kids_csv_path, index=False)
        print(f"Kids performance by city saved to: {kids_excel_path}")
    else:
        print("No kids data found.")

    if adults_results:
        adults_table = create_performance_table_per_city(adults_results, 'adults')
        adults_excel_path = os.path.join(output_dir, '4_performance_metrics_adults_by_city.xlsx')
        adults_csv_path = os.path.join(output_dir, '4_performance_metrics_adults_by_city.csv')
        save_to_excel(adults_table, str(adults_excel_path))
        adults_table.to_csv(adults_csv_path, index=False)
        print(f"Adults performance by city saved to: {adults_excel_path}")
    else:
        print("No adults data found.")

if __name__ == "__main__":
    main()
