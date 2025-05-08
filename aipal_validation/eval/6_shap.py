import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import shap
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from util import load_data
import yaml

def setup_log_capture(log_path):
    """Set up log capture to save both to console and file"""
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)

def calculate_monocyte_percent(df):
    """Calculate monocyte percentage from monocytes and WBC values"""
    # If 'Monocytes_percent' column already exists, use it
    if 'Monocytes_percent' in df.columns:
        return df

    # Otherwise calculate it
    if 'Monocytes_G_L' in df.columns and 'WBC_G_L' in df.columns:
        df['Monocytes_percent'] = (df['Monocytes_G_L'] * 100) / df['WBC_G_L']
    return df

def perform_shap_analysis(df, features, model_path, save_dir=None):
    """
    Perform SHAP analysis on the given dataframe and features.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the data
    features : list
        List of feature columns to analyze
    model_path : str
        Path to the XGBoost model file
    save_dir : str, optional
        Directory to save the plots
    """
    # Print initial class distribution
    print("\nClass distribution:")
    for class_label in sorted(df['class'].unique()):
        count = (df['class'] == class_label).sum()
        print(f"{class_label}: {count} samples ({count/len(df):.1%})")

    # Calculate monocyte percentage if needed
    df = calculate_monocyte_percent(df)

    # Prepare data for prediction
    data_for_prediction = df[features].copy()

    # Load the model
    try:
        model = xgb.Booster(model_file=model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(data_for_prediction)

    # Get predictions to verify model is working
    predictions = model.predict(dmatrix)
    print(f"Shape of predictions: {predictions.shape}")

    # Calculate SHAP values
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dmatrix)
    print(f"Shape of SHAP values: {np.shape(shap_values)}")

    # Verify SHAP values shape (n_samples, n_features, n_classes)
    if len(np.shape(shap_values)) != 3:
        print("Warning: SHAP values don't have the expected shape. Check model output.")

    # Aggregate SHAP values by taking the mean over all samples for each class
    mean_shap_values = np.mean(shap_values, axis=0)
    class_names = {0: 'ALL', 1: 'AML', 2: 'APL'}

    # Print the mean SHAP values for each feature by class
    print("\nMean SHAP values for each class:")
    for class_index in range(mean_shap_values.shape[1]):
        print(f"\nMean SHAP values for class {class_names[class_index]}:")
        for feature_index, feature_name in enumerate(features):
            print(f"{feature_name}: {mean_shap_values[feature_index, class_index]:.6f}")

    # Generate individual plots for each class
    print("\nGenerating individual SHAP summary plots for each class...")

    # For each class, create a separate visualization
    for class_idx in [0, 1, 2]:  # ALL, AML, APL in that order
        class_name = class_names[class_idx]
        print(f"Creating plot for class {class_name}")

        # Create a new figure for this class
        plt.figure(figsize=(10, 8))

        # Generate the SHAP summary plot
        shap.summary_plot(shap_values[:, :, class_idx], data_for_prediction, show=False)

        title = f"SHAP Summary Plot for {class_name}"
        plt.title(title)

        # Save individual class plot
        if save_dir:
            single_png_path = os.path.join(save_dir, f'shap_Adult_{class_name}_analysis.png')
            plt.savefig(single_png_path, dpi=300, bbox_inches='tight')
            print(f"Individual {class_name} PNG saved to: {single_png_path}")

            single_svg_path = os.path.join(save_dir, f'shap_Adult_{class_name}_analysis.svg')
            plt.savefig(single_svg_path, format='svg', bbox_inches='tight')
            print(f"Individual {class_name} SVG saved to: {single_svg_path}")

        # Close this figure
        plt.close()

    # Create a simpler combined visualization that shows feature importance
    print("\nCreating combined feature importance visualization...")
    plt.figure(figsize=(16, 10))

    # Create a DataFrame with the mean absolute SHAP values
    feature_importance = pd.DataFrame(index=features)
    for class_idx, class_name in class_names.items():
        # Get class-specific SHAP values and take mean of absolute values
        class_importance = np.abs(mean_shap_values[:, class_idx])
        feature_importance[class_name] = class_importance

    # Sort by total importance across classes
    feature_importance['Total'] = feature_importance.sum(axis=1)
    feature_importance = feature_importance.sort_values('Total', ascending=False).drop('Total', axis=1)

    # Keep only top 12 features
    feature_importance = feature_importance.head(12)

    # Plot as horizontal bar chart
    colors = {'ALL': 'blue', 'AML': 'red', 'APL': 'green'}
    feature_importance.plot(kind='barh', color=[colors[col] for col in feature_importance.columns], figsize=(16, 10))

    plt.title('Feature Importance by Class (Top 12 Features)', fontsize=16)
    plt.xlabel('Mean |SHAP Value|', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.legend(title='Leukemia Class', fontsize=12)
    plt.tight_layout()

    # Save this combined visualization
    if save_dir:
        png_path = os.path.join(save_dir, 'shap_Adult_analysis.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined PNG plot saved to: {png_path}")

        svg_path = os.path.join(save_dir, 'shap_Adult_analysis.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Combined SVG plot saved to: {svg_path}")

    return shap_values

def run_analysis(is_adult: bool):
    """Runs the SHAP analysis for a specific cohort."""
    tag = "Adult" if is_adult else "Pediatric"
    print(f"--- Running SHAP Analysis for {tag} Cohort ---")

    # Timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up directories
    config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Use tag in directory name
    plots_dir = os.path.join(config['root_results'], f'6_shap_{tag.lower()}')
    os.makedirs(plots_dir, exist_ok=True)

    # Set up log capture
    log_path = os.path.join(plots_dir, f'shap_{tag}_analysis_log_{timestamp}.txt')
    setup_log_capture(str(log_path))

    # Log start time and system info
    print(f"SHAP Analysis for {tag} started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"SHAP version: {shap.__version__}")

    # Load patient data for the specified cohort
    print(f"\nLoading {tag.lower()} patient data...")
    # Pass is_adult flag to load_data
    df, _, features = load_data(config_path=config_path, is_adult=is_adult)

    # Try multiple possible paths for the model file
    model_path = None
    possible_paths = [
        # Path relative to project root
        str(Path(__file__).parent.parent.parent / "jupyter" / "publish" / "model.json"),
        # Absolute path
        "/home/merengelke/aipal_validation/jupyter/publish/model.json",
        # Path relative to script
        str(Path(__file__).parent.parent / "jupyter" / "publish" / "model.json"),
        # Path for development env
        str(Path.cwd() / "jupyter" / "publish" / "model.json"),
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            model_path = path
            print(f"Found model at: {model_path}")
            break

    if model_path is None:
        print("Error: Could not find model.json in any of the following locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("Please specify the correct model path or place the model in one of these locations.")
        return

    # Perform SHAP analysis - update function to accept tag
    print("\nPerforming SHAP analysis...")
    # Modify perform_shap_analysis to accept tag for outputs
    perform_shap_analysis_updated(df, features, model_path, tag, save_dir=plots_dir)

    print(f"\n{tag} analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {plots_dir}")

# Update perform_shap_analysis to accept tag
def perform_shap_analysis_updated(df, features, model_path, tag, save_dir=None):
    """
    Perform SHAP analysis on the given dataframe and features.
    Includes tag for cohort-specific output naming.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the data
    features : list
        List of feature columns to analyze
    model_path : str
        Path to the XGBoost model file
    tag : str
        Cohort tag ('Adult' or 'Pediatric')
    save_dir : str, optional
        Directory to save the plots
    """
    # Print initial class distribution
    print("\nClass distribution:")
    for class_label in sorted(df['class'].unique()):
        count = (df['class'] == class_label).sum()
        print(f"{class_label}: {count} samples ({count/len(df):.1%})")

    # Calculate monocyte percentage if needed
    df = calculate_monocyte_percent(df)

    # Prepare data for prediction - ensure all required features exist
    expected_features = set(features)
    actual_features = set(df.columns)
    missing_features = expected_features - actual_features

    if missing_features:
        print(f"Warning: The following features are missing from the dataframe: {missing_features}")
        print("Using only the features that are available.")
        features = [f for f in features if f in df.columns]

    data_for_prediction = df[features].copy()

    # Handle missing values in data_for_prediction
    if data_for_prediction.isna().any().any():
        print("Warning: Missing values found in data. Filling with median values.")
        for col in data_for_prediction.columns:
            median_val = data_for_prediction[col].median()
            data_for_prediction[col].fillna(median_val, inplace=True)

    # Load the model
    try:
        model = xgb.Booster(model_file=model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}. Error details: {str(e)}")
        return None

    # Create DMatrix for XGBoost
    try:
        dmatrix = xgb.DMatrix(data_for_prediction)
        print(f"Successfully created DMatrix with shape: {data_for_prediction.shape}")
    except Exception as e:
        print(f"Error creating DMatrix: {str(e)}")
        return None

    # Get predictions to verify model is working
    try:
        predictions = model.predict(dmatrix)
        print(f"Shape of predictions: {predictions.shape}")
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None

    # Calculate SHAP values
    print("\nCalculating SHAP values...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dmatrix)
        print(f"Shape of SHAP values: {np.shape(shap_values)}")
    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        return None

    # Verify SHAP values shape (n_samples, n_features, n_classes)
    if len(np.shape(shap_values)) != 3:
        print("Warning: SHAP values don't have the expected shape. Check model output.")
        # Attempt to handle case where it might be (n_classes, n_samples, n_features)
        if len(np.shape(shap_values)) == 3 and np.shape(shap_values)[2] == len(features):
           print("Attempting to reshape SHAP values from (n_classes, n_samples, n_features) to (n_samples, n_features, n_classes)")
           try:
               shap_values = np.transpose(shap_values, (1, 2, 0))
               print(f"Reshaped SHAP values shape: {np.shape(shap_values)}")
           except Exception as e:
               print(f"Failed to reshape SHAP values: {e}")
               return None
        else:
           print("Cannot determine correct SHAP value shape. Aborting SHAP analysis.")
           return None

    # Check if the number of classes in shap_values matches expected (3)
    if np.shape(shap_values)[2] != 3:
        print(f"Warning: Expected 3 classes in SHAP values, but found {np.shape(shap_values)[2]}. Check model configuration.")
        # Adjust class names if possible, otherwise return
        if np.shape(shap_values)[2] == 1:
             print("SHAP values seem to be for a single output. Cannot plot per class.")
             # You might plot this single output if it makes sense
             class_names = {0: 'Output'}
        # Add other cases if needed
        else:
             print("Unsupported number of classes in SHAP values. Aborting.")
             return None
    else:
        class_names = {0: 'ALL', 1: 'AML', 2: 'APL'}

    # Aggregate SHAP values by taking the mean over all samples for each class
    mean_shap_values = np.mean(shap_values, axis=0)

    # Print the mean SHAP values for each feature by class
    print("\nMean SHAP values for each class:")
    for class_index in range(mean_shap_values.shape[1]):
        class_name_key = list(class_names.keys())[class_index]
        print(f"\nMean SHAP values for class {class_names[class_name_key]}:")
        for feature_index, feature_name in enumerate(features):
            print(f"{feature_name}: {mean_shap_values[feature_index, class_index]:.6f}")

    # Generate individual plots for each class
    print("\nGenerating individual SHAP summary plots for each class...")

    # For each class, create a separate visualization
    for class_idx in class_names.keys():
        class_name = class_names[class_idx]
        print(f"Creating plot for class {class_name} ({tag})")

        # Create a new figure for this class
        plt.figure(figsize=(10, 8))

        # Generate the SHAP summary plot
        shap.summary_plot(shap_values[:, :, class_idx], data_for_prediction, show=False)

        # Use tag in title
        title = f"SHAP Summary Plot for {class_name} ({tag})"
        plt.title(title)

        # Save individual class plot
        if save_dir:
            # Use tag in filenames
            single_png_path = os.path.join(save_dir, f'shap_{tag}_{class_name}_analysis.png')
            plt.savefig(single_png_path, dpi=300, bbox_inches='tight')
            print(f"Individual {class_name} ({tag}) PNG saved to: {single_png_path}")

            single_svg_path = os.path.join(save_dir, f'shap_{tag}_{class_name}_analysis.svg')
            plt.savefig(single_svg_path, format='svg', bbox_inches='tight')
            print(f"Individual {class_name} ({tag}) SVG saved to: {single_svg_path}")

        # Close this figure
        plt.close()

    # Create a simpler combined visualization that shows feature importance
    print("\nCreating combined feature importance visualization...")
    plt.figure(figsize=(16, 10))

    # Create a DataFrame with the mean absolute SHAP values
    feature_importance = pd.DataFrame(index=features)
    for class_idx, class_name in class_names.items():
        # Get class-specific SHAP values and take mean of absolute values
        class_importance = np.abs(mean_shap_values[:, class_idx])
        feature_importance[class_name] = class_importance

    # Sort by total importance across classes
    feature_importance['Total'] = feature_importance.sum(axis=1)
    feature_importance = feature_importance.sort_values('Total', ascending=False).drop('Total', axis=1)

    # Keep only top 12 features
    feature_importance = feature_importance.head(12)

    # Plot as horizontal bar chart
    colors = {'ALL': 'blue', 'AML': 'red', 'APL': 'green', 'Output': 'gray'}
    # Ensure colors are available for the columns we have
    plot_colors = [colors.get(col, 'gray') for col in feature_importance.columns]
    feature_importance.plot(kind='barh', color=plot_colors, figsize=(16, 10))

    # Use tag in title
    plt.title(f'Feature Importance by Class ({tag}, Top 12 Features)', fontsize=16)
    plt.xlabel('Mean |SHAP Value|', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.legend(title='Leukemia Class', fontsize=12)
    plt.tight_layout()

    # Save this combined visualization
    if save_dir:
        # Use tag in filenames
        png_path = os.path.join(save_dir, f'shap_{tag}_analysis.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined {tag} PNG plot saved to: {png_path}")

        svg_path = os.path.join(save_dir, f'shap_{tag}_analysis.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Combined {tag} SVG plot saved to: {svg_path}")

    plt.close() # Close the combined plot figure
    return shap_values

def main():
    """Runs SHAP analysis for both adult and pediatric cohorts."""
    print("Starting SHAP analysis...")
    run_analysis(is_adult=True)
    run_analysis(is_adult=False)
    print("\nSHAP analysis finished for both cohorts.")

if __name__ == "__main__":
    main()
