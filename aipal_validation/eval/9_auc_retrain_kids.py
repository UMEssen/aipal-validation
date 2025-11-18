import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from .util import save_roc_source_data_to_excel
# Manuscript Figure 5e

# Add proper font settings for better text rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Minimum number of samples required for a class to be included in the analysis
MIN_SAMPLES_THRESHOLD = 1

def load_config():
    """Load configuration files from config_training.yaml."""
    # Locate config file relative to script location
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "config_training.yaml"

    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Add results path if not present
    if 'root_results' not in config:
        config['root_results'] = os.path.join(config['root_dir'], 'results')

    return config

def load_predict_data(config):
    """Load the predict.csv data from the retrained model."""
    # Create path using config
    predict_path = Path(config['root_dir']) / "all_cohorts" / "retrain" / "predict.csv"

    if not predict_path.exists():
        raise FileNotFoundError(f"Could not find predict.csv at {predict_path}")

    # Load the data
    print(f"Loading prediction data from: {predict_path}")
    df = pd.read_csv(predict_path)

    # Filter for pediatric cohort (age < 18)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    pediatric_df = df[df['age'] < 18].copy()

    print(f"Loaded {len(pediatric_df)} pediatric samples out of {len(df)} total samples.")

    return pediatric_df

def filter_classes_by_count(data, min_samples=MIN_SAMPLES_THRESHOLD):
    """Filter classes with too few samples."""
    class_counts = data['class'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()

    print(f"\nFiltering classes with fewer than {min_samples} samples:")
    for cls, count in class_counts.items():
        status = "KEPT" if count >= min_samples else "EXCLUDED"
        print(f"  {cls}: {count} samples - {status}")

    return valid_classes

def calculate_roc_curves(data, valid_classes):
    """Calculate ROC curves for both base and retrained models."""
    if not valid_classes:
        print("No classes with sufficient samples found. Cannot calculate ROC curves.")
        return None

    # Ensure class is properly formatted and filter data
    data['class'] = data['class'].astype(str)
    data_filtered = data[data['class'].isin(valid_classes)].copy()

    results = {
        'base': {},
        'retrained': {}
    }

    # Process each class individually to avoid dimension issues
    for class_name in valid_classes:
        # Create binary labels for this specific class (1 for this class, 0 for others)
        y_true = (data_filtered['class'] == class_name).astype(int)

        # Get scores for this class
        base_scores = data_filtered[f"prediction.base.{class_name}"]
        retrained_scores = data_filtered[f"prediction.{class_name}"]

        # Calculate ROC curve for base model
        fpr_base, tpr_base, _ = roc_curve(y_true, base_scores)
        roc_auc_base = auc(fpr_base, tpr_base)
        results['base'][class_name] = {
            'fpr': fpr_base,
            'tpr': tpr_base,
            'auc': roc_auc_base
        }

        # Calculate ROC curve for retrained model
        fpr_retrained, tpr_retrained, _ = roc_curve(y_true, retrained_scores)
        roc_auc_retrained = auc(fpr_retrained, tpr_retrained)
        results['retrained'][class_name] = {
            'fpr': fpr_retrained,
            'tpr': tpr_retrained,
            'auc': roc_auc_retrained
        }

    return results

def plot_roc_curves(results, class_counts, valid_classes, config):
    """Create and save ROC curve plots comparing base vs. retrained models."""
    if not results:
        print("No ROC curve results to plot.")
        return

    # Create the plots directory if it doesn't exist
    plots_dir = os.path.join(config['root_results'], '9_auc_retrain_kids')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\nSaving plots to: {plots_dir}")

    # Force the specific order of classes: AML, APL, ALL (only if they exist in valid_classes)
    ordered_classes = [cls for cls in ["AML", "APL", "ALL"] if cls in valid_classes]

    # Create a combined figure with subplots for each valid class
    n_classes = len(ordered_classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 6))

    # Handle case with only one class (axes would not be an array)
    if n_classes == 1:
        axes = [axes]

    class_colors = {'AML': 'red', 'APL': 'green', 'ALL': 'blue'}
    model_styles = {
        'base': {'linestyle': '--', 'alpha': 0.7, 'linewidth': 2},
        'retrained': {'linestyle': '-', 'alpha': 0.9, 'linewidth': 2}
    }

    # Plot ROC curves for each class in separate subplots
    for i, class_name in enumerate(ordered_classes):
        ax = axes[i]
        n_samples = class_counts.get(class_name, 0)

        # Plot ROC curve for base model
        base_auc = results['base'][class_name]['auc']
        ax.plot(
            results['base'][class_name]['fpr'],
            results['base'][class_name]['tpr'],
            color=class_colors.get(class_name, 'purple'),
            linestyle=model_styles['base']['linestyle'],
            alpha=model_styles['base']['alpha'],
            linewidth=model_styles['base']['linewidth'],
            label=f'Base Model (AUC = {base_auc:.3f})'
        )

        # Retrained model metrics
        retrained_auc = results['retrained'][class_name]['auc']
        ax.plot(
            results['retrained'][class_name]['fpr'],
            results['retrained'][class_name]['tpr'],
            color=class_colors.get(class_name, 'purple'),
            linestyle=model_styles['retrained']['linestyle'],
            alpha=model_styles['retrained']['alpha'],
            linewidth=model_styles['retrained']['linewidth'],
            label=f'Retrained Model (AUC = {retrained_auc:.3f})'
        )

        # Plot the diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1)

        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve for {class_name} (n={n_samples})')
        ax.legend(loc="lower right")

        # Print AUC values
        print(f"AUC for {class_name}:")
        print(f"  Base Model: {base_auc:.4f}")
        print(f"  Retrained Model: {retrained_auc:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the main title
    plt.suptitle('ROC Curves for Pediatric Cohort: Base vs. Retrained Model on Test Set', fontsize=16)

    # Save the combined figure
    combined_path_svg = os.path.join(plots_dir, 'roc_pediatric_base_vs_retrained.svg')
    fig.savefig(combined_path_svg, format='svg', bbox_inches='tight')

    # Also save as PNG for easier viewing
    combined_path_png = os.path.join(plots_dir, 'roc_pediatric_base_vs_retrained.png')
    fig.savefig(combined_path_png, dpi=300, bbox_inches='tight')

    print(f"Saved plots to: {combined_path_svg} and {combined_path_png}")
    plt.close(fig)

def extract_and_save_roc_source_data(data, valid_classes, config):
    """Extract and save ROC source data to Excel files."""
    plots_dir = os.path.join(config['root_results'], '9_auc_retrain_kids')

    # Filter data to only include valid classes
    data_filtered = data[data['class'].isin(valid_classes)].copy()

    # Extract source data for base model
    datasets_base = {
        'base_model': ('Base Model', data_filtered)
    }
    custom_cols_base = {class_name: f'prediction.base.{class_name}' for class_name in valid_classes}
    save_roc_source_data_to_excel(datasets_base, plots_dir, 'pediatric', valid_classes, custom_cols_base)

    # Extract source data for retrained model
    datasets_retrained = {
        'retrained_model': ('Retrained Model', data_filtered)
    }
    save_roc_source_data_to_excel(datasets_retrained, plots_dir, 'pediatric', valid_classes)

def main():
    """Main function to run the analysis."""
    print("Starting AUC analysis for pediatric retraining...")

    # Load configuration
    config = load_config()
    print(f"Using root_dir: {config['root_dir']}")
    print(f"Using root_results: {config['root_results']}")

    # Load prediction data
    df = load_predict_data(config)

    # Get class counts and filter classes with insufficient samples
    class_counts = df['class'].value_counts().to_dict()
    print("\nClass distribution in pediatric data:")
    print(df['class'].value_counts())

    # Filter classes with too few samples
    valid_classes = filter_classes_by_count(df)

    # Calculate ROC curves for valid classes
    roc_results = calculate_roc_curves(df, valid_classes)

    # Generate and save plots
    plot_roc_curves(roc_results, class_counts, valid_classes, config)

    # Extract and save ROC source data
    extract_and_save_roc_source_data(df, valid_classes, config)

    print("\nAUC analysis complete.")

if __name__ == "__main__":
    main()
