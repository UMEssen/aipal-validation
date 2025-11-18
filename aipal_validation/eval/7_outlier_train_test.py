import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import logging
# FIGURE 4 d+e in Manuscript

# Import the outlier detector
from aipal_validation.outlier import MulticentricOutlierDetector

# Import utility functions
from .util import save_roc_source_data_to_excel

# Set up logging
def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'outlier_detection.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return log_path

# Add proper font settings for better text rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def calculate_roc_curves(data, classes=["AML", "APL", "ALL"]):
    """Calculate ROC curves for all classes."""
    # Filter classes to only those that exist in the data
    available_classes = [c for c in classes if c in data['class'].unique()]

    y = label_binarize(data['class'], classes=available_classes)

    # Convert predictions to a similar format for ROC calculation
    y_score = data[[f"prediction.{c}" for c in available_classes]].values

    # Compute ROC curve and ROC area for each class
    results = {}
    for i, class_name in enumerate(available_classes):
        fpr, tpr, _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        results[class_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }

    return results

def plot_roc_curves(train_results, train_counts, test_results, test_counts, classes, tag, plots_dir):
    """Create and save ROC curve plots in a 2x3 subplot layout."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    class_colors = {'AML': '#E41A1C', 'APL': '#4DAF4A', 'ALL': '#377EB8'}
    dataset_styles = {
        'Test Data': {'linestyle': '-', 'alpha': 0.9},
        'After Outlier Detection': {'linestyle': '--', 'alpha': 0.9}
    }

    # Process both sets
    for row, (set_type, results, counts) in enumerate([
        ("Training Set", train_results, train_counts),
        ("Test Set", test_results, test_counts)
    ]):
        for col, class_name in enumerate(classes):
            ax = axes[row, col]

            n_test = counts['Test Data'].get(class_name, 0)
            n_clean = counts['After Outlier Detection'].get(class_name, 0)

            if class_name not in results['test'] or class_name not in results['clean']:
                ax.text(0.5, 0.5, f"No data for {class_name}", ha='center', va='center')
                continue

            # Plot ROC curve for Test Data
            ax.plot(
                results['test'][class_name]['fpr'],
                results['test'][class_name]['tpr'],
                color=class_colors[class_name],
                linestyle=dataset_styles['Test Data']['linestyle'],
                alpha=dataset_styles['Test Data']['alpha'],
                label=f'Before Outlier Detection (AUC = {results["test"][class_name]["auc"]:.2f}, n={n_test})'
            )

            # Plot ROC curve for Clean Data
            ax.plot(
                results['clean'][class_name]['fpr'],
                results['clean'][class_name]['tpr'],
                color=class_colors[class_name],
                linestyle=dataset_styles['After Outlier Detection']['linestyle'],
                alpha=dataset_styles['After Outlier Detection']['alpha'],
                label=f'After Outlier Detection (AUC = {results["clean"][class_name]["auc"]:.2f}, n={n_clean})'
            )

            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{class_name} ({set_type})')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Log AUC values
            logging.info(f"AUC for {class_name} ({set_type}):")
            logging.info(f"  Before Outlier Detection: {results['test'][class_name]['auc']:.4f} (n={n_test})")
            logging.info(f"  After Outlier Detection: {results['clean'][class_name]['auc']:.4f} (n={n_clean})")

    plt.suptitle(f'ROC Curves for {tag} Cohorts by Leukemia Type and Dataset', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plots
    combined_path = os.path.join(plots_dir, f'roc_{tag}_outlier_combined.png')
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    combined_path_svg = os.path.join(plots_dir, f'roc_{tag}_outlier_combined.svg')
    fig.savefig(combined_path_svg, format='svg', bbox_inches='tight')
    logging.info(f"Saved combined plot to: {combined_path}")
    plt.close(fig)

def run_analysis(is_adult: bool):
    """Generate ROC plots comparing model performance before and after outlier detection for a specific cohort."""
    tag = "Adult" if is_adult else "Pediatric"
    print(f"--- Running Train/Test Outlier Analysis for {tag} Cohort ---")

    # Set up paths
    config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")
    # Load config to update is_adult
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['is_adult'] = is_adult # Ensure config reflects the current cohort

    # Use updated config for paths
    plots_dir = os.path.join(config['root_results'], f'7_outlier_train_test_{tag.lower()}')
    log_dir = os.path.join(plots_dir, 'logs')
    os.makedirs(plots_dir, exist_ok=True)

    # Setup logging for this cohort
    setup_logging(log_dir)
    logging.info(f"Starting outlier detection analysis for {tag}")
    logging.info(f"Saving plots to: {plots_dir}")
    logging.info(f"Using config from: {config_path} with is_adult={is_adult}")

    # Initialize outlier detector for training set
    logging.info(f"\nProcessing training set ({tag}, high confidence predictions > 0.9):")
    # Ensure detector uses the config reflecting the correct cohort
    detector_train = MulticentricOutlierDetector(config_path)
    # Check if data loading was successful within the detector
    if detector_train.df is None or detector_train.df.empty:
        logging.error(f"Failed to load data for {tag} cohort in detector_train. Skipping.")
        return
    # Prepare the split using the loaded df within the detector
    detector_train._prepare_train_test(confidence_threshold=0.9)
    # Check if training/test sets are created properly
    if detector_train.df_train is None or detector_train.df_train.empty or \
       detector_train.df_test is None or detector_train.df_test.empty:
       logging.warning(f"Insufficient data to create train/test split for {tag} training set outlier detection. Skipping.")
       return
    logging.info(f"Training set for outlier model training: {len(detector_train.df_train)} samples")
    logging.info(f"Test set for outlier detection (high confidence): {len(detector_train.df_test)} samples")
    detector_train.train_outlier_models()
    detector_train.detect_and_evaluate()

    train_clean = detector_train.df_test[detector_train.df_test["outlier"] == 0].copy()
    train_test = detector_train.df_test.copy()

    # Initialize new detector for test set
    logging.info(f"\nProcessing test set ({tag}, low confidence predictions < 0.9):")
    # Ensure detector uses the config reflecting the correct cohort
    detector_test = MulticentricOutlierDetector(config_path)
    # Check if data loading was successful
    if detector_test.df is None or detector_test.df.empty:
        logging.error(f"Failed to load data for {tag} cohort in detector_test. Skipping.")
        return
    # For test set, we swap the train/test split logic by using the opposite mask
    prediction_cols = ["prediction.AML", "prediction.APL", "prediction.ALL"]
    valid_pred_cols = [p for p in prediction_cols if p in detector_test.df.columns]
    if not valid_pred_cols:
        logging.error(f"No prediction columns found for {tag} cohort in detector_test. Skipping.")
        return
    test_mask = (detector_test.df[valid_pred_cols].max(axis=1) < 0.9)
    # Use correctly predicted high confidence samples for training the outlier model
    detector_test.df_train = detector_test.df[~test_mask & (detector_test.df["class"] == detector_test.df["predicted_class"])]
    detector_test.df_test = detector_test.df[test_mask]

    # Check if training/test sets are created properly
    if detector_test.df_train is None or detector_test.df_train.empty or \
       detector_test.df_test is None or detector_test.df_test.empty:
       logging.warning(f"Insufficient data to create train/test split for {tag} test set outlier detection. Skipping.")
       return
    logging.info(f"Training set for outlier model training: {len(detector_test.df_train)} samples")
    logging.info(f"Test set for outlier detection (low confidence): {len(detector_test.df_test)} samples")
    detector_test.train_outlier_models()
    detector_test.detect_and_evaluate()

    test_clean = detector_test.df_test[detector_test.df_test["outlier"] == 0].copy()
    test_test = detector_test.df_test.copy()

    # Calculate ROC curves for both sets
    train_results = {
        'test': calculate_roc_curves(train_test),
        'clean': calculate_roc_curves(train_clean)
    }
    train_counts = {
        'Test Data': train_test['class'].value_counts().to_dict(),
        'After Outlier Detection': train_clean['class'].value_counts().to_dict()
    }

    test_results = {
        'test': calculate_roc_curves(test_test),
        'clean': calculate_roc_curves(test_clean)
    }
    test_counts = {
        'Test Data': test_test['class'].value_counts().to_dict(),
        'After Outlier Detection': test_clean['class'].value_counts().to_dict()
    }

    # Create combined plot
    classes = ["AML", "APL", "ALL"]
    plot_roc_curves(
        train_results, train_counts,
        test_results, test_counts,
        classes, tag, plots_dir
    )

    # Extract and save ROC source data to Excel
    datasets = {
        'before_outlier_detection': ('Before Outlier Detection (High Confidence)', train_test),
        'after_outlier_detection': ('After Outlier Detection (High Confidence)', train_clean),
        'low_conf_before_outlier_detection': ('Low Confidence Before Outlier Detection', test_test),
        'low_conf_after_outlier_detection': ('Low Confidence After Outlier Detection', test_clean)
    }
    save_roc_source_data_to_excel(datasets, plots_dir, tag, classes)

    # Log summary statistics
    for set_type, test_df, clean_df in [
        (f"Training Set ({tag}, high conf)", train_test, train_clean),
        (f"Test Set ({tag}, low conf)", test_test, test_clean)
    ]:
        logging.info(f"\n{set_type} Summary:")
        logging.info(f"Test samples: {len(test_df)}")
        # Avoid division by zero if test_df is empty
        percentage_retained = (len(clean_df)/len(test_df)*100) if len(test_df) > 0 else 0
        logging.info(f"Samples after outlier detection: {len(clean_df)} ({percentage_retained:.1f}% of test)")

        logging.info(f"\n{set_type} class distribution in test data:")
        logging.info(test_df['class'].value_counts())

        logging.info(f"\n{set_type} class distribution after outlier detection:")
        logging.info(clean_df['class'].value_counts())

def main():
    """Generate ROC plots comparing model performance before and after outlier detection for both cohorts."""
    print("Starting Train/Test Outlier analysis...")
    run_analysis(is_adult=True)
    run_analysis(is_adult=False)
    print("\nTrain/Test Outlier analysis finished for both cohorts.")

if __name__ == "__main__":
    main()
