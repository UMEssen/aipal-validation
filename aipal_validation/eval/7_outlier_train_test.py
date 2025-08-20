import matplotlib.pyplot as plt
import yaml
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score
import logging

# Import the outlier checker
from aipal_validation.outlier.check_outlier import OutlierChecker

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

def load_data(config_path, is_adult):
    """Load and preprocess data from multiple centers"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    root_path = '/local/work/merengelke/aipal/'
    paths = [f"{root_path}{cc}/aipal/predict.csv" for cc in config['cities_countries']]

    dfs = []
    for path in paths:
        try:
            df_small = pd.read_csv(path)
            df_small['city_country'] = path.split('/')[-3]
            dfs.append(df_small)
        except FileNotFoundError:
            logging.warning(f"File not found: {path}")
            continue

    df = pd.concat(dfs, ignore_index=True)

    # Age filtering
    age_filter = df['age'] > 18 if is_adult else df['age'] <= 18
    df = df[age_filter]

    # Clean data
    df['class'] = df['class'].str.strip()
    df = df.groupby('city_country').filter(lambda x: len(x) > 30)

    # Add predicted class
    prediction_cols = ['prediction.ALL', 'prediction.AML', 'prediction.APL']
    df['predicted_class'] = df[prediction_cols].idxmax(axis=1).str.replace('prediction.', '', regex=False)

    return df, config

def apply_outlier_detection(df, outlier_checker):
    """Apply outlier detection to dataframe using OutlierChecker (batch processing)"""
    try:
        # Use the new batch processing method - much faster!
        return outlier_checker.check_dataframe(df)
    except Exception as e:
        logging.error(f"Error during batch outlier detection: {e}")
        # Fallback to sample-by-sample processing if batch fails
        logging.warning("Falling back to sample-by-sample processing...")

        outlier_flags = []
        for idx, row in df.iterrows():
            try:
                sample_data = row.to_dict()
                results = outlier_checker.check_sample(sample_data)
                actual_class = row['class']
                is_outlier = results.get(actual_class, {}).get('is_outlier', False)
                outlier_flags.append(1 if is_outlier else 0)
            except Exception as sample_error:
                logging.warning(f"Error processing sample {idx}: {sample_error}")
                outlier_flags.append(0)

        df_copy = df.copy()
        df_copy['outlier'] = outlier_flags
        return df_copy

def split_train_test_sets(df, confidence_threshold=0.9):
    """Split data into training and test sets based on confidence"""
    prediction_cols = ['prediction.ALL', 'prediction.AML', 'prediction.APL']

    # High confidence mask (>= threshold AND correctly predicted)
    high_conf_mask = (
        (df[prediction_cols].max(axis=1) >= confidence_threshold) &
        (df['class'] == df['predicted_class'])
    )

    # Low confidence mask (< threshold)
    low_conf_mask = df[prediction_cols].max(axis=1) < confidence_threshold

    train_set = df[high_conf_mask].copy()
    test_set = df[low_conf_mask].copy()

    return train_set, test_set

def calculate_comprehensive_metrics(df, classes=["AML", "APL", "ALL"]):
    """Calculate comprehensive metrics including PPV, NPV, sensitivity, specificity"""
    metrics = {}

    # Get available classes
    available_classes = [c for c in classes if c in df['class'].unique()]

    if len(available_classes) == 0:
        return metrics

    # Overall accuracy
    overall_accuracy = accuracy_score(df['class'], df['predicted_class'])
    metrics['overall'] = {'accuracy': overall_accuracy}

    # Per-class metrics
    for target_class in available_classes:
        # Create binary classification for this class
        y_true_binary = (df['class'] == target_class).astype(int)
        y_pred_binary = (df['predicted_class'] == target_class).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        # Calculate metrics
        # PPV (Positive Predictive Value) = Precision = TP / (TP + FP)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        # NPV (Negative Predictive Value) = TN / (TN + FN)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # F1 Score
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0

        # Accuracy for this class
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        metrics[target_class] = {
            'ppv': ppv,
            'npv': npv,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'accuracy': accuracy,
            'support': tp + fn,  # Total true positives for this class
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }

    return metrics

def calculate_roc_curves(data, classes=["AML", "APL", "ALL"]):
    """Calculate ROC curves for all classes."""
    # Filter classes to only those that exist in the data
    available_classes = [c for c in classes if c in data['class'].unique()]

    if len(available_classes) == 0:
        return {}

    y = label_binarize(data['class'], classes=available_classes)

    # Convert predictions to a similar format for ROC calculation
    y_score = data[[f"prediction.{c}" for c in available_classes if f"prediction.{c}" in data.columns]].values

    # Handle case where we have only one class
    if len(available_classes) == 1:
        return {}

    # Compute ROC curve and ROC area for each class
    results = {}
    for i, class_name in enumerate(available_classes):
        if i < y_score.shape[1]:
            fpr, tpr, _ = roc_curve(y[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            results[class_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }

    return results

def log_comprehensive_metrics(before_df, after_df, set_name):
    """Log comprehensive metrics comparison"""
    logging.info(f"\n{'='*60}")
    logging.info(f"COMPREHENSIVE METRICS FOR {set_name.upper()}")
    logging.info(f"{'='*60}")

    before_metrics = calculate_comprehensive_metrics(before_df)
    after_metrics = calculate_comprehensive_metrics(after_df)

    # Log overall accuracy
    if 'overall' in before_metrics and 'overall' in after_metrics:
        before_acc = before_metrics['overall']['accuracy']
        after_acc = after_metrics['overall']['accuracy']
        improvement = after_acc - before_acc
        logging.info("OVERALL ACCURACY:")
        logging.info(f"  Before: {before_acc:.4f}")
        logging.info(f"  After:  {after_acc:.4f}")
        logging.info(f"  Change: {improvement:+.4f}")

    # Log per-class metrics
    for class_name in ["AML", "APL", "ALL"]:
        if class_name in before_metrics and class_name in after_metrics:
            before = before_metrics[class_name]
            after = after_metrics[class_name]

            logging.info(f"\n{class_name} CLASS METRICS:")
            logging.info(f"  Support: {before['support']} → {after['support']} samples")

            metrics_to_compare = [
                ('PPV (Precision)', 'ppv'),
                ('NPV', 'npv'),
                ('Sensitivity (Recall)', 'sensitivity'),
                ('Specificity', 'specificity'),
                ('F1-Score', 'f1_score'),
                ('Accuracy', 'accuracy')
            ]

            for metric_name, metric_key in metrics_to_compare:
                before_val = before[metric_key]
                after_val = after[metric_key]
                improvement = after_val - before_val
                logging.info(f"  {metric_name:20s}: {before_val:.4f} → {after_val:.4f} ({improvement:+.4f})")

def plot_roc_curves(train_results, train_counts, test_results, test_counts, classes, tag, plots_dir):
    """Create and save ROC curve plots in a 2x3 subplot layout."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    class_colors = {'AML': '#E41A1C', 'APL': '#4DAF4A', 'ALL': '#377EB8'}
    dataset_styles = {
        'Before': {'linestyle': '-', 'alpha': 0.9},
        'After': {'linestyle': '--', 'alpha': 0.9}
    }

    # Process both sets
    for row, (set_type, results, counts) in enumerate([
        ("Training Set", train_results, train_counts),
        ("Test Set", test_results, test_counts)
    ]):
        for col, class_name in enumerate(classes):
            ax = axes[row, col]

            n_before = counts['before'].get(class_name, 0)
            n_after = counts['after'].get(class_name, 0)

            if class_name not in results['before'] or class_name not in results['after']:
                ax.text(0.5, 0.5, f"No data for {class_name}", ha='center', va='center')
                ax.set_title(f'{class_name} ({set_type})')
                continue

            # Plot ROC curve for Before Outlier Detection
            ax.plot(
                results['before'][class_name]['fpr'],
                results['before'][class_name]['tpr'],
                color=class_colors[class_name],
                linestyle=dataset_styles['Before']['linestyle'],
                alpha=dataset_styles['Before']['alpha'],
                label=f'Before Outlier Detection (AUC = {results["before"][class_name]["auc"]:.2f}, n={n_before})'
            )

            # Plot ROC curve for After Outlier Detection
            ax.plot(
                results['after'][class_name]['fpr'],
                results['after'][class_name]['tpr'],
                color=class_colors[class_name],
                linestyle=dataset_styles['After']['linestyle'],
                alpha=dataset_styles['After']['alpha'],
                label=f'After Outlier Detection (AUC = {results["after"][class_name]["auc"]:.2f}, n={n_after})'
            )

            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{class_name} ({set_type})')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f'ROC Curves for {tag} Cohorts by Leukemia Type and Dataset', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plots
    combined_path = os.path.join(plots_dir, f'roc_{tag}_outlier_combined.png')
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    combined_path_svg = os.path.join(plots_dir, f'roc_{tag}_outlier_combined.svg')
    fig.savefig(combined_path_svg, format='svg', bbox_inches='tight')
    logging.info(f"Saved combined plot to: {combined_path}")
    plt.close(fig)

def run_analysis(is_adult: bool, model_dir: str):
    """Generate ROC plots comparing model performance before and after outlier detection for a specific cohort."""
    tag = "Adult" if is_adult else "Pediatric"
    print(f"--- Running Train/Test Outlier Analysis for {tag} Cohort ---")

    # Set up paths
    config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")

    # Load and update config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['is_adult'] = is_adult  # Ensure config reflects the current cohort

    # Use updated config for paths
    plots_dir = os.path.join(config['root_results'], f'7_outlier_train_test_{tag.lower()}')
    log_dir = os.path.join(plots_dir, 'logs')
    os.makedirs(plots_dir, exist_ok=True)

    # Setup logging for this cohort
    setup_logging(log_dir)
    logging.info(f"Starting outlier detection analysis for {tag}")
    logging.info(f"Saving plots to: {plots_dir}")
    logging.info(f"Using config from: {config_path} with is_adult={is_adult}")
    logging.info(f"Using models from: {model_dir}")

    # Load data
    df, _ = load_data(config_path, is_adult)
    if df.empty:
        logging.error(f"No data found for {tag} cohort. Skipping.")
        return

    logging.info(f"Loaded {len(df)} samples for {tag} cohort")

    # Initialize outlier checker
    outlier_checker = OutlierChecker()
    outlier_checker.load_models(model_dir, config_path)
    logging.info(f"Loaded outlier models from {model_dir}")

    # Split into training and test sets
    train_set, test_set = split_train_test_sets(df, confidence_threshold=0.9)

    if train_set.empty or test_set.empty:
        logging.warning(f"Insufficient data to create train/test split for {tag} cohort. Skipping.")
        return

    logging.info(f"Training set: {len(train_set)} samples (high confidence)")
    logging.info(f"Test set: {len(test_set)} samples (low confidence)")

    # Apply outlier detection to both sets
    logging.info("Applying outlier detection to training set...")
    train_with_outliers = apply_outlier_detection(train_set, outlier_checker)

    logging.info("Applying outlier detection to test set...")
    test_with_outliers = apply_outlier_detection(test_set, outlier_checker)

    # Create clean datasets (without outliers)
    train_clean = train_with_outliers[train_with_outliers["outlier"] == 0].copy()
    test_clean = test_with_outliers[test_with_outliers["outlier"] == 0].copy()

    # Log comprehensive metrics
    log_comprehensive_metrics(train_set, train_clean, f"Training Set ({tag}, high conf)")
    log_comprehensive_metrics(test_set, test_clean, f"Test Set ({tag}, low conf)")

    # Calculate ROC curves for both sets
    train_results = {
        'before': calculate_roc_curves(train_set),
        'after': calculate_roc_curves(train_clean)
    }
    train_counts = {
        'before': train_set['class'].value_counts().to_dict(),
        'after': train_clean['class'].value_counts().to_dict()
    }

    test_results = {
        'before': calculate_roc_curves(test_set),
        'after': calculate_roc_curves(test_clean)
    }
    test_counts = {
        'before': test_set['class'].value_counts().to_dict(),
        'after': test_clean['class'].value_counts().to_dict()
    }

    # Create combined plot
    classes = ["AML", "APL", "ALL"]
    plot_roc_curves(
        train_results, train_counts,
        test_results, test_counts,
        classes, tag, plots_dir
    )

    # Log AUC summary
    logging.info(f"\n{'='*60}")
    logging.info(f"AUC SUMMARY FOR {tag.upper()} COHORT")
    logging.info(f"{'='*60}")

    for set_type, results in [("Training Set", train_results), ("Test Set", test_results)]:
        logging.info(f"\n{set_type}:")
        for class_name in classes:
            if class_name in results['before'] and class_name in results['after']:
                before_auc = results['before'][class_name]['auc']
                after_auc = results['after'][class_name]['auc']
                improvement = after_auc - before_auc
                logging.info(f"  {class_name}: {before_auc:.4f} → {after_auc:.4f} ({improvement:+.4f})")

def main():
    """Generate ROC plots comparing model performance before and after outlier detection for adult cohort."""
    print("Starting Train/Test Outlier analysis...")

    # Path to the trained outlier models
    adult_model_dir = "/home/merengelke/aipal_validation/aipal_validation/outlier"

    # Check if model directory exists
    if not os.path.exists(adult_model_dir):
        print(f"Adult model directory not found: {adult_model_dir}")
        return

    # Verify required model files exist
    required_files = [
        "iso_forest_ALL.pkl", "iso_forest_AML.pkl", "iso_forest_APL.pkl",
        "lof_ALL.pkl", "lof_AML.pkl", "lof_APL.pkl",
        "scaler.pkl", "imputer.pkl"
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(adult_model_dir, file)):
            missing_files.append(file)

    if missing_files:
        print(f"Missing required model files: {missing_files}")
        return

    print(f"Using models from: {adult_model_dir}")
    run_analysis(is_adult=True, model_dir=adult_model_dir)
    print("\nTrain/Test Outlier analysis finished for adult cohort.")

if __name__ == "__main__":
    main()
