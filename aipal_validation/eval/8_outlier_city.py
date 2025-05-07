import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import logging
import sys
import math

from aipal_validation.outlier import MulticentricOutlierDetector
from aipal_validation.cohort.util import load_data

# Set up path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Add proper font settings for better text rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'outlier_city_analysis.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_file

def calculate_roc_curves(data, classes=["ALL", "AML", "APL"]):
    """Calculate ROC curves for all classes."""
    # Filter classes to only those that exist in the data
    available_classes = [c for c in classes if c in data['class'].unique()]

    if not available_classes:
        logging.warning("No available classes found in the data")
        return {}

    y = label_binarize(data['class'], classes=available_classes)

    # Ensure prediction columns exist for available classes
    pred_columns = [f"prediction.{c}" for c in available_classes if f"prediction.{c}" in data.columns]
    if not pred_columns:
        logging.warning("No prediction columns found for available classes")
        return {}

    # Convert predictions to a similar format for ROC calculation
    y_score = data[pred_columns].values

    # Compute ROC curve and ROC area for each class
    results = {}
    for i, class_name in enumerate(available_classes):
        # Skip if no samples for this class or prediction column doesn't exist
        if f"prediction.{class_name}" not in data.columns:
            logging.warning(f"Prediction column for {class_name} not found in data")
            continue

        # Skip if all predictions or all labels are the same (can't calculate ROC)
        if len(np.unique(y[:, i])) < 2 or len(np.unique(y_score[:, i])) < 2:
            logging.warning(f"Cannot calculate ROC for {class_name}: not enough unique values")
            continue

        try:
            fpr, tpr, _ = roc_curve(y[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            results[class_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
        except Exception as e:
            logging.warning(f"Error calculating ROC curve for {class_name}: {e}")
            continue

    return results

def plot_city_roc_curves(city_results, city_counts, classes, tag, plots_dir):
    """Create and save ROC curve plots by city."""
    class_colors = {'ALL': '#377EB8', 'AML': '#E41A1C', 'APL': '#4DAF4A'}

    # Create individual plot for each city
    for city, results in city_results.items():
        # We'll plot even if only test or only clean data is available
        if not results.get('test') and not results.get('clean'):
            logging.warning(f"Skipping {city} - no data available")
            continue

        fig, ax = plt.subplots(figsize=(12, 10))

        # Get available classes for this city
        available_test_classes = list(results.get('test', {}).keys())
        available_clean_classes = list(results.get('clean', {}).keys())

        # Calculate average AUCs for available classes
        test_aucs = [results['test'][c]['auc'] for c in available_test_classes] if results.get('test') else []
        clean_aucs = [results['clean'][c]['auc'] for c in available_clean_classes] if results.get('clean') else []

        test_auc = np.mean(test_aucs) if test_aucs else 0
        clean_auc = np.mean(clean_aucs) if clean_aucs else 0

        # Get sample counts
        test_count = sum(city_counts.get(city, {}).get('test', {}).values())
        clean_count = sum(city_counts.get(city, {}).get('clean', {}).values())

        # Add dummy lines for the legend headers
        ax.plot(
            [0, 1], [0, 1],
            label=f"Before Outlier Detection (avg AUC={test_auc:.2f}, n={test_count})",
            alpha=0
        )
        ax.plot(
            [0, 1], [0, 1],
            label=f"After Outlier Detection (avg AUC={clean_auc:.2f}, n={clean_count})",
            alpha=0
        )

        # Plot individual class curves before outlier detection
        if results.get('test'):
            for class_name in classes:
                if class_name in results['test']:
                    test_count_class = city_counts.get(city, {}).get('test', {}).get(class_name, 0)
                    if test_count_class == 0:
                        continue

                    ax.plot(
                        results['test'][class_name]['fpr'],
                        results['test'][class_name]['tpr'],
                        color=class_colors.get(class_name, 'gray'),
                        linestyle='-',
                        alpha=0.7,
                        label=f"Before - {class_name} (AUC={results['test'][class_name]['auc']:.2f}, n={test_count_class})"
                    )

        # Plot individual class curves after outlier detection
        if results.get('clean'):
            for class_name in classes:
                if class_name in results['clean']:
                    clean_count_class = city_counts.get(city, {}).get('clean', {}).get(class_name, 0)
                    if clean_count_class == 0:
                        continue

                    ax.plot(
                        results['clean'][class_name]['fpr'],
                        results['clean'][class_name]['tpr'],
                        color=class_colors.get(class_name, 'gray'),
                        linestyle='--',
                        alpha=0.7,
                        label=f"After - {class_name} (AUC={results['clean'][class_name]['auc']:.2f}, n={clean_count_class})"
                    )

        # Check if we actually plotted any data
        if not test_aucs and not clean_aucs:
            logging.warning(f"No valid ROC data for {city}, skipping plot")
            plt.close(fig)
            continue

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves for {city.title()} Before and After Outlier Detection')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save individual city plot
        city_name = city.replace(" ", "_").replace("/", "_").replace("\\", "_")
        plot_path = os.path.join(plots_dir, f'roc_{tag}_{city_name}.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_path_svg = os.path.join(plots_dir, f'roc_{tag}_{city_name}.svg')
        fig.savefig(plot_path_svg, format='svg', bbox_inches='tight')
        logging.info(f"Saved plot for {city} to: {plot_path}")
        plt.close(fig)

    # Create a combined plot with one line per city (average of classes) for comparison
    fig, ax = plt.subplots(figsize=(12, 10))

    cities_with_data = 0

    for city, results in city_results.items():
        # Skip cities with missing data
        if not results.get('test') and not results.get('clean'):
            continue

        # Get available classes for this city
        available_test_classes = list(results.get('test', {}).keys())
        available_clean_classes = list(results.get('clean', {}).keys())

        # Calculate average AUCs for available classes
        test_aucs = [results['test'][c]['auc'] for c in available_test_classes] if results.get('test') else []
        clean_aucs = [results['clean'][c]['auc'] for c in available_clean_classes] if results.get('clean') else []

        test_auc = np.mean(test_aucs) if test_aucs else 0
        clean_auc = np.mean(clean_aucs) if clean_aucs else 0

        # Skip if no valid AUC data
        if not test_aucs and not clean_aucs:
            continue

        cities_with_data += 1

        # Get sample counts
        test_count = sum(city_counts.get(city, {}).get('test', {}).values())
        clean_count = sum(city_counts.get(city, {}).get('clean', {}).values())

        # Add dummy lines for the legend
        ax.plot(
            [0, 1], [0, 1],
            label=f"{city} - Before (AUC={test_auc:.2f}, n={test_count})",
            alpha=0
        )
        ax.plot(
            [0, 1], [0, 1],
            label=f"{city} - After (AUC={clean_auc:.2f}, n={clean_count})",
            alpha=0
        )

        # Calculate combined ROC for all classes before outlier detection
        if results.get('test'):
            for class_name in available_test_classes:
                ax.plot(
                    results['test'][class_name]['fpr'],
                    results['test'][class_name]['tpr'],
                    color='C' + str(list(city_results.keys()).index(city) % 10),
                    linestyle='-',
                    alpha=0.5
                )

        # Calculate combined ROC for all classes after outlier detection
        if results.get('clean'):
            for class_name in available_clean_classes:
                ax.plot(
                    results['clean'][class_name]['fpr'],
                    results['clean'][class_name]['tpr'],
                    color='C' + str(list(city_results.keys()).index(city) % 10),
                    linestyle='--',
                    alpha=0.8
                )

    # Only save if we have data from at least one city
    if cities_with_data > 0:
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves by City Before and After Outlier Detection')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save combined plot
        combined_path = os.path.join(plots_dir, f'roc_{tag}_city_combined.png')
        fig.savefig(combined_path, dpi=300, bbox_inches='tight')
        combined_path_svg = os.path.join(plots_dir, f'roc_{tag}_city_combined.svg')
        fig.savefig(combined_path_svg, format='svg', bbox_inches='tight')
        logging.info(f"Saved combined plot to: {combined_path}")
    else:
        logging.warning("No cities with valid ROC data, skipping combined plot")

    plt.close(fig)

def plot_city_blood_roc_curves(city_results, city_counts, classes, tag, plots_dir):
    """Create and save ROC curve plots by city and blood type."""
    class_colors = {'ALL': '#377EB8', 'AML': '#E41A1C', 'APL': '#4DAF4A'}

    # Create a subdirectory for blood type plots
    blood_plots_dir = os.path.join(plots_dir, 'blood_type')
    os.makedirs(blood_plots_dir, exist_ok=True)
    logging.info(f"Creating blood type plots in: {blood_plots_dir}")

    # For each city, create plots for each class (blood type)
    for city, results in city_results.items():
        # Skip cities with completely missing data
        if not results.get('test') and not results.get('clean'):
            continue

        for class_name in classes:
            # Skip if class doesn't exist in this city's data
            if (class_name not in results.get('test', {}) and
                class_name not in results.get('clean', {})):
                logging.info(f"Skipping {city} - {class_name}: class not present in data")
                continue

            # Get counts if available
            test_count_class = city_counts.get(city, {}).get('test', {}).get(class_name, 0)
            clean_count_class = city_counts.get(city, {}).get('clean', {}).get(class_name, 0)

            # Skip if not enough samples in both before and after
            if test_count_class < 5 and clean_count_class < 5:
                logging.info(f"Skipping {city} - {class_name} due to insufficient samples")
                continue

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Initialize AUC values
            test_auc = 0
            clean_auc = 0

            # Plot test ROC curve if available
            if class_name in results.get('test', {}):
                test_auc = results['test'][class_name]['auc']
                ax.plot(
                    results['test'][class_name]['fpr'],
                    results['test'][class_name]['tpr'],
                    color=class_colors.get(class_name, 'gray'),
                    linestyle='-',
                    linewidth=2,
                    label=f"Before Outlier Detection (AUC={test_auc:.2f}, n={test_count_class})"
                )

            # Plot clean ROC curve if available
            if class_name in results.get('clean', {}):
                clean_auc = results['clean'][class_name]['auc']
                ax.plot(
                    results['clean'][class_name]['fpr'],
                    results['clean'][class_name]['tpr'],
                    color=class_colors.get(class_name, 'gray'),
                    linestyle='--',
                    linewidth=2,
                    label=f"After Outlier Detection (AUC={clean_auc:.2f}, n={clean_count_class})"
                )

            # Skip if no ROC curves were plotted
            if class_name not in results.get('test', {}) and class_name not in results.get('clean', {}):
                logging.warning(f"No valid ROC data for {city} - {class_name}, skipping plot")
                plt.close(fig)
                continue

            # Add diagonal reference line
            ax.plot([0, 1], [0, 1], 'k--', lw=1)

            # Configure plot
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve for {city.title()} - {class_name} Before and After Outlier Detection')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add percentage of samples retained if we have both before and after data
            if test_count_class > 0 and clean_count_class > 0:
                retention_pct = (clean_count_class / test_count_class * 100)
                plt.figtext(0.5, 0.01, f"Samples retained: {retention_pct:.1f}%",
                          ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

            plt.tight_layout()

            # Save plot
            city_name = city.replace(" ", "_").replace("/", "_").replace("\\", "_")
            plot_path = os.path.join(blood_plots_dir, f'roc_{tag}_{city_name}_{class_name}.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_path_svg = os.path.join(blood_plots_dir, f'roc_{tag}_{city_name}_{class_name}.svg')
            fig.savefig(plot_path_svg, format='svg', bbox_inches='tight')
            logging.info(f"Saved plot for {city} - {class_name} to: {plot_path}")
            plt.close(fig)

def plot_city_subplots(city_results, city_counts, classes, tag, plots_dir):
    """Create a single figure with subplots for all cities that have valid ROC curves."""
    class_colors = {'ALL': '#377EB8', 'AML': '#E41A1C', 'APL': '#4DAF4A'}

    # Only use cities that have ROC curves calculated
    valid_cities = list(city_results.keys())

    logging.info(f"Creating subplot with {len(valid_cities)} cities that have valid ROC data")

    if not valid_cities:
        logging.warning("No cities with valid ROC data for subplots")
        return

    # Calculate grid dimensions
    n_cities = len(valid_cities)
    n_cols = min(3, n_cities)  # At most 3 columns
    n_rows = math.ceil(n_cities / n_cols)

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
    fig.suptitle('ROC Curves by City Before and After Outlier Detection', fontsize=20)

    # Track the overall legend items
    all_legend_items = []
    all_legend_labels = []

    # Plot each city in its own subplot
    for i, city in enumerate(valid_cities):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        results = city_results[city]

        # Get available classes for this city
        available_test_classes = list(results.get('test', {}).keys())
        available_clean_classes = list(results.get('clean', {}).keys())

        # Calculate average AUCs for available classes
        test_aucs = [results['test'][c]['auc'] for c in available_test_classes] if results.get('test') else []
        clean_aucs = [results['clean'][c]['auc'] for c in available_clean_classes] if results.get('clean') else []

        test_auc = np.mean(test_aucs) if test_aucs else 0
        clean_auc = np.mean(clean_aucs) if clean_aucs else 0

        # Get sample counts
        test_count = sum(city_counts.get(city, {}).get('test', {}).values())
        clean_count = sum(city_counts.get(city, {}).get('clean', {}).values())

        # Set title for this subplot with sample counts
        ax.set_title(f'{city} (n={test_count}→{clean_count})', fontsize=14)

        # Plot individual class curves before outlier detection
        if results.get('test'):
            for class_name in classes:
                if class_name in results['test']:
                    test_count_class = city_counts.get(city, {}).get('test', {}).get(class_name, 0)
                    if test_count_class == 0:
                        continue

                    line, = ax.plot(
                        results['test'][class_name]['fpr'],
                        results['test'][class_name]['tpr'],
                        color=class_colors.get(class_name, 'gray'),
                        linestyle='-',
                        alpha=0.7,
                        label=f"Before - {class_name}"
                    )

                    # Only add to legend for the first city to avoid duplicates
                    if i == 0:
                        all_legend_items.append(line)
                        all_legend_labels.append(f"Before - {class_name}")

        # Plot individual class curves after outlier detection
        if results.get('clean'):
            for class_name in classes:
                if class_name in results['clean']:
                    clean_count_class = city_counts.get(city, {}).get('clean', {}).get(class_name, 0)
                    if clean_count_class == 0:
                        continue

                    line, = ax.plot(
                        results['clean'][class_name]['fpr'],
                        results['clean'][class_name]['tpr'],
                        color=class_colors.get(class_name, 'gray'),
                        linestyle='--',
                        alpha=0.7,
                        label=f"After - {class_name}"
                    )

                    # Only add to legend for the first city to avoid duplicates
                    if i == 0:
                        all_legend_items.append(line)
                        all_legend_labels.append(f"After - {class_name}")

        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1)

        # Add AUC values to the plot
        ax.text(0.98, 0.02, f"Before AUC: {test_auc:.2f}\nAfter AUC: {clean_auc:.2f}",
                va='bottom', ha='right', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Configure axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True, linestyle='--', alpha=0.4)

    # Hide empty subplots
    for i in range(len(valid_cities), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    # Add a single legend for the entire figure
    fig.legend(all_legend_items, all_legend_labels, loc='lower center',
               bbox_to_anchor=(0.5, 0), ncol=len(classes)*2, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend

    # Save combined subplot figure
    subplots_path = os.path.join(plots_dir, f'roc_{tag}_city_subplots.png')
    fig.savefig(subplots_path, dpi=300, bbox_inches='tight')
    subplots_path_svg = os.path.join(plots_dir, f'roc_{tag}_city_subplots.svg')
    fig.savefig(subplots_path_svg, format='svg', bbox_inches='tight')
    logging.info(f"Saved subplot figure to: {subplots_path}")
    plt.close(fig)

def run_analysis(is_adult: bool):
    """Generate ROC plots comparing model performance before and after outlier detection by city for a specific cohort."""
    tag = "Adult" if is_adult else "Pediatric"
    print(f"--- Running City Outlier Analysis for {tag} Cohort ---")

    # Set up paths
    config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")
    # Load config to update is_adult
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['is_adult'] = is_adult # Ensure config reflects the current cohort

    # Use updated config for paths
    plots_dir = os.path.join(config['root_results'], f'8_outlier_city_{tag.lower()}')
    log_dir = os.path.join(plots_dir, 'logs')
    os.makedirs(plots_dir, exist_ok=True)

    # Setup logging for this cohort
    setup_logging(log_dir)
    logging.info(f"Starting city-based outlier detection analysis for {tag}")
    logging.info(f"Saving plots to: {plots_dir}")
    logging.info(f"Using config from: {config_path} with is_adult={is_adult}")

    # Load data using util.py function for the correct cohort
    logging.info(f"Loading {tag.lower()} data using util.py load_data function")
    # Pass the updated config path or ensure load_data uses the correct flag
    df, _, features = load_data(config_path=config_path, is_adult=is_adult)
    print(df.city_country.unique())
    class_per_city = df.groupby(['city_country', 'class']).size().reset_index(name='count')
    print(class_per_city)
    logging.info(f"Loaded {tag.lower()} data: {len(df)} samples")

    # Make sure predicted_class is available for outlier detection
    if 'predicted_class' not in df.columns:
        # Use the class with highest prediction probability as predicted_class
        pred_cols = [col for col in df.columns if col.startswith('prediction.')]
        if pred_cols:
            df['predicted_class'] = df[pred_cols].idxmax(axis=1).str.replace('prediction.', '', regex=False)
            logging.info("Added 'predicted_class' based on highest prediction.")
        else:
            # If no prediction columns, use the actual class
            df['predicted_class'] = df['class']
            logging.warning("No prediction columns found, using actual 'class' as 'predicted_class'.")

    # Initialize outlier detector and train on highly confident samples
    logging.info("Training outlier detector on highly confident samples (>0.9)")
    # Ensure detector uses the config reflecting the correct cohort
    detector = MulticentricOutlierDetector(config_path)

    # Set up training data as high confidence predictions (>0.9) where predicted class matches actual class
    prediction_cols = ["prediction.ALL", "prediction.AML", "prediction.APL"]
    confidence_threshold = 0.9
    # Ensure prediction columns exist before filtering
    valid_pred_cols = [p for p in prediction_cols if p in df.columns]
    if not valid_pred_cols:
        logging.error("No prediction columns found in the dataframe. Cannot train outlier detector.")
        return

    high_confidence_mask = (df[valid_pred_cols].max(axis=1) > confidence_threshold)
    correct_prediction_mask = (df["class"] == df["predicted_class"])

    # Use high confidence + correct predictions for training
    detector.df_train = df[high_confidence_mask & correct_prediction_mask].copy()
    logging.info(f"Using {len(detector.df_train)} samples for training outlier models")

    # Check if training data is sufficient
    if len(detector.df_train) < 10: # Arbitrary threshold, adjust if needed
        logging.warning(f"Insufficient training data ({len(detector.df_train)} samples) for {tag} cohort. Skipping outlier detection.")
        return

    # Train outlier models
    detector.train_outlier_models()
    logging.info("Finished training outlier models")

    # Group data by city
    city_results = {}
    city_counts = {}

    # Get all cities in the data
    all_cities = df['city_country'].unique()
    logging.info(f"Found {len(all_cities)} cities in the {tag} dataset: {all_cities}")

    classes = ["ALL", "AML", "APL"]

    # Process each city
    for city in all_cities:
        # Get data for this city
        city_data = df[df['city_country'] == city].copy()

        # Skip cities with very few samples
        if len(city_data) < 10:
            logging.info(f"Skipping city {city} ({tag}) with only {len(city_data)} samples")
            continue

        logging.info(f"Processing city: {city} ({tag}, samples: {len(city_data)})")
        logging.info(f"Class distribution for {city} ({tag}): {city_data['class'].value_counts().to_dict()}")

        # Test dataset is all samples for this city
        detector.df_test = city_data

        # Detect outliers in the city data
        detector.detect_and_evaluate()

        # Get the data before and after outlier detection
        df_test = detector.df_test.copy()
        df_clean = detector.df_test[detector.df_test["outlier"] == 0].copy()

        logging.info(f"After outlier detection ({city}, {tag}): {len(df_clean)}/{len(df_test)} samples remain")
        logging.info(f"Clean class distribution for {city} ({tag}): {df_clean['class'].value_counts().to_dict()}")

        # Store counts
        city_counts[city] = {
            'test': df_test['class'].value_counts().to_dict(),
            'clean': df_clean['class'].value_counts().to_dict()
        }

        # Calculate ROC curves
        try:
            # Calculate for test data
            test_results = calculate_roc_curves(df_test, classes)
            clean_results = calculate_roc_curves(df_clean, classes)

            # Store results if we have ROC curves calculated for at least one class in either set
            if test_results or clean_results:
                city_results[city] = {
                    'test': test_results if test_results else {},
                    'clean': clean_results if clean_results else {}
                }
                logging.info(f"Successfully processed ROC curves for {city} ({tag})")
                if test_results:
                    logging.info(f"  Test classes: {list(test_results.keys())}")
                if clean_results:
                    logging.info(f"  Clean classes: {list(clean_results.keys())}")
            else:
                logging.warning(f"Skipping {city} ({tag}) - Could not calculate any valid ROC curves")
                continue

            # Log results
            outliers = (df_test['outlier'] == 1).sum()
            outlier_pct = outliers / len(df_test) * 100 if len(df_test) > 0 else 0
            logging.info(f"City: {city} ({tag}) - Detected {outliers} outliers ({outlier_pct:.1f}%)")

        except Exception as e:
            logging.error(f"Error calculating ROC curves for city {city} ({tag}): {e}")
            continue

    # Debug: print all cities with ROC curves
    logging.info(f"Cities with valid ROC curves ({tag}): {list(city_results.keys())}")

    # Skip individual city plots and blood type plots, only create the combined subplot
    logging.info(f"Creating combined subplot figure for {tag}...")
    if city_results: # Only plot if there are results
        plot_city_subplots(city_results, city_counts, classes, tag, plots_dir)
    else:
        logging.warning(f"No city results to plot for {tag} cohort.")

    # Log summary statistics
    logging.info(f"\nSummary by City ({tag}):")
    for city in city_results:
        logging.info(f"\nCity: {city}")
        test_count_city = sum(city_counts[city].get('test', {}).values())
        clean_count_city = sum(city_counts[city].get('clean', {}).values())

        logging.info(f"Test samples: {test_count_city}")
        logging.info(f"Samples after outlier detection: {clean_count_city}")

        # Calculate percentage preserved
        if test_count_city > 0:
            pct = clean_count_city / test_count_city * 100
            logging.info(f"Preservation rate: {pct:.1f}%)")

        # Log AUC before/after for each class
        for class_name in classes:
            test_exists = class_name in city_results[city].get('test', {})
            clean_exists = class_name in city_results[city].get('clean', {})

            if test_exists or clean_exists:
                test_auc = city_results[city]['test'][class_name]['auc'] if test_exists else "N/A"
                clean_auc = city_results[city]['clean'][class_name]['auc'] if clean_exists else "N/A"
                test_count = city_counts[city]['test'].get(class_name, 0)
                clean_count = city_counts[city]['clean'].get(class_name, 0)

                test_auc_str = f"{test_auc:.4f}" if isinstance(test_auc, float) else test_auc
                clean_auc_str = f"{clean_auc:.4f}" if isinstance(clean_auc, float) else clean_auc

                logging.info(f"  {class_name}: Before AUC={test_auc_str} (n={test_count}), After AUC={clean_auc_str} (n={clean_count})")

def main():
    """Runs the city-based outlier analysis for both adult and pediatric cohorts."""
    print("Starting City Outlier analysis...")
    run_analysis(is_adult=True)
    run_analysis(is_adult=False)
    print("\nCity Outlier analysis finished for both cohorts.")

if __name__ == "__main__":
    main()
