import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import os
import sys
import importlib.util
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.gridspec as gridspec
import argparse
import logging
from datetime import datetime

# Add proper font settings for better text rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.titlesize'] = 10

# Add the parent directory to the path so we can import the outlier module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jupyter/publish'))

class CityOutlierAnalyzer:
    """
    Class to analyze outlier detection for each city cohort
    and generate plots comparing before and after outlier detection.
    """

    def __init__(self, config_path, city, log_dir):
        """
        Initialize the analyzer with the configuration file.

        Args:
            config_path (str): Path to the YAML configuration file
            city (str): Name of the city to analyze
            log_dir (str): Directory to save logs
        """
        self.config_path = config_path
        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        self.city = city.lower()
        self.base_classes = ["ALL", "AML", "APL"]  # All possible classes
        self.class_colors = {'AML': 'red', 'APL': 'green', 'ALL': 'blue'}

        # Set up paths
        self.root_path = "/local/work/merengelke/aipal/"

        # Set up logging
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self._setup_logger()

        # Load data
        self.city_df = self._load_city_data()
        self.logger.info(f"Loaded {len(self.city_df)} samples from {city}")

        # Determine available classes for this city
        self.classes = sorted(self.city_df["class"].unique())
        self.logger.info(f"Available classes in {city}: {self.classes}")

        # Store original data for comparison
        self.original_df = self.city_df.copy()

    def _setup_logger(self):
        """Set up logging configuration"""
        logger = logging.getLogger(f'outlier_analysis_{self.city}')
        logger.setLevel(logging.INFO)

        # Create file handler
        log_file = os.path.join(self.log_dir, f'outlier_analysis_{self.city}.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _load_city_data(self):
        """Load and preprocess data for the specific city"""
        try:
            # Try to load all data and filter for the specific city
            paths = [
                f"{self.root_path}{cc}/aipal/predict.csv"
                for cc in self.config["cities_countries"]
            ]

            dfs = []
            for path in paths:
                try:
                    df_small = pd.read_csv(path)
                    df_small["city_country"] = path.split("/")[-3]
                    dfs.append(df_small)
                except FileNotFoundError:
                    self.logger.warning(f"File not found: {path}")
                    continue

            all_df = pd.concat(dfs)

            # Filter for the specific city
            city_df = all_df[all_df["city_country"].str.lower() == self.city]

            if len(city_df) == 0:
                raise ValueError(f"No data found for {self.city} in the combined dataset")

            # Age filtering
            age_filter = city_df["age"] > 18 if self.config["is_adult"] else city_df["age"] <= 18
            city_df = city_df[age_filter]

            # Clean data
            city_df["class"] = city_df["class"].str.strip()

            return city_df

        except Exception as e:
            self.logger.error(f"Error loading {self.city} data: {e}")
            raise

    def _compute_roc_curves(self, df):
        """Compute ROC curves for each class."""
        results = {}

        # Only compute ROC curves for classes that exist in this city
        for cls in self.classes:
            # Create binary labels for this class
            y_true = (df["class"] == cls).astype(int)

            # Check if we have both positive and negative samples
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, df[f"prediction.{cls}"])
                roc_auc = auc(fpr, tpr)
                results[cls] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc
                }
            else:
                results[cls] = None
                self.logger.warning(f"Not enough samples to compute ROC curve for class {cls}")

        return results

    def run_outlier_detection(self):
        """Run the outlier detection algorithm on the city dataset."""
        try:
            # Check if we have any samples for this city
            if len(self.city_df) == 0:
                self.logger.warning(f"No samples found for {self.city}. Skipping analysis.")
                return False

            # Check if we have any classes
            if len(self.classes) == 0:
                self.logger.warning(f"No classes found for {self.city}. Skipping analysis.")
                return False

            # Try to import the outlier detection module
            script_dir = os.path.dirname(os.path.abspath(__file__))
            outlier_file = os.path.join(script_dir, "jupyter/publish/9_outlier_iso_lof.py")

            MulticentricOutlierDetector = None

            # Approach 1: Load the module directly from file if it exists
            if os.path.exists(outlier_file):
                spec = importlib.util.spec_from_file_location("outlier_module", outlier_file)
                outlier_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(outlier_module)
                MulticentricOutlierDetector = outlier_module.MulticentricOutlierDetector
                self.logger.info(f"Successfully imported outlier detection module from {outlier_file}")

            # Approach 2: Try different import techniques if file loading failed
            if MulticentricOutlierDetector is None:
                jupyter_publish_path = os.path.join(script_dir, "jupyter/publish")
                if jupyter_publish_path not in sys.path:
                    sys.path.append(jupyter_publish_path)

                try:
                    import outlier_iso_lof
                    MulticentricOutlierDetector = outlier_iso_lof.MulticentricOutlierDetector
                    self.logger.info("Successfully imported outlier_iso_lof")
                except ImportError:
                    try:
                        outlier_module = importlib.import_module("9_outlier_iso_lof")
                        MulticentricOutlierDetector = outlier_module.MulticentricOutlierDetector
                        self.logger.info("Successfully imported 9_outlier_iso_lof")
                    except ImportError:
                        raise ImportError(f"Could not find outlier module. Checked: {outlier_file}")

            # Initialize and run the outlier detector
            detector = MulticentricOutlierDetector(self.config_path)

            # Override the detector's data loading to include all data
            all_data = detector.df.copy()

            # Split into city and non-city data
            city_mask = all_data["city_country"].str.lower() == self.city
            non_city_data = all_data[~city_mask].copy()

            # Set the detector to use only non-city data for training
            detector.df = non_city_data

            # Train the model on non-city data
            self.logger.info(f"\n[ Training Information ]")
            self.logger.info(f"Training outlier detection model on non-{self.city} data...")
            self.logger.info(f"Training on {len(non_city_data)} samples from other centers")
            detector._prepare_train_test(confidence_threshold=0.0)
            detector.train_outlier_models()

            # Now set the test data to be the city data
            self.logger.info(f"\n[ Testing Information ]")
            self.logger.info(f"Evaluating outlier detection on {self.city} data...")
            detector.df_test = self.city_df.copy()

            # Add a column to track correct/incorrect predictions before outlier removal
            prediction_cols = [f"prediction.{c}" for c in self.classes]  # Only use available classes
            if len(prediction_cols) > 0:  # Only proceed if we have prediction columns
                self.city_df["predicted_class"] = (
                    self.city_df[prediction_cols]
                    .idxmax(axis=1)
                    .str.replace("prediction.", "", regex=False)
                )
                self.city_df["correctly_predicted"] = (self.city_df["class"] == self.city_df["predicted_class"]).astype(int)
            else:
                self.logger.warning("No prediction columns found. Skipping prediction analysis.")
                return False

            # Run detection
            detector.detect_and_evaluate()

            # Store the results
            self.city_df = detector.df_test.copy()
            self.clean_df = detector.df_test[detector.df_test["outlier"] == 0].copy()
            self.outlier_df = detector.df_test[detector.df_test["outlier"] == 1].copy()

            # Log detailed analysis
            self.logger.info(f"\n[ Outlier Detection Results for {self.city.title()} ]")
            total_samples = len(self.city_df)
            outlier_samples = len(self.outlier_df)
            outlier_percentage = (outlier_samples / total_samples) * 100

            self.logger.info(f"Total samples: {total_samples}")
            self.logger.info(f"Detected outliers: {outlier_samples} ({outlier_percentage:.2f}%)")
            self.logger.info(f"Remaining samples: {total_samples - outlier_samples} ({100 - outlier_percentage:.2f}%)")

            # Log class-wise analysis
            self.logger.info("\nOutliers by class:")
            for cls in self.classes:
                cls_samples = sum(self.city_df["class"] == cls)
                cls_outliers = sum((self.city_df["class"] == cls) & (self.city_df["outlier"] == 1))
                if cls_samples > 0:
                    cls_percentage = (cls_outliers / cls_samples) * 100
                    self.logger.info(f"  {cls}: {cls_outliers}/{cls_samples} ({cls_percentage:.2f}%)")

            # Analysis of wrongly filtered samples
            self.logger.info("\n[ Analysis of Wrongly Filtered Samples ]")

            # Count correctly and incorrectly predicted samples before filtering
            correct_before = sum(self.city_df["correctly_predicted"] == 1)
            incorrect_before = sum(self.city_df["correctly_predicted"] == 0)

            # Count how many correct and incorrect predictions were filtered out
            correct_filtered = sum((self.city_df["correctly_predicted"] == 1) & (self.city_df["outlier"] == 1))
            incorrect_filtered = sum((self.city_df["correctly_predicted"] == 0) & (self.city_df["outlier"] == 1))

            # Calculate percentages
            correct_filtered_pct = (correct_filtered / correct_before) * 100 if correct_before > 0 else 0
            incorrect_filtered_pct = (incorrect_filtered / incorrect_before) * 100 if incorrect_before > 0 else 0

            self.logger.info(f"Correctly predicted samples before filtering: {correct_before}")
            self.logger.info(f"Incorrectly predicted samples before filtering: {incorrect_before}")
            self.logger.info(f"Correctly predicted samples filtered out: {correct_filtered} ({correct_filtered_pct:.2f}%)")
            self.logger.info(f"Incorrectly predicted samples filtered out: {incorrect_filtered} ({incorrect_filtered_pct:.2f}%)")

            # Per-class analysis of wrongly filtered samples
            self.logger.info("\nWrongly filtered samples by class:")
            for cls in self.classes:
                cls_correct = sum((self.city_df["class"] == cls) & (self.city_df["correctly_predicted"] == 1))
                cls_correct_filtered = sum(
                    (self.city_df["class"] == cls)
                    & (self.city_df["correctly_predicted"] == 1)
                    & (self.city_df["outlier"] == 1)
                )
                cls_correct_pct = (cls_correct_filtered / cls_correct) * 100 if cls_correct > 0 else 0

                cls_incorrect = sum((self.city_df["class"] == cls) & (self.city_df["correctly_predicted"] == 0))
                cls_incorrect_filtered = sum(
                    (self.city_df["class"] == cls)
                    & (self.city_df["correctly_predicted"] == 0)
                    & (self.city_df["outlier"] == 1)
                )
                cls_incorrect_pct = (cls_incorrect_filtered / cls_incorrect) * 100 if cls_incorrect > 0 else 0

                self.logger.info(f"  {cls}:")
                self.logger.info(f"    Correctly predicted filtered: {cls_correct_filtered}/{cls_correct} ({cls_correct_pct:.2f}%)")
                self.logger.info(f"    Incorrectly predicted filtered: {cls_incorrect_filtered}/{cls_incorrect} ({cls_incorrect_pct:.2f}%)")

            # Log classification report for original data
            self.logger.info("\n[ Classification Report - Original Data ]")
            y_true = self.original_df["class"]
            y_pred = self.original_df[prediction_cols].idxmax(axis=1)
            y_pred = y_pred.str.replace("prediction.", "")
            self.logger.info("\n" + classification_report(y_true, y_pred, labels=self.classes))

            # Log classification report for clean data
            self.logger.info("\n[ Classification Report - After Outlier Removal ]")
            y_true = self.clean_df["class"]
            y_pred = self.clean_df[prediction_cols].idxmax(axis=1)
            y_pred = y_pred.str.replace("prediction.", "")
            self.logger.info("\n" + classification_report(y_true, y_pred, labels=self.classes))

            # Log AUC scores
            roc_results = self._compute_roc_curves(self.clean_df)
            auc_scores = {cls: res['auc'] for cls, res in roc_results.items() if res is not None}
            self.logger.info("\n[ AUC Scores After Outlier Removal ]")
            for cls, score in auc_scores.items():
                self.logger.info(f"  {cls}: {score:.3f}")

            return True

        except Exception as e:
            self.logger.error(f"Error running outlier detection: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _plot_roc_curves(self, ax, df, title):
        """Plot ROC curves with improved styling elements."""
        roc_results = self._compute_roc_curves(df)

        for cls in self.classes:  # Only plot available classes
            if roc_results[cls] is not None:
                ax.plot(roc_results[cls]['fpr'], roc_results[cls]['tpr'],
                        color=self.class_colors[cls],
                        lw=2.2,
                        alpha=0.9,
                        solid_capstyle='round',
                        label=f'{cls} (AUC = {roc_results[cls]["auc"]:.2f})')

        ax.plot([0, 1], [0, 1], color='#444444',
                ls=':', lw=1.2, alpha=0.7)

        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate', fontsize=7.5, labelpad=1)
        ax.set_ylabel('True Positive Rate', fontsize=7.5, labelpad=1)

        ax.tick_params(axis='both', which='major',
                    labelsize=7, pad=2, width=0.6)
        ax.grid(True, ls=':', color='#cccccc', lw=0.6)

        for spine in ax.spines.values():
            spine.set_color('#888888')
            spine.set_lw(0.8)

        ax.set_title(title, fontsize=8.5, pad=8,
                    weight='medium', color='#333333')

        for cls in self.classes:  # Only annotate available classes
            if roc_results[cls] is not None:
                last_point = (roc_results[cls]['fpr'][-1],
                            roc_results[cls]['tpr'][-1])
                ax.text(last_point[0]-0.15, last_point[1]-0.05,
                        f"AUC={roc_results[cls]['auc']:.2f}",
                        color=self.class_colors[cls],
                        fontsize=6.5, ha='right',
                        bbox=dict(facecolor='white', alpha=0.8,
                                edgecolor='none', boxstyle='round,pad=0.1'))

    def generate_plots(self, output_dir, width=7.0, height=5.0, dpi=150):
        """Generate plots showing the effect of outlier detection on the city dataset."""
        plt.style.use('seaborn-v0_8-ticks')

        # Create figure
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        fig.suptitle(f"{self.city.title()} Cohort Outlier Analysis", fontsize=12, y=0.98)

        # Create grid layout
        gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.4,
                            left=0.1, right=0.95, top=0.9, bottom=0.1)

        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        # Plot ROC curves
        self._plot_roc_curves(ax1, self.original_df, f"Original {self.city.title()} Data")
        self._plot_roc_curves(ax2, self.clean_df, f"{self.city.title()} Data After Outlier Removal")

        ax1.legend(loc="lower right", fontsize=7, framealpha=0.7)
        ax2.legend(loc="lower right", fontsize=7, framealpha=0.7)

        # Plot additional analysis
        self._plot_outlier_analysis(ax3)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the plots
        output_base = os.path.join(output_dir, f"outlier_analysis_{self.city}")
        plt.savefig(f"{output_base}.png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        self.logger.info(f"Analysis plots saved to: {output_base}.png")
        return output_base

    def _plot_outlier_analysis(self, ax):
        """Plot additional analysis about the outliers in the bottom panel."""
        if not hasattr(self, 'outlier_df') or len(self.outlier_df) == 0:
            ax.text(0.5, 0.5, "No outliers detected or analysis not run",
                   ha='center', va='center', fontsize=10)
            return

        class_counts = self.original_df['class'].value_counts()
        outlier_class_counts = self.outlier_df['class'].value_counts()

        class_percentages = {}
        for cls in self.classes:  # Only plot available classes
            if cls in class_counts and cls in outlier_class_counts:
                class_percentages[cls] = (outlier_class_counts[cls] / class_counts[cls]) * 100
            else:
                class_percentages[cls] = 0

        x = np.arange(len(self.classes))
        bars = ax.bar(x, [class_percentages.get(cls, 0) for cls in self.classes],
                     color=[self.class_colors[cls] for cls in self.classes], alpha=0.7)

        overall_percentage = (len(self.outlier_df) / len(self.original_df)) * 100
        ax.axhline(y=overall_percentage, color='black', linestyle='--',
                  alpha=0.7, label=f'Overall ({overall_percentage:.1f}%)')

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

        for i, cls in enumerate(self.classes):
            if cls in outlier_class_counts and cls in class_counts:
                ax.text(i, 1, f'{outlier_class_counts[cls]}/{class_counts[cls]}',
                       ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(self.classes)
        ax.set_ylabel('Outlier Percentage (%)', fontsize=9)
        ax.set_title(f'Outlier Percentage by Class in {self.city.title()} Cohort', fontsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        ax.legend()
        ax.set_ylim(bottom=0)

def main():
    """Main function to run outlier analysis for all cities"""
    parser = argparse.ArgumentParser(description="Analyze outlier detection for all cities")
    parser.add_argument("--config", type=str, default="../aipal_validation/config/config_outlier.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Directory to save the output figures")
    parser.add_argument("--width", type=float, default=7.0,
                        help="Width of the figure in inches")
    parser.add_argument("--height", type=float, default=5.0,
                        help="Height of the figure in inches")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI (dots per inch) for the output figure")

    args = parser.parse_args()

    # Load configuration to get list of cities
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create output and log directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"outlier_analysis_{timestamp}")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set up main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'main.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('main')

    # Process each city
    processed_cities = []
    skipped_cities = []
    failed_cities = []

    for city in config['cities_countries']:
        logger.info(f"\nProcessing {city}...")
        try:
            analyzer = CityOutlierAnalyzer(args.config, city, log_dir)

            # Check if city has data before proceeding
            if len(analyzer.city_df) == 0:
                logger.warning(f"No data found for {city}. Skipping...")
                skipped_cities.append(city)
                continue

            success = analyzer.run_outlier_detection()

            if success:
                output_path = analyzer.generate_plots(output_dir, args.width, args.height, args.dpi)
                logger.info(f"Analysis complete for {city}. Results saved to: {output_path}")
                processed_cities.append(city)
            else:
                logger.warning(f"Outlier detection skipped for {city} due to insufficient data.")
                skipped_cities.append(city)

        except Exception as e:
            logger.error(f"Error processing {city}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_cities.append(city)
            continue

    # Print summary
    logger.info("\n=== Analysis Summary ===")
    logger.info(f"Successfully processed cities ({len(processed_cities)}): {', '.join(processed_cities)}")
    logger.info(f"Skipped cities due to insufficient data ({len(skipped_cities)}): {', '.join(skipped_cities)}")
    logger.info(f"Failed cities ({len(failed_cities)}): {', '.join(failed_cities)}")
    logger.info("\nAnalysis complete.")

if __name__ == "__main__":
    main()
