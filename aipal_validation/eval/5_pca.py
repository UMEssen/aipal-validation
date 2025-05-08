import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from util import load_data
import yaml

def select_features(X, y, k=None):
    """Select best features using ANOVA F-value"""
    if k is None:
        k = X.shape[1]  # Use all features if k is not specified

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    return selector.get_support(), selector.scores_

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

def perform_pca_analysis(df, features, save_dir=None):
    """
    Perform PCA analysis on the given dataframe and features with improved preprocessing.
    """
    # Prepare the data
    X = df[features].copy()
    y = df['class']

    # Print initial class distribution
    print("\nClass distribution:")
    for class_label in sorted(df['class'].unique()):
        count = (df['class'] == class_label).sum()
        print(f"{class_label}: {count} samples ({count/len(df):.1%})")

    # Handle missing values using robust statistics
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Select features
    feature_mask, f_scores = select_features(X_imputed, y)
    selected_features = np.array(features)[feature_mask]
    X_selected = X_imputed[:, feature_mask]

    # Print feature importance
    print("\nFeature Importance:")
    for feature, score in sorted(zip(features, f_scores), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {score:.2f}")

    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Perform PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Calculate explained variance ratios
    explained_variance_ratio = pca.explained_variance_ratio_

    # Create color mapping for classes
    class_colors = {'AML': 'red', 'APL': 'green', 'ALL': 'blue'}

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('PCA of Adult Leukemia Samples', fontsize=14, y=1.05)

    # Get the overall min and max for consistent axes
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    # Plot each class separately
    for idx, (class_name, color) in enumerate(class_colors.items()):
        # Plot only the current class
        mask = y == class_name
        count = mask.sum()
        axes[idx].scatter(X_pca[mask, 0], X_pca[mask, 1],
                         c=color, label=f'{class_name} (n={count})', alpha=0.6)

        # Set consistent axes limits
        axes[idx].set_xlim(x_min, x_max)
        axes[idx].set_ylim(y_min, y_max)

        # Add labels and title
        axes[idx].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
        axes[idx].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
        axes[idx].set_title(f'{class_name} Distribution')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plots if directory is provided
    if save_dir:
        # Save PNG
        png_path = os.path.join(save_dir, 'pca_Adult_analysis.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"\nPNG plot saved to: {png_path}")

        # Save SVG
        svg_path = os.path.join(save_dir, 'pca_Adult_analysis.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"SVG plot saved to: {svg_path}")

    # Print explained variance information
    print("\nExplained Variance by Principal Components:")
    total_var = 0
    for i, var in enumerate(explained_variance_ratio, 1):
        total_var += var
        print(f"PC{i}: {var:.2%} (Cumulative: {total_var:.2%})")

    # Print component loadings
    print("\nFeature contributions to principal components:")
    loadings = pca.components_.T
    for i, feature in enumerate(selected_features):
        contributions = [f"PC{j+1}: {loading:.3f}" for j, loading in enumerate(loadings[i])]
        print(f"{feature}: {', '.join(contributions)}")

    return pca, X_pca, selected_features

def run_analysis(is_adult: bool):
    """Runs the PCA analysis for a specific cohort."""
    tag = "Adult" if is_adult else "Pediatric"
    print(f"--- Running PCA Analysis for {tag} Cohort ---")

    # Timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up directories
    config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Use tag in directory name
    plots_dir = os.path.join(config['root_results'], f'5_pca_{tag.lower()}')
    os.makedirs(plots_dir, exist_ok=True)

    # Set up log capture
    log_path = os.path.join(plots_dir, f'pca_{tag}_analysis_log_{timestamp}.txt')
    setup_log_capture(str(log_path))

    # Log start time and system info
    print(f"PCA Analysis for {tag} started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")

    # Load patient data for the specified cohort
    print(f"\nLoading {tag.lower()} patient data...")
    # Use the is_adult flag passed to the function
    df, _, features = load_data(config_path=config_path, is_adult=is_adult)

    # Perform PCA analysis - update function to accept tag for plotting
    print("\nPerforming PCA analysis...")
    # Need to pass the tag to perform_pca_analysis to update plot titles/filenames
    # Modify perform_pca_analysis signature and implementation
    pca, X_pca, selected_features = perform_pca_analysis_updated(df, features, tag, save_dir=plots_dir)

    print(f"\n{tag} analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {plots_dir}")

# Update perform_pca_analysis to accept tag
def perform_pca_analysis_updated(df, features, tag, save_dir=None):
    """
    Perform PCA analysis on the given dataframe and features with improved preprocessing.
    Includes tag for cohort-specific output naming.
    """
    # Prepare the data
    X = df[features].copy()
    y = df['class']

    # Print initial class distribution
    print("\nClass distribution:")
    for class_label in sorted(df['class'].unique()):
        count = (df['class'] == class_label).sum()
        print(f"{class_label}: {count} samples ({count/len(df):.1%})")

    # Handle missing values using robust statistics
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Select features
    feature_mask, f_scores = select_features(X_imputed, y)
    selected_features = np.array(features)[feature_mask]
    X_selected = X_imputed[:, feature_mask]

    # Print feature importance
    print("\nFeature Importance:")
    for feature, score in sorted(zip(features, f_scores), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {score:.2f}")

    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Perform PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Calculate explained variance ratios
    explained_variance_ratio = pca.explained_variance_ratio_

    # Create color mapping for classes
    class_colors = {'AML': 'red', 'APL': 'green', 'ALL': 'blue'}

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Use tag in title
    fig.suptitle(f'PCA of {tag} Leukemia Samples', fontsize=14, y=1.05)

    # Get the overall min and max for consistent axes
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    # Plot each class separately
    for idx, (class_name, color) in enumerate(class_colors.items()):
        # Plot only the current class
        mask = y == class_name
        count = mask.sum()
        axes[idx].scatter(X_pca[mask, 0], X_pca[mask, 1],
                         c=color, label=f'{class_name} (n={count})', alpha=0.6)

        # Set consistent axes limits
        axes[idx].set_xlim(x_min, x_max)
        axes[idx].set_ylim(y_min, y_max)

        # Add labels and title
        axes[idx].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
        axes[idx].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
        axes[idx].set_title(f'{class_name} Distribution')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plots if directory is provided
    if save_dir:
        # Use tag in filenames
        png_path = os.path.join(save_dir, f'pca_{tag}_analysis.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"\n{tag} PNG plot saved to: {png_path}")

        svg_path = os.path.join(save_dir, f'pca_{tag}_analysis.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"{tag} SVG plot saved to: {svg_path}")

    # Print explained variance information
    print("\nExplained Variance by Principal Components:")
    total_var = 0
    for i, var in enumerate(explained_variance_ratio, 1):
        total_var += var
        print(f"PC{i}: {var:.2%} (Cumulative: {total_var:.2%})")

    # Print component loadings
    print("\nFeature contributions to principal components:")
    loadings = pca.components_.T
    for i, feature in enumerate(selected_features):
        contributions = [f"PC{j+1}: {loading:.3f}" for j, loading in enumerate(loadings[i])]
        print(f"{feature}: {', '.join(contributions)}")

    return pca, X_pca, selected_features

def main():
    """Runs PCA analysis for both adult and pediatric cohorts."""
    print("Starting PCA analysis...")
    run_analysis(is_adult=True)
    run_analysis(is_adult=False)
    print("\nPCA analysis finished for both cohorts.")

if __name__ == "__main__":
    main()
