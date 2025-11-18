import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from .util import load_data, save_roc_source_data_to_excel
# Manuscript Figure 5b + c

# Add proper font settings for better text rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def prediction_data_pruner(df, threshold=0):
    """Remove data rows based on the threshold of missing values."""
    data = df.copy()
    mandatory_columns = [
        'Fibrinogen_g_L', 'LDH_UI_L', 'WBC_G_L', 'Lymphocytes_G_L',
        'MCHC_g_L', 'MCV_fL', 'Monocytes_G_L', 'Platelets_G_L',
        'PT_percent'
    ]

    data["nan_percentage"] = data[mandatory_columns].isna().mean(axis=1)
    data = data[data["nan_percentage"] <= threshold]
    data.drop(columns=["nan_percentage"], inplace=True)
    return data

def plot_confusion_matrix(df, plots_dir, tag):
    """Generate and save a normalized confusion matrix."""
    df["max_pred"] = df[["prediction.AML", "prediction.APL", "prediction.ALL"]].idxmax(axis=1)
    df["max_pred"] = df["max_pred"].str.split(".").str[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        df["class"],
        df["max_pred"],
        normalize='true',
        ax=ax,
        cmap='Blues',
        values_format='.2f'
    )

    disp.ax_.set_title(f"Confusion Matrix - {tag} Cohort (Normalized)")

    cm_path_png = os.path.join(plots_dir, f'confusion_matrix_{tag.lower()}.png')
    cm_path_svg = os.path.join(plots_dir, f'confusion_matrix_{tag.lower()}.svg')
    fig.savefig(cm_path_png, dpi=300, bbox_inches='tight')
    fig.savefig(cm_path_svg, format='svg', bbox_inches='tight')
    print(f"Saved confusion matrix to: {cm_path_png} and {cm_path_svg}")
    plt.close(fig)

    # Save the data used for confusion matrix to Excel
    cm_data = df[['class', 'max_pred', 'prediction.AML', 'prediction.APL', 'prediction.ALL']].copy()
    cm_data_path = os.path.join(plots_dir, f'confusion_matrix_data_{tag.lower()}.xlsx')
    cm_data.to_excel(cm_data_path, index=False, engine='openpyxl')
    print(f"Saved confusion matrix data to: {cm_data_path}")

def calculate_roc_curves(df, classes=["AML", "APL", "ALL"]):
    """Calculate ROC curves for all classes."""
    y = label_binarize(df['class'], classes=classes)

    y_score = df[[f"prediction.{cls}" for cls in classes]].values

    results = {}
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        results[class_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }

    return results

def plot_roc_curves(roc_results, df, plots_dir, tag):
    """Create and save a single ROC curve plot with all classes."""
    classes = ["AML", "APL", "ALL"]
    class_colors = {'AML': 'red', 'APL': 'green', 'ALL': 'blue'}

    class_counts = df['class'].value_counts().to_dict()

    fig, ax = plt.subplots(figsize=(10, 8))

    for class_name in classes:
        n_samples = class_counts.get(class_name, 0)
        ax.plot(
            roc_results[class_name]['fpr'],
            roc_results[class_name]['tpr'],
            color=class_colors[class_name],
            lw=2,
            label=f'ROC curve of {class_name} (AUC = {roc_results[class_name]["auc"]:.2f}, n={n_samples})'
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves for {tag} Cohort - All Classes')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    roc_path_png = os.path.join(plots_dir, f'roc_all_classes_{tag.lower()}.png')
    roc_path_svg = os.path.join(plots_dir, f'roc_all_classes_{tag.lower()}.svg')
    fig.savefig(roc_path_png, dpi=300, bbox_inches='tight')
    fig.savefig(roc_path_svg, format='svg', bbox_inches='tight')

    print(f"\nSaved ROC curve to: {roc_path_png} and {roc_path_svg}")

    print("\nAUC values:")
    for class_name in classes:
        n_samples = class_counts.get(class_name, 0)
        print(f"  {class_name}: {roc_results[class_name]['auc']:.4f} (n={n_samples})")

    plt.close(fig)

def extract_and_save_roc_source_data(df, plots_dir, tag):
    """Extract and save ROC source data to Excel."""
    classes = ["AML", "APL", "ALL"]
    datasets = {
        'all_classes': ('All Classes', df)
    }
    save_roc_source_data_to_excel(datasets, plots_dir, tag, classes)

def run_analysis(is_adult: bool):
    """Run the analysis for a specific cohort."""
    tag = "Adult" if is_adult else "Pediatric"
    print(f"\n--- Running analysis for {tag} Cohort ---")

    analysis_config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")

    with open(analysis_config_path, 'r') as f:
        config = yaml.safe_load(f)

    df, config, features = load_data(config_path=analysis_config_path, root_path=config['root_dir'] + '/', is_adult=is_adult)

    print(f"Length of df: {len(df)}")
    print(f"Class distribution: {df['class'].value_counts()}")

    config['is_adult'] = is_adult

    df[["prediction.AML", "prediction.APL", "prediction.ALL"]] = df[["prediction.AML", "prediction.APL", "prediction.ALL"]].astype(float)

    df = prediction_data_pruner(df, threshold=0.2)

    plots_dir = os.path.join(config['root_results'], f'13_auc_roc_one_plot_{tag.lower()}')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")

    print(f"\nData Summary ({tag}):")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    print(df['class'].value_counts())

    plot_confusion_matrix(df.copy(), plots_dir, tag)

    roc_results = calculate_roc_curves(df)

    plot_roc_curves(roc_results, df, plots_dir, tag)

    extract_and_save_roc_source_data(df, plots_dir, tag)

def main():
    """Run the analysis for both adult and pediatric cohorts."""
    print("Starting ROC and Confusion Matrix analysis...")
    run_analysis(is_adult=False)
    run_analysis(is_adult=True)
    print("\nAnalysis complete for both cohorts.")

if __name__ == "__main__":
    main()
