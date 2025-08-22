import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve, auc


# Style to match existing plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18


MIN_SAMPLES_PER_CLASS = 10


def load_config():
    """Load configuration and normalize paths similarly to other eval scripts."""
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "config_training.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Normalize root_dir like other scripts to work outside container
    if config['root_dir'] == "/data":
        config['root_dir'] = '/local/work/merengelke/aipal'

    if 'root_results' not in config:
        config['root_results'] = os.path.join(config['root_dir'], 'results')

    return config


def load_predict_data(config):
    """Load merged predict.csv produced by age_binning all_cohorts flow."""
    predict_path = Path(config['root_dir']) / "all_cohorts" / "age_binning" / "predict.csv"
    if not predict_path.exists():
        raise FileNotFoundError(f"Could not find predict.csv at {predict_path}")

    df = pd.read_csv(predict_path)
    # Ensure expected columns and types
    for col in ["prediction.AML", "prediction.APL", "prediction.ALL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            raise KeyError(f"Missing expected prediction column: {col}")

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age']).copy()
    df['class'] = df['class'].astype(str)

    # Create decade labels (e.g., 0-9, 10-19, ...)
    decade_start = (df['age'] // 10 * 10).astype(int)
    df['age_decade_label'] = decade_start.astype(str) + '-' + (decade_start + 9).astype(str)

    return df


def calculate_roc_for_decade(data_decade, class_name):
    """Compute ROC curve and AUC for a specific class within a decade subset.

    Returns (fpr, tpr, auc_val) or None if not computable.
    """
    y_true = (data_decade['class'] == class_name).astype(int)
    # Need both positive and negative samples to compute ROC
    pos = int(y_true.sum())
    neg = int((1 - y_true).sum())
    if pos < 1 or neg < 1 or pos < MIN_SAMPLES_PER_CLASS:
        return None

    score_col = f"prediction.{class_name}"
    if score_col not in data_decade.columns:
        return None
    y_score = data_decade[score_col].astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)
    return fpr, tpr, auc_val


def plot_age_binning_roc(df, config):
    """Plot ROC curves per age decade for each class (AML, APL, ALL)."""
    plots_dir = os.path.join(config['root_results'], '10_auc_age_binning')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")

    classes = ["AML", "APL", "ALL"]

    # Sort decades numerically by start value
    def decade_sort_key(lbl):
        try:
            return int(lbl.split('-')[0])
        except Exception:
            return 0

    decades = sorted(df['age_decade_label'].unique().tolist(), key=decade_sort_key)

    # Color map for decades
    cmap = plt.get_cmap('tab10') if len(decades) <= 10 else plt.get_cmap('tab20')

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, class_name in enumerate(classes):
        ax = axes[i]
        legend_entries = []

        for idx, decade_label in enumerate(decades):
            subset = df[df['age_decade_label'] == decade_label]

            res = calculate_roc_for_decade(subset, class_name)
            if res is None:
                # Skip decades with insufficient samples for this class
                continue
            fpr, tpr, auc_val = res
            color = cmap(idx % cmap.N)
            ax.plot(
                fpr,
                tpr,
                color=color,
                linestyle='-',
                alpha=0.9,
                linewidth=2,
                label=f"{decade_label} (AUC={auc_val:.2f}, n={len(subset)})",
            )

        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC by Age Decade – {class_name}')
        ax.legend(loc='lower right', ncol=1, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle('ROC Curves by Age Decade for Leukemia Types')

    # Save combined figure
    save_svg = os.path.join(plots_dir, 'roc_age_binning_combined.svg')
    fig.savefig(save_svg, format='svg', bbox_inches='tight')
    save_png = os.path.join(plots_dir, 'roc_age_binning_combined.png')
    fig.savefig(save_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_svg}\nSaved: {save_png}")
    plt.close(fig)


def main():
    print("Starting ROC analysis with age binning (decades)...")
    config = load_config()
    print(f"Using root_dir: {config['root_dir']}")
    print(f"Using root_results: {config['root_results']}")

    df = load_predict_data(config)

    # Report decade and class distribution
    print("\nSample distribution by age decade:")
    print(df['age_decade_label'].value_counts().sort_index())
    print("\nClass distribution overall:")
    print(df['class'].value_counts())

    plot_age_binning_roc(df, config)
    print("\nAUC age-binning analysis complete.")


if __name__ == "__main__":
    main()


