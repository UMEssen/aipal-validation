import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from util import load_data
from aipal_validation.outlier.check_outlier import OutlierChecker


# Visual defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'
]
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16


def calculate_roc_curves(df: pd.DataFrame, classes=("ALL", "AML", "APL")):
    """Calculate one-vs-rest ROC curves and AUC per available class.

    Returns a dict: {class_name: {fpr, tpr, auc}}
    """
    results = {}
    present = set(df['class'].dropna().unique())
    for c in classes:
        if c not in present:
            continue
        pred_col = f"prediction.{c}"
        if pred_col not in df.columns:
            continue
        # Ensure numeric scores
        scores = pd.to_numeric(df[pred_col], errors='coerce').values
        y_true = (df['class'] == c).astype(int).values
        # Need both positive and negative labels, and non-constant scores
        if len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            results[c] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': float(auc(fpr, tpr))
            }
        except Exception:
            continue
    return results


def run_analysis(is_adult: bool = True):
    tag = 'Adult' if is_adult else 'Pediatric'

    # Load data
    analysis_config_path = str(Path(__file__).parent.parent / 'config' / 'config_analysis.yaml')
    df, config, _ = load_data(config_path=analysis_config_path, is_adult=is_adult, filter_by_size=False)
    config['is_adult'] = is_adult

    # Load pretrained outlier models
    model_dir = Path(__file__).resolve().parent.parent / 'outlier'
    outlier_config_path = Path(__file__).resolve().parent.parent / 'config' / 'config_outlier.yaml'

    checker = OutlierChecker()
    checker.load_models(model_dir, outlier_config_path)

    # Run batch outlier detection (returns a copy with an 'outlier' column)
    df_with_outliers = checker.check_dataframe(df)

    # Cities present in the filtered dataset
    ordered_cities = sorted(df_with_outliers['city_country'].dropna().unique())
    classes = ("ALL", "AML", "APL")

    # Pre-filter to only cities that will yield at least one ROC curve to plot
    valid_cities = []
    for city in ordered_cities:
        df_city = df_with_outliers[df_with_outliers['city_country'] == city].copy()
        if len(df_city) < 5:
            continue
        df_city_after = df_city[df_city.get('outlier', 0) == 0].copy()
        roc_before_tmp = calculate_roc_curves(df_city, classes=classes)
        roc_after_tmp = calculate_roc_curves(df_city_after, classes=classes) if len(df_city_after) >= 5 else {}
        if roc_before_tmp or roc_after_tmp:
            valid_cities.append(city)

    # Prepare outputs
    plots_dir = os.path.join(config['root_results'], f'11_roc_auc_plot_by_center_{tag.lower()}')
    os.makedirs(plots_dir, exist_ok=True)

    # Accumulate AUC table
    auc_rows = []

    # Subplot grid
    n_cities = len(valid_cities)
    if n_cities == 0:
        print(f"No cities available for {tag} cohort after preprocessing.")
        return

    n_cols = min(3, max(1, n_cities))
    n_rows = math.ceil(n_cities / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle(f'ROC Curves by City Before vs After Outlier Filtering ({tag})', fontsize=16)

    class_colors = {'ALL': '#377EB8', 'AML': '#E41A1C', 'APL': '#4DAF4A'}

    for idx, city in enumerate(valid_cities):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        df_city = df_with_outliers[df_with_outliers['city_country'] == city].copy()

        df_city_before = df_city.copy()
        df_city_after = df_city[df_city.get('outlier', 0) == 0].copy()

        roc_before = calculate_roc_curves(df_city_before, classes=classes)
        roc_after = calculate_roc_curves(df_city_after, classes=classes) if len(df_city_after) >= 5 else {}

        # Gather counts and AUCs for CSV
        for c in classes:
            before_auc = roc_before.get(c, {}).get('auc', np.nan)
            after_auc = roc_after.get(c, {}).get('auc', np.nan)
            n_before = int((df_city_before['class'] == c).sum()) if 'class' in df_city_before.columns else 0
            n_after = int((df_city_after['class'] == c).sum()) if 'class' in df_city_after.columns else 0
            auc_rows.append({
                'city_country': city,
                'class': c,
                'auc_before': before_auc,
                'auc_after': after_auc,
                'n_before': n_before,
                'n_after': n_after
            })

        # Plot per-class ROC curves (before solid, after dashed)
        any_plotted = False
        for c in classes:
            color = class_colors.get(c, 'gray')
            if c in roc_before:
                ax.plot(roc_before[c]['fpr'], roc_before[c]['tpr'], color=color, linestyle='-', linewidth=2,
                        label=f"Before - {c} (AUC={roc_before[c]['auc']:.2f})")
                any_plotted = True
            if c in roc_after:
                ax.plot(roc_after[c]['fpr'], roc_after[c]['tpr'], color=color, linestyle='--', linewidth=2,
                        label=f"After - {c} (AUC={roc_after[c]['auc']:.2f})")
                any_plotted = True

        if not any_plotted:
            ax.axis('off')
            continue

        # Diagonal and formatting
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        total_before = len(df_city_before)
        total_after = len(df_city_after)
        ax.set_title(f"{city} (n={total_before}→{total_after})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=8)

    # Hide any unused axes
    for i in range(n_cities, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save figure
    fig_path_svg = os.path.join(plots_dir, f'roc_by_city_{tag.lower()}.svg')
    fig_path_png = os.path.join(plots_dir, f'roc_by_city_{tag.lower()}.png')
    fig.savefig(fig_path_svg, format='svg', bbox_inches='tight')
    fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save AUC CSV
    auc_df = pd.DataFrame(auc_rows)
    auc_csv_path = os.path.join(plots_dir, f'auc_by_city_{tag.lower()}.csv')
    auc_df.to_csv(auc_csv_path, index=False)

    print(f"Saved figure: {fig_path_svg}")
    print(f"Saved AUC table: {auc_csv_path}")


def main():
    # Adults only by default as requested; run pediatric as well if needed
    # Essen has no adults
    # Turkey has no adults
    run_analysis(is_adult=True)


if __name__ == '__main__':
    main()


