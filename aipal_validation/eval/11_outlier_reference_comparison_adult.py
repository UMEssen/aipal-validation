import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Standardized features and units (kept aligned with 1_reference_comparison.py)
STANDARDIZED_FEATURES = {
    'MCV_fL': {'unit': 'fl'},
    'PT_percent': {'unit': '%'},
    'LDH_UI_L': {'unit': 'U/l'},
    'MCHC_g_L': {'unit': 'g/l'},
    'WBC_G_L': {'unit': '/nl'},
    'Fibrinogen_g_L': {'unit': 'g/l'},
    'Monocytes_G_L': {'unit': '/nl'},
    'Platelets_G_L': {'unit': '/nl'},
    'Lymphocytes_G_L': {'unit': '/nl'},
    'age': {'unit': 'years'}
}


# Reference values from 1_reference_comparison.py
REFERENCE_VALUES = {
    'ALL': {
        'age': {'mean': 37.74, 'sd': 24.95, 'median': 36.48, 'iqr': [17, 59.98]},
        'WBC_G_L': {'mean': 39.24, 'sd': 81.18, 'median': 9.74, 'iqr': [3.8, 32.26]},
        'MCV_fL': {'mean': 86.64, 'sd': 8.61, 'median': 86.4, 'iqr': [81.57, 91.5]},
        'MCHC_g_L': {'mean': 288.34, 'sd': 113.32, 'median': 336, 'iqr': [319, 346.25]},
        'Monocytes_G_L': {'mean': 0.49, 'sd': 1.22, 'median': 0.1, 'iqr': [0, 0.4]},
        'Lymphocytes_G_L': {'mean': 3.72, 'sd': 5.19, 'median': 2.49, 'iqr': [1.23, 4.11]},
        'Platelets_G_L': {'mean': 91.63, 'sd': 92.59, 'median': 52.5, 'iqr': [30, 131]},
        'Fibrinogen_g_L': {'mean': 4.13, 'sd': 1.66, 'median': 3.85, 'iqr': [3.08, 5]},
        'LDH_UI_L': {'mean': 1068.5, 'sd': 1593.61, 'median': 566, 'iqr': [351, 1112]},
        'PT_percent': {'mean': 84.32, 'sd': 13.66, 'median': 86, 'iqr': [77, 94]}
    },
    'AML': {
        'age': {'mean': 60.99, 'sd': 17.2, 'median': 64, 'iqr': [51.1, 73]},
        'WBC_G_L': {'mean': 37.75, 'sd': 65.18, 'median': 9.46, 'iqr': [2.5, 42.3]},
        'MCV_fL': {'mean': 95.93, 'sd': 8.77, 'median': 95.7, 'iqr': [90.1, 101.3]},
        'MCHC_g_L': {'mean': 278.89, 'sd': 122.22, 'median': 336, 'iqr': [318, 346]},
        'Monocytes_G_L': {'mean': 5.34, 'sd': 17.95, 'median': 0.24, 'iqr': [0.03, 2.4]},
        'Lymphocytes_G_L': {'mean': 3.37, 'sd': 4.96, 'median': 1.89, 'iqr': [1.05, 3.8]},
        'Platelets_G_L': {'mean': 88.82, 'sd': 77.09, 'median': 66, 'iqr': [36, 120]},
        'Fibrinogen_g_L': {'mean': 4.18, 'sd': 1.67, 'median': 3.92, 'iqr': [3.1, 5.1]},
        'LDH_UI_L': {'mean': 742.53, 'sd': 1084.42, 'median': 396.5, 'iqr': [235.75, 763.75]},
        'PT_percent': {'mean': 76.03, 'sd': 16.58, 'median': 78, 'iqr': [66, 88]}
    },
    'APL': {
        'age': {'mean': 49.13, 'sd': 20.74, 'median': 51, 'iqr': [33.08, 63.39]},
        'WBC_G_L': {'mean': 16.1, 'sd': 32.14, 'median': 3.2, 'iqr': [1.18, 17.68]},
        'MCV_fL': {'mean': 89.36, 'sd': 7.23, 'median': 89.7, 'iqr': [84.05, 93.85]},
        'MCHC_g_L': {'mean': 317.78, 'sd': 98.68, 'median': 350, 'iqr': [342, 357.75]},
        'Monocytes_G_L': {'mean': 0.14, 'sd': 0.4, 'median': 0, 'iqr': [0, 0.1]},
        'Lymphocytes_G_L': {'mean': 1.33, 'sd': 1.36, 'median': 0.9, 'iqr': [0.5, 1.65]},
        'Platelets_G_L': {'mean': 41.8, 'sd': 36.64, 'median': 30.5, 'iqr': [17, 52.25]},
        'Fibrinogen_g_L': {'mean': 2.51, 'sd': 8.42, 'median': 1.57, 'iqr': [1, 2.2]},
        'LDH_UI_L': {'mean': 474.3, 'sd': 387.43, 'median': 363, 'iqr': [239, 553]},
        'PT_percent': {'mean': 61.78, 'sd': 16.15, 'median': 62, 'iqr': [53, 71]}
    }
}


def _get_class_position_map(plot_df: pd.DataFrame) -> dict:
    actual_classes = plot_df['class'].dropna().unique()
    return {cls: i for i, cls in enumerate(actual_classes)}


def _expand_iqr_bounds_with_refs(plot_df: pd.DataFrame, feature: str) -> Tuple[float, float]:
    Q1 = plot_df[feature].quantile(0.25)
    Q3 = plot_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    ref_means_for_feature = []
    for c in ['ALL', 'AML', 'APL']:
        if feature in REFERENCE_VALUES.get(c, {}):
            ref_means_for_feature.append(REFERENCE_VALUES[c][feature]['mean'])

    if ref_means_for_feature:
        min_ref_mean = min(ref_means_for_feature)
        max_ref_mean = max(ref_means_for_feature)
        lower_bound = min(lower_bound, min_ref_mean - abs(min_ref_mean * 0.05))
        upper_bound = max(upper_bound, max_ref_mean + abs(max_ref_mean * 0.05))

    return lower_bound, upper_bound


def plot_outlier_distributions_big_subplot(df_with_outliers: pd.DataFrame, features: List[str], save_dir: Optional[str] = None) -> None:
    """Create a large grid of violin plots for OUTLIERS ONLY across classes, with reference means overlaid."""
    outliers_df = df_with_outliers[df_with_outliers.get('outlier', 0) == 1].copy()
    if outliers_df.empty:
        print("No outliers found to plot.")
        return

    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))

    for idx, feature in enumerate(features, 1):
        if feature not in outliers_df.columns:
            continue

        plt.subplot(n_rows, n_cols, idx)

        plot_df = outliers_df.reset_index(drop=True)
        lower_bound, upper_bound = _expand_iqr_bounds_with_refs(plot_df, feature)
        mask = (plot_df[feature] >= lower_bound) & (plot_df[feature] <= upper_bound)
        plot_df_filtered = plot_df[mask].copy()

        if plot_df_filtered.empty:
            plt.title(f'{feature} (no data)', fontsize=10)
            plt.axis('off')
            continue

        # Prepare data for matplotlib's violinplot grouped by class
        actual_classes = plot_df_filtered['class'].dropna().unique().tolist()
        class_to_position = {cls: i for i, cls in enumerate(actual_classes)}
        data_by_class = [
            plot_df_filtered.loc[plot_df_filtered['class'] == cls, feature].dropna().values
            for cls in actual_classes
        ]
        parts = plt.violinplot(data_by_class, positions=list(range(len(actual_classes))), showmeans=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('#87CEFA')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        ref_means = []
        x_positions = []
        for c in ['ALL', 'AML', 'APL']:
            if feature in REFERENCE_VALUES.get(c, {}) and c in class_to_position:
                ref_means.append(REFERENCE_VALUES[c][feature]['mean'])
                x_positions.append(class_to_position[c])

        if ref_means and x_positions:
            plt.plot(x_positions, ref_means, 'r*', markersize=10, label='Reference Mean')

        plt.title(f'{feature}', fontsize=11, fontweight='bold')
        unit = STANDARDIZED_FEATURES.get(feature, {}).get('unit', '')
        plt.ylabel(f"{feature} ({unit})" if unit else feature, fontsize=9)
        plt.xlabel('Disease Class', fontsize=9)
        plt.xticks(ticks=list(range(len(actual_classes))), labels=actual_classes, rotation=30)
        plt.ylim(lower_bound * 0.95, upper_bound * 1.05)

        if idx == 1 and ref_means:
            plt.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path_png = os.path.join(save_dir, 'outliers_big_subplot.png')
        out_path_pdf = os.path.join(save_dir, 'outliers_big_subplot.pdf')
        plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_path_pdf, bbox_inches='tight')

    plt.close()


def compute_outlier_percentage_per_country(df_with_outliers: pd.DataFrame) -> pd.DataFrame:
    """Compute outlier percentages per cohort (city/country)."""
    city_col = 'city_country' if 'city_country' in df_with_outliers.columns else ('city' if 'city' in df_with_outliers.columns else None)
    if city_col is None:
        raise ValueError("Neither 'city_country' nor 'city' column present")

    grp = df_with_outliers.groupby(city_col).agg(
        total=('outlier', 'count'),
        outliers=('outlier', lambda s: int((s == 1).sum()))
    ).reset_index()
    grp['outlier_pct'] = grp['outliers'] / grp['total'] * 100.0
    grp = grp.rename(columns={city_col: 'cohort'})
    return grp


def compute_feature_missingness_outliers_vs_non(df_with_outliers: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Compare missingness rates for features: outliers vs non-outliers."""
    mask_out = df_with_outliers['outlier'] == 1
    mask_in = df_with_outliers['outlier'] == 0

    out_rates = df_with_outliers.loc[mask_out, feature_cols].isna().mean().rename('outliers_missing_rate')
    in_rates = df_with_outliers.loc[mask_in, feature_cols].isna().mean().rename('non_outliers_missing_rate')

    summary = pd.concat([out_rates, in_rates], axis=1)
    summary['delta_out_minus_non'] = summary['outliers_missing_rate'] - summary['non_outliers_missing_rate']
    summary = summary.reset_index().rename(columns={'index': 'feature'})
    return summary


def run_analysis_adult_outliers() -> None:
    print("--- Running Adult Outlier Reference Comparison ---")

    # Paths
    config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")
    precomputed_outliers_csv = "/data/all_cohorts/outlier_investigation_adults/adults_with_outlier.csv"

    df_with_outliers: Optional[pd.DataFrame] = None
    output_dir: str

    # Prefer and require the existing outlier outputs from the prior container run
    if os.path.exists(precomputed_outliers_csv):
        df_with_outliers = pd.read_csv(precomputed_outliers_csv)
        output_dir = str(Path(precomputed_outliers_csv).parent)
    else:
        print(f"Missing precomputed outlier CSV: {precomputed_outliers_csv}. Run the CLI first: python -m aipal_validation --adults_outliers --step all")
        return

    # Use a stable list of features present in the dataframe and standardized mapping
    analysis_features = [f for f in STANDARDIZED_FEATURES.keys() if f in df_with_outliers.columns]

    # 1) Big subplot of outlier distributions with reference means
    print("Creating big subplot for outlier distributions...")
    plots_dir = os.path.join(output_dir, 'distribution_plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_outlier_distributions_big_subplot(df_with_outliers, analysis_features, plots_dir)

    # 2) Outlier percentages per country
    print("Computing outlier percentages per cohort...")
    outlier_pct_df = compute_outlier_percentage_per_country(df_with_outliers)
    outlier_pct_csv = os.path.join(output_dir, 'outlier_percentages_per_cohort.csv')
    outlier_pct_df.to_csv(outlier_pct_csv, index=False)

    # 3) Missing features outliers vs non-outliers (percentage)
    print("Computing feature missingness: outliers vs non-outliers...")
    missing_summary = compute_feature_missingness_outliers_vs_non(df_with_outliers, analysis_features)
    missing_csv = os.path.join(output_dir, 'feature_missingness_outliers_vs_non_outliers.csv')
    missing_summary.to_csv(missing_csv, index=False)

    # Also save a formatted Excel sheet with the two key tables
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # Combine into one Excel with separate sheets for convenience
    try:
        excel_path = os.path.join(tables_dir, 'outlier_summary_adult.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            outlier_pct_df.to_excel(writer, sheet_name='Outlier_pct_per_cohort', index=False)
            missing_summary.to_excel(writer, sheet_name='Missingness_out_vs_non', index=False)
        print(f"Saved: {outlier_pct_csv}\nSaved: {missing_csv}\nExcel: {excel_path}")
    except Exception as e:
        print(f"Saved: {outlier_pct_csv}\nSaved: {missing_csv}\nExcel export skipped: {e}")
    print("Adult outlier reference comparison completed.")


def main():
    run_analysis_adult_outliers()


if __name__ == "__main__":
    main()


