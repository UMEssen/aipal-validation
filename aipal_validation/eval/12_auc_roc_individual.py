import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from util import load_data

# Ensure Matplotlib and fontconfig caches point to writable locations (e.g., inside containers)
TEMP_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_CACHE_ROOT / "mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_ROOT / "xdg_cache"))
for env_var in ("MPLCONFIGDIR", "XDG_CACHE_HOME"):
    Path(os.environ[env_var]).mkdir(parents=True, exist_ok=True)

# Matplotlib defaults for a consistent look across evaluation scripts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
    "Bitstream Vera Sans",
    "sans-serif",
]
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16

CLASSES: List[str] = ["AML", "APL", "ALL"]
CLASS_COLORS: Dict[str, str] = {"AML": "#d62728", "APL": "#2ca02c", "ALL": "#1f77b4"}
DEFAULT_CITY_ORDER: List[str] = [
    "Hannover",
    "Bochum",
    "Barcelona",
    "Salamanca",
    "Milano",
    "Rome",
    "Maastricht",
    "Wroclaw",
    "Suzhou",
    "Kolkata",
    "Melbourne",
    "Sao paulo",
    "Buenos aires",
    "Dallas",
    "Antananarivo",
    "Lagos",
]
MIN_SAMPLES_PER_CITY = 10


def load_adult_predictions() -> Tuple[pd.DataFrame, Dict, List[str]]:
    """Load adult cohort predictions using the shared util module."""
    analysis_config_path = str(
        Path(__file__).resolve().parent.parent / "config" / "config_analysis.yaml"
    )
    df, config, features = load_data(config_path=analysis_config_path, is_adult=True, filter_missing_values=False)
    config["is_adult"] = True
    return df, config, features


def ensure_prediction_columns_are_float(df: pd.DataFrame) -> pd.DataFrame:
    """Convert prediction columns to float to avoid downstream issues."""
    prediction_cols = ["prediction.AML", "prediction.APL", "prediction.ALL"]
    df[prediction_cols] = df[prediction_cols].astype(float)
    return df


def compute_city_roc_curves(
    df_city: pd.DataFrame,
) -> Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]]:
    """Compute ROC curve data for each class within a city subset.

    Returns a mapping from class name to (fpr, tpr, auc) tuples. If there are
    insufficient positive/negative samples for a class, the value will be None.
    """
    if df_city.empty:
        return {cls: None for cls in CLASSES}

    y_true = label_binarize(df_city["class"], classes=CLASSES)
    y_scores = df_city[["prediction.AML", "prediction.APL", "prediction.ALL"]].values

    roc_data: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]] = {}
    for idx, cls in enumerate(CLASSES):
        positives = y_true[:, idx]
        # We need at least one positive and one negative sample to compute ROC
        if len(np.unique(positives)) < 2:
            roc_data[cls] = None
            continue

        fpr, tpr, _ = roc_curve(positives, y_scores[:, idx])
        roc_auc = auc(fpr, tpr)
        roc_data[cls] = (fpr, tpr, roc_auc)

    return roc_data


def order_cities(cities: List[str]) -> List[str]:
    """Return cities ordered by the predefined layout while keeping unknown ones."""
    ordered = [city for city in DEFAULT_CITY_ORDER if city in cities]
    remaining = [city for city in cities if city not in ordered]
    return ordered + sorted(remaining)


def summarize_city_metrics(
    city: str, df_city: pd.DataFrame, roc_data: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]]
) -> List[Dict]:
    """Create summary rows for the metrics table."""
    rows: List[Dict] = []
    total_samples = len(df_city)
    for cls in CLASSES:
        auc_value = roc_data[cls][2] if roc_data[cls] is not None else np.nan
        rows.append(
            {
                "city_country": city,
                "class": cls,
                "samples_total": total_samples,
                "samples_positive": int((df_city["class"] == cls).sum()),
                "auc": auc_value,
            }
        )
    return rows


def get_results_directory(config: Dict, subdirectory: str) -> str:
    """Resolve a writable results directory, falling back if needed."""
    requested_root = config.get("root_results")
    fallback_candidates = [
        requested_root,
        "/data/results",
        str(Path(__file__).resolve().parents[2] / "results"),
    ]
    candidate_roots = [path for path in fallback_candidates if path]
    for root in candidate_roots:
        target = os.path.join(root, subdirectory)
        try:
            os.makedirs(target, exist_ok=True)
            if root != requested_root:
                print(
                    f"Warning: falling back to writable results directory at {target} "
                    f"(requested root '{requested_root}' not accessible)."
                )
            return target
        except PermissionError:
            continue

    raise PermissionError(
        "Unable to create a writable results directory. "
        f"Tried: {candidate_roots}. Please adjust permissions or config."
    )


def plot_city_roc(
    ax: plt.Axes,
    city: str,
    roc_data: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]],
    total_samples: int,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    add_legend: bool,
) -> None:
    """Plot ROC curves for all classes in a single subplot."""
    ax.set_title(f"{city} (n={total_samples})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate" if show_xlabel else "")
    ax.set_ylabel("True Positive Rate" if show_ylabel else "")
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    plotted_any = False
    handles = []
    labels = []
    for cls in CLASSES:
        curve = roc_data[cls]
        if curve is None:
            continue
        fpr, tpr, auc_value = curve
        line, = ax.plot(
            fpr,
            tpr,
            color=CLASS_COLORS[cls],
            lw=2,
            label=f"{cls} (AUC = {auc_value:.2f})",
        )
        plotted_any = True
        handles.append(line)
        labels.append(f"{cls} (AUC = {auc_value:.2f})")

    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    if not show_ylabel:
        ax.tick_params(labelleft=False)

    if plotted_any and add_legend:
        ax.legend(handles, labels, loc="lower right")
    elif plotted_any and not add_legend:
        ax.legend(handles, labels, loc="lower right", fontsize=9, handlelength=1.8)
    elif not plotted_any:
        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )


def main() -> None:
    """Generate ROC curves and AUC metrics for each adult city/country."""
    print("Loading adult cohort predictions...")
    df, config, _ = load_adult_predictions()
    df = ensure_prediction_columns_are_float(df)

    available_cities = order_cities(sorted(df["city_country"].unique()))
    print(f"Found {len(available_cities)} adult cohorts.")

    rows_per_city: List[Dict] = []
    roc_results_by_city: Dict[str, Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]]] = {}

    for city in available_cities:
        df_city = df[df["city_country"] == city].copy()
        roc_data = compute_city_roc_curves(df_city)
        roc_results_by_city[city] = roc_data
        rows_per_city.extend(summarize_city_metrics(city, df_city, roc_data))

    metrics_df = pd.DataFrame(rows_per_city)

    # Prepare output directory
    plots_dir = get_results_directory(config, "12_auc_roc_individual")
    print(f"Results will be saved to: {plots_dir}")

    # Plot layout: 3 columns, enough rows for all cities
    n_cities = len(available_cities)
    n_cols = 3
    n_rows = math.ceil(n_cities / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes_array = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, (ax, city) in enumerate(zip(axes_array, available_cities)):
        df_city = df[df["city_country"] == city]
        total_samples = len(df_city)
        if total_samples < MIN_SAMPLES_PER_CITY:
            ax.axis("off")
            ax.set_visible(True)
            ax.text(
                0.5,
                0.5,
                f"{city}\nInsufficient samples (n={total_samples})",
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
                transform=ax.transAxes,
            )
            continue

        row_idx = idx // n_cols
        col_idx = idx % n_cols
        show_xlabel = row_idx == n_rows - 1
        show_ylabel = col_idx == 0
        add_legend = col_idx == 0

        plot_city_roc(
            ax,
            city,
            roc_results_by_city[city],
            total_samples,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            add_legend=add_legend,
        )

    # Hide leftover axes if any
    for ax in axes_array[n_cities:]:
        ax.axis("off")

    cohort_tag = "Adult"
    fig.suptitle(f"ROC Curves by Center ({cohort_tag} Cohort)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = os.path.join(plots_dir, f"roc_curves_{cohort_tag.lower()}.svg")
    fig.savefig(plot_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC figure to: {plot_path}")

    # Save metrics table
    metrics_path = os.path.join(plots_dir, f"auc_metrics_{cohort_tag.lower()}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved AUC metrics to: {metrics_path}")

    # Display summary in console
    print("\nPer-city AUC summary:")
    summary = (
        metrics_df.groupby(["city_country", "class"])
        .agg(samples_total=("samples_total", "first"), auc=("auc", "first"))
        .reset_index()
    )
    for _, row in summary.iterrows():
        auc_value = row["auc"]
        auc_str = f"{auc_value:.3f}" if np.isfinite(auc_value) else "n/a"
        print(
            f"  {row['city_country']:>15s} | {row['class']:>3s} | "
            f"n={int(row['samples_total'])} | AUC={auc_str}"
        )


if __name__ == "__main__":
    main()
