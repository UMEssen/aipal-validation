from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from aipal_validation.eval.util import load_data


def calculate_auc(data):
    """Calculate AUC scores for each class."""
    classes = ["ALL", "AML", "APL"]
    y_true = label_binarize(data["class"], classes=classes)
    auc_scores = {
        cat: (
            roc_auc_score(y_true[:, i], data[f"prediction.{cat}"])
            if len(np.unique(y_true[:, i])) > 1
            else float("nan")
        )
        for i, cat in enumerate(classes)
    }
    return auc_scores


def calculate_metrics(data, min_class_size=10):
    """Calculate performance metrics for each class."""
    # Calculate AUC scores
    auc_scores = calculate_auc(data)

    prediction_columns = [f"prediction.{cat}" for cat in ["ALL", "AML", "APL"]]

    # Get predicted class
    data = data.copy()
    data["predicted_class"] = (
        data[prediction_columns]
        .idxmax(axis=1)
        .str.replace("prediction.", "", regex=False)
    )

    # Compute metrics for each category
    categories = ["ALL", "AML", "APL"]
    metrics = {}

    for cat in categories:
        y_true = data["class"] == cat
        y_pred = data["predicted_class"] == cat
        class_size = np.sum(y_true)

        if class_size < min_class_size:
            metrics[cat] = {
                "AUC": float("nan"),
                "Accuracy": float("nan"),
                "Precision": float("nan"),
                "Recall": float("nan"),
                "F1 Score": float("nan"),
            }
            continue

        true_positives = np.sum(y_true & y_pred)
        true_negatives = np.sum(~y_true & ~y_pred)
        false_positives = np.sum(~y_true & y_pred)
        false_negatives = np.sum(y_true & ~y_pred)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics[cat] = {
            "AUC": auc_scores[cat],
            "Accuracy": (true_positives + true_negatives) / len(data),
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }

    return metrics


def print_metrics(metrics, cohort_name):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Performance Metrics - {cohort_name}")
    print(f"{'='*60}")

    metric_names = ["AUC", "Accuracy", "Precision", "Recall", "F1 Score"]
    categories = ["ALL", "AML", "APL"]

    # Print header
    print(f"{'Metric':<15}", end="")
    for cat in categories:
        print(f"{cat:>15}", end="")
    print()
    print("-" * 60)

    # Print each metric
    for metric in metric_names:
        print(f"{metric:<15}", end="")
        for cat in categories:
            value = metrics[cat][metric]
            if np.isnan(value):
                print(f"{'N/A':>15}", end="")
            else:
                print(f"{value:>15.4f}", end="")
        print()

    print(f"{'='*60}\n")


config_path = str(Path(__file__).parent.parent / "config" / "config_analysis.yaml")

# 1. Load all data kids and adults
df_kids, _, features = load_data(config_path=config_path, is_adult=False, filter_missing_values=True)
df_adults, _, features = load_data(config_path=config_path, is_adult=True, filter_missing_values=True)

print(f"Vessen has {df_adults[df_adults['city_country'] == 'Vessen'].shape[0]} ALL samples")

print(f"Kids cohort by class: {df_kids['class'].value_counts()}")
print(f"Kids cohort total: {df_kids['class'].value_counts().sum()}")
print(f"Adults cohort by class: {df_adults['class'].value_counts()}")
print(f"Adults cohort total: {df_adults['class'].value_counts().sum()}")

# 2. Get counts of kids cohort by class

counts_kids_per_class = df_kids['class'].value_counts()
percentage_kids_per_class = counts_kids_per_class / len(df_kids)
percentage_adults_per_class = df_adults['class'].value_counts() / len(df_adults)

print(f"Percentage of kids per class: {percentage_kids_per_class}")
print(f"Percentage of adults per class: {percentage_adults_per_class}")

# 3. Random sample adults dataframe to match the absolute counts of kids by class

# Store original adults dataframe for reference
df_adults_original = df_adults.copy()

# Sample each class to match kids absolute counts (with oversampling if needed)
def sample_class_to_match_counts(group):
    class_name = group.name
    if class_name in counts_kids_per_class.index:
        desired_sample_size = counts_kids_per_class[class_name]
        if desired_sample_size > 0:
            # If we need more samples than available, oversample with replacement
            if desired_sample_size > len(group):
                return group.sample(n=desired_sample_size, replace=True, random_state=42)
            else:
                return group.sample(n=desired_sample_size, replace=False, random_state=42)
    return pd.DataFrame()

df_adults = df_adults_original.groupby('class', group_keys=False).apply(sample_class_to_match_counts).reset_index(drop=True)

print(f"Ad adults cohort by class: {df_adults['class'].value_counts()}")
print(f"Kids cohort counts by class: {counts_kids_per_class}")
print(f"Match check - Adults vs Kids counts: {df_adults['class'].value_counts() - counts_kids_per_class}")

# 4. Evaluate performance of kids and adults cohorts
print("\n" + "="*60)
print("EVALUATING PERFORMANCE")
print("="*60)

# Check if prediction columns exist
prediction_cols = ["prediction.ALL", "prediction.AML", "prediction.APL"]
missing_cols_kids = [col for col in prediction_cols if col not in df_kids.columns]
missing_cols_adults = [col for col in prediction_cols if col not in df_adults.columns]

if missing_cols_kids:
    print(f"Warning: Missing prediction columns in kids data: {missing_cols_kids}")
if missing_cols_adults:
    print(f"Warning: Missing prediction columns in adults data: {missing_cols_adults}")

# Calculate metrics for kids cohort
if not missing_cols_kids:
    metrics_kids = calculate_metrics(df_kids)
    print_metrics(metrics_kids, "Kids Cohort")
else:
    print("Skipping kids cohort evaluation due to missing prediction columns")

# Calculate metrics for adults cohort (original, non-balanced)
if not missing_cols_adults:
    metrics_adults_original = calculate_metrics(df_adults_original)
    print_metrics(metrics_adults_original, "Adults Cohort (Original - No Rebalancing)")
else:
    print("Skipping adults cohort evaluation due to missing prediction columns")

# Calculate metrics for adults cohort (balanced)
if not missing_cols_adults:
    metrics_adults = calculate_metrics(df_adults)
    print_metrics(metrics_adults, "Adults Cohort (Balanced - Matched to Kids)")
else:
    print("Skipping adults cohort evaluation due to missing prediction columns")
