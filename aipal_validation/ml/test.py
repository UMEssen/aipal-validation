import logging

import numpy as np
import pandas as pd
import wandb
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample

from aipal_validation.ml.util import init_wandb


class LeukemiaModelEvaluator:
    def __init__(self, data, config, ds_name):
        self.data = data
        self.config = config
        self.cutoffs = {c["category"]: c for c in config["cutoffs"]}
        wandb.log({"Evaluating": ds_name})
        wandb.log({"Cohort Size": len(self.data)})
        wandb.log({"Cohort Distribution": data["age"].describe().to_dict()})

    def log_to_wandb(self, results, phase="Evaluation"):
        columns = ["Category", "Metric", "Mean", "CI Lower-Upper"]
        wandb_table = wandb.Table(columns=columns)

        for cat, metrics in results.items():
            for metric, values in metrics.items():
                mean_val, ci_lower, ci_upper = values
                wandb_table.add_data(
                    cat,
                    metric,
                    np.round(mean_val, 3),
                    f"[{np.round(ci_lower, 3)}, {np.round(ci_upper, 3)}]",
                )
        wandb.log({f"{phase}": wandb_table})

    def calculate_auc(self, data):
        classes = ["ALL", "AML", "APL"]
        y_true = label_binarize(data["class"], classes=classes)
        auc_scores = {}

        for i, cat in enumerate(classes):
            y_score = data[f"prediction.{cat}"]
            if (
                len(np.unique(y_true[:, i])) == 1
            ):  # AUC not defined in cases with one label only
                auc_scores[cat] = float("nan")
            else:
                auc_scores[cat] = roc_auc_score(y_true[:, i], y_score)
        return auc_scores

    def apply_cutoffs(self, data, cutoff_type):
        """Apply different cutoff types based on the configuration for confident and overall predictions."""
        for cat in ["ALL", "AML", "APL"]:
            cutoff = self.cutoffs[cat]
            if cutoff_type == "overall cutoff":
                cutoff_value = cutoff["ACC"]
            elif cutoff_type in ["confident cutoff"]:
                cutoff_value = cutoff["PPV"]
            elif cutoff_type in ["confident not cutoff"]:
                cutoff_value = cutoff["NPV"]

            if cutoff_type in ["overall cutoff", "confident cutoff"]:
                data[f"{cutoff_type}.{cat}"] = data[f"prediction.{cat}"].apply(
                    lambda x: (True if x >= cutoff_value else False)
                )
            elif cutoff_type in ["confident not cutoff"]:
                data[f"{cutoff_type}.{cat}"] = data[f"prediction.{cat}"].apply(
                    lambda x: (True if x < cutoff_value else False)
                )
        return data

    def calculate_metrics(self, data, cutoff_type):
        if cutoff_type in [
            "overall cutoff",
            "confident cutoff",
            "confident not cutoff",
        ]:
            """Calculate performance metrics based on different cutoffs."""
            data = self.apply_cutoffs(data, cutoff_type)
            confident_columns = [
                f"{cutoff_type}.{cat}" for cat in ["ALL", "AML", "APL"]
            ]
            data["predicted_class"] = (
                data[confident_columns]
                .idxmax(axis=1)
                .str.replace(f"{cutoff_type}.", "", regex=True)
            )
        elif cutoff_type == "no cutoff":
            prediction_columns = [f"prediction.{cat}" for cat in ["ALL", "AML", "APL"]]
            data["predicted_class"] = (
                data[prediction_columns]
                .idxmax(axis=1)
                .str.replace("prediction.", "", regex=True)
            )
        else:
            raise ValueError(f"Invalid cutoff type: {cutoff_type}")

        auc_scores = self.calculate_auc(data)
        metrics = {}

        for cat in ["ALL", "AML", "APL"]:
            y_true = data["class"] == cat
            y_pred = data["predicted_class"] == cat

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

    def bootstrap_metrics(self, cutoff_type, iterations=1000, confidence_level=0.95):
        results = {
            cat: {
                metric: []
                for metric in ["AUC", "Accuracy", "Precision", "Recall", "F1 Score"]
            }
            for cat in ["ALL", "AML", "APL"]
        }

        for _ in range(iterations):
            sampled_data = resample(self.data)
            metrics = self.calculate_metrics(sampled_data, cutoff_type)

            for cat in metrics:
                for metric in metrics[cat]:
                    results[cat][metric].append(metrics[cat][metric])

        # Compute mean, CI lower, and CI upper for each metric
        final_results = {}
        for cat, metrics in results.items():
            final_results[cat] = {}
            for metric, values in metrics.items():
                mean_val = np.mean(values)
                std_dev = np.std(values)
                ci_width = norm.ppf((1 + confidence_level) / 2) * (
                    std_dev / np.sqrt(iterations)
                )
                ci_lower = mean_val - ci_width
                ci_upper = mean_val + ci_width
                final_results[cat][metric] = (mean_val, ci_lower, ci_upper)

        return final_results

    def prediction_data_pruner(self, threshold=0):
        """Remove data rows based on the threshold of missing values."""
        data = self.data.copy()
        mandatory_columns = [value[3] for value in self.config["obs_codes_si"].values()]
        data["nan_percentage"] = data[mandatory_columns].isna().mean(axis=1)

        # Prune data where the percentage of NaN values is above the threshold
        data = data[data["nan_percentage"] <= threshold]
        data.drop(columns=["nan_percentage"], inplace=True)
        logging.info(f"Pruned {len(self.data) - len(data)} rows")

        mandatory_columns += [
            "prediction.ALL",
            "prediction.AML",
            "prediction.APL",
            "class",
        ]
        self.data = data[mandatory_columns]
        return self.data


def main(config):
    init_wandb(config)
    data = pd.read_csv(config["task_dir"] / "predict.csv")
    data_pediatric = data.where(data["age"] < 18).dropna(subset=["age"])
    data_adults = data.where(data["age"] >= 18).dropna(subset=["age"])

    if len(data_pediatric) <= 10:
        logging.warning(f"Only {len(data_pediatric)} pediatric samples, skipping...")
        ds_dict = {"adults": data_adults}
    else:
        ds_dict = {"kids": data_pediatric, "adults": data_adults}

    for ds_name, ds in ds_dict.items():
        evaluator = LeukemiaModelEvaluator(ds, config, ds_name)
        ds = evaluator.prediction_data_pruner(threshold=0.2)

        results = evaluator.bootstrap_metrics("no cutoff")
        ds.to_csv(config["task_dir"] / f"{ds_name}_pruned.csv", index=False)
        ds_counts = (
            ds["class"]
            .value_counts()
            .to_string()
            .strip()
            .replace("\n", "  ")
            .replace("   ", ":")
        )
        lg_string = f"{ds_name}/{config['run_id']} {results}, {len(ds)} samples classes {ds_counts}"
        tbl_string = f"{ds_name}/{config['run_id']} - {len(ds)}: {ds_counts}"

        logging.info(f"Results no cutoff: {lg_string}")
        evaluator.log_to_wandb(results, phase=f"No Cutoff - {tbl_string}")

        results = evaluator.bootstrap_metrics("overall cutoff")
        logging.info(f"Results overall cutoff: {lg_string}")
        evaluator.log_to_wandb(results, phase=f"Overall Cutoff - {tbl_string}")

        results = evaluator.bootstrap_metrics("confident cutoff")
        logging.info(f"Results confident cutoff: {lg_string}")
        # True positives are certain to be true
        evaluator.log_to_wandb(results, phase=f"Confident Cutoff - {tbl_string}")

        results = evaluator.bootstrap_metrics("confident not cutoff")
        logging.info(f"Results confident cutoff: {lg_string}")
        # True negatives are certain to be true
        evaluator.log_to_wandb(results, phase=f"Confident NOT Cutoff - {tbl_string}")
