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
        # one vs all fastion
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

    def apply_cutoffs(self, data, cutoff_type: str):
        for cat in ["ALL", "AML", "APL"]:
            cutoff = self.cutoffs[cat]
            cutoff_fl = cutoff[cutoff_type]

            # Confident predictions are TRUE
            # Non-confident predictions are FALSE
            # "The PPV was then shown for the predicted category, and the NPV for the excluded categories."
            if cutoff_type == "PPV" or cutoff_type == "ACC":
                data[f"confident.{cat}"] = data[f"prediction.{cat}"].apply(
                    lambda x: (True if x >= cutoff_fl else False)
                )
            elif cutoff_type == "NPV":
                data[f"confident.{cat}"] = data[f"prediction.{cat}"].apply(
                    lambda x: (True if x < cutoff_fl else False)
                )
            else:
                raise ValueError(f"Invalid cutoff type: {cutoff_type}")
        return data

    def calculate_metrics(self, data, cutoff_type: str):

        if cutoff_type:
            data = self.apply_cutoffs(data, cutoff_type)
            confident_columns = [f"confident.{cat}" for cat in ["ALL", "AML", "APL"]]
            # check if any of the confident columns are True then take the column name as predicted class
            data["predicted_class"] = (
                data[confident_columns]
                .idxmax(axis=1)
                .str.replace("confident.", "", regex=True)
            )

        else:
            prediction_columns = [f"prediction.{cat}" for cat in ["ALL", "AML", "APL"]]
            data["predicted_class"] = (
                data[prediction_columns]
                .idxmax(axis=1)
                .str.replace("prediction.", "", regex=True)
            )

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

        # Compute mean, CI lower and CI upper for each metric
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
        # Prune rows where the percentage of NaN values is above the threshold
        # Here, threshold is defined as the maximum allowed percentage of missing values

        data = self.data.copy()
        mandatory_columns = [value[3] for value in self.config["obs_codes_si"].values()]
        data["nan_percentage"] = data[mandatory_columns].isna().mean(axis=1)

        # If threshold=0.2, we want at least 80% of data, so we drop rows with >20% missing
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

    ds_dict = {"kids": data_pediatric, "adults": data_adults}

    for ds_name, ds in ds_dict.items():
        evaluator = LeukemiaModelEvaluator(ds, config, ds_name)
        ds = evaluator.prediction_data_pruner(threshold=0.2)

        results = evaluator.bootstrap_metrics(cutoff_type="")
        ds.to_csv(config["task_dir"] / f"{ds_name}_pruned.csv", index=False)
        logging.info(
            f"Results no cutoffs: {ds_name} {results}, {len(ds)} samples classes {ds['class'].value_counts()}"
        )
        class_counts = ds["class"].value_counts().to_string().replace("\n", ", ")

        evaluator.log_to_wandb(
            results, phase=f"No Cutoffs - {ds_name} - {len(ds)} samples: {class_counts}"
        )

        # cutoff_type = ["ACC", "PPV", "NPV"]
        cutoff_type = ["ACC"]
        for c_type in cutoff_type:
            results = evaluator.bootstrap_metrics(c_type)
            logging.info(
                f"Results cutoffs:{cutoff_type} - {ds_name}, {results}, {len(ds)} samples"
            )
            evaluator.log_to_wandb(
                results,
                phase=f"Cutoffs - {ds_name} - {len(ds)} samples: {class_counts}",
            )


if __name__ == "__main__":
    main()
