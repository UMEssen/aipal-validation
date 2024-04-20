import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import wandb
from aipal_validation.ml.util import init_wandb


class LeukemiaModelEvaluator:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.cutoffs = {c["category"]: c for c in config["cutoffs"]}

    def log_to_wandb(self, results, phase="Evaluation"):
        # Create a table with columns
        columns = ["Category", "AUC", "Accuracy", "Precision", "Recall", "F1 Score"]
        wandb_table = wandb.Table(columns=columns)

        # Fill the table with data
        for cat, metrics in results.items():
            row = [cat] + [
                metrics[metric] for metric in columns[1:]
            ]  # List comprehension to retrieve metrics
            wandb_table.add_data(*row)

        # Log the table to wandb
        wandb.log({f"{phase} Metrics": wandb_table})

    def calculate_auc(self):
        classes = ["ALL", "AML", "APL"]
        y_true = label_binarize(self.data["class"], classes=classes)
        auc_scores = {}

        for i, cat in enumerate(classes):
            y_score = self.data[f"prediction.{cat}"]
            if (
                len(np.unique(y_true[:, i])) == 1
            ):  # AUC not defined in cases with one label only
                auc_scores[cat] = float("nan")
            else:
                auc_scores[cat] = roc_auc_score(y_true[:, i], y_score)
        return auc_scores

    def apply_cutoffs(self):
        for cat in ["ALL", "AML", "APL"]:
            cutoff = self.cutoffs[cat]
            ppv_cutoff = cutoff["PPV"]
            npv_cutoff = cutoff["NPV"]

            self.data[f"confident.{cat}"] = self.data[f"prediction.{cat}"].apply(
                lambda x: (
                    "Confident"
                    if x >= ppv_cutoff
                    else ("Non-confident" if x <= npv_cutoff else "Uncertain")
                )
            )

    def calculate_metrics(self, is_cutoff=False):
        if is_cutoff:
            self.apply_cutoffs()

        auc_scores = self.calculate_auc()
        metrics = {}
        for cat in ["ALL", "AML", "APL"]:
            if is_cutoff:
                cond = self.data[f"confident.{cat}"] == "Confident"
            else:
                cond = pd.Series([True] * len(self.data), index=self.data.index)

            y_true = self.data["class"] == cat
            y_pred = cond

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
                "Accuracy": (true_positives + true_negatives) / len(self.data),
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }
        return metrics

    def summarize_results(self, is_cutoff=False):
        return self.calculate_metrics(is_cutoff=is_cutoff)

    def prediction_data_pruner(self, threshold=0):
        # prune na values, more than threshold percentage of na values must be present in a row
        data = self.data.copy()
        mandatory_columns = [value[3] for value in self.config["obs_codes_si"].values()]
        data["nan_percentage"] = data[mandatory_columns].isna().mean(axis=1)
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


def main(config):
    init_wandb(config)
    data = pd.read_csv(config["task_dir"] / "predict.csv")
    evaluator = LeukemiaModelEvaluator(data, config)
    evaluator.prediction_data_pruner(threshold=0.4)
    results = evaluator.summarize_results(is_cutoff=False)
    logging.info(f"Results without applying cutoffs: {results}")
    evaluator.log_to_wandb(results, phase="No Cutoffs")
    results = evaluator.summarize_results(is_cutoff=True)
    logging.info(f"Results after applying cutoffs: {results}")
    evaluator.log_to_wandb(results, phase="Cutoffs")


if __name__ == "__main__":
    main()
