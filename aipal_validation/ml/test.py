import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample

import wandb
import yaml
from aipal_validation.ml.util import init_wandb


class LeukemiaModelEvaluator:
    def __init__(self, data, config, ds_name):
        self.data = data
        self.config = config
        self.cutoffs = {c["category"]: c for c in config["cutoffs"]}
        wandb.log(
            {
                "Evaluating": ds_name,
                "Cohort Size": len(self.data),
                "Cohort Distribution": data["age"].describe().to_dict(),
            }
        )
        self.ds_name = ds_name

    @staticmethod
    def log_to_wandb(results, phase="Evaluation"):
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
        wandb.log({phase: wandb_table})

    @staticmethod
    def calculate_auc(data):
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

    def apply_cutoffs(self, data, cutoff_metric):
        deleted_count = 0
        # Extract the max predicted value for each row and the corresponding class
        prediction_columns = [f"prediction.{cat}" for cat in ["ALL", "AML", "APL"]]

        if cutoff_metric == "ISO":  # Apply Isolation Forest and Local Outlier Factor
            # Drop rows with missing values (or alternatively impute them)
            features = [
                "age",
                "MCV_fL",
                "PT_percent",
                "LDH_UI_L",
                "MCHC_g_L",
                "WBC_G_L",
                "Fibrinogen_g_L",
                "Monocytes_G_L",
                "Platelets_G_L",
                "Lymphocytes_G_L",
                "Monocytes_percent",
            ]
            data_cleaned = data.dropna(subset=features).copy()

            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            data_cleaned.loc[:, "outlier_iso"] = iso_forest.fit_predict(
                data_cleaned[features]
            )

            # Apply Local Outlier Factor
            lof = LocalOutlierFactor(n_neighbors=20)
            data_cleaned.loc[:, "outlier_lof"] = lof.fit_predict(data_cleaned[features])

            # Combine the outlier detections
            data_cleaned.loc[:, "outlier"] = (
                (data_cleaned["outlier_iso"] == -1)
                | (data_cleaned["outlier_lof"] == -1)
            ).astype(int)
            data = data_cleaned[data_cleaned["outlier"] == 0]
            deleted_count = len(data_cleaned) - len(data)

        # Apply cutoff based on max predicted class and value, in NPV case, the cutoff is reversed
        if cutoff_metric == "NPV":
            NotImplementedError

        if cutoff_metric == "PPV" or cutoff_metric == "ACC":
            data["max_pred_value"] = data[prediction_columns].max(axis=1)
            data["max_pred_class"] = (
                data[prediction_columns]
                .idxmax(axis=1)
                .str.replace("prediction.", "", regex=False)
            )
            data["above_cutoff"] = data.apply(
                lambda row: row["max_pred_value"]
                >= self.cutoffs[row["max_pred_class"]][cutoff_metric],
                axis=1,
            )

            deleted_count = len(data[~data["above_cutoff"]])
            data = data[data["above_cutoff"]]

            # Clean up temporary columns
            data = data.copy()
            data.drop(
                columns=["max_pred_value", "max_pred_class", "above_cutoff"],
                inplace=True,
            )

        # Log the count of deleted rows
        logging.info(f"\n\n\nRows deleted: {deleted_count}, metric {cutoff_metric}")
        logging.info(
            f"Class distribution after cutoff: {data['class'].value_counts().to_dict()}"
        )
        logging.info(f"Remaining rows: {len(data)}")

        return data

    def calculate_metrics(self, data, cutoff_type, min_class_size=10):
        # Calculate AUC scores
        auc_scores = self.calculate_auc(data)

        prediction_columns = [f"prediction.{cat}" for cat in ["ALL", "AML", "APL"]]

        data["predicted_class"] = (
            data[prediction_columns]
            .idxmax(axis=1)
            .str.replace(f"{cutoff_type}.", "", regex=True)
            .str.replace("prediction.", "", regex=True)
        )

        # Compute other metrics based on predicted and actual classes
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

    def compute_statistical_metrics(self, results, iterations, confidence_level):
        final_results = {}
        for cat, metrics in results.items():
            final_results[cat] = {}
            for metric, values in metrics.items():
                # Ignore NaN values in calculations
                mean_val = np.nanmean(values)
                std_dev = np.nanstd(values)
                ci_width = norm.ppf((1 + confidence_level) / 2) * (
                    std_dev / np.sqrt(iterations)
                )
                final_results[cat][metric] = (
                    mean_val,
                    mean_val - ci_width,
                    mean_val + ci_width,
                )
        return final_results

    def bootstrap_metrics(
        self, cutoff_type, cutoff_metric=None, iterations=1, confidence_level=0.95
    ):
        results = {
            cat: {
                metric: []
                for metric in ["AUC", "Accuracy", "Precision", "Recall", "F1 Score"]
            }
            for cat in ["ALL", "AML", "APL"]
        }

        # Apply cutoffs based on the specified type, if applicable
        if cutoff_type != "no cutoff" and cutoff_type != "outlier":
            data = self.apply_cutoffs(self.data, cutoff_metric)
            if data.empty:
                logging.warning("No data remaining after applying cutoffs")
                return None
        elif cutoff_type == "outlier":
            data = self.apply_cutoffs(self.data, cutoff_metric)
        else:
            data = self.data
            logging.info(f"Shape of data: {data.shape}")

        wandb.log({"Cohort Size": len(data)})
        wandb.log({"Cohort Distribution": data["class"].value_counts().to_dict()})
        data.to_csv(self.config["task_dir"] / f"predictions_{cutoff_type}.csv")

        # Filter out classes with fewer than 10 samples
        class_counts = data["class"].value_counts()
        valid_classes = class_counts[class_counts >= 10].index.tolist()
        data = data[data["class"].isin(valid_classes)]
        if data.empty:
            logging.warning("No data remaining after filtering out small classes")
            return None

        for _ in range(iterations):
            sampled_data = resample(data)
            metrics = self.calculate_metrics(sampled_data, cutoff_type)
            for cat, met in metrics.items():
                for metric, value in met.items():
                    results[cat][metric].append(value)
        results = self.compute_statistical_metrics(
            results, iterations, confidence_level
        )

        # log to wandb table
        ds_counts = (
            data["class"]
            .value_counts()
            .to_string()
            .strip()
            .replace("\n", "  ")
            .replace("   ", ":")
        )

        tbl_string = f"{cutoff_type} - {self.ds_name}/{self.config['run_id']} - {len(data)}: {ds_counts}"
        self.log_to_wandb(results, phase=tbl_string)
        return results

    def prediction_data_pruner(self, threshold=0):
        """Remove data rows based on the threshold of missing values."""
        data = self.data.copy()
        len_brfore = len(data)
        logging.info(
            f"Class distribution before pruning: {data['class'].value_counts().to_dict()}"
        )

        mandatory_columns = [value[3] for value in self.config["obs_codes_si"].values()]
        mandatory_columns += ["Monocytes_percent"]
        data["nan_percentage"] = data[mandatory_columns].isna().mean(axis=1)

        # Prune data where the percentage of NaN values is above the threshold
        data = data[data["nan_percentage"] <= threshold]
        data.drop(columns=["nan_percentage"], inplace=True)

        mandatory_columns += [
            "prediction.ALL",
            "prediction.AML",
            "prediction.APL",
            "class",
            "age",
        ]
        self.data = data[mandatory_columns]
        logging.info(f"Pruned {len_brfore - len(self.data)} rows")
        logging.info(
            f"Class distribution after pruning (base cohort): {self.data['class'].value_counts().to_dict()}"
        )

        return self.data


def main(config):
    # Idea: Run each center seperately and see how the performance varies
    # Minimum cohort size is 30
    # Minimum class size is 10
    init_wandb(config)

    cutoff_dict = {
        "no cutoff": None,
        "overall cutoff": "ACC",
        "confident cutoff": "PPV",
    }

    if config['step'] == 'all_cohorts':
        # make relative to the package directory
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config_analysis.yaml')
        config_analysis = yaml.safe_load(open(path))
        cities_countries = config_analysis['cities_countries']
        root = config["root_dir"].parent
        paths = [
            f"{root}/{city_country}/aipal/predict.csv"
            for city_country in cities_countries
        ]
        data = pd.DataFrame()
        print(f"Paths: {paths}")
        for path in paths:
            df_small = pd.read_csv(path)
            data = pd.concat([data, df_small])
        logging.info(f"Data shape: {data.shape}")
        all_dir = config["root_dir"] / "all"
        os.makedirs(all_dir, exist_ok=True)
        data.to_csv(all_dir / "predict.csv", index=False)

    else:
        data = pd.read_csv(config["task_dir"] / "predict.csv")
    data_pediatric = data[data["age"] < 18].dropna(subset=["age"])
    data_adults = data[data["age"] >= 18].dropna(subset=["age"])
    minimum_cohort_size = 30
    if (
        len(data_pediatric) > minimum_cohort_size
        and len(data_adults) >= minimum_cohort_size
    ):
        ds_dict = {"kids": data_pediatric, "adults": data_adults}
    elif len(data_pediatric) > minimum_cohort_size:
        ds_dict = {"kids": data_pediatric}
    elif len(data_adults) >= minimum_cohort_size:
        ds_dict = {"adults": data_adults}

    results_df = pd.DataFrame()
    for ds_name, ds in ds_dict.items():
        evaluator = LeukemiaModelEvaluator(ds, config, ds_name)
        ds = evaluator.prediction_data_pruner(threshold=0.2)
        LeukemiaModelEvaluator.calculate_auc(ds)

        for cutoff_type, cutoff_metric in cutoff_dict.items():
            results = evaluator.bootstrap_metrics(
                cutoff_type,
                cutoff_metric,
                iterations=2000 if not config["debug"] else 1,
            )

            if results is None:
                continue

            logging.info(f"{cutoff_type} - {ds_name}")
            results_copy = results.copy()
            # round all numbers to 2 decimals
            for cat, metrics in results_copy.items():
                for metric, values in metrics.items():
                    results_copy[cat][metric] = [round(val, 2) for val in values]
            logging.info(
                f"AUC Scores {ds_name} ALL: {results_copy['ALL']['AUC'][0]}, AML: {results_copy['AML']['AUC'][0]}, APL: {results_copy['APL']['AUC'][0]}"
            )
            logging.info(f"Results: {results_copy}")

            results_dict = {
                f"{cutoff_type} - {ds_name}": results_copy,
            }
            results_df = pd.concat([results_df, pd.DataFrame.from_dict(results_dict)])

    results_df.to_csv(config["task_dir"] / "results.csv")
