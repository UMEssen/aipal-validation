import numpy as np
import pandas as pd
import yaml
import random
import os
import pickle
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(123)
random.seed(123)
os.environ['PYTHONHASHSEED'] = '123'


class MulticentricOutlierDetector:
    def __init__(self, config_path):
        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        self.df = None
        self.features = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.outlier_models = {}
        self._load_data()

    def _load_data(self):
        """Load and preprocess data from multiple centers"""
        root_path = "/local/work/merengelke/aipal/"
        paths = [
            f"{root_path}{cc}/aipal/predict.csv"
            for cc in self.config["cities_countries"]
        ]

        dfs = []
        for path in paths:
            df_small = pd.read_csv(path)
            df_small["city_country"] = path.split("/")[-3]
            dfs.append(df_small)

        self.df = pd.concat(dfs)

        # Age filtering
        age_filter = (
            self.df["age"] > 18 if self.config["is_adult"] else self.df["age"] <= 18
        )
        self.df = self.df[age_filter]

        # Clean data
        self.df["class"] = self.df["class"].str.strip()
        self.features = self.config["feature_columns"]
        self.df = self.df.groupby("city_country").filter(lambda x: len(x) > 30)

        # Add predictions
        prediction_cols = ["prediction.ALL", "prediction.AML", "prediction.APL"]
        self.df["predicted_class"] = (
            self.df[prediction_cols]
            .idxmax(axis=1)
            .str.replace("prediction.", "", regex=False)
        )

    def _prepare_train_test(self, confidence_threshold=0.9):
        """Split data into high-confidence train and low-confidence test sets"""
        prediction_cols = ["prediction.ALL", "prediction.AML", "prediction.APL"]
        high_conf_mask = (
            self.df[prediction_cols].max(axis=1) >= confidence_threshold
        ) & (self.df["class"] == self.df["predicted_class"])

        self.df_train = self.df[high_conf_mask]
        self.df_test = self.df[~high_conf_mask]

        # Stratify training data
        self._stratify_training_data(min_samples=5)

    def _stratify_training_data(self, min_samples=5):
        """Ensure minimum representation from each class-cohort group"""
        stratified = []
        for (_cls, _cohort), group in self.df_train.groupby(["class", "city_country"]):
            stratified.append(
                group if len(group) <= min_samples else group.sample(min_samples, random_state=123)
            )
        self.df_train = pd.concat(stratified)

    def train_outlier_models(self):
        """Train isolation forest and LOF models for each class"""
        # Preprocess training data
        X_train = self.imputer.fit_transform(self.df_train[self.features])
        X_train = self.scaler.fit_transform(X_train)
        self.df_train[self.features] = X_train

        # Train per-class models
        for cls in self.df_train["class"].unique():
            cls_data = self.df_train[self.df_train["class"] == cls][self.features]

            iso = IsolationForest(random_state=123, contamination="auto").fit(cls_data)
            lof = LocalOutlierFactor(
                n_neighbors=20, contamination="auto", novelty=True
            ).fit(cls_data)

            self.outlier_models[cls] = {"iso_forest": iso, "lof": lof}

    def save_models(self, output_dir):
        """Save trained models and scalers to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for cls, models in self.outlier_models.items():
            with open(output_dir / f"iso_forest_{cls}.pkl", "wb") as f:
                pickle.dump(models["iso_forest"], f)
            with open(output_dir / f"lof_{cls}.pkl", "wb") as f:
                pickle.dump(models["lof"], f)

        # Save scaler and imputer
        with open(output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(output_dir / "imputer.pkl", "wb") as f:
            pickle.dump(self.imputer, f)

    def detect_and_evaluate(self):
        """Run full outlier detection pipeline"""
        # Preprocess test data
        X_test = self.imputer.transform(self.df_test[self.features])
        X_test = self.scaler.transform(X_test)
        self.df_test[self.features] = X_test

        # Baseline evaluation
        logger.info("\n[ Baseline Performance ]")
        self._evaluate(self.df_test)

        # Add a column to track correct/incorrect predictions before outlier removal
        prediction_cols = ["prediction.ALL", "prediction.AML", "prediction.APL"]
        self.df_test["predicted_class"] = (
            self.df_test[prediction_cols]
            .idxmax(axis=1)
            .str.replace("prediction.", "", regex=False)
        )
        self.df_test["correctly_predicted"] = (self.df_test["class"] == self.df_test["predicted_class"]).astype(int)

        # Detect outliers
        self.df_test["outlier"] = 0
        for cls in self.df_test["class"].unique():
            cls_mask = self.df_test["class"] == cls
            if cls not in self.outlier_models:
                continue

            iso_pred = self.outlier_models[cls]["iso_forest"].predict(X_test[cls_mask])
            lof_pred = self.outlier_models[cls]["lof"].predict(X_test[cls_mask])
            self.df_test.loc[cls_mask, "outlier"] = (
                (iso_pred == -1) | (lof_pred == -1)
            ).astype(int)

        # Count and report outliers
        total_samples = len(self.df_test)
        outlier_samples = self.df_test["outlier"].sum()
        outlier_percentage = (outlier_samples / total_samples) * 100

        logger.info("\n[ Outlier Detection Results ]")
        logger.info(f"Total training samples: {len(self.df_train)}")
        logger.info(f"Total test samples: {total_samples}")
        logger.info(f"Detected outliers: {outlier_samples} ({outlier_percentage:.2f}%)")
        logger.info(
            f"Remaining samples: {total_samples - outlier_samples} ({100 - outlier_percentage:.2f}%)"
        )

        # Per-class outlier statistics
        logger.info("\nOutliers by class:")
        for cls in self.df_test["class"].unique():
            cls_samples = sum(self.df_test["class"] == cls)
            cls_outliers = sum(
                (self.df_test["class"] == cls) & (self.df_test["outlier"] == 1)
            )
            cls_percentage = (
                (cls_outliers / cls_samples) * 100 if cls_samples > 0 else 0
            )
            logger.info(f"  {cls}: {cls_outliers}/{cls_samples} ({cls_percentage:.2f}%)")

        # Analysis of wrongly filtered samples
        logger.info("\n[ Analysis of Wrongly Filtered Samples ]")

        # Count correctly and incorrectly predicted samples before filtering
        correct_before = sum(self.df_test["correctly_predicted"] == 1)
        incorrect_before = sum(self.df_test["correctly_predicted"] == 0)

        # Count how many correct and incorrect predictions were filtered out
        correct_filtered = sum((self.df_test["correctly_predicted"] == 1) & (self.df_test["outlier"] == 1))
        incorrect_filtered = sum((self.df_test["correctly_predicted"] == 0) & (self.df_test["outlier"] == 1))

        # Calculate percentages
        correct_filtered_pct = (correct_filtered / correct_before) * 100 if correct_before > 0 else 0
        incorrect_filtered_pct = (incorrect_filtered / incorrect_before) * 100 if incorrect_before > 0 else 0

        logger.info(f"Correctly predicted samples before filtering: {correct_before}")
        logger.info(f"Incorrectly predicted samples before filtering: {incorrect_before}")
        logger.info(f"Correctly predicted samples filtered out: {correct_filtered} ({correct_filtered_pct:.2f}%)")
        logger.info(f"Incorrectly predicted samples filtered out: {incorrect_filtered} ({incorrect_filtered_pct:.2f}%)")

        # Per-class analysis of wrongly filtered samples
        logger.info("\nWrongly filtered samples by class:")
        for cls in self.df_test["class"].unique():
            cls_correct = sum((self.df_test["class"] == cls) & (self.df_test["correctly_predicted"] == 1))
            cls_correct_filtered = sum(
                (self.df_test["class"] == cls)
                & (self.df_test["correctly_predicted"] == 1)
                & (self.df_test["outlier"] == 1)
            )
            cls_correct_pct = (cls_correct_filtered / cls_correct) * 100 if cls_correct > 0 else 0

            cls_incorrect = sum((self.df_test["class"] == cls) & (self.df_test["correctly_predicted"] == 0))
            cls_incorrect_filtered = sum(
                (self.df_test["class"] == cls)
                & (self.df_test["correctly_predicted"] == 0)
                & (self.df_test["outlier"] == 1)
            )
            cls_incorrect_pct = (cls_incorrect_filtered / cls_incorrect) * 100 if cls_incorrect > 0 else 0

            logger.info(f"  {cls}:")
            logger.info(f"    Correctly predicted filtered: {cls_correct_filtered}/{cls_correct} ({cls_correct_pct:.2f}%)")
            logger.info(f"    Incorrectly predicted filtered: {cls_incorrect_filtered}/{cls_incorrect} ({cls_incorrect_pct:.2f}%)")

        # Filter outliers and re-evaluate
        clean_df = self.df_test[self.df_test["outlier"] == 0]
        logger.info("\n[ Performance After Outlier Removal ]")
        self._evaluate(clean_df)


    def _evaluate(self, df):
        """Helper method for performance evaluation"""
        # Get available classes in the data
        available_classes = sorted(df["class"].unique())

        # Calculate AUC only for available classes
        auc_scores = {}
        for cls in available_classes:
            # Create binary labels for this class
            y_true = (df["class"] == cls).astype(int)
            if len(np.unique(y_true)) > 1:  # Check if we have both positive and negative samples
                auc_scores[cls] = roc_auc_score(y_true, df[f"prediction.{cls}"])
            else:
                auc_scores[cls] = np.nan
        logger.info("AUC Scores:", auc_scores)

        # Classification report
        prediction_cols = [f"prediction.{c}" for c in available_classes]
        y_pred = df[prediction_cols].idxmax(axis=1)
        y_pred = y_pred.str.replace("prediction.", "", regex=False)
        logger.info("\nClassification Report:")
        logger.info(classification_report(df["class"], y_pred, labels=available_classes))

        # Calculate PPV and NPV for each class
        logger.info("\nPPV and NPV by class:")
        ppv_npv_scores = {}
        for cls in available_classes:
            # For each class, calculate binary classification metrics (one-vs-rest)
            y_true_binary = (df["class"] == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)

            # Calculate confusion matrix components
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))  # True Positives
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))  # False Positives
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))  # True Negatives
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))  # False Negatives

            # Calculate PPV (Positive Predictive Value) = Precision = TP / (TP + FP)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Calculate NPV (Negative Predictive Value) = TN / (TN + FN)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            ppv_npv_scores[cls] = {'PPV': ppv, 'NPV': npv}
            logger.info(f"  {cls}: PPV = {ppv:.4f}, NPV = {npv:.4f}")

        return {
            'auc_scores': auc_scores,
            'ppv_npv_scores': ppv_npv_scores,
            'accuracy': classification_report(df["class"], y_pred, labels=available_classes, output_dict=True)['accuracy']
        }
