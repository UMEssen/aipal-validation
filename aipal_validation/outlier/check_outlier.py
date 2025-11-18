import json
import pickle
import pandas as pd
import yaml
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
from aipal_validation.eval.util import post_filter


class OutlierChecker:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.outlier_models = {}
        self.features = None

    def load_models(self, model_dir, config_path):
        """Load trained models and scalers from disk"""
        model_dir = Path(model_dir)

        # Load config
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.features = config["feature_columns"]

        # Load models
        self.outlier_models = {}
        for cls in ["ALL", "AML", "APL"]:
            with open(model_dir / f"iso_forest_{cls}.pkl", "rb") as f:
                iso_forest = pickle.load(f)
            with open(model_dir / f"lof_{cls}.pkl", "rb") as f:
                lof = pickle.load(f)
            self.outlier_models[cls] = {"iso_forest": iso_forest, "lof": lof}

        # Load scaler and imputer
        with open(model_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(model_dir / "imputer.pkl", "rb") as f:
            self.imputer = pickle.load(f)

    def check_sample(self, sample_data):
        """Check if a single sample is an outlier"""
        # Calculate Monocytes_percent
        if "Monocytes_G_L" in sample_data and "WBC_G_L" in sample_data:
            mono_percent = (sample_data["Monocytes_G_L"] * 100) / sample_data["WBC_G_L"]
            sample_data["Monocytes_percent"] = mono_percent

        # Create DataFrame with all required features
        sample_df = pd.DataFrame([sample_data])

        # Ensure all required features are present
        missing_features = [f for f in self.features if f not in sample_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Preprocess sample
        X_sample = self.imputer.transform(sample_df[self.features])
        X_sample = self.scaler.transform(X_sample)

        # Check for each class
        results = {}
        for cls, models in self.outlier_models.items():
            iso_pred = models["iso_forest"].predict(X_sample)
            lof_pred = models["lof"].predict(X_sample)
            is_outlier = (iso_pred == -1) or (lof_pred == -1)
            results[cls] = {
                "is_outlier": bool(is_outlier),
                "iso_forest_score": float(models["iso_forest"].score_samples(X_sample)[0]),
                "lof_score": float(models["lof"].score_samples(X_sample)[0])
            }

        # Apply post-filters to the sample
        filtered_sample = post_filter(sample_data, logger=None)

        # Add post-filter results to the results dict
        results['post_filter'] = {
            'post_filter_outlier': bool(filtered_sample['post_filter_outlier']),
            'post_filter_flag': filtered_sample['post_filter_flag']
        }

        # Save results to temporary file
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/outlier_results.json", "w") as f:
            json.dump(results, f)

        return results

    def check_dataframe(self, df, class_column='class'):
        """Check if multiple samples in a DataFrame are outliers

        Args:
            df: DataFrame containing samples with feature columns
            class_column: Name of column containing class labels (default: 'class')
                         If present, uses specific class models. Otherwise checks all classes.

        Returns:
            DataFrame with original data plus outlier detection results
        """
        result_df = df.copy()

        # Calculate Monocytes_percent if not already present
        if "Monocytes_percent" not in result_df.columns:
            if "Monocytes_G_L" in result_df.columns and "WBC_G_L" in result_df.columns:
                result_df["Monocytes_percent"] = (result_df["Monocytes_G_L"] * 100) / result_df["WBC_G_L"]

        # Ensure all required features are present
        missing_features = [f for f in self.features if f not in result_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Preprocess all samples
        X = self.imputer.transform(result_df[self.features])
        X = self.scaler.transform(X)

        # If class column exists, use it to determine which model to use for each sample
        if class_column in result_df.columns:
            result_df['outlier'] = 0
            result_df['iso_forest_score'] = 0.0
            result_df['lof_score'] = 0.0

            for cls in result_df[class_column].unique():
                if cls not in self.outlier_models:
                    print(f"Warning: No model found for class {cls}, skipping these samples")
                    continue

                # Get indices for this class
                cls_mask = result_df[class_column] == cls
                cls_indices = result_df.index[cls_mask].tolist()

                if len(cls_indices) == 0:
                    continue

                # Get preprocessed data for this class
                X_cls = X[cls_mask]

                # Run predictions for this class
                models = self.outlier_models[cls]
                iso_pred = models["iso_forest"].predict(X_cls)
                lof_pred = models["lof"].predict(X_cls)

                # A sample is an outlier if either model flags it (prediction == -1)
                is_outlier = (iso_pred == -1) | (lof_pred == -1)

                # Get scores
                iso_scores = models["iso_forest"].score_samples(X_cls)
                lof_scores = models["lof"].score_samples(X_cls)

                # Update results for this class
                result_df.loc[cls_mask, 'outlier'] = is_outlier.astype(int)
                result_df.loc[cls_mask, 'iso_forest_score'] = iso_scores
                result_df.loc[cls_mask, 'lof_score'] = lof_scores
        else:
            # No class column, check against all models and return results for each
            for cls, models in self.outlier_models.items():
                iso_pred = models["iso_forest"].predict(X)
                lof_pred = models["lof"].predict(X)
                is_outlier = (iso_pred == -1) | (lof_pred == -1)

                result_df[f'outlier_{cls}'] = is_outlier.astype(int)
                result_df[f'iso_forest_score_{cls}'] = models["iso_forest"].score_samples(X)
                result_df[f'lof_score_{cls}'] = models["lof"].score_samples(X)

            # Overall outlier flag (outlier in any class)
            outlier_cols = [f'outlier_{cls}' for cls in self.outlier_models.keys()]
            result_df['outlier'] = result_df[outlier_cols].max(axis=1)

        # Apply post-filters to all samples
        result_df = post_filter(result_df, logger=None)

        return result_df
