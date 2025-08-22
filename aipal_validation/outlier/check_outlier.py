import json
import pickle
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os


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
            if sample_data["WBC_G_L"] == 0:
                sample_data["Monocytes_percent"] = float('nan')
            else:
                mono_percent = (sample_data["Monocytes_G_L"] * 100) / sample_data["WBC_G_L"]
                sample_data["Monocytes_percent"] = mono_percent

        # Create DataFrame with all required features
        sample_df = pd.DataFrame([sample_data])

        # Ensure all required features are present
        missing_features = [f for f in self.features if f not in sample_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Handle infinity values and extremely large numbers before imputation
        features_data = sample_df[self.features].copy()
        
        # Replace infinite values with NaN
        features_data = features_data.replace([float('inf'), float('-inf')], float('nan'))
        
        # Replace extremely large values that could cause overflow

        max_float64 = np.finfo(np.float64).max / 1e6  # Use a safe margin
        features_data = features_data.clip(lower=-max_float64, upper=max_float64)

        # Preprocess sample
        X_sample = self.imputer.transform(features_data)
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

        # Save results to temporary file
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/outlier_results.json", "w") as f:
            json.dump(results, f)

        return results

    def check_dataframe(self, df):
        """Check outliers for entire DataFrame (much faster than sample-by-sample)"""
        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Calculate Monocytes_percent if needed
        if "Monocytes_G_L" in df_copy.columns and "WBC_G_L" in df_copy.columns:
            df_copy["Monocytes_percent"] = (df_copy["Monocytes_G_L"] * 100) / df_copy["WBC_G_L"]

        # Ensure all required features are present
        missing_features = [f for f in self.features if f not in df_copy.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Handle infinity values and extremely large numbers before imputation
        features_data = df_copy[self.features].copy()
        
        # Replace infinite values with NaN
        features_data = features_data.replace([float('inf'), float('-inf')], float('nan'))
        
        # Replace extremely large values that could cause overflow

        max_float64 = np.finfo(np.float64).max / 1e6  # Use a safe margin
        features_data = features_data.clip(lower=-max_float64, upper=max_float64)
        
        # Preprocess all samples at once
        X_data = self.imputer.transform(features_data)
        X_data = self.scaler.transform(X_data)

        # Convert back to DataFrame to preserve feature names (fixes sklearn warnings)
        X_df = pd.DataFrame(X_data, columns=self.features, index=df_copy.index)

        # Initialize outlier column
        df_copy["outlier"] = 0

        # Apply outlier detection for each class
        for cls in df_copy["class"].unique():
            if cls not in self.outlier_models:
                continue

            cls_mask = df_copy["class"] == cls
            cls_data = X_df[cls_mask]

            if len(cls_data) == 0:
                continue

            # Get models for this class
            models = self.outlier_models[cls]

            # Predict outliers using both models
            iso_pred = models["iso_forest"].predict(cls_data)
            lof_pred = models["lof"].predict(cls_data)

            # Mark as outlier if either model flags it
            outlier_mask = (iso_pred == -1) | (lof_pred == -1)
            df_copy.loc[cls_mask, "outlier"] = outlier_mask.astype(int)

        return df_copy
