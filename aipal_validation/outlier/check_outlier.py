import json
import pickle
import pandas as pd
import yaml
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

        # Save results to temporary file
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/outlier_results.json", "w") as f:
            json.dump(results, f)

        return results
