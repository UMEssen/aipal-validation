# AIPAL Validation

External validation pipeline for [AIPAL](https://github.com/VincentAlcazer/AIPAL) (Acute Leukemia Prediction), a machine learning model that predicts acute leukemia subtypes (ALL, AML, APL) from routine laboratory measurements. This tool provides a FHIR-based data extraction pipeline, outlier detection, and model retraining capabilities for multicentric validation.

## Citation

If you use this software in your research, please cite:

> *Citation will be added upon publication.*

## Local Setup

### Prerequisites

- **Python** >= 3.10
- **R** with packages: `dplyr`, `tidyr`, `yaml`, `caret`, `xgboost`

  ```bash
  sudo apt-get install r-base
  R -e "install.packages(c('dplyr', 'tidyr', 'yaml', 'caret', 'xgboost'), repos='https://cran.r-project.org/')"
  ```

1. Install dependencies:

    ```bash
    poetry install
    ```

2. Run the validation pipeline:

    ```bash
    poetry run aipal_validation --task aipal --step [all,data,sampling,test]
    ```

## Docker Setup

1. Set your data directory and run the container:

    ```bash
    export DATA_DIR=/path/to/your/data
    docker compose run aipal bash
    ```

2. Inside the container:

    ```bash
    python -m aipal_validation --task aipal --step [all,data,sampling,test]
    ```

## Project Structure

- `aipal_validation/` — Main package
  - `r/` — R scripts for prediction and model training
  - `config/` — Configuration files
  - `data_preprocessing/` — Data preprocessing modules
  - `eval/` — Evaluation and analysis scripts
  - `fhir/` — FHIR data extraction, filtering, and validation
  - `ml/` — Machine learning modules
  - `outlier/` — Outlier detection (Isolation Forest, LOF)
  - `helper/` — Utility functions

## Importing Data from Excel

If you don't have a FHIR server and want to import data from an Excel sheet:

1. Update the `run_id` in the config to match your cohort name.
2. In your `root_dir`, create `<cohort_name>/aipal/` and place your Excel sheet there.
3. Generate samples:
   ```bash
   python -m aipal_validation --task aipal --step sampling
   ```
   Ensure column names in your Excel file match those expected by `generate_custom_samples.py`.
4. Run the validation pipeline:
   ```bash
   python -m aipal_validation --task aipal --step test
   ```

## Outlier Detection

Run outlier detection using Isolation Forest and Local Outlier Factor (LOF):

```bash
poetry run aipal_validation --task outlier --step detect
```

## Model Retraining (Pediatric Subset)

Retrain the AIPAL model on a pediatric subset (age < 18):

```bash
poetry run aipal_validation --task retrain --step all
```

This will split data into training/testing sets, train an XGBoost model, save the retrained model to `aipal_validation/r/`, and evaluate on the test set.

## Credits

The XGBoost model file (`221003_Final_model_res_list.rds`) was obtained from [VincentAlcazer/AIPAL](https://github.com/VincentAlcazer/AIPAL) (MIT License). No source code from the original repository is reused; only the trained model asset is redistributed for validation purposes.

## License

This project is licensed under the [MIT License](LICENSE).
