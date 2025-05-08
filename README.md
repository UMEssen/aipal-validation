# AIPAL Validator

AIPAL Validator is a tool designed to streamline the validation process for [AIPAL](https://github.com/VincentAlcazer/AIPAL). Below you'll find instructions on how to set up and run this validator both locally and with Docker.


## Local Setup

### Prerequisites

- **R Installation**: Ensure R is installed on your system. If not, install it using:

  ```bash
  sudo apt-get install r-base
  ```

  Also ensure to install the following packages within R: 'dplyr', 'tidyr', 'yaml', 'caret', 'xgboost'

1. Install the necessary dependencies:

    ```bash
    poetry install
    ```

2. Run the validation process. You can specify the step to run (all, data, sampling, test):

    ```bash
    poetry run aipal_validation --task aipal --step [all,data,sampling,test]
    ```

## Docker Setup

1. Run the Docker container:

    ```bash
    docker compose run aipal bash
    ```

2. Inside the Docker container, execute the validation script:

    ```bash
    python -m aipal_validation --task aipal --step [all,data,sampling,test]
    ```

# Importing Data from Excel Without a Firemetrics Server

If you don't have a Firemetrics server running and want to import data from an Excel sheet, follow these steps:

## Steps to Import Data

1. **Set the `run_id`:**
   - Update the `run_id` to match your cohort name.

2. **Prepare Your Directory:**
   - In your `root_dir`, create a folder named after your cohort.
   - Inside this folder, create another folder named `aipal`.
   - Place your Excel sheet in the `aipal` folder.

3. **Generate Custom Samples:**
   - Run the following command:
     ```bash
     python -m aipal_validation --task aipal --step sampling
     ```
   - This command invokes the `generate_custom_samples.py` class.
   - Ensure the column names in your Excel file exactly match the expected names in the script.
   - Alternatively, perform necessary data transformations within the script.

4. **Run the Validation Pipeline:**
   - Once the `samples.csv` file is successfully created, execute the following command to run the validation pipeline:
     ```bash
     python -m aipal_validation --task aipal --step test
     ```

# Outlier Detection

To run outlier detection on your dataset and identify potential anomalies:

1. **Local Setup:**
   ```bash
   poetry run aipal_validation --task outlier --step detect
   ```

2. **Docker Setup:**
   ```bash
   docker compose run aipal bash
   python -m aipal_validation --task outlier --step detect
   ```

The outlier detection uses isolation forest and local outlier factor (LOF) algorithms to identify samples that deviate significantly from the expected patterns in each class.

# Model Retraining (on pediatric subset)

To retrain the AIPAL model with your dataset:

1. **Local Setup:**
   ```bash
   poetry run aipal_validation --task retrain --step all
   ```

2. **Docker Setup:**
   ```bash
   docker compose run aipal bash
   python -m aipal_validation --task retrain --step all
   ```

The retraining process will:
- Split your data into training and testing sets
- Train an XGBoost model on the pediatric subset (age < 18)
- Save the retrained model and prediction outputs
- Perform evaluation on the test set
