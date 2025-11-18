#!/usr/bin/env python3
"""
Batch outlier detection script for processing CSV files with multiple samples.
Implements the same logic as single sample prediction:
1. Make prediction to get most likely class (argmax)
2. Check outlier status for that predicted class only
3. Flag as outlier only if it's an outlier for its predicted class
"""

import argparse
import pandas as pd
import logging
import numpy as np
from pathlib import Path
from aipal_validation.outlier.check_outlier import OutlierChecker
from aipal_validation.eval.util import load_data, post_filter
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def setup_task_logging(task_dir):
    """Set up logging to save to task directory"""
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler for task directory
    fh = logging.FileHandler(task_dir / "batch_outlier_log.txt")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(levelname)s %(asctime)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s"
    )
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

    logger.info(f"Batch outlier detection logs will be saved to: {task_dir / 'batch_outlier_log.txt'}")
    return task_dir


def analyze_accuracy_by_column(result_df, column_name, task_dir):
    """Analyze outlier detection accuracy by a specific column (category or diagnosis)"""
    if column_name not in result_df.columns:
        logger.warning(f"Column '{column_name}' not found in results. Skipping analysis.")
        return

    logger.info(f"\nACCURACY ANALYSIS BY {column_name.upper()}")
    logger.info("-" * 60)

    # Filter out samples without the column data
    valid_mask = result_df[column_name].notna()
    valid_df = result_df[valid_mask]

    if len(valid_df) == 0:
        logger.warning(f"No valid data found for {column_name} analysis")
        return

    logger.info(f"Analyzing {len(valid_df)} samples with valid {column_name} data")

    # Calculate accuracy metrics for each unique value
    accuracy_results = {}
    excluded_values = []

    for value in valid_df[column_name].unique():
        mask = valid_df[column_name] == value
        subset = valid_df[mask]

        if len(subset) == 0:
            continue

        total_samples = len(subset)

        # Prerequisite: n must be greater than 1 for meaningful statistical analysis
        if total_samples <= 1:
            excluded_values.append(f"{value} (n={total_samples})")
            logger.info(f"Excluding {value}: insufficient sample size (n={total_samples})")
            continue

        outliers = subset['final_outlier_decision'].sum()
        outlier_rate = outliers / total_samples * 100

        # Calculate average prediction confidence
        avg_confidence = subset['predicted_probability'].mean()

        # Get predicted class distribution
        pred_dist = subset['predicted_class'].value_counts().to_dict()

        accuracy_results[value] = {
            'total_samples': total_samples,
            'outliers': outliers,
            'outlier_rate': outlier_rate,
            'avg_confidence': avg_confidence,
            'predicted_classes': pred_dist
        }

        logger.info(f"{value}:")
        logger.info(f"  Samples: {total_samples}")
        logger.info(f"  Outliers: {outliers} ({outlier_rate:.1f}%)")
        logger.info(f"  Avg prediction confidence: {avg_confidence:.3f}")
        logger.info(f"  Predicted classes: {pred_dist}")

    # Log excluded values
    if excluded_values:
        logger.info(f"Excluded from analysis (n≤1): {', '.join(excluded_values)}")
    else:
        logger.info("All categories/diagnoses have sufficient sample sizes (n>1)")

    # Save detailed analysis to file
    analysis_file = task_dir / f"accuracy_analysis_{column_name}.txt"
    with open(analysis_file, 'w') as f:
        f.write(f"ACCURACY ANALYSIS BY {column_name.upper()}\n")
        f.write("=" * 50 + "\n\n")

        # Add information about excluded values
        if excluded_values:
            f.write(f"EXCLUDED FROM ANALYSIS (n≤1): {', '.join(excluded_values)}\n\n")

        for value, metrics in accuracy_results.items():
            f.write(f"{value}:\n")
            f.write(f"  Total samples: {metrics['total_samples']}\n")
            f.write(f"  Outliers: {metrics['outliers']} ({metrics['outlier_rate']:.1f}%)\n")
            f.write(f"  Average prediction confidence: {metrics['avg_confidence']:.3f}\n")
            f.write(f"  Predicted classes: {metrics['predicted_classes']}\n\n")

    logger.info(f"Detailed {column_name} analysis saved to: {analysis_file}")
    return accuracy_results


def load_excel_data(excel_file, task_dir):
    """Load Excel or CSV file with all data including category and diagnosis"""
    file_path = Path(excel_file)
    file_ext = file_path.suffix.lower()

    logger.info(f"Loading data file: {excel_file} (extension: {file_ext})")

    if not file_path.exists():
        logger.error(f"Data file not found at {excel_file}")
        raise FileNotFoundError(f"Data file not found: {excel_file}")

    try:
        # Load data based on file extension
        if file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(excel_file)
            logger.info(f"Loaded Excel data with {len(df)} samples")
        elif file_ext == '.csv':
            df = pd.read_csv(excel_file)
            logger.info(f"Loaded CSV data with {len(df)} samples")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .xlsx, .xls, .csv")

        logger.info(f"Data columns: {list(df.columns)}")

        # Create ID column from index if it doesn't exist
        if 'ID' not in df.columns:
            df['ID'] = df.index.astype(str)
            logger.info(f"Created ID column from index (range: {df['ID'].iloc[0]} to {df['ID'].iloc[-1]})")

        # Check for optional columns and log distributions
        for col in ['category', 'diagnosis']:
            if col not in df.columns:
                logger.warning(f"Missing optional column: {col}. Analysis will be skipped.")
            else:
                logger.info(f"{col.capitalize()} distribution:")
                for val, count in df[col].value_counts().items():
                    logger.info(f"  {val}: {count} samples")

        return df

    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        raise


def load_cohort_data(config_path, root_path, is_adult, filter_by_size, filter_missing_values, task_dir):
    """Load data from all cohorts using load_data from util.py"""
    logger.info("Loading data from all cohorts using load_data...")
    logger.info(f"  Config path: {config_path}")
    logger.info(f"  Root path: {root_path}")
    logger.info(f"  Is adult: {is_adult}")
    logger.info(f"  Filter by size: {filter_by_size}")
    logger.info(f"  Filter missing values: {filter_missing_values}")

    try:
        # Load data using load_data function
        df, config, features = load_data(
            config_path=config_path,
            root_path=root_path,
            filter_by_size=filter_by_size,
            is_adult=is_adult,
            filter_missing_values=filter_missing_values
        )

        logger.info(f"Loaded cohort data with {len(df)} samples")
        logger.info(f"Data columns: {list(df.columns)}")
        # Log exclusion statistics
        logger.info("\nEXCLUSION STATISTICS:")
        logger.info("-" * 60)

        # Count samples by city_country before and after filtering
        # Note: load_data already applies filters, so we log what we have
        logger.info(f"Total samples after filtering: {len(df)}")

        if 'city_country' in df.columns:
            logger.info("\nSamples by city_country:")
            city_counts = df['city_country'].value_counts().sort_index()
            for city, count in city_counts.items():
                logger.info(f"  {city}: {count} samples")
            logger.info(f"  Total cities: {len(city_counts)}")

        # Log age statistics
        if 'age' in df.columns:
            logger.info("\nAge statistics:")
            logger.info(f"  Min age: {df['age'].min():.1f}")
            logger.info(f"  Max age: {df['age'].max():.1f}")
            logger.info(f"  Mean age: {df['age'].mean():.1f}")
            if is_adult:
                logger.info(f"  Adult samples (age > 18): {len(df)}")
            else:
                logger.info(f"  Children samples (age <= 18): {len(df)}")

        # Log sex distribution
        if 'sex' in df.columns:
            logger.info("\nSex distribution:")
            for sex, count in df['sex'].value_counts().items():
                logger.info(f"  {sex}: {count} samples")

        # Check if ID column exists, if not create one
        if 'ID' not in df.columns:
            logger.warning("ID column not found in loaded data. Creating sequential IDs...")
            df['ID'] = [f"COHORT_{i+1:06d}" for i in range(len(df))]

        # Note that category and diagnosis may not be in the loaded data
        if 'category' not in df.columns:
            logger.info("Note: 'category' column not found in cohort data. Category analysis will be skipped.")
        if 'diagnosis' not in df.columns:
            logger.info("Note: 'diagnosis' column not found in cohort data. Diagnosis analysis will be skipped.")

        return df

    except Exception as e:
        logger.error(f"Error loading cohort data: {e}")
        raise


def predict_classes_with_r(df, task_dir):
    """Use R script to predict classes for all samples in DataFrame"""
    logger.info("Running R prediction script to determine class probabilities...")

    try:
        # Create temporary CSV file for R script
        temp_csv = task_dir / "temp_data_for_r.csv"

        # Select only the columns needed for prediction (exclude category/diagnosis for R)
        prediction_columns = ['ID', 'age', 'sex', 'WBC_G_L', 'Monocytes_G_L', 'Lymphocytes_G_L',
                             'Platelets_G_L', 'MCHC_g_L', 'MCV_fL', 'LDH_UI_L', 'PT_percent', 'Fibrinogen_g_L']

        # Filter to only include columns that exist in the dataframe
        available_columns = [col for col in prediction_columns if col in df.columns]
        temp_df = df[available_columns].copy()

        # Ensure no inf/nan values get to R (XGBoost doesn't handle inf well)
        numeric_cols_temp = temp_df.select_dtypes(include=[np.number]).columns
        temp_df[numeric_cols_temp] = temp_df[numeric_cols_temp].replace([np.inf, -np.inf], np.nan)

        # Save temporary CSV
        temp_df.to_csv(temp_csv, index=False)
        logger.info(f"Created temporary CSV for R script: {temp_csv}")

        # Run R prediction script using console_predict.R approach
        cmd = [
            "Rscript", "-e",
            f"""
            library(dplyr)
            res_list <- readRDS("aipal_validation/r/221003_Final_model_res_list.rds")
            model <- res_list$final_model
            new_data <- read.csv("{temp_csv}")

            # Add calculated column for Monocytes_percent if not present (avoid division by zero)
            if (!"Monocytes_percent" %in% names(new_data)) {{
                new_data$Monocytes_percent <- ifelse(new_data$WBC_G_L > 0,
                                                     new_data$Monocytes_G_L * 100 / new_data$WBC_G_L,
                                                     NA)
            }}

            # Make prediction
            prediction <- predict(model, newdata = new_data, type = "prob", na.action = na.pass)

            # Save predictions to temporary file
            write.csv(prediction, "{task_dir}/predictions.csv", row.names = FALSE)
            cat("Predictions saved to {task_dir}/predictions.csv\n")
            """
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")

        if result.returncode != 0:
            logger.error(f"R script failed with return code {result.returncode}")
            logger.error(f"R script stderr: {result.stderr}")
            raise RuntimeError(f"R prediction failed: {result.stderr}")

        logger.info(f"R script output: {result.stdout}")

        # Load the predictions
        pred_file = Path(task_dir) / "predictions.csv"
        if not pred_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_file}")

        predictions_df = pd.read_csv(pred_file)
        logger.info(f"Loaded predictions for {len(predictions_df)} samples")
        logger.info(f"Prediction columns: {list(predictions_df.columns)}")

        # Clean up temporary file
        temp_csv.unlink()

        return predictions_df

    except Exception as e:
        logger.error(f"Error running R prediction: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Batch outlier detection for Excel/CSV files with prediction-based logic")
    parser.add_argument("--excel_file", type=str, help="Path to Excel or CSV file containing samples with category and diagnosis (required if --use_cohort_data is not set)")
    parser.add_argument("--use_cohort_data", action="store_true", help="Load data from all cohorts using load_data from util.py instead of Excel file")
    parser.add_argument("--config_path", type=str, default="aipal_validation/config/config_outlier.yaml", help="Path to config yaml (used with --use_cohort_data)")
    parser.add_argument("--root_path", type=str, default="/data/", help="Root path for cohort data files (used with --use_cohort_data)")
    parser.add_argument("--children", action="store_true", help="Load children data (age <= 18) instead of adult data (used with --use_cohort_data)")
    parser.add_argument("--no_filter_by_size", action="store_true", help="Do not filter cohorts by minimum size (used with --use_cohort_data)")
    parser.add_argument("--no_filter_missing", action="store_true", help="Do not filter samples with >20%% missing values (used with --use_cohort_data)")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained outlier detection models")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output", type=str, help="Output CSV file path (default: adds '_with_outliers' to input filename)")
    parser.add_argument("--task_dir", type=str, help="Task directory for logging (default: creates dir next to input Excel file)")

    args = parser.parse_args()

    # Validate arguments
    if not args.use_cohort_data and not args.excel_file:
        parser.error("Either --excel_file or --use_cohort_data must be provided")

    # Set up task directory and logging
    if args.task_dir:
        task_dir = setup_task_logging(args.task_dir)
    else:
        # Create task directory
        if args.use_cohort_data:
            # Create task directory in root_path
            base_path = Path(args.root_path)
            task_dir_name = f"batch_outlier_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            task_dir = setup_task_logging(base_path / task_dir_name)
        else:
            # Create task directory in the same location as the input Excel file
            excel_path = Path(args.excel_file)
            task_dir_name = f"batch_outlier_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            task_dir = setup_task_logging(excel_path.parent / task_dir_name)

    logger.info("="*80)
    logger.info("STARTING BATCH OUTLIER DETECTION WITH PREDICTION-BASED LOGIC")
    logger.info("="*80)

    # Load data from either Excel file or cohort data
    if args.use_cohort_data:
        is_adult = not args.children  # Default to adult unless --children is specified
        df = load_cohort_data(
            config_path=args.config_path,
            root_path=args.root_path,
            is_adult=is_adult,
            filter_by_size=not args.no_filter_by_size,
            filter_missing_values=not args.no_filter_missing,
            task_dir=task_dir
        )
        logger.info(f"Loaded {len(df)} samples from cohort data")
    else:
        df = load_excel_data(args.excel_file, task_dir)
        logger.info(f"Loaded {len(df)} samples from {args.excel_file}")

    logger.info(f"Sample columns: {list(df.columns)}")

    # Log sample statistics
    logger.info(f"Sample ID range: {df['ID'].iloc[0]} to {df['ID'].iloc[-1]}")
    logger.info(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
    logger.info(f"Sex distribution: {df['sex'].value_counts().to_dict()}")

    # Clean data: replace infinity values with NaN and handle problematic calculations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    nan_count = df[numeric_cols].isna().sum().sum()

    if inf_count > 0 or nan_count > 0:
        logger.warning(f"Found {inf_count} infinity values and {nan_count} NaN values. Cleaning data...")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Handle potential division by zero issues (e.g., Monocytes_percent calculation)
        if 'Monocytes_G_L' in df.columns and 'WBC_G_L' in df.columns:
            # Calculate Monocytes_percent safely
            wbc_valid = df['WBC_G_L'] > 0
            df.loc[wbc_valid, 'Monocytes_percent'] = (df.loc[wbc_valid, 'Monocytes_G_L'] * 100) / df.loc[wbc_valid, 'WBC_G_L']
            df.loc[~wbc_valid, 'Monocytes_percent'] = np.nan

    # Step 1: Get predictions for all samples to determine predicted classes
    logger.info("\nSTEP 1: RUNNING PREDICTION TO DETERMINE PREDICTED CLASSES")
    logger.info("-" * 60)
    try:
        predictions_df = predict_classes_with_r(df, task_dir)

        # Get predicted class for each sample (argmax)
        predicted_classes = predictions_df.idxmax(axis=1)
        logger.info("Predicted class distribution:")
        for cls, count in predicted_classes.value_counts().items():
            logger.info(f"  {cls}: {count} samples ({count/len(predicted_classes)*100:.1f}%)")

        # Add predictions and predicted classes to dataframe
        # Reset indices to avoid reindexing issues with duplicate indices
        df = df.reset_index(drop=True)
        predictions_df = predictions_df.reset_index(drop=True)
        predicted_classes = predicted_classes.reset_index(drop=True)

        df = pd.concat([df, predictions_df], axis=1)
        df['predicted_class'] = predicted_classes
        df['predicted_probability'] = predictions_df.max(axis=1)

        logger.info(f"Average prediction confidence: {df['predicted_probability'].mean():.3f}")

    except Exception as e:
        logger.error(f"Failed to run predictions: {e}")
        logger.info("Falling back to setting all samples to 'ALL' class...")
        df['predicted_class'] = 'ALL'
        df['predicted_probability'] = 1.0

    # Step 2: Initialize and load the outlier checker
    logger.info("\nSTEP 2: LOADING OUTLIER DETECTION MODELS")
    logger.info("-" * 60)
    logger.info(f"Loading outlier detection models from: {args.model_dir}")
    checker = OutlierChecker()
    checker.load_models(args.model_dir, args.config)
    logger.info(f"Available outlier models for classes: {list(checker.outlier_models.keys())}")

    # Step 3: Run outlier detection for all classes for each sample
    logger.info("\nSTEP 3: RUNNING OUTLIER DETECTION FOR ALL CLASSES")
    logger.info("-" * 60)
    logger.info("Running comprehensive outlier detection (all classes)...")

    # Temporarily set all samples to have each class to get outlier scores for all classes
    df_temp = df.copy()
    df_temp['class'] = 'ALL'  # Start with ALL class
    result_df = checker.check_dataframe(df_temp)

    # Get outlier results for each class
    outlier_results = {}
    for cls in ['ALL', 'AML', 'APL']:
        if cls in checker.outlier_models:
            logger.info(f"Processing outlier detection for class: {cls}")
            df_cls = df.copy()
            df_cls['class'] = cls
            cls_result = checker.check_dataframe(df_cls)
            outlier_results[cls] = cls_result['outlier'].values
            logger.info(f"  {cls} outliers: {cls_result['outlier'].sum()}/{len(cls_result)} ({cls_result['outlier'].sum()/len(cls_result)*100:.1f}%)")

    # Step 4: Apply prediction-based outlier logic
    logger.info("\nSTEP 4: APPLYING PREDICTION-BASED OUTLIER LOGIC")
    logger.info("-" * 60)

    result_df = df.copy()
    result_df['final_outlier_decision'] = 0
    result_df['outlier_reason'] = 'Normal'

    # Add outlier flags for all classes
    for cls in outlier_results:
        result_df[f'outlier_{cls}'] = outlier_results[cls]

    prediction_based_outliers = 0
    for idx, row in result_df.iterrows():
        predicted_cls = row['predicted_class']

        # Check if sample is outlier for its predicted class
        if predicted_cls in outlier_results:
            is_outlier_for_predicted = outlier_results[predicted_cls][idx]
            result_df.loc[idx, 'final_outlier_decision'] = int(is_outlier_for_predicted)

            if is_outlier_for_predicted:
                result_df.loc[idx, 'outlier_reason'] = f'Outlier_for_predicted_class_{predicted_cls}'
                prediction_based_outliers += 1
            else:
                result_df.loc[idx, 'outlier_reason'] = f'Normal_for_predicted_class_{predicted_cls}'
        else:
            logger.warning(f"No outlier model found for predicted class {predicted_cls} for sample {row['ID']}")
            result_df.loc[idx, 'final_outlier_decision'] = 0
            result_df.loc[idx, 'outlier_reason'] = f'No_model_for_{predicted_cls}'

    logger.info("Prediction-based outlier detection completed:")
    logger.info(f"  Total samples: {len(result_df)}")
    logger.info(f"  Final outliers: {prediction_based_outliers} ({prediction_based_outliers/len(result_df)*100:.1f}%)")

    # Step 5: Apply post-filters based on clinical criteria
    logger.info("\nSTEP 5: APPLYING POST-FILTERS (CLINICAL CRITERIA)")
    logger.info("-" * 60)
    result_df = post_filter(result_df, logger)

    # Combine prediction-based outliers with post-filter flags for final decision
    logger.info("Combining prediction-based outliers with post-filter flags...")
    prediction_outliers = result_df['final_outlier_decision'].copy()
    post_filter_outliers = result_df['post_filter_outlier'].copy()

    # Update final_outlier_decision: flag as outlier if EITHER prediction-based OR post-filter flags it
    result_df['final_outlier_decision'] = ((prediction_outliers == 1) | (post_filter_outliers == 1)).astype(int)

    # Update outlier_reason to include post-filter information
    for idx in result_df.index:
        pred_flag = prediction_outliers.iloc[idx] == 1
        post_flag = post_filter_outliers.iloc[idx] == 1

        if pred_flag and post_flag:
            result_df.loc[idx, 'outlier_reason'] = f"{result_df.loc[idx, 'outlier_reason']}; PostFilter:{result_df.loc[idx, 'post_filter_flag']}"
        elif post_flag:
            result_df.loc[idx, 'outlier_reason'] = f"PostFilter:{result_df.loc[idx, 'post_filter_flag']}"

    combined_outliers = result_df['final_outlier_decision'].sum()
    logger.info(f"Combined outliers (prediction OR post-filter): {combined_outliers}/{len(result_df)} ({combined_outliers/len(result_df)*100:.1f}%)")
    logger.info(f"  Prediction-based only: {(prediction_outliers == 1).sum()}")
    logger.info(f"  Post-filter only: {((post_filter_outliers == 1) & (prediction_outliers == 0)).sum()}")
    logger.info(f"  Both: {((prediction_outliers == 1) & (post_filter_outliers == 1)).sum()}")

    # Step 6: Analyze accuracy by category and diagnosis
    logger.info("\nSTEP 6: ANALYZING ACCURACY BY CATEGORY AND DIAGNOSIS")
    logger.info("-" * 60)

    category_accuracy = None
    diagnosis_accuracy = None

    if 'category' in result_df.columns:
        category_accuracy = analyze_accuracy_by_column(result_df, 'category', task_dir)

    if 'diagnosis' in result_df.columns:
        diagnosis_accuracy = analyze_accuracy_by_column(result_df, 'diagnosis', task_dir)

    # Count outliers
    outlier_count = result_df["final_outlier_decision"].sum()
    total_count = len(result_df)
    logger.info("\nFINAL OUTLIER SUMMARY:")
    logger.info(f"  Total samples processed: {total_count}")
    logger.info(f"  Combined outliers (prediction OR post-filter): {outlier_count} ({outlier_count/total_count*100:.1f}%)")

    # Detailed breakdown by predicted class
    logger.info("\nBreakdown by predicted class:")
    for cls in result_df['predicted_class'].unique():
        cls_mask = result_df['predicted_class'] == cls
        cls_count = cls_mask.sum()
        cls_outliers = result_df.loc[cls_mask, 'final_outlier_decision'].sum()
        logger.info(f"  {cls}: {cls_outliers}/{cls_count} outliers ({cls_outliers/cls_count*100:.1f}%)")

    # Compare with naive approach (ALL class for everyone)
    if 'outlier_ALL' in result_df.columns:
        naive_outliers = result_df['outlier_ALL'].sum()
        logger.info("\nComparison with naive approach (ALL class for all):")
        logger.info(f"  Naive outliers: {naive_outliers}/{total_count} ({naive_outliers/total_count*100:.1f}%)")
        logger.info(f"  Combined outliers (prediction OR post-filter): {outlier_count}/{total_count} ({outlier_count/total_count*100:.1f}%)")
        logger.info(f"  Difference: {abs(naive_outliers - outlier_count)} samples ({abs(naive_outliers - outlier_count)/total_count*100:.1f}%)")

    # Set up output file path
    if args.output:
        output_path = args.output
    else:
        if args.use_cohort_data:
            # Create output filename based on cohort data parameters
            cohort_type = "children" if args.children else "adult"
            output_filename = f"cohort_data_{cohort_type}_with_outliers.csv"
            output_path = task_dir / output_filename
        else:
            input_path = Path(args.excel_file)
            output_path = input_path.parent / f"{input_path.stem}_with_outliers.csv"

    # Step 7: Save results
    logger.info("\nSTEP 7: SAVING RESULTS")
    logger.info("-" * 60)
    logger.info(f"Saving results to: {output_path}")

    # Reorder columns for better readability
    column_order = ['ID'] + [col for col in result_df.columns if col != 'ID']
    result_df = result_df.reindex(columns=[col for col in column_order if col in result_df.columns])

    result_df.to_csv(output_path, index=False)
    logger.info(f"Results saved successfully with {len(result_df.columns)} columns")

    # Save detailed analysis to task directory
    analysis_file = task_dir / "outlier_analysis_summary.txt"
    with open(analysis_file, 'w') as f:
        f.write("BATCH OUTLIER DETECTION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        if args.use_cohort_data:
            f.write("Data source: Cohort data (load_data from util.py)\n")
            f.write(f"  Config path: {args.config_path}\n")
            f.write(f"  Root path: {args.root_path}\n")
            f.write(f"  Is adult: {not args.children}\n")
            f.write(f"  Filter by size: {not args.no_filter_by_size}\n")
            f.write(f"  Filter missing values: {not args.no_filter_missing}\n")
        else:
            f.write(f"Input file: {args.excel_file}\n")
        f.write(f"Output file: {output_path}\n")
        f.write(f"Total samples: {total_count}\n")
        f.write(f"Combined outliers (prediction OR post-filter): {outlier_count} ({outlier_count/total_count*100:.1f}%)\n\n")

        f.write("Breakdown by predicted class:\n")
        for cls in result_df['predicted_class'].unique():
            cls_mask = result_df['predicted_class'] == cls
            cls_count = cls_mask.sum()
            cls_outliers = result_df.loc[cls_mask, 'final_outlier_decision'].sum()
            f.write(f"  {cls}: {cls_outliers}/{cls_count} ({cls_outliers/cls_count*100:.1f}%)\n")

        if 'outlier_ALL' in result_df.columns:
            naive_outliers = result_df['outlier_ALL'].sum()
            f.write("\nComparison with naive approach:\n")
            f.write(f"  Naive (ALL class): {naive_outliers}/{total_count} ({naive_outliers/total_count*100:.1f}%)\n")
            f.write(f"  Combined (prediction OR post-filter): {outlier_count}/{total_count} ({outlier_count/total_count*100:.1f}%)\n")

        # Add accuracy analysis summaries
        if category_accuracy:
            f.write("\nACCURACY ANALYSIS BY CATEGORY:\n")
            for category, metrics in category_accuracy.items():
                f.write(f"  {category}: {metrics['outliers']}/{metrics['total_samples']} outliers ({metrics['outlier_rate']:.1f}%)\n")
            f.write("  Note: Categories with n≤1 are excluded from analysis\n")

        if diagnosis_accuracy:
            f.write("\nACCURACY ANALYSIS BY DIAGNOSIS:\n")
            for diagnosis, metrics in diagnosis_accuracy.items():
                f.write(f"  {diagnosis}: {metrics['outliers']}/{metrics['total_samples']} outliers ({metrics['outlier_rate']:.1f}%)\n")
            f.write("  Note: Diagnoses with n≤1 are excluded from analysis\n")

    logger.info(f"Detailed analysis saved to: {analysis_file}")

    # Print final summary to console
    print("\n" + "="*80)
    print("COMBINED OUTLIER DETECTION SUMMARY (PREDICTION + POST-FILTER)")
    print("="*80)
    print(f"Total samples processed: {total_count}")
    print(f"Combined outliers (prediction OR post-filter): {outlier_count} ({outlier_count/total_count*100:.1f}%)")

    print("\nBreakdown by predicted class:")
    for cls in result_df['predicted_class'].unique():
        cls_mask = result_df['predicted_class'] == cls
        cls_count = cls_mask.sum()
        cls_outliers = result_df.loc[cls_mask, 'final_outlier_decision'].sum()
        print(f"  {cls}: {cls_outliers}/{cls_count} ({cls_outliers/cls_count*100:.1f}%)")

    if 'outlier_ALL' in result_df.columns:
        naive_outliers = result_df['outlier_ALL'].sum()
        print("\nComparison with naive approach (ALL class for all):")
        print(f"  Naive: {naive_outliers}/{total_count} ({naive_outliers/total_count*100:.1f}%)")
        print(f"  Combined (prediction OR post-filter): {outlier_count}/{total_count} ({outlier_count/total_count*100:.1f}%)")

    # Print accuracy analysis summaries
    if category_accuracy:
        print("\nACCURACY ANALYSIS BY CATEGORY:")
        for category, metrics in category_accuracy.items():
            print(f"  {category}: {metrics['outliers']}/{metrics['total_samples']} outliers ({metrics['outlier_rate']:.1f}%)")
        # Note: Categories with n≤1 are excluded from analysis

    if diagnosis_accuracy:
        print("\nACCURACY ANALYSIS BY DIAGNOSIS:")
        for diagnosis, metrics in diagnosis_accuracy.items():
            print(f"  {diagnosis}: {metrics['outliers']}/{metrics['total_samples']} outliers ({metrics['outlier_rate']:.1f}%)")
        # Note: Diagnoses with n≤1 are excluded from analysis

    print(f"\nResults saved to: {output_path}")
    print(f"Logs saved to: {task_dir}")
    print("="*80)

    logger.info("\nBATCH OUTLIER DETECTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
