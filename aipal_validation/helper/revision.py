import logging
from pathlib import Path

import pandas as pd
import yaml
import numpy as np

from aipal_validation.helper.util import run_r_script
from aipal_validation.ml import test
from aipal_validation.outlier.check_outlier import OutlierChecker
from typing import Optional


def prepare_and_run_no_monocytes(config: dict) -> None:
    """Prepare merged samples without Monocytes_G_L and run predict + eval.

    This replicates the previous inline flow from cli.py but is extracted to
    keep the CLI lean. It writes a merged samples.csv under task_dir, crafts a
    temporary YAML that points R to the same layout, runs the R prediction,
    and evaluates using test.py on the produced predict.csv.
    """
    logger = logging.getLogger(__name__)

    logger.info("Preparing merged samples without Monocytes_G_L for prediction and evaluation")

    # Merge samples.csv from all cohorts
    # Load list of cities/countries
    analysis_cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'config_analysis.yaml'
    config_analysis = yaml.safe_load(analysis_cfg_path.open())
    cities_countries = config_analysis['cities_countries']

    # Root one level above current run_id (R will use root_dir/run_id/task)
    root_parent = config["root_dir"].parent

    merged_df = pd.DataFrame()
    paths = [root_parent / city_country / 'aipal' / 'samples.csv' for city_country in cities_countries]
    logger.info(f"Merging samples from: {[str(p) for p in paths]}")
    for p in paths:
        if not p.exists():
            logger.warning(f"Missing samples.csv: {p}")
            continue
        df_small = pd.read_csv(p)
        df_small['city'] = p.parent.parent.name
        if df_small['city'].str.contains('Vessen').any():
            df_small = df_small[df_small['age'] < 18]
        merged_df = pd.concat([merged_df, df_small], ignore_index=True)

    if merged_df.empty:
        raise FileNotFoundError("No samples.csv files found to merge.")

    # Drop Monocytes_G_L feature if present
    if 'Monocytes_G_L' in merged_df.columns:
        merged_df.drop(columns=['Monocytes_G_L'], inplace=True)

    # Persist merged samples for R script
    merged_path = config["task_dir"] / 'samples.csv'
    merged_df.to_csv(merged_path, index=False)
    logger.info(f"Wrote merged samples to {merged_path}")

    # Prepare a temporary YAML for R with correct root/task/run layout
    base_yaml_path = Path(config.get("config", "aipal_validation/config/config_training.yaml"))
    base_yaml = yaml.safe_load(base_yaml_path.open())

    # Ensure R sees root_dir as parent of current run_id so that root_dir/run_id/task == task_dir
    base_yaml['root_dir'] = str(root_parent)
    base_yaml['run_id'] = config.get('run_id', base_yaml.get('run_id'))
    base_yaml['task'] = 'no_monocytes'

    temp_yaml_path = config["task_dir"] / 'config_no_monocytes.yaml'
    with temp_yaml_path.open('w') as f:
        yaml.safe_dump(base_yaml, f, sort_keys=False)

    # Run predictions with R using the temporary YAML
    config["r_script"] = config.get("r_script", "aipal_validation/r/predict.R")
    config["r_script_args"] = [str(temp_yaml_path)]
    run_r_script(config)

    # Evaluate using test.py on the freshly created predict.csv
    # Force a simple test step so test.py reads task_dir/predict.csv directly
    prev_step = config.get('step')
    config['step'] = 'test'
    test.main(config)
    # Restore step in case callers rely on it later
    config['step'] = prev_step



def merge_all_cohorts_predicts_and_eval(config: dict) -> None:
    """Merge existing predict.csv from all cohorts, write to task_dir, and evaluate.

    - Reads cohort list from config_analysis.yaml
    - Loads each root_parent/<cohort>/aipal/predict.csv
    - Concats into a single DataFrame, prunes extra columns to match test.py expectations
    - Writes task_dir/predict.csv
    - Calls test.main with current config (honors age_binning, e.g., decade)
    """
    logger = logging.getLogger(__name__)

    analysis_cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'config_analysis.yaml'
    config_analysis = yaml.safe_load(analysis_cfg_path.open())
    cities_countries = config_analysis['cities_countries']

    root_parent = config["root_dir"].parent
    paths = [root_parent / city_country / 'aipal' / 'predict.csv' for city_country in cities_countries]
    logger.info(f"Merging predictions from: {[str(p) for p in paths]}")

    merged_df = pd.DataFrame()
    for p in paths:
        if not p.exists():
            logger.warning(f"Missing predict.csv: {p}")
            continue
        df_small = pd.read_csv(p)
        df_small['city'] = p.parent.parent.name
        if df_small['city'].str.contains('Vessen').any():
            df_small = df_small[df_small['age'] < 18]
        merged_df = pd.concat([merged_df, df_small], ignore_index=True)

    if merged_df.empty:
        raise FileNotFoundError("No predict.csv files found to merge.")

    # Drop columns not expected by evaluation, if present
    drop_cols = ['recorded_date','encounter_start','encounter_end','patient_id','birth_date','condition_codes','condition_id','encounter_id']
    existing_drop = [c for c in drop_cols if c in merged_df.columns]
    if existing_drop:
        merged_df.drop(columns=existing_drop, inplace=True)

    merged_path = config["task_dir"] / 'predict.csv'
    merged_df.to_csv(merged_path, index=False)
    logger.info(f"Wrote merged predictions to {merged_path}")

    # Ensure evaluation reads from task_dir/predict.csv
    prev_step = config.get('step')
    config['step'] = 'test'
    test.main(config)
    config['step'] = prev_step



def merge_adult_samples_across_cohorts(config: dict) -> pd.DataFrame:
    """Merge `samples.csv` across all cohorts and return only adults (age >= 18).

    - Reads cohort list from `config_analysis.yaml`
    - Loads each root_parent/<cohort>/aipal/samples.csv
    - Adds `city` column derived from folder name
    - Filters to adult cohort
    - Writes merged adults to task_dir/adult_samples.csv
    - Returns the merged adult DataFrame
    """
    logger = logging.getLogger(__name__)

    analysis_cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'config_analysis.yaml'
    config_analysis = yaml.safe_load(analysis_cfg_path.open())
    cities_countries = config_analysis['cities_countries']

    root_parent = config["root_dir"].parent
    paths = [root_parent / city_country / 'aipal' / 'samples.csv' for city_country in cities_countries]
    logger.info(f"Merging adult samples from: {[str(p) for p in paths]}")

    merged_df = pd.DataFrame()
    for p in paths:
        if not p.exists():
            logger.warning(f"Missing samples.csv: {p}")
            continue
        df_small = pd.read_csv(p)
        df_small['city'] = p.parent.parent.name
        # Adults only
        if 'age' not in df_small.columns:
            raise ValueError(f"Column 'age' not found in {p}")
        df_small = df_small[df_small['age'] >= 18]
        merged_df = pd.concat([merged_df, df_small], ignore_index=True)

    if merged_df.empty:
        raise FileNotFoundError("No adult samples found to merge.")

    # Persist merged adult samples for reproducibility
    adult_samples_path = config["task_dir"] / 'adult_samples.csv'
    merged_df.to_csv(adult_samples_path, index=False)
    logger.info(f"Wrote merged adult samples to {adult_samples_path}")

    return merged_df


def run_outlier_detector_on_adults(config: dict, adults_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run pretrained outlier detector on adult samples and persist outputs.

    - Requires config['outlier_model_dir'] and config['outlier_config_path'] to load models
    - Uses OutlierChecker.check_dataframe for efficient batch scoring
    - Writes the following under task_dir:
        * adults_with_outlier.csv  (full adults with 'outlier' column)
        * adults_excluded.csv      (outlier == 1)
        * adults_included.csv      (outlier == 0)
    - Returns the full DataFrame with the 'outlier' column
    """
    logger = logging.getLogger(__name__)

    if adults_df is None:
        adults_df = merge_adult_samples_across_cohorts(config)

    # Hardcode paths as they're part of the package structure
    model_dir = Path(__file__).resolve().parent.parent / 'outlier'
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config_outlier.yaml'

    checker = OutlierChecker()
    checker.load_models(model_dir, config_path)
    logger.info(f"Loaded pretrained outlier models from {model_dir}")

    # Clean data before outlier detection - robust coercion for model feature columns
    adults_df_clean = adults_df.copy()

    feature_cols = list(checker.features)

    def _coerce_numeric(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        # If value uses decimal comma and no dot, convert comma to dot; otherwise remove thousands commas
        mask_dec_comma = s.str.contains(',', regex=False) & ~s.str.contains('.', regex=False)
        s = s.where(~mask_dec_comma, s.str.replace(',', '.', regex=False))
        s = s.where(mask_dec_comma, s.str.replace(',', '', regex=False))
        s = s.str.replace(' ', '', regex=False)
        s = pd.to_numeric(s, errors='coerce')
        s = s.where(np.isfinite(s), np.nan)
        return s

    for col in feature_cols:
        if col in adults_df_clean.columns:
            adults_df_clean[col] = _coerce_numeric(adults_df_clean[col])

    # Calculate Monocytes_percent if part of model features
    if 'Monocytes_percent' in feature_cols:
        if 'Monocytes_G_L' in adults_df_clean.columns and 'WBC_G_L' in adults_df_clean.columns:
            denom = _coerce_numeric(adults_df_clean['WBC_G_L']).replace(0, np.nan)
            numer = _coerce_numeric(adults_df_clean['Monocytes_G_L'])
            adults_df_clean['Monocytes_percent'] = (numer * 100) / denom
        else:
            adults_df_clean['Monocytes_percent'] = pd.NA

    # Ensure finiteness and clip extreme values per feature
    for col in feature_cols:
        if col in adults_df_clean.columns:
            col_series = adults_df_clean[col]
            if not pd.api.types.is_numeric_dtype(col_series):
                col_series = _coerce_numeric(col_series)
            col_series = col_series.where(np.isfinite(col_series), np.nan)
            if col_series.notna().any():
                q_low = col_series.quantile(0.0001)
                q_high = col_series.quantile(0.9999)
                col_series = col_series.clip(lower=q_low, upper=q_high)
            adults_df_clean[col] = col_series



    # Batch detection; check_outlier handles Monocytes_percent if source columns exist
    if 'class' in adults_df_clean.columns:
        df_with_outliers = checker.check_dataframe(adults_df_clean)
    else:
        # Fallback: if no class labels, evaluate outlier status under each class model and OR-combine
        all_flags = []
        for cls in ["ALL", "AML", "APL"]:
            df_tmp = adults_df_clean.copy()
            df_tmp['class'] = cls
            flagged = checker.check_dataframe(df_tmp)
            all_flags.append(flagged['outlier'].rename(f'outlier_{cls}'))
        df_with_outliers = adults_df_clean.copy()
        for s in all_flags:
            df_with_outliers = df_with_outliers.join(s)
        # Combine any-class outlier
        outlier_cols = [c for c in df_with_outliers.columns if c.startswith('outlier_')]
        df_with_outliers['outlier'] = df_with_outliers[outlier_cols].max(axis=1)

    # Persist outputs
    out_path_all = config["task_dir"] / 'adults_with_outlier.csv'
    out_path_excluded = config["task_dir"] / 'adults_excluded.csv'
    out_path_included = config["task_dir"] / 'adults_included.csv'

    df_with_outliers.to_csv(out_path_all, index=False)
    df_with_outliers[df_with_outliers['outlier'] == 1].to_csv(out_path_excluded, index=False)
    df_with_outliers[df_with_outliers['outlier'] == 0].to_csv(out_path_included, index=False)

    logger.info(f"Saved outlier annotations: {out_path_all}")
    logger.info(f"Saved excluded adults (outliers): {out_path_excluded}")
    logger.info(f"Saved included adults: {out_path_included}")

    return df_with_outliers


def analyze_outlier_patterns_for_adults(config: dict, df_with_outliers: Optional[pd.DataFrame] = None) -> None:
    """Analyze patterns among excluded vs included adult samples and persist summaries.

    Produces CSVs under task_dir:
    - adult_pattern_missingness.csv: per-column missing rates (included vs excluded, and delta)
    - adult_pattern_numeric_extremes.csv: per-numeric-column rate of |z|>3, negatives, and summary stats
    - adult_pattern_categorical_enrichment.csv: categorical value frequencies by group with delta
    - adult_pattern_overview.csv: high-level counts by city and class if available
    """
    logger = logging.getLogger(__name__)

    if df_with_outliers is None:
        adults_with_outlier_path = config["task_dir"] / 'adults_with_outlier.csv'
        if not adults_with_outlier_path.exists():
            df_with_outliers = run_outlier_detector_on_adults(config)
        else:
            df_with_outliers = pd.read_csv(adults_with_outlier_path)

    if 'outlier' not in df_with_outliers.columns:
        raise ValueError("Expected column 'outlier' not found in adults_with_outlier DataFrame")

    included = df_with_outliers[df_with_outliers['outlier'] == 0].copy()
    excluded = df_with_outliers[df_with_outliers['outlier'] == 1].copy()

    # 1) Missingness patterns
    def _missing_rates(df: pd.DataFrame) -> pd.Series:
        return df.isna().mean()

    miss_inc = _missing_rates(included)
    miss_exc = _missing_rates(excluded)
    missing_summary = pd.concat([miss_inc.rename('included_missing_rate'),
                                 miss_exc.rename('excluded_missing_rate')], axis=1)
    missing_summary['delta_excluded_minus_included'] = (
        missing_summary['excluded_missing_rate'] - missing_summary['included_missing_rate']
    )
    missing_path = config["task_dir"] / 'adult_pattern_missingness.csv'
    missing_summary.sort_values('delta_excluded_minus_included', ascending=False).to_csv(missing_path)

    # 2) Numeric extremes and negatives
    numeric_cols = df_with_outliers.select_dtypes(include=['number']).columns.tolist()
    # Exclude the outlier marker itself from numeric analysis
    numeric_cols = [c for c in numeric_cols if c != 'outlier']

    if numeric_cols:
        # Compute global mean/std based on included cohort to avoid leakage from outliers
        means = included[numeric_cols].mean()
        stds = included[numeric_cols].std(ddof=0).replace(0, float('nan'))

        def _rate_abs_z_gt3(df: pd.DataFrame) -> pd.Series:
            z = (df[numeric_cols] - means) / stds
            return (z.abs() > 3).mean()

        def _rate_negative(df: pd.DataFrame) -> pd.Series:
            return (df[numeric_cols] < 0).mean()

        rate_z_inc = _rate_abs_z_gt3(included).rename('included_rate_|z|>3')
        rate_z_exc = _rate_abs_z_gt3(excluded).rename('excluded_rate_|z|>3')
        rate_neg_inc = _rate_negative(included).rename('included_rate_negative')
        rate_neg_exc = _rate_negative(excluded).rename('excluded_rate_negative')

        # Basic stats for reference
        mean_inc = means.rename('included_mean')
        mean_exc = excluded[numeric_cols].mean().rename('excluded_mean')
        std_inc = stds.rename('included_std')
        std_exc = excluded[numeric_cols].std(ddof=0).rename('excluded_std')

        numeric_summary = pd.concat([
            mean_inc, mean_exc, std_inc, std_exc,
            rate_z_inc, rate_z_exc, rate_neg_inc, rate_neg_exc
        ], axis=1)
        numeric_summary['delta_rate_|z|>3'] = numeric_summary['excluded_rate_|z|>3'] - numeric_summary['included_rate_|z|>3']
        numeric_summary['delta_rate_negative'] = numeric_summary['excluded_rate_negative'] - numeric_summary['included_rate_negative']
    else:
        numeric_summary = pd.DataFrame()

    numeric_path = config["task_dir"] / 'adult_pattern_numeric_extremes.csv'
    numeric_summary.sort_values('delta_rate_|z|>3', ascending=False, na_position='last').to_csv(numeric_path)

    # 3) Categorical enrichment (including 'city' if present)
    categorical_cols = df_with_outliers.select_dtypes(include=['object', 'category']).columns.tolist()
    # Ensure 'city' is included if present
    if 'city' in df_with_outliers.columns and 'city' not in categorical_cols:
        categorical_cols.append('city')

    cat_frames = []
    for col in categorical_cols:
        inc_freq = included[col].value_counts(normalize=True, dropna=False).rename('included_freq')
        exc_freq = excluded[col].value_counts(normalize=True, dropna=False).rename('excluded_freq')
        cat_summary = pd.concat([inc_freq, exc_freq], axis=1).fillna(0)
        cat_summary['delta_excluded_minus_included'] = cat_summary['excluded_freq'] - cat_summary['included_freq']
        cat_summary['feature'] = col
        cat_summary['value'] = cat_summary.index
        cat_frames.append(cat_summary.reset_index(drop=True))
    categorical_summary = pd.concat(cat_frames, ignore_index=True) if cat_frames else pd.DataFrame()
    categorical_path = config["task_dir"] / 'adult_pattern_categorical_enrichment.csv'
    categorical_summary.sort_values(['delta_excluded_minus_included'], ascending=False).to_csv(categorical_path, index=False)

    # 4) Overview by city and class if available
    overview_groups = []
    for group_col in ['city', 'class']:
        if group_col in df_with_outliers.columns:
            grp = df_with_outliers.groupby(group_col).agg(
                total=('outlier', 'count'),
                excluded=('outlier', lambda s: int((s == 1).sum())),
            ).reset_index()
            grp['excluded_rate'] = grp['excluded'] / grp['total']
            grp['group'] = group_col
            overview_groups.append(grp)
    overview_df = pd.concat(overview_groups, ignore_index=True) if overview_groups else pd.DataFrame()
    overview_path = config["task_dir"] / 'adult_pattern_overview.csv'
    overview_df.to_csv(overview_path, index=False)

    logger.info(f"Wrote pattern analyses to: {missing_path}, {numeric_path}, {categorical_path}, {overview_path}")


def adults_outlier_investigation(config: dict) -> None:
    """End-to-end helper to address the revision request for adult outliers.

    Steps:
    1. Merge adult samples across cohorts
    2. Run pretrained outlier detector and save excluded samples to CSV
    3. Analyze and save pattern summaries
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting adults outlier investigation")

    adults_df = merge_adult_samples_across_cohorts(config)
    df_with_outliers = run_outlier_detector_on_adults(config, adults_df)
    analyze_outlier_patterns_for_adults(config, df_with_outliers)
    logger.info("Completed adults outlier investigation")
