import logging
from pathlib import Path

import pandas as pd
import yaml

from aipal_validation.helper.util import run_r_script
from aipal_validation.ml import test


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


