import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def is_main_process():
    return "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0


# Time decorator for function execution time measurement
def timed(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = (time.perf_counter() - start) / 60
        logger.info(
            f"Time taken for {func.__name__}: {np.round(elapsed_time,2)} minutes"
        )
        return result

    return wrapper


def name_from_model(
    model_name: Union[str, Path], roformer: bool = False
) -> Tuple[str, str, bool]:
    if isinstance(model_name, Path):
        name = model_name.parent.parent.name
        plain_name = name.replace("_", "/")
        if not model_name.exists():
            raise ValueError(f"Model {model_name} does not exist.")
        load = True
    elif Path(model_name).exists():
        name = Path(model_name).name
        if name == "best" or name.startswith("checkpoint"):
            name = Path(model_name).parent.parent.name
        plain_name = name.replace("_", "/")
        load = True
    else:
        plain_name = model_name
        name = model_name.replace("/", "_")
        if roformer:
            name = "roformer_" + name
        load = False
    return plain_name, name, load


def get_labels_info(
    labels: list,
    additional_string: str = "",
    stop_after: int = 20,
):
    if isinstance(labels[0], list):
        logger.info(
            f"Average number of labels per sample: {np.mean([len(x) for x in labels]):.2f}"
        )
        labels = [item for sublist in labels for item in sublist]

    counts = list(pd.Series(labels).value_counts().to_dict().items())
    logger.info(
        f"Label counts {'(' + additional_string + ')' if additional_string else ''}"
    )
    if stop_after:
        counts = counts[:stop_after]
    for label, count in counts:
        logger.info(f"{label}: {count}")


def clear_process_data(config):
    if not config["is_live_prediction"]:
        if input("Do you really want to delete the cache (y:yes)?:") != "y":
            exit()
    folders = config["folders_to_clear"]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                logger.info(f"deleting: {file_path}")
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.info("Failed to delete %s. Reason: %s" % (file_path, e))


def run_r_script(config):
    """Run R script for prediction."""
    if config.get("debug"):
        logger.warning("WARNING!!! Running in debug mode. Switch off debug for production.")

    # Get R script path from config, default to predict.R
    r_script_path = config.get("r_script", "aipal_validation/r/predict.R")

    # Get additional arguments for R script
    r_script_args = config.get("r_script_args", [])
    
    # For synthetic data, automatically pass the config file path
    if config.get("run_id") == "synthetic_test_data" and r_script_path == "aipal_validation/r/predict.R":
        r_script_args = ["aipal_validation/config/config_synthetic.yaml"] + r_script_args

    # Command to run the R script
    command = ["Rscript", r_script_path] + r_script_args

    try:
        # Skip predict.csv check for outlier prediction script
        if r_script_path == "aipal_validation/r/predict_with_outlier.R":
            logging.info("Running outlier prediction script...")
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logging.info("R script output: %s", result.stdout)
            logging.info("R script errors: %s", result.stderr)
            return

        # check for predict.csv existance for regular prediction
        if (
            os.path.exists(config["task_dir"] / Path("predict.csv"))
            and not config["rerun_cache"]
        ):
            logging.info(
                "Predictions already exist in predict.csv. Skipping R script execution."
            )
            return
        else:
            logging.info("Running R script to generate predictions.")
            # Running the command
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            # Correctly formatted logging output
            logging.info("R script output: %s", result.stdout)
            logging.info("R script errors: %s", result.stderr)

    except subprocess.CalledProcessError as e:
        # Correctly formatted logging error
        logging.error("Error running R script: %s", e)
        logging.error("R script output: %s", e.stdout)
        logging.error("R script errors: %s", e.stderr)
