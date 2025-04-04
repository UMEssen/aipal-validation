import argparse
import json
import logging
import pickle
from pathlib import Path

import yaml

from aipal_validation.data_preprocessing import (
    generate_custom_samples,
    generate_samples,
)
from aipal_validation.fhir import FHIRExtractor, FHIRFilter, FHIRValidator
from aipal_validation.helper.util import is_main_process, run_r_script, timed
from aipal_validation.ml import test
from aipal_validation.outlier import OutlierChecker
from aipal_validation.outlier.train_outlier import MulticentricOutlierDetector

pipelines = {
    "aipal": {
        "generate": generate_samples.main,
        "generate_custom": generate_custom_samples.main,
        "test": test.main,
    },
}

LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(levelname)s %(asctime)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
    level=LOG_LEVEL,
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger().setLevel(LOG_LEVEL)
logger = logging.getLogger(__name__)


def load_config(file_path: str = "aipal_validation/config/config_training.yaml"):
    path = Path.cwd() / file_path
    if path.suffix == ".yaml":
        return yaml.safe_load(path.open())
    elif path.suffix == ".json":
        return json.load(path.open())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


@timed
def build_cache(config: dict):
    extract = FHIRExtractor(config)
    filt = FHIRFilter(config)
    validator = FHIRValidator(config)

    config["task_dir"].mkdir(parents=True, exist_ok=True)

    resources = ["patient_condition", "observation"]
    for resource in resources:
        logger.info(f"Extracting {resource}...")
        extract.build(resource)
        logger.info(f"Filtering {resource}...")
        filt.filter(resource)
        logger.info(f"Validating {resource}...")
        validator.validate(resource)


def parse_args_local(config) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, default=config["root_dir"])
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument(
        "--task",
        type=str,
        default=config["task"],
        required=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=config["debug"],
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        help="Step to run, all_cohorts is for all_cohorts in one go",
    )
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=False,
        help="Evaluate all cohorts in on go plus try to filter outliers",
    )
    parser.add_argument(
        "--sample",
        type=str,
        help="Path to sample JSON file for outlier detection",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing trained outlier detection models",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="aipal_validation/config/config_training.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--r_script",
        type=str,
        default="r/predict.R",
        help="Path to R script for prediction",
    )
    parser.add_argument(
        "--train_outlier",
        action="store_true",
        default=False,
        help="Train outlier detection models",
    )
    parser.add_argument(
        "--outlier_config",
        type=str,
        default="aipal_validation/config/config_outlier.yaml",
        help="Path to outlier configuration file",
    )
    parser.add_argument(
        "--outlier_output_dir",
        type=str,
        help="Directory to save trained outlier models",
    )

    return parser.parse_args()


def run():
    config = load_config()

    args = parse_args_local(config)
    config.update(vars(args))
    if config["debug"]:
        logger.warning(
            "WARNING!!! Running in debug mode. Switch off debug for production."
        )

    config["root_dir"] = config["root_dir"] / config["run_id"]

    if config["step"] == "all_cohorts":
        config["task_dir"] = config["root_dir"] / "all"
    else:
        config["task_dir"] = config["root_dir"] / config["task"]
    config["task_dir"].mkdir(parents=True, exist_ok=True)
    logger.info(f"The outputs will be stored in {config['task_dir']}.")

    fh = logging.FileHandler(config["task_dir"] / "log.txt")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(levelname)s %(asctime)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s"
    )
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

    # Handle outlier model training
    if config.get("train_outlier"):
        logger.info("Training outlier detection models...")
        if not config.get("outlier_output_dir"):
            raise ValueError("--outlier_output_dir must be specified when training outlier models")

        detector = MulticentricOutlierDetector(config["outlier_config"])
        detector.train_outlier_models()
        detector.save_models(config["outlier_output_dir"])
        detector.detect_and_evaluate()
        return

    assert config["task"] in pipelines, f"Task {config['task']} not found."

    # Handle outlier detection on a sample
    if config.get("sample") and config.get("model_dir"):
        logger.info("Running outlier detection on sample...")
        checker = OutlierChecker()
        checker.load_models(config["model_dir"], config["config"])

        with open(config["sample"], "r") as f:
            sample_data = json.load(f)

        results = checker.check_sample(sample_data)
        logger.info("\nOutlier Detection Results:")
        for cls, result in results.items():
            logger.info(f"\n{cls}:")
            logger.info(f"  Is Outlier: {result['is_outlier']}")
            logger.info(f"  Isolation Forest Score: {result['iso_forest_score']:.4f}")
            logger.info(f"  LOF Score: {result['lof_score']:.4f}")

        # Only skip if ALL classes are outliers
        is_outlier_all = all(result["is_outlier"] for result in results.values())

        if not is_outlier_all:
            logger.info("Sample is not an outlier for all classes, running prediction...")
            config["r_script"] = "r/predict_with_outlier.R"  # Use outlier-specific R script
            config["r_script_args"] = [config["sample"]]  # Pass sample file path as argument
            run_r_script(config)
        else:
            logger.warning("Sample is an outlier for all classes, skipping prediction.")
        return

    # Generic FHIR-pipeline
    if config["run_id"].startswith("V"):
        if config["step"] == "all":
            config["data_dir"] = config["root_dir"] / "data_raw"
            config["data_dir"].mkdir(parents=True, exist_ok=True)
            config["step"] = "data+sampling+test"
            config["step"] = config["step"].split("+")

        if "data" in config["step"] and is_main_process():
            build_cache(config)
            with (config["task_dir"] / "config_data.pkl").open("wb") as of:
                pickle.dump(config, of)

        if "sampling" in config["step"] and is_main_process():
            if isinstance(pipelines[config["task"]]["generate"], list):
                for pipeline in pipelines[config["task"]]["generate"]:
                    pipeline(config)
            else:
                pipelines[config["task"]]["generate"](config)
            with (config["task_dir"] / "config_sampling.pkl").open("wb") as of:
                pickle.dump(config, of)
    else:
        # Custom pipeline in case data already exists
        if config["step"] == "all":
            config["step"] = "sampling+test"
            config["step"] = config["step"].split("+")

        if "sampling" in config["step"]:
            pipelines[config["task"]]["generate_custom"](config)

    if "test" in config["step"]:
        run_r_script(config)
        test.main(config)

    if "all_cohorts" in config["step"]:
        test.main(config)

    if config.get("eval_all"):
        test.main(config)
