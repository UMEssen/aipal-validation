import argparse
import logging
import pickle
from pathlib import Path

import yaml

from aipal_validation.data_preprocessing import generate_samples
from aipal_validation.fhir import FHIRExtractor, FHIRFilter, FHIRValidator
from aipal_validation.helper.util import is_main_process, run_r_script, timed
from aipal_validation.ml import test

pipelines = {
    "aipal": {"generate": generate_samples.main, "test": test.main},
}


# Set up logging
LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(levelname)s %(asctime)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
    level=LOG_LEVEL,
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger().setLevel(LOG_LEVEL)

logger = logging.getLogger(__name__)


# Load config file
def load_config(file_path: str = "aipal_validation/config/config_training.yaml"):
    return yaml.safe_load((Path.cwd() / file_path).open())


@timed
def build_cache(config):
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
        required=True,
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
    )
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def run():
    config = load_config()

    args = parse_args_local(config)
    config.update(vars(args))
    if config["debug"]:
        logger.warning(
            "WARNING!!! You are running aipal_validation in debug mode, "
            "please change this when you are done testing."
        )

    config["root_dir"] = config["root_dir"] / config["run_id"]
    config["data_dir"] = config["root_dir"] / "data_raw"
    config["data_dir"].mkdir(parents=True, exist_ok=True)

    config["task_dir"] = config["root_dir"] / config["task"]

    config["sample_dir"] = (
        config["task_dir"] / f"sampled_{config['data_id'][config['task']]}"
    )

    if config["step"] == "all":
        config["step"] = "data+sampling+test"

    config["step"] = config["step"].split("+")

    if "data" in config["step"] and is_main_process():
        build_cache(config)
        with (config["task_dir"] / "config_data.pkl").open("wb") as of:
            pickle.dump(config, of)

    assert config["task"] in pipelines, f"Task {config['task']} not found."

    config["task_dir"].mkdir(parents=True, exist_ok=True)
    logger.info(f"The outputs will be stored in {config['task_dir']}.")

    if "sampling" in config["step"] and is_main_process():
        if isinstance(pipelines[config["task"]]["generate"], list):
            for pipeline in pipelines[config["task"]]["generate"]:
                pipeline(config)
        else:
            pipelines[config["task"]]["generate"](config)
        with (config["task_dir"] / "config_sampling.pkl").open("wb") as of:
            pickle.dump(config, of)

    if "test" in config["step"]:
        run_r_script()
        test.main(config)
