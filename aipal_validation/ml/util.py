import os

import numpy as np
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score


def get_param_for_task_model(config, param: str, task: str, model: str):
    if task in config[param]:
        if isinstance(config[param][task], dict) and model in config[param][task]:
            return config[param][task][model]
        else:
            return config[param][task]["default"]
    return config[param]["default"]


def init_wandb(config):
    wandb.init(
        tags=["Essen"],
        project="aipal_validation",
        mode="disabled" if config["debug"] else "online",
        entity="ship-ai-autopilot",
        config=config,
    )
    wandb.run.log_code(".")


def get_evaluation_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    single_label=False,
    zero_division=0,
):
    metrics = {
        "accuracy": (predictions == labels).mean(),
    }
    if single_label:
        metrics["precision"] = precision_score(
            labels,
            predictions,
            zero_division=zero_division,
        )
        metrics["recall"] = recall_score(
            labels,
            predictions,
            zero_division=zero_division,
        )
        metrics["f1"] = f1_score(
            labels,
            predictions,
            zero_division=zero_division,
        )
    else:
        for average in ["macro", "micro", "weighted"]:
            metrics[f"{average}_precision"] = precision_score(
                labels,
                predictions,
                average=average,
                zero_division=zero_division,
            )
            metrics[f"{average}_recall"] = recall_score(
                labels,
                predictions,
                average=average,
                zero_division=zero_division,
            )
            metrics[f"{average}_f1"] = f1_score(
                labels,
                predictions,
                average=average,
                zero_division=zero_division,
            )
    return metrics


def resolve_paths(input_dict):
    def resolve_path(path):
        return os.path.abspath(path)

    def resolve_dict_paths(d):
        for key, value in d.items():
            if isinstance(value, str) and os.path.exists(value):
                d[key] = resolve_path(value)
            elif isinstance(value, dict):
                d[key] = resolve_dict_paths(value)
        return d

    return resolve_dict_paths(input_dict)
