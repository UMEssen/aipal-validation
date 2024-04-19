from datetime import datetime

import numpy as np
import pandas as pd

from aipal_validation.data_preprocessing.util import skip_build
from aipal_validation.fhir.util import OUTPUT_FORMAT, check_and_read, store_df


def pivot_obs(config):
    obs = check_and_read(config["data_dir"] / f"observation{OUTPUT_FORMAT}")
    obs_pivoted = obs.pivot_table(
        index="encounter_id", columns="code", values="value", aggfunc="first"
    )

    for code, (_old_unit, factor, _new_unit, aipal_label) in config[
        "obs_codes_si"
    ].items():
        if code in obs_pivoted.columns:
            obs_pivoted[code] *= factor
            # obs_pivoted.rename(columns={code: f"{code}_{_new_unit}"}, inplace=True)
            obs_pivoted.rename(columns={code: aipal_label}, inplace=True)
        else:
            obs_pivoted[aipal_label] = np.nan

    return obs_pivoted.reset_index()


def condition_to_codes(config, df: pd.DataFrame) -> pd.DataFrame:
    code_to_class = {
        code: key for key, codes in config["LK_codes"].items() for code in codes
    }
    df["class"] = df["condition_codes"].map(code_to_class)
    return df


def calculate_age(df, birth_date_column):
    today = datetime.today()

    def get_age(birth_date):
        return (
            today.year
            - birth_date.year
            - ((today.month, today.day) < (birth_date.month, birth_date.day))
        )

    # Ensure the birth date column is in datetime format
    df[birth_date_column] = pd.to_datetime(df[birth_date_column])
    df["age"] = df[birth_date_column].apply(get_age)

    return df


def add_missing_obs_columns(df, config):
    _, _, _, mandatory_columns = config["obs_codes_si"].values()
    for code in mandatory_columns:
        if code not in df.columns:
            df[code] = np.nan
    return df


def merge_pats_conds_obs(config, df_obs):
    pats_cond = check_and_read(config["task_dir"] / f"patient_condition{OUTPUT_FORMAT}")
    return pd.merge(pats_cond, df_obs, on="encounter_id")


def main(config):
    if skip_build(config):
        return
    df_obs = pivot_obs(config)
    df_merged = merge_pats_conds_obs(config, df_obs)
    df = condition_to_codes(config, df_merged)
    df = calculate_age(df, "birth_date")
    df = df.round(2)
    store_df(df, config["task_dir"] / "samples.csv", "samples")
