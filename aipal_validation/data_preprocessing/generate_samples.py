import pandas as pd

from aipal_validation.data_preprocessing.util import skip_build
from aipal_validation.fhir.util import OUTPUT_FORMAT, check_and_read, store_df


def pivot_obs(config):
    obs = check_and_read(config["data_dir"] / f"observation{OUTPUT_FORMAT}")
    obs_pivoted = obs.pivot_table(
        index="encounter_id", columns="code", values="value", aggfunc="first"
    )

    for code, (_old_unit, factor, new_unit) in config["obs_codes_si"].items():
        if code in obs_pivoted.columns:
            obs_pivoted[code] *= factor
            obs_pivoted.rename(columns={code: f"{code}_{new_unit}"}, inplace=True)

    return obs_pivoted.reset_index()


def merge_pats_conds_obs(config, df_obs):
    pats_cond = check_and_read(config["task_dir"] / f"patient_condition{OUTPUT_FORMAT}")
    pats_cond_obs = pd.merge(pats_cond, df_obs, on="encounter_id")
    return pats_cond_obs


def main(config):
    if skip_build(config):
        return
    df_obs = pivot_obs(config)
    store_df(
        merge_pats_conds_obs(config, df_obs),
        config["task_dir"] / f"samples{OUTPUT_FORMAT}",
        "samples",
    )
