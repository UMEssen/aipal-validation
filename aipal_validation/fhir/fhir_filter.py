import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from aipal_validation.fhir.util import (
    OUTPUT_FORMAT,
    check_and_read,
    reduce_cardinality,
    store_df,
)

logger = logging.getLogger(__name__)


# Class FHIRExtractor
class FHIRFilter:
    def __init__(self, config):
        self.config = config

    def filter(self, resource: str):
        resource = resource.lower()
        if resource == "patient_condition":
            self.filter_conditions()
        elif resource == "observation":
            self.filter_observation()
        else:
            raise NotImplementedError(f"Resource {resource} not supported")

    # TODO: Datetime filtering for each metrics resource
    @staticmethod
    def filter_date(
        start: datetime, end: datetime, resource: pd.DataFrame, date_col: str
    ) -> pd.DataFrame:
        df = resource[
            ((start <= resource[date_col]) & (resource[date_col] <= end))
        ].sort_values([date_col])

        return df

    def skip_filter(self, path: Path):
        if self.config["rerun_cache"] or not path.exists():
            return False
        else:
            return path.exists()

    def observations_to_si(self, df_obs: pd.DataFrame) -> pd.DataFrame:
        for unit, factor in self.config["obs_codes_si"].items():
            df_obs.loc[df_obs.code == unit, "value"] = (
                df_obs.loc[df_obs.code == unit, "value"].astype(float) * factor[1]
            )
            df_obs.loc[df_obs.code == unit, "unit"] = factor[2]

        return df_obs

    @staticmethod
    def to_datetime(df, col_format):
        for k, v in tqdm(col_format.items(), desc="Converting DateTime"):
            df[k] = pd.to_datetime(df[k], format=v, utc=True, errors="coerce")
        return df

    def basic_filtering(
        self, name: str, output_name: str = None, save: bool = True, is_patient_df=False
    ) -> Optional[pd.DataFrame]:
        output_name = name if output_name is None else output_name
        output_path = self.config["task_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        if self.skip_filter(output_path):
            return None
        df = check_and_read(self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}")
        if save:
            store_df(df, output_path)
        else:
            return df

    def filter_patient_info(self):
        output_path = self.config["task_dir"] / f"patient{OUTPUT_FORMAT}"
        if (
            df := self.basic_filtering("patient", save=False, is_patient_df=True)
        ) is None:
            return

        df.drop_duplicates(subset=["patient_id"], inplace=True)

        df["insurance_type"] = (
            df["insurance_type"]
            .map({"PKV": "privat", "GKV": "gesetzlich"})
            .fillna("unbekannt")
        )
        df.drop(columns=["original_patient_id"], inplace=True)

        store_df(df, output_path)

    def filter_conditions(self) -> None:
        output_path = self.config["task_dir"] / f"patient_condition{OUTPUT_FORMAT}"
        enc_pat_cond = self.basic_filtering("patient_condition", save=False)

        # Returns the earliest condition for each patient
        first_conditions = []
        for pat in enc_pat_cond.patient_id.unique():
            pat_cond = enc_pat_cond[enc_pat_cond.patient_id == pat]
            if not pat_cond.empty:
                first_pat_cond = pd.DataFrame(
                    [pat_cond.loc[pat_cond.recorded_date.idxmin()]]
                )
                first_conditions.append(first_pat_cond)

        if first_conditions:
            pats_cond = pd.concat(first_conditions).reset_index(drop=True)

        pats_cond = pats_cond.dropna(
            subset=["encounter_id", "patient_id", "condition_id"]
        )
        pats_cond["sex"] = reduce_cardinality(pats_cond["sex"], set_to_none=True)
        pats_cond["condition_codes"] = reduce_cardinality(
            pats_cond["condition_codes"], set_to_none=True
        )

        store_df(pats_cond, output_path)

    def filter_observation(self):
        output_path = self.config["task_dir"] / f"observation{OUTPUT_FORMAT}"
        obs = self.basic_filtering("observation", save=False)
        if obs is None:
            return

        # Initialize an empty list to collect DataFrames
        all_first_obs = []

        # For each encounter, get the earliest observation based on each code
        for enc_id in obs.encounter_id.unique():
            enc_obs = obs[obs.encounter_id == enc_id]
            for code in enc_obs.code.unique():
                code_obs = enc_obs[enc_obs.code == code]
                if not code_obs.empty:  # Check if the filtered DataFrame is not empty
                    first_code_obs = pd.DataFrame(
                        [code_obs.loc[code_obs.effectiveDateTime.idxmin()]]
                    )
                    all_first_obs.append(first_code_obs)

        # Concatenate all the DataFrames in the list, if the list is not empty
        if all_first_obs:
            obs_by_enc = pd.concat(all_first_obs).reset_index(drop=True)
            df_obs = self.observations_to_si(obs_by_enc)
            store_df(df_obs, output_path)
        else:
            logging.ERROR("No observations found")
