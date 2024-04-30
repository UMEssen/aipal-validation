import logging
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
from fhir_pyrate.pirate import Pirate
from sqlalchemy import text
from tqdm import tqdm

from aipal_validation.fhir.util import (
    OUTPUT_FORMAT,
    auth,
    check_and_read,
    engine,
    store_df,
)
from aipal_validation.helper.util import timed

logger = logging.getLogger(__name__)


class FHIRExtractor:
    def __init__(self, config):
        self.config = config
        self.search = Pirate(
            auth=auth,
            base_url=os.environ["SEARCH_URL"],
            num_processes=30,
        )

    def skip_build(self, path: Path):
        if self.config["rerun_cache"] or not path.exists():
            return False
        else:
            if path.exists():
                logger.info(
                    f"Skipping build, file {str(path).split('/')[-1]} already exists."
                )
            return path.exists()

    @timed
    def build(self, resource: str):
        if resource == "patient_condition":
            self.build_patient_condition()
        elif resource == "observation":
            self.build_observation()
        else:
            raise NotImplementedError(f"Resource {resource} not supported")

    @timed
    def df_from_query(self, query: str, chunk_size: int = 1000) -> pd.DataFrame:
        query_all = query.strip()
        logger.debug(f"Running query: {query_all}")
        dfs = []
        with engine.connect() as connection:
            # Chunk connection
            start = time.perf_counter()
            for chunk in tqdm(
                pd.read_sql_query(
                    sql=text(query_all),
                    con=connection.execution_options(stream_results=True),
                    chunksize=chunk_size,
                )
            ):
                logger.debug(f"Query relative {time.perf_counter() - start} seconds")
                dfs.append(chunk)
        return pd.concat(dfs, ignore_index=True)

    def check_and_build_file(self, name: str) -> pd.DataFrame:
        file_path = self.config["data_dir"] / f"{name}{OUTPUT_FORMAT}"
        if not file_path.exists():
            raise ValueError(f"Name {name} not recognized for building")

        df = check_and_read(file_path)
        return df

    def default_metrics_extraction(
        self,
        output_name: str,
        query: str,
        timestamp_columns: List[str] = None,
        store: bool = True,
    ):
        output_path = self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        resource_name = output_name.title().replace("_", "")
        if self.skip_build(output_path):
            return
        df = self.df_from_query(query)
        for col in timestamp_columns or []:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        logger.info(f"Extracted {len(df)} {resource_name}s")
        if store:
            store_df(df, output_path, resource_name)
        else:
            return df

    def large_metrics_extraction(
        self, query_template: str, output_name: str, store: bool = True
    ):
        output_path = self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        resource_name = output_name.title().replace("_", "")
        if self.skip_build(output_path):
            return

        time_frame_length = timedelta(days=1)
        start_datetime = pd.to_datetime(self.config["start_datetime"])
        end_datetime = pd.to_datetime(self.config["end_datetime"])

        # Initialize an empty list to store DataFrames
        dfs = []
        counter = 0
        # Generate time frames and query for each
        current_start = start_datetime
        while current_start < end_datetime:
            current_end = current_start + time_frame_length
            query = query_template.format(current_start, current_end)

            # Execute the query and append the resulting DataFrame to the list
            df = self.df_from_query(query, chunk_size=10000)
            logger.info(
                f"Extracted {len(df)} {resource_name}s from time frame {current_start} to {current_end}"
            )
            dfs.append(df)
            counter += len(df)
            current_start = current_end

        final_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Extracted {len(final_df)} {resource_name}s")
        if store:
            store_df(final_df, output_path, resource_name)
        else:
            return final_df

    def default_pyrate_extraction(
        self,
        output_name: str,
        process_function: Callable = None,
        fhir_paths: Union[List[str], List[Tuple[str, str]]] = None,
        request_params: Dict[str, str] = None,
        time_attribute_name: str = None,
        explode: List = None,
        disable_parallel: bool = False,
    ):
        output_path = self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        resource_name = output_name.title().replace("_", "")
        if self.skip_build(output_path):
            return
        params = {
            "resource_type": resource_name,
        }
        if fhir_paths is not None:
            params["fhir_paths"] = fhir_paths
        elif process_function is not None:
            params["process_function"] = process_function
        if disable_parallel:
            new_request_params = request_params.copy()
            new_request_params[time_attribute_name] = (
                f"ge{self.config['start_datetime']}",
                f"le{self.config['end_datetime']}",
            )
            params["request_params"] = new_request_params
            df = self.search.steal_bundles_to_dataframe(**params)
        else:
            params.update(
                dict(
                    request_params=request_params,
                    time_attribute_name=time_attribute_name,
                    date_init=self.config["start_datetime"],
                    date_end=self.config["end_datetime"],
                )
            )
            df = self.search.sail_through_search_space_to_dataframe(
                **params,
            )
        if explode:
            df = df.explode(explode)

        store_df(df, output_path, resource_name)

    def build_observation(self):
        pats_cond = check_and_read(
            self.config["task_dir"] / f"patient_condition{OUTPUT_FORMAT}"
        )
        pats_cond_ids = "', '".join(pats_cond["encounter_id"].unique().tolist())
        obs_codes_str = "', '".join(list(self.config["obs_codes_si"].keys()))
        obs_codes_str += "', '".join(
            [
                ", ".join(map(str, v)) if isinstance(v, list) else str(v)
                for v in self.config["merge_codes"].values()
            ]
        )
        self.default_metrics_extraction(
            output_name="observation",
            query=f"""
            SELECT
                o0.id "observation_id",
                fhirql_code (occ0.code) "code",
                occ0.display "display",
                lower(o0."effectiveDateTime") "effectiveDateTime",
                ov0.value "value",
                ov0.unit "unit",
                e1.id "encounter_id"
            FROM
                observation o0
                JOIN observation_code oc0 ON (oc0._resource = o0._id)
                JOIN observation_code_coding occ0 ON (occ0._resource = o0._id)
                JOIN "observation_valueQuantity" ov0 ON (ov0._resource = o0._id)
                JOIN observation_encounter oe0 ON (oe0._resource = o0._id)
                JOIN encounter e1 ON (
                    e1._id = oe0._reference_id
                    and oe0._reference_type = 'Encounter'
                )
            WHERE
                e1.id IN ('{pats_cond_ids}') and fhirql_code(occ0.code) IN ('{obs_codes_str}')
            """,
        )

    def build_patient_condition(self):
        Lk_code_values = [
            item for sublist in self.config["LK_codes"].values() for item in sublist
        ]
        Lk_code_values_str = "', '".join(Lk_code_values)
        self.default_metrics_extraction(
            output_name="patient_condition",
            query=f"""
                    SELECT
                        LOWER(c0."recordedDate") AS recorded_date,
                        LOWER(ep1.start) AS encounter_start,
                        LOWER(ep1.end) AS encounter_end,
                        p1.id AS patient_id,
                        LOWER(p1."birthDate") AS birth_date,
                        fhirql_read_codes(p1.gender) as sex,
                        fhirql_read_codes(ccc0.code) AS condition_codes,
                        c0.id AS condition_id,
                        e1.id AS encounter_id
                    FROM
                        condition c0
                    JOIN
                        condition_code_coding ccc0 ON ccc0._resource = c0._id
                    JOIN
                        condition_subject cs0 ON cs0._resource = c0._id
                    JOIN
                        patient p1 ON p1._id = cs0._reference_id AND cs0._reference_type = 'Patient'
                    JOIN
                        condition_encounter ce0 ON ce0._resource = c0._id
                    JOIN
                        encounter e1 ON e1._id = ce0._reference_id AND ce0._reference_type = 'Encounter'
                    JOIN
                        encounter_period ep1 ON ep1._resource = e1._id
                    WHERE
                        fhirql_code(ccc0.code) IN ('{Lk_code_values_str}')
                    """,
        )
