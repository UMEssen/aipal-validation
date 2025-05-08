import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

from fhir_pyrate.pirate import Pirate

from aipal_validation.fhir.util import OUTPUT_FORMAT, auth, store_df
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

    def default_pyrate_extraction(
        self,
        output_name: str,
        fhir_paths: Union[List[str], List[Tuple[str, str]]] = None,
        request_params: Dict[str, str] = None,
        time_attribute_name: str = None,
        explode: List = None,
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
        if request_params is not None:
            params["request_params"] = request_params
        if time_attribute_name:
            params["time_attribute_name"] = time_attribute_name
            params["date_init"] = self.config["start_datetime"]
            params["date_end"] = self.config["end_datetime"]

        df = self.search.sail_through_search_space_to_dataframe(**params)
        if explode:
            df = df.explode(explode)
        store_df(df, output_path, resource_name)

    def build_patient_condition(self):
        output_path = self.config["data_dir"] / f"patient_condition{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return

        # Extract Condition resources with related Patient and Encounter data
        df = self.search.steal_bundles_to_dataframe(
            resource_type="Condition",
            request_params={
                "_lastUpdated": f"ge{self.config['start_datetime']}",
                "_count": "100",
                "_include": ["Condition:subject", "Condition:encounter"],
            },
            fhir_paths=[
                ("recorded_date", "recordedDate"),
                ("encounter_start", "encounter.period.start"),
                ("encounter_end", "encounter.period.end"),
                ("patient_id", "subject.reference.replace('Patient/', '')"),
                ("birth_date", "subject.birthDate"),
                ("sex", "subject.gender"),
                ("condition_codes", "code.coding.code"),
                ("condition_id", "id"),
                ("encounter_id", "encounter.reference.replace('Encounter/', '')"),
                ("patient_name_family", "subject.name.family"),
                ("patient_name_given", "subject.name.given"),
                ("patient_deceased", "subject.deceasedBoolean"),
            ],
        )

        store_df(df, output_path, "PatientCondition")

    def build_observation(self):
        self.default_pyrate_extraction(
            output_name="observation",
            fhir_paths=[
                ("observation_id", "id"),
                ("code", "code.coding.code"),
                ("display", "code.coding.display"),
                ("effective_date", "effectiveDateTime"),
                ("value", "valueQuantity.value"),
                ("unit", "valueQuantity.unit"),
                ("encounter_id", "encounter.reference.replace('Encounter/', '')"),
            ],
            request_params={
                "code": self.config["obs_codes_si"],
                "encounter": self.config["encounter_ids"],
            },
            time_attribute_name="date",
        )
