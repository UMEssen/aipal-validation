import logging

import pandas as pd

from aipal_validation.fhir.util import OUTPUT_FORMAT, check_and_read, col_to_datetime

logger = logging.getLogger(__name__)


class FHIRValidator:
    def __init__(self, config):
        self.config = config
        self.resource_schemas = {
            "observation": [("encounter_id", True), ("value", True)],
            "patient_condition": [("encounter_id", True)],
        }

    def validate(self, resource: str):
        resource = resource.lower()
        if self.config["skip_validation"]:
            pass
        elif resource in self.resource_schemas:
            self.generic_validate(resource, self.resource_schemas[resource])
        else:
            raise NotImplementedError(f"Resource {resource} not supported")

    def generic_validate(self, resource, schema):
        df = check_and_read(self.config["task_dir"] / f"{resource}{OUTPUT_FORMAT}")

        if resource == "biologically_derived_product":
            df["ausgabe_datetime"] = col_to_datetime(df.ausgabe_datetime)
            df_count = df.ausgabe_datetime.value_counts().sort_index()[:-1]
            if (df_count == 0).any():
                logger.warning("BDP count for one or more imported days = 0")

        na_counts = df.isna().sum()

        for field_name, is_error in schema:
            self.na_checker(field_name, na_counts, is_error)

    @staticmethod
    def na_checker(field_name: str, na_counts: pd.Series, is_error: bool) -> None:
        if na_counts[field_name] and is_error:
            logger.error(f"At least one {field_name} is zero")
            raise ValueError(f"At least one {field_name} is zero")
        elif na_counts[field_name]:
            logger.warning(f"At least one {field_name} is zero")
        else:
            logger.info(f"Validation for {field_name} passed")
