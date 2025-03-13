import logging

import numpy as np
import pandas as pd

from aipal_validation.data_preprocessing.util import skip_build
from aipal_validation.fhir.util import store_df


def filter_irrelevant_columns(df, config):
    obs_codes = {v[3] for v in config["obs_codes_si"].values()}
    obs_codes.add("age")
    obs_codes.add("sex")
    obs_codes.add("class")
    obs_codes.add("ID")
    df.drop(columns=[col for col in df.columns if col not in obs_codes], inplace=True)
    return df


def transform_age_diagnosis(df) -> pd.DataFrame:
    df.rename({"age (years)": "age"}, axis=1, inplace=True)
    df.rename({"diagnose": "class"}, axis=1, inplace=True)
    return df


def df_to_si(df, config, dict_key_name=None):
    if dict_key_name is None:
        si_dict_key = config["run_id"] + "_codes_si"
    else:
        si_dict_key = dict_key_name
    si_dict = config[si_dict_key]
    rename_dict = {k: v[3] for k, v in si_dict.items()}
    multipliers = {k: v[1] for k, v in si_dict.items()}

    # Multiply the columns with the corresponding multipliers
    for k, v in multipliers.items():
        if k in df.columns:
            # Clean the column values
            df[k] = df[k].replace("", np.nan)  # Replace empty strings with NaN
            df[k] = df[k].astype(str).str.replace(",", ".")  # Handle commas as decimals
            df[k] = pd.to_numeric(
                df[k], errors="coerce"
            )  # Convert to numeric, invalid values become NaN
            df[k] = df[k] * v  # Apply the multiplier

    # Change col names to new units
    df.rename(columns=rename_dict, inplace=True)
    return df


def parse_to_numeric(df, config, names=None, dropna=True):
    column_names = names if names else {v[3] for v in config["obs_codes_si"].values()}
    if not column_names:
        logging.warning("No columns specified for numeric parsing")
        return df

    # Only process columns that actually exist in the dataframe
    columns_to_process = [col for col in column_names if col in df.columns]
    if not columns_to_process:
        logging.warning(f"None of the specified columns {column_names} found in dataframe")
        return df

    for col in columns_to_process:
        df[col] = df[col].astype(str).str.replace(",", ".")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logging.info(
        f"Identified {df.shape[0] - df.dropna(subset=columns_to_process).shape[0]} rows with missing values"
    )
    if dropna:
        df = df.dropna(subset=columns_to_process)
        logging.info(f"Dropped rows with missing values, new shape: {df.shape}")
    else:
        df = df.fillna(np.nan)
        logging.info(f"Filled missing values with NaN, shape: {df.shape}")
    return df


def parse_classes_maastricht(df):
    df["class"] = df["class"].str.strip()

    valid_classes = ["AML", "APL", "ALL"]
    df["parsed_class"] = df["class"].apply(
        lambda x: next((cls for cls in valid_classes if x.endswith(cls)), None)
    )

    df = df[df["parsed_class"].notna()]
    df["class"] = df["parsed_class"]
    df.drop(columns=["parsed_class"], inplace=True)

    return df


# Helper functions for specific run_id operations
def parse_maastricht(df, config):
    df.dropna(subset=["class"], inplace=True)
    df = df_to_si(df, config)
    return parse_classes_maastricht(df)


def parse_melbourne(df, config):
    df.dropna(subset=["class"], inplace=True)
    df = df_to_si(df, config)
    print(df.columns)
    return df


def transform_age_poland(df, config):
    df = transform_age_diagnosis(df)
    return df_to_si(df, config)


def parse_sao_paulo(df, config):
    df = parse_to_numeric(df, config)
    df.rename(columns={"LDH_IU_L": "LDH_UI_L"}, inplace=True)
    df["LDH_UI_L"] = pd.to_numeric(df["LDH_UI_L"], errors="coerce")
    return df


def parse_buenos_aires(df, config):
    df.columns = df.columns.str.strip()
    return parse_to_numeric(df, config, dropna=False)


def parse_suzhou(df, config):
    df.columns = df.columns.str.strip()
    df = df[df["class"] != "MPAL"]
    df = df[df["class"] != "AUL"]
    df = df_to_si(df, config, "warsaw_suzhou_codes_si")
    df = parse_to_numeric(df, config, dropna=False)
    df["PT_%_AVG"] = np.nan
    return df


def parse_bochum(df, config):
    df.rename(columns={"Age": "age"}, inplace=True)
    return parse_to_numeric(df, config, dropna=False)


def parse_newcastle(df, config):
    print("Newcastle: Insufficient columns provided")
    df["PT_percent"] = pd.to_numeric(df["PT_percent"], errors="coerce")
    df["PT_percent"].iloc[0] = 0
    return df


def parse_antananarivo(df, config):
    df["sex"] = ["Male" if x == "MALE" else "Female" for x in df["sex"]]
    return df


def parse_lagos(df, config):
    df.rename(columns={"LDH_IU_L": "LDH_UI_L"}, inplace=True)
    return df


def parse_hannover(df, config):
    df = df[df["out of measurment span"] == 0]
    df.rename(columns={"LDH_IU_L": "LDH_UI_L"}, inplace=True)
    df["Fibrinogen_g_L"] = pd.to_numeric(df["Fibrinogen_g_L"], errors="coerce")
    return df


def parse_wroclaw(df, config):
    df = df_to_si(df, config, "wroclaw_codes_si")
    df.index = [f"WC{i}" for i in range(len(df))]
    df.ID = df.index
    df.sex = ["Male" if x == 0 else "Female" for x in df.sex]
    return df


def main(config):
    if skip_build(config):
        return

    # Load and clean initial data
    df = pd.read_excel(config["task_dir"] / "data.xlsx")
    df["class"] = df["class"].str.strip()
    df.columns = df.columns.str.strip()

    # Define operations for each run_id
    run_operations = {
        "rome": lambda df, config: (print("Rome: Nothing to do"), df)[1],
        "barcelona": lambda df, config: (print("Barcelona: Nothing to do"), df)[1],
        "maastricht": lambda df, config: parse_maastricht(df, config),
        "dallas": lambda df, config: df_to_si(df, config),
        "melbourne": lambda df, config: parse_melbourne(df, config),
        "poland": lambda df, config: transform_age_poland(df, config),
        "sao_paulo": lambda df, config: parse_sao_paulo(df, config),
        "salamanca": lambda df, config: (print("Salamanca: Nothing to do"), df)[1],
        "turkey": lambda df, config: (print("Turkey: Nothing to do"), df)[1],
        "buenos_aires": lambda df, config: parse_buenos_aires(df, config),
        "kalkutta": lambda df, config: (print("Kalkutta: Nothing to do"), df)[1],
        "suzhou": lambda df, config: parse_suzhou(df, config),
        "bochum": lambda df, config: parse_bochum(df, config),
        "milano": lambda df, config: (print("Milano: Nothing to do"), df)[1],
        "newcastle": lambda df, config: parse_newcastle(df, config),
        "wroclaw": lambda df, config: parse_wroclaw(df, config),
        "antananarivo": lambda df, config: parse_antananarivo(df, config),
        "lagos": lambda df, config: parse_lagos(df, config),
        "madagascar": lambda df, config: (print("Madagascar: Nothing to do"), df)[1],
        "hannover": lambda df, config: parse_hannover(df, config),
    }
    try:
        operation = run_operations[config["run_id"]]
        df = operation(df, config)
    except KeyError:
        raise NotImplementedError(f"Unknown run_id: {config['run_id']}")

    # Check required columns
    column_names = {v[3] for v in config["obs_codes_si"].values()}
    if not column_names.issubset(df.columns):
        raise ValueError(
            f"Columns {column_names} not found in the data, parsing failed."
        )

    # Final processing
    filter_irrelevant_columns(df, config)
    store_df(df, config["task_dir"] / "samples.csv")
