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


def transform_gender_age_class_italy(df) -> pd.DataFrame:
    df["sex"] = df["Gender"].apply(lambda x: "Male" if x == "M" else "Female")
    df["age"] = round(df["Age at diagnosis"], 0).astype(int)
    df.drop(columns=["Gender", "Age at diagnosis"], inplace=True)
    df["class"] = df["Diagnosis"].apply(
        lambda x: (
            "ALL"
            if "ALL" in x
            else "AML"
            if "AML" in x
            else "APL"
            if "APL" in x
            else "other"
        )
    )
    df = df[~df["class"].isin(["other"])]
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
            df[k] = df[k].astype(float) * v

    # Change col names to new units
    df.rename(columns=rename_dict, inplace=True)
    return df


def parse_to_numeric(df, config, names=None, dropna=True):
    column_names = names if names else {v[3] for v in config["obs_codes_si"].values()}
    for col in column_names:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logging.info(
        f"Identified {df.shape[0] - df.dropna(subset=column_names).shape[0]} rows with missing values"
    )
    if dropna:
        df = df.dropna(subset=column_names)
        logging.info(f"Dropped rows with missing values, new shape: {df.shape}")
    else:
        df = df.fillna(np.nan)
        logging.info(f"Filled missing values with NaN, shape: {df.shape}")
    return df


def main(config):
    if skip_build(config):
        return
    df = pd.read_excel(config["task_dir"] / "data.xlsx")

    # todo make more generic if needed
    if config["run_id"] == "italy" or config["run_id"] == "rome":
        df = transform_gender_age_class_italy(df)
        df = df_to_si(df, config)
    elif config["run_id"] == "poland":
        df = transform_age_diagnosis(df)
        df = df_to_si(df, config)
    elif config["run_id"] == "sao_paulo":
        df = parse_to_numeric(df, config)
        df.rename(columns={"Age": "age"}, inplace=True)
    elif config["run_id"] == "salamanca":
        df = df_to_si(df, config)  # obs since it is the same as the obs_codes_si
        df.rename(columns={"Gender": "sex"}, inplace=True)
        df.rename(columns={"Age": "age"}, inplace=True)
        df.rename(columns={"Class": "class"}, inplace=True)
    elif config["run_id"] == "turkey":
        df.rename(columns={"Age": "age"}, inplace=True)
        df = parse_to_numeric(df, config, dropna=False)
    elif config["run_id"] == "buenos_aires":
        df["sex"] = df["sex"].str.strip()
        df["class"] = df["class"].str.strip()
        df = parse_to_numeric(df, config, dropna=False)
        df.drop(columns=["Unnamed: 13", "comment"], inplace=True)
    elif config["run_id"] == "kalkutta":
        df = parse_to_numeric(df, config, dropna=False)
    elif config["run_id"] == "suzhou":
        df.columns = df.columns.str.strip()
        df = df[df["class"] != "MPAL"]
        df = df[df["class"] != "AUL"]
        df = df_to_si(df, config, "warsaw_suzhou_codes_si")
        df = parse_to_numeric(df, config, dropna=False)
        df["PT_%_AVG"] = np.nan
    elif config["run_id"] == "bochum":
        df.rename(columns={"Age": "age"}, inplace=True)
        df["sex"] = df["sex"].apply(lambda x: "Male" if x == "male" else "Female")
        df = parse_to_numeric(df, config, dropna=False)
    elif config["run_id"] == "milano":
        df["sex"] = df["sex"].apply(lambda x: "Male" if x == "M" else "Female")
        df = parse_to_numeric(df, config, dropna=False)
    elif config["run_id"] == "warsaw":
        df.columns = df.columns.str.replace("\n", "").str.strip()
        df = df_to_si(df, config, "warsaw_suzhou_codes_si")

    else:
        raise NotImplementedError(f"Unknown run_id: {config['run_id']}")

    column_names = {v[3] for v in config["obs_codes_si"].values()}
    if not column_names.issubset(df.columns):
        raise ValueError(
            f"Columns {column_names} not found in the data, parsing failed."
        )

    filter_irrelevant_columns(df, config)

    store_df(df, config["task_dir"] / "samples.csv")
