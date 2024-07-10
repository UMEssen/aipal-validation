import pandas as pd

from aipal_validation.data_preprocessing.util import skip_build
from aipal_validation.fhir.util import store_df


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


def df_to_si(df, config):
    si_dict_key = config["run_id"] + "_codes_si"
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


def parse_to_numeric(df, config, names=None):
    column_names = names if names else {v[3] for v in config["obs_codes_si"].values()}
    for col in column_names:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=column_names)
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
        # df.rename(columns={"MCHC_g_dL": "MCHC_g_L"}, inplace=True)
        # df.rename(columns={"Fibrinogen_mg_dL": "Fibrinogen_g_L"}, inplace=True)
        df.rename(columns={"Age": "age"}, inplace=True)
        df.rename(columns={"Class": "class"}, inplace=True)
    else:
        raise NotImplementedError(f"Unknown run_id: {config['run_id']}")

    column_names = {v[3] for v in config["obs_codes_si"].values()}
    if not column_names.issubset(df.columns):
        raise ValueError(
            f"Columns {column_names} not found in the data, parsing failed."
        )

    store_df(df, config["task_dir"] / "samples.csv")
