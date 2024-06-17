import pandas as pd

from aipal_validation.data_preprocessing.util import skip_build
from aipal_validation.fhir.util import store_df


def transform_gender_age_class_italy(df) -> pd.DataFrame:
    df["sex"] = df["Gender"].apply(lambda x: "Male" if x == "M" else "Female")
    df["age"] = round(df["Age at diagnosis"], 0).astype(int)
    df.drop(columns=["Gender", "Age at diagnosis"], inplace=True)
    df["class"] = df["Diagnosis"].apply(
        lambda x: "ALL"
        if "ALL" in x
        else "AML"
        if "AML" in x
        else "APL"
        if "APL" in x
        else "other"
    )
    df = df[~df["class"].isin(["other"])]
    return df


def transform_age_diagnosis(df) -> pd.DataFrame:
    df.rename({"age (years)": "age"}, axis=1, inplace=True)
    df.rename({"diagnose": "class"}, axis=1, inplace=True)
    return df


def df_to_si(df, config):
    # Access the dictionary in config using the key
    si_dict_key = config["run_id"] + "_codes_si"
    si_dict = config[si_dict_key]
    print(si_dict_key)
    rename_dict = {k: v[3] for k, v in si_dict.items()}
    df.rename(columns=rename_dict, inplace=True)
    return df


def main(config):
    if skip_build(config):
        return
    df = pd.read_excel(config["task_dir"] / "data.xlsx")

    # todo make more generic if needed
    if config["run_id"] == "italy":
        df = transform_gender_age_class_italy(df)
        df = df_to_si(df, config)

    if config["run_id"] == "poland":
        df = transform_age_diagnosis(df)
        df = df_to_si(df, config)

    store_df(df, config["task_dir"] / "samples.csv")
