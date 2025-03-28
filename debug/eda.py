import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

columns_of_interest = ['age', 'MCV_fL', 'PT_percent', 'LDH_UI_L',
                       'MCHC_g_L', 'WBC_G_L', 'Fibrinogen_g_L', 'Monocytes_G_L',
                       'Platelets_G_L', 'Lymphocytes_G_L'
                       ]

# Load data
data_files = glob('../data/**/aipal/samples.csv', recursive=True)
data_per_city = {fn.split(os.path.sep)[2]: fn for fn in data_files}

city_dfs = []
for city_name, file in data_per_city.items():
    df = pd.read_csv(file)
    df['origin'] = city_name
    city_dfs.append(df)

master_df = pd.concat(city_dfs).reset_index(drop=True)
master_df["origin"] = master_df["origin"].astype(str).astype("category")
# master_df.to_csv('./all_origins.csv', index=False)

# age
# print(master_df[master_df.age > 100].origin.value_counts())
# print(master_df[master_df.age < 0].origin.value_counts())
# MCV_fL
# print(master_df[master_df.MCV_fL > 200].origin.value_counts())
# MCHC_g_L
# print(master_df[master_df['MCHC_g_L'] > 3000].origin.value_counts())

for col in columns_of_interest:
    if col in master_df.columns and master_df[col].dtype == 'object':
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

    plt.figure(figsize=(10, 6))
    try:
        ax = sns.kdeplot(
            data=master_df,
            x=col,
            hue="origin",
            fill=True,
            alpha=0.4,
            common_norm=False,
            legend=True,
            warn_singular=False
        )

        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"{col} Distribution by City")

        plt.tight_layout()
        plt.savefig(f"./distributions/{col}.png")
    except TypeError:
        print(col)