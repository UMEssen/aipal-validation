#%%


import pandas as pd
import os
from util import load_data

df_adults, config, features  = load_data(is_adult=True)
df_kids, config, features  = load_data(is_adult=False)


#%%
df = pd.concat([df_kids, df_adults], axis=0)

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load config file using relative path
features.append('sex')
features.append('class')


# overall
df = df[features]
df.sex = df.sex.str.lower().str.strip()
df.sex = df.sex.replace('male', 'male')
df.sex = df.sex.replace('female', 'female')
print(f"Number of samples: {len(df)+10}")

## inital
print("#########################")
print("Initial")
male, female = df.sex.value_counts()
print(f"Number of males: {male}")
print(f"Number of females: {female}")
print(f"Ratio: {male/female}")
print(f"Audult: {len(df[df.age > 18])}")
print(f"Children: {len(df[df.age <= 18])}")
print(f"Classes: \n{df['class'].value_counts()}")


# # get samples with more than 20 % missing values
df_input_features = df[features]
missing_samples = df_input_features.isna().sum(axis=1) > 0.2 * len(features)

df_input_features = df_input_features[df_input_features.isna().sum(axis=1) < 0.2 * len(features)]

percentage_missing = missing_samples.sum() / len(df_input_features)

print(f"Number of samples with more than 20 % missing values: {missing_samples.sum()}")
print(f"Percentage of samples with more than 20 % missing values: {percentage_missing}")


## after removing samples with more than 20 % missing values
print("#########################")
print("After removing samples with more than 20 % missing values")
df = df[~missing_samples]
male, female = df.sex.value_counts()
print(f"Number of males: {male}")
print(f"Number of females: {female}")
print(f"Ratio: {male/female}")
print(f"Audult: {len(df[df.age > 18])}")
print(f"Children: {len(df[df.age <= 18])}")
print(f"Classes: \n{df['class'].value_counts()}")


# %%

df_adults = df[df.age > 18]
df_adults['class'].value_counts()

# %%
# get children classes
df_kids = df[df.age <= 18]
df_kids['class'].value_counts()

# %%
