import numpy as np
import pandas as pd
import yaml
import os

def load_data(config_path='cfg.yaml', root_path='/local/work/merengelke/aipal/', filter_by_size=True):
    """
    Load and preprocess data from multiple cohorts based on configuration.

    Parameters:
    -----------
    config_path : str
        Path to the configuration YAML file
    root_path : str
        Root path for data files
    filter_by_size : bool
        Whether to filter cohorts by minimum size (default: True)

    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe with all cohorts
    dict
        Configuration dictionary
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get paths for all cohorts
    cities_countries = config['cities_countries']

    if config['is_adult']:
        cities_countries = [city_country for city_country in cities_countries if city_country != 'newcastle' and city_country != 'turkey' and city_country != 'Vessen']

    paths = [f"{root_path}{city_country}/aipal/predict.csv" for city_country in cities_countries]

    # Load and concatenate data
    df = pd.DataFrame()
    for path in paths:
        try:
            df_small = pd.read_csv(path)
            df_small['city_country'] = path.split('/')[-3]
            df = pd.concat([df, df_small])
        except Exception as e:
            print(f"Error loading {path}: {e}")

    # Filter by age based on configuration
    if config['is_adult']:
        df = df[df['age'] > 18]
    else:
        df = df[df['age'] <= 18]

    df['city_country'] = df['city_country'].str.replace('_', ' ')
    df['city_country'] = df['city_country'].str.capitalize()

    # map M and F to Male and Female
    df['sex'] = df['sex'].replace({'M': 'Male', 'F': 'Female'})
    df['sex'] = df['sex'].replace('I', 'Male')

    # Drop unnecessary columns
    df.drop(columns=['ELN', 'Diagnosis', 'additional.diagnosis.details..lineage.etc', 'lineage.details'],
            inplace=True, errors='ignore')

    # Clean class values
    df['class'] = df['class'].str.strip()

    # Filter cohorts by size if requested
    if filter_by_size:
        df = df.groupby('city_country').filter(lambda x: len(x) > 30)

    # Get feature columns from config
    features = config['feature_columns']

    return df, config, features
