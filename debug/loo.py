import pandas as pd
import xgboost as xgb
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

columns_of_interest = ['age', 'MCV_fL', 'PT_percent', 'LDH_UI_L',
                       'MCHC_g_L', 'WBC_G_L', 'Fibrinogen_g_L', 'Monocytes_G_L',
                       'Platelets_G_L', 'Lymphocytes_G_L', 'class', 'origin'
                       ]


def fit_and_score(estimator, X_train, X_test, y_train, y_test):
    """Fit the estimator on the train set and score it on both sets"""
    estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)

    return estimator, train_score, test_score


master_df = pd.read_csv('./all_origins.csv', usecols=columns_of_interest)
master_df['class'] = pd.Categorical(master_df['class']).codes.astype(float)
master_df.dropna(inplace=True)

origins = master_df.origin.unique()

for o in origins:
    logger.info(f'LOO: {o}')
