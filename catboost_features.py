"""
catboost_features.py — CatBoost specific feature layer
============================================
Add CatBoost-specific features on top of the base features from preprocessing_shared:
  - ratio / log transforms (handle right-skewed spending columns)
  - interaction features (CryoSpend, Age_x_Cryo, etc.)
  - boolean flags (AwakeZeroSpend, HasSpent, HighSpa)
  - SpendDiversity, MaxSpend

CatBoost handles cat_features natively, so LabelEncode is not required.
"""

import numpy as np
import pandas as pd

CAT_FEATURES = ['HomePlanet', 'Destination', 'Deck', 'Side']
SPEND_COLS   = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def make_catboost_data(base_train, base_test):
    """
    Returns
    -------
    X, y, X_test, CAT_FEATURES
    """
    train = base_train.copy()
    test  = base_test.copy()

    for df in [train, test]:
        cryo_val = df['CryoSleep'].fillna(0)

        df['HasSpent']       = (df['TotalSpend'] > 0).astype(int)
        for col in SPEND_COLS:
            df[f'{col}_ratio'] = df[col] / (df['TotalSpend'] + 1)
        for col in SPEND_COLS + ['TotalSpend']:
            df[f'{col}_log']   = np.log1p(df[col])
        df['NecessitiesSpend_log']   = np.log1p(df['NecessitiesSpend'])
        df['EntertainmentSpend_log'] = np.log1p(df['EntertainmentSpend'])

        df['Side_bin']       = (df['Side'] == 'S').astype(int)
        df['AwakeZeroSpend'] = ((cryo_val == 0) & (df['TotalSpend'] == 0)).astype(int)
        df['CryoSpend']      = cryo_val * df['TotalSpend_log']
        df['Age_x_Cryo']     = df['Age'] * cryo_val
        df['NonCryo_Spend']  = (1 - cryo_val) * df['TotalSpend_log']
        df['HighSpa']        = (df['Spa'] > 1000).astype(int)

        log_cols = [f'{c}_log' for c in SPEND_COLS]
        df['SpendDiversity'] = (df[log_cols] > 0).sum(axis=1)
        df['MaxSpend']       = df[SPEND_COLS].max(axis=1)
        df['MaxSpend_log']   = np.log1p(df['MaxSpend'])

        df.drop(columns=['Surname'], inplace=True)

    X      = train.drop(columns=['Transported'])
    y      = train['Transported']
    X_test = test

    return X, y, X_test, CAT_FEATURES


if __name__ == '__main__':
    from preprocessing import load_and_preprocess
    base_train, base_test, groups = load_and_preprocess()
    X, y, X_test, cat_features = make_catboost_data(base_train, base_test)
    print(f"CatBoost Number of features: {X.shape[1]}")
    print(f"Feature columns: {list(X.columns)}")
