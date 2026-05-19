"""
rf_features.py — Random Forest specific feature layer
==========================================
Add RF-specific features on top of the base features from core_preprocess:
  - CabinRegion (CabinNum // 300, cabin section, widely recognized as a strong feature)
  - SurnameFreq (family-size proxy, counted on train+test together without leakage)
  - Fill CryoSleep / VIP NaN with 0
  - One-Hot Encoding(HomePlanet, Destination, Deck, Side)
  - No StandardScaler (tree models are not sensitive to feature scale)
  - Drop Surname after OHE (high-cardinality, not used directly by the tree model)

Return DataFrames with column names and support feature_importances_ visualization.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CAT_COLS = ['HomePlanet', 'Destination', 'Deck', 'Side']


def make_rf_data(base_train, base_test):
    """
    Parameters
    ----------
    base_train : pd.DataFrame, training set output by core_preprocess
    base_test  : pd.DataFrame, test set output by core_preprocess

    Returns
    -------
    X      : pd.DataFrame, training features (with column names)
    y      : pd.Series,    binary labels (0/1)
    X_test : pd.DataFrame, test features (with column names)
    """
    train = base_train.copy()
    test  = base_test.copy()

    # ── RF-specific additional features ────────────────────────────────────────
    # CabinRegion: bucket CabinNum by 300 (distinguish cabin section positions)
    train['CabinRegion'] = (train['CabinNum'] // 300).fillna(-1).astype(int)
    test['CabinRegion']  = (test['CabinNum']  // 300).fillna(-1).astype(int)

    # SurnameFreq: number of passengers with the same surname (family-size proxy, counted on train+test together)
    surname_counts = pd.concat([train['Surname'], test['Surname']]).value_counts()
    train['SurnameFreq'] = train['Surname'].map(surname_counts).fillna(1).astype(int)
    test['SurnameFreq']  = test['Surname'].map(surname_counts).fillna(1).astype(int)

    # CryoSleep / VIP fill missing values with 0
    for df in [train, test]:
        for col in ['CryoSleep', 'VIP']:
            df[col] = df[col].fillna(0)

    y     = train['Transported']
    train = train.drop(columns=['Transported', 'Surname'])
    test  = test.drop(columns=['Surname'])

    # ── One-Hot Encoding(tree models do not need scaling)────────────────
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    ohe_feat_names = ohe.get_feature_names_out(CAT_COLS)

    ohe_tr = pd.DataFrame(ohe.transform(train[CAT_COLS]), columns=ohe_feat_names)
    ohe_te = pd.DataFrame(ohe.transform(test[CAT_COLS]),  columns=ohe_feat_names)

    num_cols = [c for c in train.columns if c not in CAT_COLS]
    X      = pd.concat([train[num_cols].reset_index(drop=True), ohe_tr], axis=1)
    X_test = pd.concat([test[num_cols].reset_index(drop=True),  ohe_te], axis=1)

    return X, y.reset_index(drop=True), X_test


if __name__ == '__main__':
    from preprocessing import load_and_preprocess
    base_train, base_test, _ = load_and_preprocess()
    X, y, X_test = make_rf_data(base_train, base_test)
    print(f"RF Number of features: {X.shape[1]}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Training set: {X.shape},  Test set: {X_test.shape}")
