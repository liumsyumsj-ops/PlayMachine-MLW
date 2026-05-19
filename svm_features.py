"""
svm_features.py — SVM specific feature layer
=======================================
Add SVM-specific processing on top of the base features from core_preprocess:
  - Fill CryoSleep / VIP NaN with 0
  - LabelEncoder(HomePlanet, Destination, Deck, Side, Surname)
  - Global StandardScaler standardization (SVM is highly scale-sensitive, so this is required)
  - SelectKBest(f_classif, k=50) ANOVA feature selection
    (SVM is sensitive to high-dimensional data, so select the top-k univariate significant features)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

CAT_COLS = ['HomePlanet', 'Destination', 'Deck', 'Side']


def make_svm_data(base_train, base_test, k=50):
    """
    Parameters
    ----------
    base_train : pd.DataFrame, training set output by core_preprocess
    base_test  : pd.DataFrame, test set output by core_preprocess
    k          : int, number of features retained by SelectKBest (default 50)

    Returns
    -------
    X              : np.ndarray, selected training features
    y              : pd.Series,  binary labels (0/1)
    X_test         : np.ndarray, selected test features
    selected_feats : list[str],  names of retained features
    """
    train = base_train.copy()
    test  = base_test.copy()

    # CryoSleep / VIP fill missing values with 0
    for df in [train, test]:
        for col in ['CryoSleep', 'VIP']:
            df[col] = df[col].fillna(0)

    y     = train['Transported']
    train = train.drop(columns=['Transported'])

    # LabelEncoder (including Surname; SVM can use ordinal encoding)
    svm_cat_cols = CAT_COLS + ['Surname']
    for col in svm_cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    # StandardScaler (all features, including encoded categorical column values)
    scaler = StandardScaler()
    X_all  = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    Xt_all = pd.DataFrame(scaler.transform(test),       columns=test.columns)

    # SelectKBest selection
    k_real   = min(k, X_all.shape[1])
    selector = SelectKBest(f_classif, k=k_real)
    X      = selector.fit_transform(X_all, y)
    X_test = selector.transform(Xt_all)
    selected_feats = list(X_all.columns[selector.get_support()])

    return X, y, X_test, selected_feats


if __name__ == '__main__':
    from preprocessing import load_and_preprocess
    base_train, base_test, _ = load_and_preprocess()
    X, y, X_test, feats = make_svm_data(base_train, base_test)
    print(f"Number of SVM features after selection: {X.shape[1]}")
    print(f"Retained features: {feats}")
    print(f"Training set: {X.shape},  Test set: {X_test.shape}")
