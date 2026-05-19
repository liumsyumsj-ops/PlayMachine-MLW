"""
knn_features.py — KNN specific feature layer
=======================================
Add KNN-specific processing on top of the base features from core_preprocess:
  - Remove Surname (high-cardinality categorical variable that KNN cannot use effectively)
  - Fill CryoSleep / VIP NaN with 0
  - One-Hot Encoding(HomePlanet, Destination, Deck, Side)
  - Global StandardScaler standardization (KNN is highly scale-sensitive, so this is required)
  - PCA dimensionality reduction (keeps 95% variance by default to reduce the curse of dimensionality)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

CAT_COLS = ['HomePlanet', 'Destination', 'Deck', 'Side']


def make_knn_data(base_train, base_test, pca_variance=0.95):
    """
    Parameters
    ----------
    base_train   : pd.DataFrame, training set output by core_preprocess
    base_test    : pd.DataFrame, test set output by core_preprocess
    pca_variance : float, PCA retained variance ratio (default 0.95)

    Returns
    -------
    X      : np.ndarray, training features after PCA dimensionality reduction
    y      : pd.Series,  binary labels (0/1)
    X_test : np.ndarray, test features after PCA dimensionality reduction
    n_comp : int,        actual number of principal components retained by PCA
    """
    train = base_train.copy()
    test  = base_test.copy()

    # CryoSleep / VIP fill missing values with 0
    for df in [train, test]:
        for col in ['CryoSleep', 'VIP']:
            df[col] = df[col].fillna(0)

    y     = train['Transported']
    train = train.drop(columns=['Transported', 'Surname'])
    test  = test.drop(columns=['Surname'])

    num_cols = [c for c in train.columns if c not in CAT_COLS]

    # One-Hot Encoding(train+test fit on train+test together to avoid leakage)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    enc_train = ohe.transform(train[CAT_COLS])
    enc_test  = ohe.transform(test[CAT_COLS])

    X_raw  = np.hstack([train[num_cols].values, enc_train])
    Xt_raw = np.hstack([test[num_cols].values,  enc_test])

    # StandardScaler
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_raw)
    Xt_scaled = scaler.transform(Xt_raw)

    # PCA dimensionality reduction
    pca    = PCA(n_components=pca_variance, random_state=42)
    X      = pca.fit_transform(X_scaled)
    X_test = pca.transform(Xt_scaled)

    return X, y, X_test, pca.n_components_


if __name__ == '__main__':
    from preprocessing import load_and_preprocess
    base_train, base_test, _ = load_and_preprocess()
    X, y, X_test, n = make_knn_data(base_train, base_test)
    print(f"KNN Number of features (PCA after): {X.shape[1]}  (retaining 95% variance, {n} principal components)")
    print(f"Training set: {X.shape},  Test set: {X_test.shape}")
