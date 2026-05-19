"""
mlp_features.py — MLP specific feature layer(Embedding + Residual)
=========================================================
Add MLP-specific processing on top of the base features from core_preprocess:

  Categorical features (HomePlanet, Destination, Deck, Side, Surname): 
    LabelEncoder -> integer indices for Embedding lookup
  Numerical features (all remaining columns): 
    StandardScaler standardization
  Fill CryoSleep / VIP NaN with 0

  Return the merged DataFrame, with cat_cols / num_cols listed separately,
  so the caller can separately extract X[:, num] and X[:, cat] for the model.

Example of dynamic Embedding dimension calculation (see __main__): 
  cat_cardinalities = [int(max(X[c].max(), X_test[c].max()) + 1) for c in cat_cols]
  emb_dims          = [min(50, (count // 2) + 1) for count in cat_cardinalities]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

CAT_COLS     = ['HomePlanet', 'Destination', 'Deck', 'Side']
CAT_COLS_MLP = CAT_COLS + ['Surname']   # MLP additionally uses Surname for Embedding


def make_mlp_data(base_train, base_test):
    """
    Parameters
    ----------
    base_train : pd.DataFrame, training set output by core_preprocess
    base_test  : pd.DataFrame, test set output by core_preprocess

    Returns
    -------
    X        : pd.DataFrame, training features (numeric columns standardized, categorical columns as integer indices)
    y        : pd.Series,    binary labels (0/1)
    X_test   : pd.DataFrame, test features (same format as above)
    cat_cols : list[str],    categorical column names (for the Embedding layer)
    num_cols : list[str],    numeric column names (for the fully connected layers)
    """
    train = base_train.copy()
    test  = base_test.copy()

    # CryoSleep / VIP fill missing values with 0
    for df in [train, test]:
        for col in ['CryoSleep', 'VIP']:
            df[col] = df[col].fillna(0)

    y     = train['Transported']
    train = train.drop(columns=['Transported'])

    # ── Categorical features: LabelEncode to integers (for Embedding lookup)────
    for col in CAT_COLS_MLP:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    # ── Numerical features: StandardScaler ──────────────────────────────
    num_cols = [c for c in train.columns if c not in CAT_COLS_MLP]
    scaler = StandardScaler()
    train_num = pd.DataFrame(
        scaler.fit_transform(train[num_cols]), columns=num_cols)
    test_num  = pd.DataFrame(
        scaler.transform(test[num_cols]), columns=num_cols)

    # ── Merge (column order: numeric columns first, categorical columns last)────────────────
    X      = pd.concat(
        [train_num, train[CAT_COLS_MLP].reset_index(drop=True)], axis=1)
    X_test = pd.concat(
        [test_num,  test[CAT_COLS_MLP].reset_index(drop=True)],  axis=1)

    return X, y.reset_index(drop=True), X_test, CAT_COLS_MLP, num_cols


if __name__ == '__main__':
    from preprocessing import load_and_preprocess
    base_train, base_test, _ = load_and_preprocess()
    X, y, X_test, cat_cols, num_cols = make_mlp_data(base_train, base_test)
    print(f"MLP Numeric features: {len(num_cols)}  | Categorical features: {len(cat_cols)} ")
    print(f"Numeric columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")
    cat_cardinalities = [int(max(X[c].max(), X_test[c].max()) + 1) for c in cat_cols]
    emb_dims = [min(50, (count // 2) + 1) for count in cat_cardinalities]
    for col, card, dim in zip(cat_cols, cat_cardinalities, emb_dims):
        print(f"  {col:15s}: cardinality={card}, emb_dim={dim}")
    print(f"Training set: {X.shape},  Test set: {X_test.shape}")
