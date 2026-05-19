"""
lgb_features.py — LightGBM specific feature layer
=======================================
Add LGB-specific features on top of the base features from preprocessing_shared:
  - log transforms (handle right-skewed spending columns)
  - SpendDiversity, AwakeZeroSpend, HasSpent
  - Europa_Cryo(strong interaction)
  - LabelEncode -> category dtype(LGB native categorical support)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SPEND_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def make_lgb_data(base_train, base_test):
    """
    Returns
    -------
    X, y, X_test
    """
    train = base_train.copy()
    test  = base_test.copy()

    for df in [train, test]:
        for col in SPEND_COLS + ['TotalSpend']:
            df[f'{col}_log'] = np.log1p(df[col])

        df['SpendDiversity'] = df[SPEND_COLS].gt(0).sum(axis=1)
        df['AwakeZeroSpend'] = ((df['CryoSleep'].fillna(0) == 0) &
                                (df['TotalSpend'] == 0)).astype(int)
        df['HasSpent']       = (df['TotalSpend'] > 0).astype(int)

        df['Europa_Cryo'] = (
            (df['HomePlanet'] == 'Europa') & (df['CryoSleep'].fillna(0) == 1)
        ).astype(int)

        df.drop(columns=['Surname'], inplace=True)

    # CryoSleep / VIP NaN fill with -1
    for col in ['CryoSleep', 'VIP']:
        train[col] = train[col].fillna(-1)
        test[col]  = test[col].fillna(-1)

    # LabelEncode -> category dtype
    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]]).astype(str)
        le.fit(combined)
        encoded = le.transform(train[col].astype(str)); train[col] = pd.Categorical(encoded)
        test[col]  = pd.Categorical(le.transform(test[col].astype(str)))

    X      = train.drop(columns=['Transported'])
    y      = train['Transported']
    X_test = test

    return X, y, X_test


if __name__ == '__main__':
    from preprocessing import load_and_preprocess
    base_train, base_test, groups = load_and_preprocess()
    X, y, X_test = make_lgb_data(base_train, base_test)
    print(f"LGB Number of features: {X.shape[1]}")
    print(f"Feature columns: {list(X.columns)}")
