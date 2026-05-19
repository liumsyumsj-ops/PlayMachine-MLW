"""
preprocessing.py — unified data cleaning + model-specific encoding layers
=================================================
All models share the same core cleaning logic.

Core functions: 
  load_data(train_path, test_path)          -> train_raw, test_raw
  core_preprocess(train_raw, test_raw)      -> base_train, base_test
  load_and_preprocess(train_path, test_path)-> base_train, base_test, groups

Model-specific encoders: 
  encode_for_knn(base_train, base_test)     -> X, y, X_test  (OHE + StandardScaler)
  encode_for_svm(base_train, base_test)     -> X, y, X_test  (LabelEncoder + StandardScaler)

Base features: 
  GroupSize, IsAlone, IsChild
  Deck, CabinNum, Side, CabinSize
  HomePlanet, Destination (str)
  CryoSleep, VIP (0/1/NaN)
  Age
  RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
  TotalSpend, NecessitiesSpend, EntertainmentSpend
  AllGroupCryo, NoGroupCryo
  Surname
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

TRAIN_PATH = "spaceship-titanic/train.csv"
TEST_PATH  = "spaceship-titanic/test.csv"

CAT_COLS = ['HomePlanet', 'Destination', 'Deck', 'Side']


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)
    return train_raw, test_raw


def _preprocess(train_raw, test_raw):
    train = train_raw.copy(); test = test_raw.copy()
    train['is_train'] = 1; test['is_train'] = 0; test['Transported'] = np.nan
    df = pd.concat([train, test], ignore_index=True)

    df['GroupId']   = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum']  = pd.to_numeric(df['CabinNum'], errors='coerce')

    for col in ['HomePlanet', 'Destination', 'Cabin']:
        df[col] = df.groupby('GroupId')[col].transform(
            lambda x: x.fillna(x.mode()[0]) if x.notna().any() else x)
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')

    has_cabin = df['Deck'].notna() & df['CabinNum'].notna() & df['Side'].notna()
    cabin_key = (df['Deck'].astype(str) + '_' +
                 df['CabinNum'].astype(str) + '_' +
                 df['Side'].astype(str))
    cabin_key[~has_cabin] = np.nan
    df['CabinSize'] = cabin_key.map(cabin_key.dropna().value_counts())
    df['CabinSize'] = df['CabinSize'].fillna(1).astype(int)

    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cryo_true = df['CryoSleep'] == True
    for col in spend_cols:
        df.loc[cryo_true & df[col].isna(), col] = 0.0
    has_spend = df[spend_cols].gt(0).any(axis=1)
    df.loc[has_spend & df['CryoSleep'].isna(), 'CryoSleep'] = False

    deck_map = {'Europa': 'B', 'Earth': 'G', 'Mars': 'F'}
    mask = df['Deck'].isna() & df['HomePlanet'].notna()
    df.loc[mask, 'Deck'] = df.loc[mask, 'HomePlanet'].map(deck_map)

    for col in ['Deck', 'Side']:
        df[col] = df.groupby(['HomePlanet', 'CryoSleep'])[col].transform(
            lambda x: x.fillna(x.mode()[0]) if x.notna().any() else x)
        df[col] = df[col].fillna(df[col].mode()[0])

    df['CabinNum'] = df.groupby('Deck')['CabinNum'].transform(
        lambda x: x.fillna(x.median()))

    for col in spend_cols:
        df[col] = df[col].fillna(0.0)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df['VIP']       = df['VIP'].map({True: 1, False: 0, 'True': 1, 'False': 0})

    for col in CAT_COLS:
        df[col] = df[col].astype(str)

    df['Surname'] = df['Name'].str.split().str[-1].fillna('Unknown')

    df['TotalSpend']         = df[spend_cols].sum(axis=1)
    df['NecessitiesSpend']   = df['RoomService'] + df['FoodCourt']
    df['EntertainmentSpend'] = df['ShoppingMall'] + df['Spa'] + df['VRDeck']

    group_cryo = df.groupby('GroupId')['CryoSleep'].mean().rename('_gcr')
    df = df.join(group_cryo, on='GroupId')
    df['AllGroupCryo'] = (df['_gcr'] == 1.0).astype(int)
    df['NoGroupCryo']  = (df['_gcr'] == 0.0).astype(int)
    df = df.drop(columns=['_gcr'])

    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    df['IsChild'] = (df['Age'] <= 12).astype(int)

    group_ids = df['GroupId'].copy()
    df = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'GroupId'])

    train_mask = df['is_train'] == 1
    base_train = df[train_mask].drop(columns=['is_train']).reset_index(drop=True)
    base_test  = df[~train_mask].drop(columns=['is_train', 'Transported']).reset_index(drop=True)
    base_train['Transported'] = base_train['Transported'].astype(int)
    groups_train = group_ids[train_mask].reset_index(drop=True).values

    return base_train, base_test, groups_train


def core_preprocess(train_raw, test_raw):
    base_train, base_test, _ = _preprocess(train_raw, test_raw)
    return base_train, base_test


def load_and_preprocess(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train_raw, test_raw = load_data(train_path, test_path)
    return _preprocess(train_raw, test_raw)


def encode_for_knn(base_train, base_test):
    train = base_train.copy()
    test  = base_test.copy()

    y = train['Transported']
    train = train.drop(columns=['Transported', 'Surname'])
    test  = test.drop(columns=['Surname'])

    for col in ['CryoSleep', 'VIP']:
        train[col] = train[col].fillna(0)
        test[col]  = test[col].fillna(0)

    num_cols = [c for c in train.columns if c not in CAT_COLS]

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    enc_train = ohe.transform(train[CAT_COLS])
    enc_test  = ohe.transform(test[CAT_COLS])

    X_num  = train[num_cols].values
    Xt_num = test[num_cols].values
    X_raw  = np.hstack([X_num, enc_train])
    Xt_raw = np.hstack([Xt_num, enc_test])

    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)
    X_test = scaler.transform(Xt_raw)

    return X, y, X_test


def encode_for_svm(base_train, base_test):
    train = base_train.copy()
    test  = base_test.copy()

    y = train['Transported']
    train = train.drop(columns=['Transported'])

    for col in ['CryoSleep', 'VIP']:
        train[col] = train[col].fillna(0)
        test[col]  = test[col].fillna(0)

    svm_cat_cols = CAT_COLS + ['Surname']
    for col in svm_cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    scaler = StandardScaler()
    X      = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    X_test = pd.DataFrame(scaler.transform(test),      columns=test.columns)

    return X, y, X_test


def encode_for_rf(base_train, base_test):
    """OHE for categoricals, no scaling, Surname dropped. Returns DataFrames with column names."""
    train = base_train.copy()
    test  = base_test.copy()

    for col in ['CryoSleep', 'VIP']:
        train[col] = train[col].fillna(0)
        test[col]  = test[col].fillna(0)

    y     = train['Transported']
    train = train.drop(columns=['Transported', 'Surname'])
    test  = test.drop(columns=['Surname'])

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(pd.concat([train[CAT_COLS], test[CAT_COLS]], axis=0))
    ohe_feat_names = ohe.get_feature_names_out(CAT_COLS)

    ohe_tr = pd.DataFrame(ohe.transform(train[CAT_COLS]), columns=ohe_feat_names)
    ohe_te = pd.DataFrame(ohe.transform(test[CAT_COLS]),  columns=ohe_feat_names)

    num_cols = [c for c in train.columns if c not in CAT_COLS]
    X      = pd.concat([train[num_cols].reset_index(drop=True), ohe_tr], axis=1)
    X_test = pd.concat([test[num_cols].reset_index(drop=True),  ohe_te], axis=1)

    return X, y.reset_index(drop=True), X_test


def encode_for_mlp(base_train, base_test):
    """LabelEncode cats (for embedding) + StandardScale nums. Returns DataFrames + col lists."""
    train = base_train.copy()
    test  = base_test.copy()

    for col in ['CryoSleep', 'VIP']:
        train[col] = train[col].fillna(0)
        test[col]  = test[col].fillna(0)

    y     = train['Transported']
    train = train.drop(columns=['Transported'])

    mlp_cat_cols = CAT_COLS + ['Surname']
    for col in mlp_cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    num_cols = [c for c in train.columns if c not in mlp_cat_cols]
    scaler = StandardScaler()
    train_num = pd.DataFrame(scaler.fit_transform(train[num_cols]), columns=num_cols)
    test_num  = pd.DataFrame(scaler.transform(test[num_cols]),       columns=num_cols)

    X      = pd.concat([train_num, train[mlp_cat_cols].reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_num,  test[mlp_cat_cols].reset_index(drop=True)],  axis=1)

    return X, y.reset_index(drop=True), X_test, mlp_cat_cols, num_cols


if __name__ == '__main__':
    base_train, base_test, groups = load_and_preprocess()
    print(f"base_train: {base_train.shape}, base_test: {base_test.shape}")
    print(f"Columns: {list(base_train.columns)}")
