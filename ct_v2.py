"""
ct_v2.py — CatBoost best-parameter version
===================================
Parameters: lr=0.03, l2=2, rsm=1.0, depth=6(LB 0.81529)
Outputs: oof_ct_v2.npy / test_ct_v2.npy
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import accuracy_score

from preprocessing import load_and_preprocess
from catboost_features import make_catboost_data

RANDOM_SEED = 42
TRAIN_PATH  = "spaceship-titanic/train.csv"
TEST_PATH   = "spaceship-titanic/test.csv"

base_train, base_test, groups = load_and_preprocess(TRAIN_PATH, TEST_PATH)
X, y, X_test, cat_features = make_catboost_data(base_train, base_test)
gkf = GroupKFold(n_splits=5)

params = dict(
    iterations       = 1000,
    learning_rate    = 0.03,
    depth            = 6,
    l2_leaf_reg      = 2,
    rsm              = 1.0,
    min_data_in_leaf = 1,
    random_seed      = RANDOM_SEED,
    verbose          = 0,
)

print("--- CT v2 GroupKFold CV (lr=0.03, l2=2) ---")
oof_proba   = np.zeros(len(X))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
    m = CatBoostClassifier(**params)
    m.fit(Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_features),
          eval_set=Pool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_features),
          early_stopping_rounds=50)
    proba = m.predict_proba(X.iloc[va_idx])[:, 1]
    oof_proba[va_idx] = proba
    acc = accuracy_score(y.iloc[va_idx], (proba >= 0.5).astype(int))
    fold_scores.append(acc)
    print(f"  Fold {fold+1}: acc={acc:.4f}")

cv_mean = np.mean(fold_scores)
cv_std  = np.std(fold_scores)
print(f"\nCT v2 CV: {cv_mean:.4f} ± {cv_std:.4f}")
np.save("oof_ct_v2.npy", oof_proba)

print("\n--- Full retraining ---")
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
cb = CatBoostClassifier(**params)
cb.fit(Pool(X_tr, y_tr, cat_features=cat_features),
       eval_set=Pool(X_val, y_val, cat_features=cat_features),
       early_stopping_rounds=50)
best_iter = cb.best_iteration_
print(f"best_iter: {best_iter}")

cb_final = CatBoostClassifier(**{**params, 'iterations': best_iter})
cb_final.fit(Pool(X, y, cat_features=cat_features))
test_proba = cb_final.predict_proba(Pool(X_test, cat_features=cat_features))[:, 1]
np.save("test_ct_v2.npy", test_proba)

print(f"\nSaved: oof_ct_v2.npy, test_ct_v2.npy")
