"""
xgb_best.py — XGBoost Model
=============================
Features: ct9 feature set + one-hot encoded categorical features
Training: GroupKFold 5-fold CV + holdout early stopping + full retraining
Parameters: n_estimators=1000, lr=0.05, max_depth=6, subsample=0.8, colsample=0.8
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

from preprocessing import load_and_preprocess
from catboost_features import make_catboost_data

RANDOM_SEED = 42
TRAIN_PATH  = "spaceship-titanic/train.csv"
TEST_PATH   = "spaceship-titanic/test.csv"

# ── Data ──────────────────────────────────────────────────────
base_train, base_test, groups = load_and_preprocess(TRAIN_PATH, TEST_PATH)
X_raw, y, X_test_raw, cat_features = make_catboost_data(base_train, base_test)

X_enc      = pd.get_dummies(X_raw,      columns=cat_features)
X_test_enc = pd.get_dummies(X_test_raw, columns=cat_features)
X_test_enc = X_test_enc.reindex(columns=X_enc.columns, fill_value=0)
print(f"Number of features after one-hot encoding: {X_enc.shape[1]}")

XGB_PARAMS = dict(
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = 'error',
    random_state     = RANDOM_SEED,
    tree_method      = 'hist',
    verbosity        = 0,
)

# ── GroupKFold CV ─────────────────────────────────────────────
print("\n--- GroupKFold CV (5 folds) ---")
gkf = GroupKFold(n_splits=5)
oof_proba   = np.zeros(len(X_enc))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_enc, y, groups)):
    m = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=50)
    m.fit(X_enc.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X_enc.iloc[va_idx], y.iloc[va_idx])],
          verbose=False)
    proba = m.predict_proba(X_enc.iloc[va_idx])[:, 1]
    oof_proba[va_idx] = proba
    acc = accuracy_score(y.iloc[va_idx], (proba >= 0.5).astype(int))
    fold_scores.append(acc)
    print(f"  Fold {fold+1}: acc={acc:.4f}, best_iter={m.best_iteration}")

fold_scores = np.array(fold_scores)
print(f"\nXGB GroupKFold CV : {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
np.save("oof_xgb.npy", oof_proba)

oof_ct9 = np.load("oof_ct_v2.npy")
r, _ = pearsonr(oof_proba, oof_ct9)
print(f"XGB vs CT correlation: {r:.4f}")

# ── Full retraining: holdout early stopping -> fixed iteration ──
print("\n--- Full retraining (seed=42) ---")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_enc, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
cb = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=50)
cb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_iter = cb.best_iteration
val_acc   = accuracy_score(y_val, cb.predict(X_val))
print(f"Holdout val acc: {val_acc:.4f}, best_iter: {best_iter}")

m_final = xgb.XGBClassifier(**{**XGB_PARAMS, 'n_estimators': best_iter})
m_final.fit(X_enc, y)

# ── Prediction ─────────────────────────────────────────────────
test_proba = m_final.predict_proba(X_test_enc)[:, 1]
np.save("test_xgb.npy", test_proba)

print(f"\n====== xgb_best completed ======")
print(f"GroupKFold CV : {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
print(f"Holdout val   : {val_acc:.4f}, best_iter={best_iter}")
print(f"XGB vs CT correlation: {r:.4f}")
print(f"Saved: oof_xgb.npy, test_xgb.npy")
