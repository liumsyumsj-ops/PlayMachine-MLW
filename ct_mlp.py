"""
ct_mlp.py — sklearn MLP (256,128,64) OOF/test probability generation
========================================================
solver=adam(default), lr_init=0.001, early_stopping=True
Outputs: oof_MLP-wide.npy / test_mlp.npy(for integration in final_submission.py)
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

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
print(f"Number of features after encoding: {X_enc.shape[1]}")

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('mlp',     MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_SEED,
    )),
])

# ── GroupKFold CV ─────────────────────────────────────────────
print("\n--- MLP GroupKFold CV (5 folds) ---")
gkf = GroupKFold(n_splits=5)
oof_mlp   = np.zeros(len(X_enc))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_enc, y, groups)):
    pipe.fit(X_enc.iloc[tr_idx], y.iloc[tr_idx])
    proba = pipe.predict_proba(X_enc.iloc[va_idx])[:, 1]
    oof_mlp[va_idx] = proba
    acc = accuracy_score(y.iloc[va_idx], (proba >= 0.5).astype(int))
    fold_scores.append(acc)
    print(f"  Fold {fold+1}: acc={acc:.4f}")

print(f"\nMLP CV : {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
np.save("oof_MLP-wide.npy", oof_mlp)

# ── Train MLP on the full dataset ──────────────────────────────────────────────
print("\n--- Train MLP on the full dataset ---")
pipe.fit(X_enc, y)
mlp_test_proba = pipe.predict_proba(X_test_enc)[:, 1]
np.save("test_mlp.npy", mlp_test_proba)

print(f"\n====== ct_mlp completed ======")
print(f"MLP CV   : {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"Output files: oof_MLP-wide.npy / test_mlp.npy")
