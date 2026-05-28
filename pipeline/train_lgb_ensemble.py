"""LightGBM lgb_base for ensemble (same params as lgb_grid_v2 baseline)."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lgb_features import make_lgb_data
from preprocessing import load_and_preprocess

RANDOM_SEED = 42
BASE_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    reg_lambda=1.0,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    subsample_freq=1,
    objective="binary",
    metric="binary_error",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=-1,
)


def main():
    tr = ROOT / "spaceship-titanic" / "train.csv"
    te = ROOT / "spaceship-titanic" / "test.csv"
    base_train, base_test, groups = load_and_preprocess(str(tr), str(te))
    X, y, X_test = make_lgb_data(base_train, base_test)
    gkf = GroupKFold(n_splits=5)

    oof_proba = np.zeros(len(X))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        m = lgb.LGBMClassifier(**BASE_PARAMS)
        m.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        proba = m.predict_proba(X.iloc[va_idx])[:, 1]
        oof_proba[va_idx] = proba
        print(f"  LGB fold {fold + 1}: {accuracy_score(y.iloc[va_idx], (proba >= 0.5).astype(int)):.4f}")

    np.save(ROOT / "oof_lgb_base.npy", oof_proba)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    m_hold = lgb.LGBMClassifier(**BASE_PARAMS)
    m_hold.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    m_final = lgb.LGBMClassifier(**{**BASE_PARAMS, "n_estimators": m_hold.best_iteration_})
    m_final.fit(X, y)
    test_proba = m_final.predict_proba(X_test)[:, 1]
    np.save(ROOT / "test_lgb_base.npy", test_proba)

    print("Saved: oof_lgb_base.npy, test_lgb_base.npy")


if __name__ == "__main__":
    main()
