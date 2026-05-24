"""
xgb_grid.py — XGBoost 第一轮参数矩阵搜索
======================================

功能：
1. 使用 preprocessing_shared.py + catboost_features.py 生成特征
2. 对约 15 组 XGBoost 参数做 GroupKFold CV
3. 每组保存：
   - oof_xgb_<id>.npy
   - test_xgb_<id>.npy
   - submission_xgb_<id>.csv
4. 汇总保存：
   - xgb_grid_results.csv

注意：
这里不是 720 个全排列，而是根据给定范围挑选 15 个代表性组合。
第一轮目标是观察方向，不是暴力搜索。
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

from preprocessing_shared import load_and_preprocess
from catboost_features import make_catboost_data


warnings.filterwarnings("ignore")


# ============================================================
# 0. Basic Config
# ============================================================

RANDOM_SEED = 42

TRAIN_PATH = "spaceship-titanic/train.csv"
TEST_PATH = "spaceship-titanic/test.csv"

N_SPLITS = 5
THRESHOLD = 0.5

OUTPUT_DIR = "xgb_grid_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. First-round parameter matrix
# ============================================================
# Given ranges:
# max_depth: 4, 5, 6, 7, 8
# min_child_weight: 5, 10, 15, 20
# colsample_bytree: 0.6, 0.7, 0.8, 0.9
# subsample: 0.6, 0.7, 0.8
# reg_lambda: 2, 5, 10
#
# Full Cartesian product would be 5*4*4*3*3 = 720 combinations.
# This first round uses 15 representative combinations.
PARAM_GRID = [
    # 013 原始参数，作为 baseline 保留
    {
        "id": "201",
        "max_depth": 7,
        "min_child_weight": 20,
        "colsample_bytree": 0.9,
        "subsample": 0.7,
        "reg_lambda": 10,
    },

    # 调 max_depth：看 6/7/8 哪个更稳
    {
        "id": "202",
        "max_depth": 6,
        "min_child_weight": 20,
        "colsample_bytree": 0.9,
        "subsample": 0.7,
        "reg_lambda": 10,
    },
    {
        "id": "203",
        "max_depth": 8,
        "min_child_weight": 20,
        "colsample_bytree": 0.9,
        "subsample": 0.7,
        "reg_lambda": 10,
    },

    # 调 min_child_weight：看强正则是否还能更强
    {
        "id": "204",
        "max_depth": 7,
        "min_child_weight": 15,
        "colsample_bytree": 0.9,
        "subsample": 0.7,
        "reg_lambda": 10,
    },
    {
        "id": "205",
        "max_depth": 7,
        "min_child_weight": 25,
        "colsample_bytree": 0.9,
        "subsample": 0.7,
        "reg_lambda": 10,
    },
    {
        "id": "206",
        "max_depth": 7,
        "min_child_weight": 30,
        "colsample_bytree": 0.9,
        "subsample": 0.7,
        "reg_lambda": 10,
    },

    # 调 colsample_bytree：013 的 0.9 很高，看看 0.85/0.95
    {
        "id": "207",
        "max_depth": 7,
        "min_child_weight": 20,
        "colsample_bytree": 0.85,
        "subsample": 0.7,
        "reg_lambda": 10,
    },
    {
        "id": "208",
        "max_depth": 7,
        "min_child_weight": 20,
        "colsample_bytree": 0.95,
        "subsample": 0.7,
        "reg_lambda": 10,
    },

    # 调 subsample：看 0.65/0.75 是否更适合 LB
    {
        "id": "209",
        "max_depth": 7,
        "min_child_weight": 20,
        "colsample_bytree": 0.9,
        "subsample": 0.65,
        "reg_lambda": 10,
    },
    {
        "id": "210",
        "max_depth": 7,
        "min_child_weight": 20,
        "colsample_bytree": 0.9,
        "subsample": 0.75,
        "reg_lambda": 10,
    },

    # 调 reg_lambda：看 L2 正则是否需要更强
    {
        "id": "211",
        "max_depth": 7,
        "min_child_weight": 20,
        "colsample_bytree": 0.9,
        "subsample": 0.7,
        "reg_lambda": 8,
    },
    {
        "id": "212",
        "max_depth": 7,
        "min_child_weight": 20,
        "colsample_bytree": 0.9,
        "subsample": 0.7,
        "reg_lambda": 12,
    },
    {
        "id": "213",
        "max_depth": 7,
        "min_child_weight": 20,
        "colsample_bytree": 0.9,
        "subsample": 0.7,
        "reg_lambda": 15,
    },

    # 组合微调：保留深树 + 强正则方向
    {
        "id": "214",
        "max_depth": 7,
        "min_child_weight": 25,
        "colsample_bytree": 0.85,
        "subsample": 0.7,
        "reg_lambda": 12,
    },
    {
        "id": "215",
        "max_depth": 8,
        "min_child_weight": 25,
        "colsample_bytree": 0.85,
        "subsample": 0.65,
        "reg_lambda": 15,
    },
]
# PARAM_GRID = [
#     # Conservative shallow models
#     {
#         "id": "001",
#         "max_depth": 4,
#         "min_child_weight": 5,
#         "colsample_bytree": 0.8,
#         "subsample": 0.8,
#         "reg_lambda": 2,
#     },
#     {
#         "id": "002",
#         "max_depth": 4,
#         "min_child_weight": 10,
#         "colsample_bytree": 0.7,
#         "subsample": 0.8,
#         "reg_lambda": 5,
#     },
#     {
#         "id": "003",
#         "max_depth": 4,
#         "min_child_weight": 15,
#         "colsample_bytree": 0.6,
#         "subsample": 0.7,
#         "reg_lambda": 10,
#     },

#     # Medium-depth stable models
#     {
#         "id": "004",
#         "max_depth": 5,
#         "min_child_weight": 5,
#         "colsample_bytree": 0.8,
#         "subsample": 0.8,
#         "reg_lambda": 2,
#     },
#     {
#         "id": "005",
#         "max_depth": 5,
#         "min_child_weight": 10,
#         "colsample_bytree": 0.7,
#         "subsample": 0.7,
#         "reg_lambda": 5,
#     },
#     {
#         "id": "006",
#         "max_depth": 5,
#         "min_child_weight": 15,
#         "colsample_bytree": 0.9,
#         "subsample": 0.8,
#         "reg_lambda": 5,
#     },
#     {
#         "id": "007",
#         "max_depth": 5,
#         "min_child_weight": 20,
#         "colsample_bytree": 0.6,
#         "subsample": 0.6,
#         "reg_lambda": 10,
#     },

#     # Around current depth
#     {
#         "id": "008",
#         "max_depth": 6,
#         "min_child_weight": 5,
#         "colsample_bytree": 0.8,
#         "subsample": 0.8,
#         "reg_lambda": 2,
#     },
#     {
#         "id": "009",
#         "max_depth": 6,
#         "min_child_weight": 10,
#         "colsample_bytree": 0.7,
#         "subsample": 0.7,
#         "reg_lambda": 5,
#     },
#     {
#         "id": "010",
#         "max_depth": 6,
#         "min_child_weight": 15,
#         "colsample_bytree": 0.6,
#         "subsample": 0.8,
#         "reg_lambda": 10,
#     },

#     # Deeper but more regularized models
#     {
#         "id": "011",
#         "max_depth": 7,
#         "min_child_weight": 10,
#         "colsample_bytree": 0.8,
#         "subsample": 0.7,
#         "reg_lambda": 5,
#     },
#     {
#         "id": "012",
#         "max_depth": 7,
#         "min_child_weight": 15,
#         "colsample_bytree": 0.7,
#         "subsample": 0.6,
#         "reg_lambda": 10,
#     },
#     {
#         "id": "013",
#         "max_depth": 7,
#         "min_child_weight": 20,
#         "colsample_bytree": 0.9,
#         "subsample": 0.7,
#         "reg_lambda": 10,
#     },

#     # Very deep, heavily regularized models
#     {
#         "id": "014",
#         "max_depth": 8,
#         "min_child_weight": 15,
#         "colsample_bytree": 0.8,
#         "subsample": 0.7,
#         "reg_lambda": 10,
#     },
#     {
#         "id": "015",
#         "max_depth": 8,
#         "min_child_weight": 20,
#         "colsample_bytree": 0.6,
#         "subsample": 0.6,
#         "reg_lambda": 10,
#     },
# ]


# ============================================================
# 2. Fixed XGBoost parameters
# ============================================================

BASE_XGB_PARAMS = dict(
    n_estimators=1200,
    learning_rate=0.05,
    eval_metric="error",
    random_state=RANDOM_SEED,
    tree_method="hist",
    verbosity=0,
    n_jobs=-1,
)


# ============================================================
# 3. Helper functions
# ============================================================

def save_submission(test_path, test_proba, out_path, threshold=0.5):
    test_raw = pd.read_csv(test_path)

    submission = pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Transported": (test_proba >= threshold).astype(bool),
    })

    submission.to_csv(out_path, index=False)
    return submission


def safe_corr_with_ct(oof_proba):
    """
    Optional:
    If oof_ct9.npy exists, calculate correlation with CatBoost OOF.
    If not, return NaN.
    """
    if not os.path.exists("oof_ct9.npy"):
        return np.nan

    try:
        oof_ct9 = np.load("oof_ct9.npy")
        if len(oof_ct9) != len(oof_proba):
            return np.nan
        r, _ = pearsonr(oof_proba, oof_ct9)
        return float(r)
    except Exception:
        return np.nan


def run_one_param_set(param_set, X, y, X_test, groups):
    model_id = param_set["id"]

    xgb_params = {
        **BASE_XGB_PARAMS,
        "max_depth": param_set["max_depth"],
        "min_child_weight": param_set["min_child_weight"],
        "colsample_bytree": param_set["colsample_bytree"],
        "subsample": param_set["subsample"],
        "reg_lambda": param_set["reg_lambda"],
    }

    print("\n" + "=" * 80)
    print(f"Running XGB {model_id}")
    print(json.dumps(xgb_params, indent=2))
    print("=" * 80)

    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_proba = np.zeros(len(X))
    test_fold_probas = []

    fold_scores = []
    fold_best_iters = []

    start_time = time.perf_counter()

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        model = xgb.XGBClassifier(
            **xgb_params,
            early_stopping_rounds=50,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        val_proba = model.predict_proba(X_va)[:, 1]
        val_pred = (val_proba >= THRESHOLD).astype(int)

        fold_acc = accuracy_score(y_va, val_pred)
        fold_scores.append(fold_acc)

        best_iter = model.best_iteration + 1 if model.best_iteration is not None else xgb_params["n_estimators"]
        fold_best_iters.append(best_iter)

        oof_proba[va_idx] = val_proba

        test_fold_proba = model.predict_proba(X_test)[:, 1]
        test_fold_probas.append(test_fold_proba)

        print(
            f"Fold {fold}: "
            f"acc={fold_acc:.5f}, "
            f"best_iter={best_iter}"
        )

    elapsed = time.perf_counter() - start_time

    fold_scores = np.array(fold_scores)
    fold_best_iters = np.array(fold_best_iters)

    oof_pred = (oof_proba >= THRESHOLD).astype(int)
    oof_acc = accuracy_score(y, oof_pred)

    test_proba = np.mean(test_fold_probas, axis=0)

    pred_true_ratio = float(np.mean(test_proba >= THRESHOLD))
    oof_true_ratio = float(np.mean(oof_pred == 1))

    corr_ct = safe_corr_with_ct(oof_proba)

    # Save files
    oof_path = os.path.join(OUTPUT_DIR, f"oof_xgb_{model_id}.npy")
    test_path = os.path.join(OUTPUT_DIR, f"test_xgb_{model_id}.npy")
    sub_path = os.path.join(OUTPUT_DIR, f"submission_xgb_{model_id}.csv")

    np.save(oof_path, oof_proba)
    np.save(test_path, test_proba)
    submission = save_submission(TEST_PATH, test_proba, sub_path, threshold=THRESHOLD)

    print(f"\nXGB {model_id} finished")
    print(f"OOF Accuracy      : {oof_acc:.5f}")
    print(f"Fold mean Accuracy: {fold_scores.mean():.5f} ± {fold_scores.std():.5f}")
    print(f"Mean best_iter    : {fold_best_iters.mean():.1f}")
    print(f"OOF True Ratio    : {oof_true_ratio:.5f}")
    print(f"Test True Ratio   : {pred_true_ratio:.5f}")
    print(f"CT correlation    : {corr_ct:.5f}" if not np.isnan(corr_ct) else "CT correlation    : N/A")
    print(f"Time seconds      : {elapsed:.2f}")
    print(f"Saved: {oof_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {sub_path}")
    print("Submission distribution:")
    print(submission["Transported"].value_counts())

    result = {
        "id": model_id,
        "oof_acc": oof_acc,
        "fold_mean_acc": float(fold_scores.mean()),
        "fold_std_acc": float(fold_scores.std()),
        "fold_1": float(fold_scores[0]),
        "fold_2": float(fold_scores[1]),
        "fold_3": float(fold_scores[2]),
        "fold_4": float(fold_scores[3]),
        "fold_5": float(fold_scores[4]),
        "mean_best_iter": float(fold_best_iters.mean()),
        "min_best_iter": int(fold_best_iters.min()),
        "max_best_iter": int(fold_best_iters.max()),
        "oof_true_ratio": oof_true_ratio,
        "test_true_ratio": pred_true_ratio,
        "ct_corr": corr_ct,
        "time_seconds": elapsed,
        "oof_path": oof_path,
        "test_path": test_path,
        "submission_path": sub_path,
        **param_set,
    }

    return result


# ============================================================
# 4. Main
# ============================================================

def main():
    print("Loading and preprocessing data...")

    base_train, base_test, groups = load_and_preprocess(TRAIN_PATH, TEST_PATH)

    X_raw, y, X_test_raw, cat_features = make_catboost_data(base_train, base_test)

    X = pd.get_dummies(X_raw, columns=cat_features)
    X_test = pd.get_dummies(X_test_raw, columns=cat_features)

    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    print(f"Train shape after one-hot: {X.shape}")
    print(f"Test shape after one-hot : {X_test.shape}")
    print(f"Target distribution:")
    print(y.value_counts(normalize=True))

    all_results = []

    for param_set in PARAM_GRID:
        result = run_one_param_set(param_set, X, y, X_test, groups)
        all_results.append(result)

        # Save intermediate results after every run
        temp_df = pd.DataFrame(all_results)
        temp_df = temp_df.sort_values("oof_acc", ascending=False)
        temp_df.to_csv(os.path.join(OUTPUT_DIR, "xgb_grid_results_partial.csv"), index=False)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("oof_acc", ascending=False)

    final_result_path = os.path.join(OUTPUT_DIR, "xgb_grid_results.csv")
    results_df.to_csv(final_result_path, index=False)

    print("\n" + "=" * 80)
    print("XGB GRID SEARCH FINISHED")
    print("=" * 80)

    display_cols = [
        "id",
        "oof_acc",
        "fold_mean_acc",
        "fold_std_acc",
        "mean_best_iter",
        "max_depth",
        "min_child_weight",
        "colsample_bytree",
        "subsample",
        "reg_lambda",
        "test_true_ratio",
        "ct_corr",
        "time_seconds",
        "submission_path",
    ]

    print(results_df[display_cols].to_string(index=False))

    print(f"\nSaved summary: {final_result_path}")

    best = results_df.iloc[0]
    print("\nBest OOF parameter set:")
    print(best[display_cols].to_string())


if __name__ == "__main__":
    main()