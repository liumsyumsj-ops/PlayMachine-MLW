"""
lgb_grid_v2.py — LightGBM one-parameter-at-a-time controlled search (round 2)
=======================================================
Method: change only one parameter at a time, keep the others fixed, and verify each direction.
Ensemble evaluation: CT65% + MLP25% + LGB9% (current best, LB 0.81669)
  - Replace lgb_base in the ensemble with the new LGB and observe the ensemble OOF change
  - Fixed weights: CT=65/99, MLP=25/99, LGB=9/99

Baseline LGB: lgb_base(num_leaves=31, lr=0.05, lambda=1, sub=0.8, col=0.8, mcw=20)
Current best ensemble LB: 0.81669

Search grid: narrow the range from round 1 and focus on promising directions
  num_leaves:        [31, 47, 63, 95, 127]
  learning_rate:     [0.03, 0.05, 0.08, 0.10]
  reg_lambda:        [0.1, 0.5, 1.0, 2.0]
  min_child_samples: [10, 20, 40, 80]
  subsample:         [0.7, 0.8, 0.9, 1.0]
  colsample_bytree:  [0.7, 0.8, 0.9, 1.0]

Usage: python3 lgb_grid_v2.py
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import accuracy_score

from preprocessing import load_and_preprocess
from lgb_features import make_lgb_data

RANDOM_SEED = 42
TRAIN_PATH  = "spaceship-titanic/train.csv"
TEST_PATH   = "spaceship-titanic/test.csv"

# ── Baseline LGB Parameter ─────────────────────────────────────────────
BASE_PARAMS = dict(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    num_leaves        = 31,
    reg_lambda        = 1.0,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    subsample_freq    = 1,
    objective         = 'binary',
    metric            = 'binary_error',
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    verbose           = -1,
)

# ── Search grid ──────────────────────────────────────────────────
PARAM_GROUPS = [
    ('num_leaves',        [31, 47, 63, 95, 127]),
    ('learning_rate',     [0.03, 0.05, 0.08, 0.10]),
    ('reg_lambda',        [0.1, 0.5, 1.0, 2.0]),
    ('min_child_samples', [10, 20, 40, 80]),
    ('subsample',         [0.7, 0.8, 0.9, 1.0]),
    ('colsample_bytree',  [0.7, 0.8, 0.9, 1.0]),
]

COMBOS = []
for param_name, values in PARAM_GROUPS:
    for v in values:
        if BASE_PARAMS.get(param_name) == v:
            continue
        COMBOS.append((param_name, v))

# ── Data ─────────────────────────────────────────────────────
base_train, base_test, groups = load_and_preprocess(TRAIN_PATH, TEST_PATH)
X, y, X_test = make_lgb_data(base_train, base_test)
print(f"Number of features: {X.shape[1]}")

gkf      = GroupKFold(n_splits=5)
test_raw = pd.read_csv(TEST_PATH)
y_arr    = y.values

# ── Ensemble baseline: CT65% + MLP25% + LGB9% ─────────────────────────
W_CT, W_MLP, W_LGB = 65/99, 25/99, 9/99
oof_ct   = np.load("oof_ct_v2.npy")
oof_mlp  = np.load("oof_MLP-wide.npy")
test_ct  = np.load("test_ct_v2.npy")
test_mlp = np.load("test_mlp.npy")

# ── Helper: run one LGB configuration ─────────────────────────────────────
def run_lgb(params, name):
    t0 = time.time()
    oof_proba   = np.zeros(len(X))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        proba = m.predict_proba(X.iloc[va_idx])[:, 1]
        oof_proba[va_idx] = proba
        fold_scores.append(accuracy_score(y.iloc[va_idx], (proba >= 0.5).astype(int)))

    cv_mean = np.mean(fold_scores)
    cv_std  = np.std(fold_scores)

    # Ensemble OOF: CT65% + MLP25% + new LGB x 9%
    blend_acc = accuracy_score(y_arr,
        (W_CT*oof_ct + W_MLP*oof_mlp + W_LGB*oof_proba >= 0.5).astype(int))
    delta = blend_acc - blend_base_acc

    # Full retraining
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    m_hold = lgb.LGBMClassifier(**params)
    m_hold.fit(X_tr, y_tr,
               eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(50, verbose=False),
                          lgb.log_evaluation(-1)])
    best_iter = m_hold.best_iteration_

    m_final = lgb.LGBMClassifier(**{**params, 'n_estimators': best_iter})
    m_final.fit(X, y)
    test_proba = m_final.predict_proba(X_test)[:, 1]

    np.save(f"oof_{name}.npy",  oof_proba)
    np.save(f"test_{name}.npy", test_proba)

    # Create submission file
    blend_test = W_CT*test_ct + W_MLP*test_mlp + W_LGB*test_proba
    pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Transported': (blend_test >= 0.5),
    }).to_csv(f"spaceship-titanic/submission_{name}.csv", index=False)

    elapsed = time.time() - t0
    return cv_mean, cv_std, blend_acc, delta, best_iter, elapsed

# ── Run the baseline first and save as oof_lgb_base.npy / test_lgb_base.npy ─────
blend_base_acc = 0  # placeholder, updated after the base run finishes
print("=" * 75)
print(f"[baseline] num_leaves=31, lr=0.05, lambda=1.0, sub=0.8, col=0.8, mcw=20")
cv_base, cv_std_base, b_acc_base, _, iter_base, t_base = run_lgb(BASE_PARAMS, "lgb_base")
blend_base_acc = b_acc_base
print(f"  CV={cv_base:.4f}±{cv_std_base:.4f}  Ensemble OOF={b_acc_base:.4f}  iter={iter_base}  [{t_base:.0f}s]")
print()

results = [("lgb_base(baseline)", "—", cv_base, cv_std_base, b_acc_base, 0.0, iter_base)]

# ── Main loop ────────────────────────────────────────────────────
for param_name, val in COMBOS:
    params  = {**BASE_PARAMS, param_name: val}
    name    = f"lgb_v2_{param_name}{str(val).replace('.', '')}"
    tag_str = f"{param_name}={val}"
    print(f"[{tag_str}]")

    cv, cv_std, b_acc, delta, best_iter, elapsed = run_lgb(params, name)
    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else ' ')
    print(f"  CV={cv:.4f}±{cv_std:.4f}  Ensemble OOF={b_acc:.4f}({arrow}{abs(delta):.4f})  "
          f"iter={best_iter}  [{elapsed:.0f}s]\n")
    results.append((name, tag_str, cv, cv_std, b_acc, delta, best_iter))

# ── Summary table (sorted by ensemble OOF)──────────────────────────────────────
results_sorted = sorted(results[1:], key=lambda x: -x[4])

print("\n" + "="*80)
print(f"{'Name':<35} {'Parameter':>22} {'CV':>7} {'std':>6} {'Ensemble OOF':>9} {'Δ':>8} {'iter':>6}")
print("-"*80)
print(f"{'lgb_base(baseline)':<35} {'—':>22} {cv_base:.4f}  {cv_std_base:.4f}  {b_acc_base:.4f}    0.0000  {iter_base:>6}")
print("-"*80)
for name, tag, cv, cv_std, b_acc, delta, bi in results_sorted:
    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else ' ')
    best_mark = " ← best" if name == results_sorted[0][0] else ""
    print(f"{name:<35} {str(tag):>22} {cv:.4f}  {cv_std:.4f}  {b_acc:.4f}  {arrow}{abs(delta):.4f}  {bi:>6}{best_mark}")
print("="*80)

# ── Summary by parameter direction ────────────────────────────────────────────
print("\n── Summary by parameter direction ───────────────────────────────────────")
for param_name, values in PARAM_GROUPS:
    relevant = [(r[1], r[2], r[4], r[5]) for r in results[1:]
                if r[1].startswith(f"{param_name}=")]
    if not relevant:
        continue
    print(f"\n{param_name}(baseline={BASE_PARAMS[param_name]}):")
    for tag, cv, b_acc, delta in relevant:
        arrow = '↑' if delta > 0 else ('↓' if delta < 0 else ' ')
        print(f"  {tag:<32} CV={cv:.4f}  Ensemble OOF={b_acc:.4f}({arrow}{abs(delta):.4f})")

print(f"\nNote: Ensemble OOF = CT65%+MLP25%+new LGBx9%, compared with the baseline ensemble(CT65+MLP25+lgb_basex9%)")
print(f"    Current best LB: 0.81669(CT65+MLP25+lgb_basex9%)")
print(f"    Submission files generated to spaceship-titanic/submission_lgb_v2_*.csv")
