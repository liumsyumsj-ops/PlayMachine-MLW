"""
final_submission.py — Final submission file generation
========================================
Pipeline:
  1. Run ct_v2.py        -> oof_ct_v2.npy  / test_ct_v2.npy
  2. Run ct_mlp.py       -> oof_MLP-wide.npy / test_mlp.npy
  3. Run lgb_grid_v2.py  -> oof_lgb_base.npy / test_lgb_base.npy
  4. Run this file           -> submission_final.csv(LB 0.81786)

Ensemble weights: 65/99 CatBoost + 25/99 MLP + 9/99 LightGBM(normalized weights sum to 1)
Post-processing: Earth + CryoSleep=True + TRAPPIST-1e -> force Transported=True
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

TRAIN_PATH = "spaceship-titanic/train.csv"
TEST_PATH  = "spaceship-titanic/test.csv"

# ── Load OOF and test-set probabilities ──────────────────────────────────────
oof_ct  = np.load("oof_ct_v2.npy")
oof_mlp = np.load("oof_MLP-wide.npy")
oof_lgb = np.load("oof_lgb_base.npy")

test_ct  = np.load("test_ct_v2.npy")
test_mlp = np.load("test_mlp.npy")
test_lgb = np.load("test_lgb_base.npy")

# ── Ensemble weights ──────────────────────────────────────────────────
W_CT, W_MLP, W_LGB = 65/99, 25/99, 9/99

# ── OOF accuracy (validate ensemble performance)────────────────────────────────
train_raw = pd.read_csv(TRAIN_PATH)
y = train_raw['Transported'].astype(int).values

oof_blend = W_CT*oof_ct + W_MLP*oof_mlp + W_LGB*oof_lgb
oof_acc = accuracy_score(y, (oof_blend >= 0.5).astype(int))
print(f"Ensemble OOF accuracy: {oof_acc:.4f}  (target ~ 0.8131, LB 0.81786)")

# ── Test-set ensemble probabilities ─────────────────────────────────────────────
test_raw   = pd.read_csv(TEST_PATH)
blend_test = W_CT*test_ct + W_MLP*test_mlp + W_LGB*test_lgb
pred       = (blend_test >= 0.5)

# ── Post-processing: Earth + CryoSleep=True + TRAPPIST-1e -> True ───────
cryo = test_raw['CryoSleep'].map(
    {True: True, False: False, 'True': True, 'False': False}
).fillna(False).infer_objects(copy=False)

mask = (
    (test_raw['HomePlanet'] == 'Earth') &
    cryo &
    (test_raw['Destination'] == 'TRAPPIST-1e') &
    (~pred)
)
pred[mask] = True
print(f"Post-processing flips: {mask.sum()} rows(Earth+CryoSleep+TRAPPIST-1e)")

# ── Create submission file ───────────────────────────────────────────────
out = pd.DataFrame({
    'PassengerId': test_raw['PassengerId'],
    'Transported': pred,
})
out.to_csv("spaceship-titanic/submission_final.csv", index=False)
print(f"Generated: spaceship-titanic/submission_final.csv")
print(f"Transported=True ratio: {pred.mean():.4f}")
