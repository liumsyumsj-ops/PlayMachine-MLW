"""Final ensemble CSV (CT65 + MLP25 + LGB9 + post-processing)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "submissions" / "submission_final.csv"

W_CT, W_MLP, W_LGB = 65 / 99, 25 / 99, 9 / 99


def main():
    oof_ct = np.load(ROOT / "oof_ct_v2.npy")
    oof_mlp = np.load(ROOT / "oof_MLP-wide.npy")
    oof_lgb = np.load(ROOT / "oof_lgb_base.npy")
    test_ct = np.load(ROOT / "test_ct_v2.npy")
    test_mlp = np.load(ROOT / "test_mlp.npy")
    test_lgb = np.load(ROOT / "test_lgb_base.npy")

    train_raw = pd.read_csv(ROOT / "spaceship-titanic" / "train.csv")
    y = train_raw["Transported"].astype(int).values
    oof_blend = W_CT * oof_ct + W_MLP * oof_mlp + W_LGB * oof_lgb
    print(f"Ensemble OOF accuracy: {accuracy_score(y, (oof_blend >= 0.5).astype(int)):.4f}")

    test_raw = pd.read_csv(ROOT / "spaceship-titanic" / "test.csv")
    blend_test = W_CT * test_ct + W_MLP * test_mlp + W_LGB * test_lgb
    pred = (blend_test >= 0.5)

    cryo = test_raw["CryoSleep"].map(
        {True: True, False: False, "True": True, "False": False}
    ).fillna(False).infer_objects(copy=False)
    mask = (
        (test_raw["HomePlanet"] == "Earth")
        & cryo
        & (test_raw["Destination"] == "TRAPPIST-1e")
        & (~pred)
    )
    pred[mask] = True
    print(f"Post-processing flips: {mask.sum()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Transported": pred,
    }).to_csv(OUT, index=False)
    also = ROOT / "spaceship-titanic" / "submission_final.csv"
    pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Transported": pred,
    }).to_csv(also, index=False)
    print(f"Saved: {OUT}")
    print(f"Saved: {also}")


if __name__ == "__main__":
    main()
