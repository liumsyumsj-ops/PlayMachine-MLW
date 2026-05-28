"""Prepare csv paths expected by legacy scripts (no changes to source models)."""

from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from preprocessing import load_and_preprocess

ST = ROOT / "spaceship-titanic"
TRAIN = ST / "train.csv"
TEST = ST / "test.csv"


def main():
    for name, src in [("train.csv", TRAIN), ("test.csv", TEST)]:
        dst = ROOT / name
        if not dst.exists():
            dst.symlink_to(src.resolve())

    base_train, base_test, _ = load_and_preprocess(str(TRAIN), str(TEST))
    # svm.py reads these csv files directly (no fillna in its local encoder)
    for df in (base_train, base_test):
        for col in ("CryoSleep", "VIP"):
            if col in df.columns:
                df[col] = df[col].fillna(0)
    base_train.to_csv(ROOT / "train_processed.csv", index=False)
    base_test.to_csv(ROOT / "test_processed.csv", index=False)
    print(f"Prepared: train_processed.csv {base_train.shape}, test_processed.csv {base_test.shape}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
