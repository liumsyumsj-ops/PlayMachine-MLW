"""Single-model CSV from ct_mlp.py test probabilities (sklearn MLP used in ensemble)."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "submissions" / "submission_mlp.csv"


def main():
    test_proba = np.load(ROOT / "test_mlp.npy")
    test_raw = pd.read_csv(ROOT / "spaceship-titanic" / "test.csv")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Transported": (test_proba >= 0.5).astype(bool),
    }).to_csv(OUT, index=False)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
