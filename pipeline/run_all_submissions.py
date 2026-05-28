#!/usr/bin/env python3
"""
Run all single-model scripts + final ensemble. Original source files are not edited.

Outputs (pipeline/submissions/):
  submission_knn.csv
  submission_svm.csv
  submission_random_forest.csv
  submission_catboost.csv
  submission_mlp.csv
  submission_final.csv

Usage:
  python3 run_all.py
  python3 run_all.py --quick
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE = Path(__file__).resolve().parent
SUBMISSIONS = PIPE / "submissions"
PYTHON = os.environ.get(
    "MLW_PYTHON",
    "/opt/anaconda3/bin/python3" if Path("/opt/anaconda3/bin/python3").exists() else sys.executable,
)
ENV = {
    **os.environ,
    "MPLBACKEND": "Agg",
    "MPLCONFIGDIR": str(PIPE / ".mplconfig"),
}


def run(label, args, cwd=None):
    cwd = cwd or ROOT
    print(f"\n{'=' * 60}\n>>> {label}\n{'=' * 60}")
    r = subprocess.run([PYTHON] + args, cwd=str(cwd), env=ENV)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {r.returncode})")


def copy_submission(src: Path, dst_name: str):
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    dst = SUBMISSIONS / dst_name
    if not src.is_file():
        raise FileNotFoundError(f"Missing expected output: {src}")
    shutil.copy2(src, dst)
    print(f"Copied -> {dst}")


def main():
    parser = argparse.ArgumentParser(description="Run all models and collect submission CSVs")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow mlp_1.4.py; write submission_mlp.csv from ct_mlp (sklearn) instead",
    )
    args = parser.parse_args()

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for old in SUBMISSIONS.glob("*.csv"):
        old.unlink()

    print(f"Project root: {ROOT}")
    print(f"Python: {PYTHON}")
    print(f"Output dir: {SUBMISSIONS}")

    run("Prepare data", [str(PIPE / "prepare_data.py")])

    run("KNN", ["KNN.py"])
    copy_submission(ROOT / "submission_knn.csv", "submission_knn.csv")

    run("SVM", ["svm.py"])
    copy_submission(ROOT / "svm_submission.csv", "submission_svm.csv")

    run("Random Forest", ["random_forest.py"])
    copy_submission(
        ROOT / "spaceship-titanic" / "submission_rf3.csv",
        "submission_random_forest.csv",
    )

    run("CatBoost (ct_v2)", ["ct_v2.py"])
    run("CatBoost CSV", [str(PIPE / "export_catboost_csv.py")])

    run("MLP for ensemble (ct_mlp)", ["ct_mlp.py"])
    run("LightGBM for ensemble (lgb_base)", [str(PIPE / "train_lgb_ensemble.py")])

    if args.quick:
        run("MLP single CSV (sklearn, quick)", [str(PIPE / "export_mlp_sklearn_csv.py")])
    else:
        run("MLP single model (mlp_1.4 PyTorch)", ["mlp_1.4.py"])
        copy_submission(
            ROOT / "submission_pseudo_label_blend.csv",
            "submission_mlp.csv",
        )

    run("Final ensemble", [str(PIPE / "build_final_ensemble.py")])

    expected = [
        "submission_knn.csv",
        "submission_svm.csv",
        "submission_random_forest.csv",
        "submission_catboost.csv",
        "submission_mlp.csv",
        "submission_final.csv",
    ]
    missing = [n for n in expected if not (SUBMISSIONS / n).is_file()]
    if missing:
        raise SystemExit(f"Missing outputs: {missing}")

    print(f"\n{'=' * 60}")
    print("Done. Submit any of:")
    for name in expected:
        print(f"  {SUBMISSIONS / name}")
    print(f"Also: {ROOT / 'spaceship-titanic' / 'submission_final.csv'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
