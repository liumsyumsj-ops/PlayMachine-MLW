# pipeline — run all models & submissions

Does **not** modify original scripts in the project root (`KNN.py`, `svm.py`, `ct_v2.py`, etc.).

## Outputs (`pipeline/submissions/`)

| File | Model |
|------|--------|
| `submission_knn.csv` | KNN |
| `submission_svm.csv` | SVM |
| `submission_random_forest.csv` | Random Forest |
| `submission_catboost.csv` | CatBoost (`ct_v2`) |
| `submission_mlp.csv` | MLP (`mlp_1.4` PyTorch; or sklearn if `--quick`) |
| `submission_final.csv` | Ensemble CT65% + MLP25% + LGB9% + post-processing |

Also writes `spaceship-titanic/submission_final.csv`.

## Run

From project root:

```bash
python3 run_all.py
```

Fast smoke test (skips slow PyTorch MLP):

```bash
python3 run_all.py --quick
```

Uses `/opt/anaconda3/bin/python3` when present; override with `MLW_PYTHON=...`.

Timing: without `mlp_1.4`, ~15–25 minutes; full run adds ~10–20 minutes for PyTorch MLP.
