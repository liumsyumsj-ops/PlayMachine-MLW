# Spaceship Titanic — Kaggle Competition Project Documentation

## Project Overview

Spaceship Titanic is a binary classification competition task on Kaggle. The goal is to predict whether a passenger was transported during an interstellar accident. The training set contains 8,693 rows, the test set contains 4,277 rows, and the evaluation metric is Accuracy.

**Final score: LB 0.81786**

---

## File Structure

```
mlw/
├── spaceship-titanic/
│   ├── train.csv                  ← Training data
│   ├── test.csv                   ← Test data
│   └── submission_final.csv       ← Final submission file (LB 0.81786)
│
├── preprocessing.py               ← Shared preprocessing + encoders (KNN / SVM / RF / MLP)
├── catboost_features.py           ← CatBoost + ensemble MLP feature layer
├── lgb_features.py                ← LightGBM feature layer (ensemble)
│
├── Single-model scripts
│   ├── KNN.py                     ← KNN
│   ├── svm.py                     ← SVM
│   ├── random_forest.py           ← Random Forest
│   ├── ct_v2.py                   ← CatBoost (also feeds ensemble)
│   └── mlp_1.4.py                 ← MLP PyTorch single model
│
├── Ensemble (training scripts)
│   ├── ct_mlp.py                  ← sklearn MLP → OOF / test probabilities for ensemble
│   └── pipeline/train_lgb_ensemble.py  ← LightGBM lgb_base for ensemble
│
├── run_all.py                     ← **Recommended:** all single-model CSVs + final ensemble
├── pipeline/                      ← Orchestration (see “One-Click Run”)
│   └── submissions/               ← Collected Kaggle CSVs
│
├── experiments_log.md             ← Experiment record
└── references.txt                 ← References (IEEE format)
```

---

## One-Click Run (All Submissions)

Run every single model and the final ensemble **without editing** the original training scripts (`KNN.py`, `svm.py`, `random_forest.py`, `ct_v2.py`, `mlp_1.4.py`, etc.). Orchestration lives under `pipeline/`.

### Usage

From the project root:

```bash
# Full run (includes slow PyTorch MLP single model mlp_1.4.py; ~30–45 min total)
python3 run_all.py

# Faster smoke test: MLP single model uses sklearn ct_mlp export instead (~15–25 min)
python3 run_all.py --quick
```

Recommended Python: Anaconda with `catboost`, `lightgbm`, and `torch` installed. On macOS, if the default `python3` lacks `lightgbm`, use:

```bash
/opt/anaconda3/bin/python3 run_all.py
```

Override the interpreter:

```bash
MLW_PYTHON=/path/to/python3 python3 run_all.py
```

Equivalent entry point:

```bash
python3 pipeline/run_all_submissions.py
```

Run log (optional): `pipeline/run.log` when using `tee pipeline/run.log`.

### Generated Files

All submission CSVs for Kaggle are collected under **`pipeline/submissions/`**:

| Output file | Model / purpose |
|-------------|-----------------|
| `pipeline/submissions/submission_knn.csv` | KNN single model |
| `pipeline/submissions/submission_svm.csv` | SVM single model |
| `pipeline/submissions/submission_random_forest.csv` | Random Forest single model |
| `pipeline/submissions/submission_catboost.csv` | CatBoost single model (`ct_v2`) |
| `pipeline/submissions/submission_mlp.csv` | MLP single model (`mlp_1.4` in full run; sklearn `ct_mlp` if `--quick`) |
| `pipeline/submissions/submission_final.csv` | Final ensemble (CT 65% + MLP 25% + LGB 9% + post-processing) |

The same final ensemble is also written to:

- **`spaceship-titanic/submission_final.csv`**

Intermediate files used for blending (project root, overwritten each run):

- `oof_ct_v2.npy`, `test_ct_v2.npy`
- `oof_MLP-wide.npy`, `test_mlp.npy`
- `oof_lgb_base.npy`, `test_lgb_base.npy`
- `train_processed.csv`, `test_processed.csv` (for SVM / Random Forest)

More detail: `pipeline/README.md`.

---

## Single-Model Descriptions

### KNN (`KNN.py`)

K-Nearest Neighbors is sensitive to feature scale, so standardization and dimensionality reduction are required before training.

| Item | Description |
|------|-------------|
| Feature encoding | OHE for categorical columns + StandardScaler |
| Dimensionality reduction | PCA, retaining 95% variance (22 principal components) |
| Hyperparameter search | GridSearchCV over k, distance metric, and weights |
| Cross-validation | 5-fold StratifiedKFold |

```bash
python3 KNN.py
```

---

### SVM (`svm.py`)

Support Vector Machine classifier with an RBF kernel. Since SVM is sensitive to feature scale, standardization is required.

| Item | Description |
|------|-------------|
| Feature encoding | LabelEncoder for categorical columns including Surname + StandardScaler |
| Feature selection | SelectKBest with f_classif, retaining 23 features |
| Hyperparameter search | GridSearchCV over C and gamma |
| Cross-validation | 5-fold StratifiedKFold |

```bash
python3 svm.py
```

---

### Random Forest (`random_forest.py`)

Random Forest is a tree-based model and is not sensitive to feature scale, so standardization is not required.

| Item | Description |
|------|-------------|
| Added features | CabinRegion (CabinNum // 300), SurnameFreq as a family-size proxy |
| Feature encoding | OHE for categorical columns, without StandardScaler |
| Number of features | 38 |
| Cross-validation | 5-fold GroupKFold grouped by GroupId to prevent family-level leakage |

```bash
python3 random_forest.py
```

---

### MLP (`mlp_1.4.py`)

This PyTorch-based Multilayer Perceptron uses Entity Embedding to map categorical features into low-dimensional dense vectors and also includes residual connections.

| Item | Description |
|------|-------------|
| Numerical features | 18 columns, standardized with StandardScaler |
| Categorical features | 5 columns (HomePlanet, Destination, Deck, Side, Surname), LabelEncoder → Embedding layer |
| Model architecture | Embedding + Pre-layer(128) + ResBlock(128×2) + Post-layer(64→1) |
| Dropout | 0.485, the best value found by Optuna |
| Loss function | BCEWithLogitsLoss |
| Cross-validation | 5-fold StratifiedKFold |

```bash
python3 mlp_1.4.py
```

---

## Ensemble Model Description

### How to Run

**Recommended:** use [One-Click Run](#one-click-run-all-submissions) (`python3 run_all.py`).

Manual ensemble steps (same logic as `pipeline/`):

```bash
python3 ct_v2.py
python3 ct_mlp.py
python3 pipeline/train_lgb_ensemble.py
python3 pipeline/build_final_ensemble.py
```

Final output: `pipeline/submissions/submission_final.csv` and `spaceship-titanic/submission_final.csv`

---

### Feature Engineering

- **Missing value handling**: numerical columns are filled with medians, categorical columns are filled with group-level modes; CryoSleep and spending values are inferred from each other.
- **Spending features**: TotalSpend, NecessitiesSpend (RoomService + FoodCourt), EntertainmentSpend (ShoppingMall + Spa + VRDeck).
- **Group features**: GroupSize, IsAlone, AllGroupCryo, and NoGroupCryo.
- **Other features**: CabinSize, IsChild (Age ≤ 12), and Surname.

### Models and Ensemble Weights

| Model | Key parameters | Ensemble weight |
|------|----------------|-----------------|
| CatBoost (ct_v2) | lr=0.03, depth=6, l2=2, iter=3000 | 65% |
| MLP (ct_mlp) | See the MLP single-model description | 25% |
| LightGBM (lgb_base) | lr=0.05, num_leaves=31, λ=1.0 | 9% |

All models use **5-fold GroupKFold cross-validation** grouped by GroupId to prevent leakage among passengers from the same family/group.

Weighted average blending:

```
blend = CT × (65/99) + MLP × (25/99) + LGB × (9/99)
```

Ensemble OOF accuracy: **0.8131**

### Post-Processing Rule

In the training set, passengers with **Earth + CryoSleep=True + Destination=TRAPPIST-1e** were almost always transported. However, the model predicted 33 such test cases as False. After forcing these cases to True, the LB score increased from 0.81669 to **0.81786** (+0.00117).

---

## Experiment Timeline (Key Milestones)

| Method | LB |
|------|----|
| CatBoost single-model baseline | ~0.803 |
| CT + MLP two-model ensemble | 0.81318 |
| CT + MLP + LGB three-model ensemble | 0.81622 |
| CT v2 parameter optimization | 0.81669 |
| **+ Earth+CryoSleep+TRAPPIST-1e post-processing (final)** | **0.81786** |

See `experiments_log.md` for the detailed experiment record.

---

## Runtime Environment

This project is a **Python 3** machine-learning pipeline (pandas / scikit-learn / CatBoost / LightGBM / PyTorch). A full one-click run (`python3 run_all.py`) needs **all** packages below installed in the same interpreter.

### Requirements (summary)

| Category | Requirement |
|----------|-------------|
| OS | macOS, Linux, or Windows (tested on **macOS arm64**) |
| Python | **3.9+** recommended; **3.13** with Anaconda used for the verified snapshot below |
| RAM | ≥ 8 GB (16 GB recommended for full run + CatBoost) |
| Disk | ~500 MB for data, model caches, and submission outputs |
| GPU | Optional; PyTorch MLP (`mlp_1.4.py`) runs on CPU. CUDA/MPS not required for the final LB score |

### Python packages

Install missing packages (use the same `python3` you will run with):

```bash
python3 -m pip install numpy pandas scipy scikit-learn matplotlib seaborn catboost lightgbm torch
```

On macOS, the system `/usr/bin/python3` often lacks `lightgbm` and `torch`. Use Anaconda instead:

```bash
/opt/anaconda3/bin/python3 run_all.py
# or
MLW_PYTHON=/opt/anaconda3/bin/python3 python3 run_all.py
```

### Verified environment snapshot

The table below was generated on this machine with the recommended Anaconda interpreter. Regenerate after upgrading packages:

```bash
/opt/anaconda3/bin/python3 scripts/collect_env.py -o docs/environment_snapshot.md
```

<!-- env:snapshot:start -->
<!-- AUTO-GENERATED by scripts/collect_env.py — do not edit by hand -->
Generated: 2026-05-28 10:55:56 UTC

| Item | Value |
|------|-------|
| Python | 3.13.9 |
| Executable | `/opt/anaconda3/bin/python3` |
| Platform | macOS-26.5-arm64-arm-64bit-Mach-O |
| Machine | arm64 |
| pip | pip 25.3 from /opt/anaconda3/lib/python3.13/site-packages/pip (python 3.13) |

#### Python packages

| Package | Version |
|---------|---------|
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scipy | 1.16.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.6 |
| seaborn | 0.13.2 |
| catboost | 1.2.10 |
| lightgbm | 4.6.0 |
| torch | 2.12.0 |

#### PyTorch device

- CUDA: no; MPS: no
<!-- env:snapshot:end -->

Canonical copy (same content): [`docs/environment_snapshot.md`](docs/environment_snapshot.md).

### Minimal dependency list

```
catboost
lightgbm
scikit-learn
pandas
numpy
torch
scipy
matplotlib
seaborn
```

---

## References

See `references.txt`, which includes 15 references covering KNN, SVM, Random Forest, MLP Entity Embedding, XGBoost, LightGBM, CatBoost, stacking ensembles, and probability calibration.
