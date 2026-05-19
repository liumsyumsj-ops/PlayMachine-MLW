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
├── preprocessing.py               ← Shared data preprocessing used by all models
│
├── Single-model feature engineering
│   ├── knn_features.py            ← KNN-specific: OHE + StandardScaler + PCA
│   ├── svm_features.py            ← SVM-specific: LabelEncoder + StandardScaler + SelectKBest
│   ├── rf_features.py             ← RF-specific: CabinRegion + SurnameFreq + OHE
│   ├── mlp_features.py            ← MLP-specific: LabelEncoder for Embedding + StandardScaler
│   ├── lgb_features.py            ← LightGBM-specific: OHE reused from the RF-style pipeline
│   └── catboost_features.py       ← Feature engineering for CatBoost and ensemble MLP
│
├── Single-model scripts
│   ├── KNN.py                     ← KNN single-model training and submission
│   ├── svm.py                     ← SVM single-model training and submission
│   ├── random_forest3(2).py       ← Random Forest single-model training and submission
│   ├── mlp_1.4.py                 ← MLP single-model training and submission (PyTorch, Optuna-tuned)
│   └── LightGBM(1).py             ← LightGBM single-model training and submission
│
├── Ensemble pipeline
│   ├── ct_v2.py                   ← Step 1: Train the CatBoost model and generate OOF + test probabilities
│   ├── ct_mlp.py                  ← Step 2: Train the MLP model and generate OOF + test probabilities
│   ├── lgb_grid_v2.py             ← Step 3: Train the LightGBM model and generate OOF + test probabilities
│   └── final_submission.py        ← Step 4: Ensemble + post-processing → generate the submission file
│
├── oof_ct_v2.npy                  ← CatBoost OOF predicted probabilities
├── oof_MLP-wide.npy               ← MLP OOF predicted probabilities
├── oof_lgb_base.npy               ← LightGBM OOF predicted probabilities
├── test_ct_v2.npy                 ← CatBoost test-set predicted probabilities
├── test_mlp.npy                   ← MLP test-set predicted probabilities
├── test_lgb_base.npy              ← LightGBM test-set predicted probabilities
│
├── eda.py                         ← Exploratory data analysis
├── experiments_log.md             ← Full experiment log
└── references.txt             ← References in IEEE format
```

---

## Single-Model Descriptions

### KNN (`KNN.py` + `knn_features.py`)

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

### SVM (`svm.py` + `svm_features.py`)

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

### Random Forest (`random_forest3(2).py` + `rf_features.py`)

Random Forest is a tree-based model and is not sensitive to feature scale, so standardization is not required.

| Item | Description |
|------|-------------|
| Added features | CabinRegion (CabinNum // 300), SurnameFreq as a family-size proxy |
| Feature encoding | OHE for categorical columns, without StandardScaler |
| Number of features | 38 |
| Cross-validation | 5-fold GroupKFold grouped by GroupId to prevent family-level leakage |

```bash
python3 "random_forest3(2).py"
```

---

### MLP (`mlp_1.4.py` + `mlp_features.py`)

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

### LightGBM Single Model (`LightGBM(1).py` + `lgb_features.py`)

A single LightGBM gradient boosting tree model that reuses RF-style feature encoding.

| Item | Description |
|------|-------------|
| Feature encoding | OHE for categorical columns, without StandardScaler |
| Main parameters | n_estimators=1000, lr=0.01, max_depth=8, num_leaves=64, subsample=0.8 |
| Cross-validation | 5-fold StratifiedKFold |

```bash
python3 "LightGBM(1).py"
```

---

## Ensemble Model Description

### How to Run

Run the following scripts in order. Each step generates the corresponding `.npy` files:

```bash
python3 ct_v2.py           # Around 10-15 minutes
python3 ct_mlp.py          # Around 5-10 minutes
python3 lgb_grid_v2.py     # Around 10-15 minutes
python3 final_submission.py
```

Final output: `spaceship-titanic/submission_final.csv`

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
| LightGBM (lgb_grid_v2) | lr=0.05, num_leaves=31, λ=1.0 | 9% |

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

## Dependencies

```
catboost
lightgbm
scikit-learn
pandas
numpy
torch
scipy
```

---

## References

See `references.txt`, which includes 15 references covering KNN, SVM, Random Forest, MLP Entity Embedding, XGBoost, LightGBM, CatBoost, stacking ensembles, and probability calibration.
