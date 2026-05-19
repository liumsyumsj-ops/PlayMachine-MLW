# ======================
# 1. Import Libraries
# ======================

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

from preprocessing import load_data, core_preprocess, encode_for_rf

# ======================
# 2. Load Data
# ======================

train_raw, test_raw = load_data("train.csv", "test.csv")

test_ids = test_raw["PassengerId"]

# =========================================================
# 3. Preprocessing
# =========================================================
# LightGBM is a tree-based model, so here we reuse the RF-style encoder:
# - core_preprocess(): shared feature engineering and missing value handling
# - encode_for_rf(): one-hot encoding, no StandardScaler

train_clean, test_clean = core_preprocess(train_raw, test_raw)

X, y, X_test = encode_for_rf(train_clean, test_clean)

# =========================================================
# 4. Build LightGBM Model
# =========================================================

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=8,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# =========================================================
# 5. Cross Validation
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("CV Accuracy Scores:")
print(scores)

print("\nMean CV Accuracy:")
print(scores.mean())

# =========================================================
# 6. Train Final Model
# =========================================================

model.fit(X, y)

# =========================================================
# 7. Predict Test Set
# =========================================================

preds = model.predict(X_test)

# Convert to boolean
preds = preds.astype(bool)

# =========================================================
# 8. Create Submission
# =========================================================

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": preds
})

submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully!")
print(submission.head())
