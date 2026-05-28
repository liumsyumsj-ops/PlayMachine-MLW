import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif


TRAIN_PROCESSED_PATH = "train_processed.csv"
TEST_PROCESSED_PATH = "test_processed.csv"

TEST_RAW_PATH = "test.csv"

# Read the preprocessed data
train_clean = pd.read_csv(TRAIN_PROCESSED_PATH)
test_clean = pd.read_csv(TEST_PROCESSED_PATH)

test_raw = pd.read_csv(TEST_RAW_PATH)
test_ids = test_raw['PassengerId']


def encode_for_svm(train, test):

    train = train.copy(); test = test.copy()
    cat_cols = ['HomePlanet','Destination','Deck','Side','Surname']

    # LabelEncoder encodes categorical features (fit on train+test together to avoid data leakage)
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    # Drop unnecessary columns
    drop = ['AgeGroup_qcut']
    train = train.drop(columns=drop, errors='ignore')
    test  = test.drop(columns=drop, errors='ignore')

    # Split features and labels
    X = train.drop(columns=['Transported'])
    y = train['Transported']

    # Standardization (SVM is scale-sensitive, so this is required)
    scaler = StandardScaler()
    X_scaled      = pd.DataFrame(scaler.fit_transform(X),      columns=X.columns)
    test_scaled   = pd.DataFrame(scaler.transform(test),        columns=test.columns)
    return X_scaled, y, test_scaled

# Run SVM encoding
X_svm, y_svm, test_svm = encode_for_svm(train_clean, test_clean)

# ============================================================
# 3. Feature optimization (SVM is sensitive to high-dimensional data, so select core features)
# ============================================================
selector = SelectKBest(f_classif, k=min(50, X_svm.shape[1]))  # limit to at most 50 dimensions
X_svm_selected = selector.fit_transform(X_svm, y_svm)
test_svm_selected = selector.transform(test_svm)

# Output feature selection results
selected_features = X_svm.columns[selector.get_support()]
print(f"Number of retained features after selection: {len(selected_features)}")
print(f"Core feature list: {list(selected_features)[:10]}...")

# ============================================================
# 4. Hyperparameter grid search (core tuning step)
# ============================================================
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter space (focus on the RBF kernel, which performs best)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf'],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
}

# Initialize and run grid search
svm_base = SVC(probability=True)
grid_search = GridSearchCV(
    estimator=svm_base,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_svm_selected, y_svm)

# Output best parameters
print("\n=== Best hyperparameters ===")
for k, v in grid_search.best_params_.items():
    print(f"{k}: {v}")
cv_score = grid_search.best_score_
print(f"\nMean 5-fold cross-validation accuracy: {cv_score:.4f}")

# ============================================================
# 5. Final model training and prediction
# ============================================================
svm_best = grid_search.best_estimator_
svm_best.fit(X_svm_selected, y_svm)  # train on the full dataset
y_pred = svm_best.predict(test_svm_selected)

# Convert to the boolean format required by the competition (0->False, 1->True)
y_pred_bool = np.where(y_pred == 1, True, False)

# ============================================================
# 6. Create submission file
# ============================================================
SUBMIT_PATH = "svm_submission.csv"
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': y_pred_bool
})
submission.to_csv(SUBMIT_PATH, index=False)
print(f"\nSubmission file generated: {SUBMIT_PATH}")

# ============================================================
# 7. Model evaluation (on the training set)
# ============================================================
y_train_pred = svm_best.predict(X_svm_selected)
train_accuracy = accuracy_score(y_svm, y_train_pred)

print("\n=== Model training-set evaluation ===")
print(f"Training accuracy: {train_accuracy:.4f}")
print("\nClassification report: ")
print(classification_report(y_svm, y_train_pred, target_names=['Not Transported', 'Transported']))