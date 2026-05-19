import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from preprocessing import (
    load_data,
    core_preprocess,
    encode_for_knn
)

# Make sure the following files are in the same directory:
#* preprocessing.py
#* train.csv
#* test.csv
#* KNN.py

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)

# Save PassengerId for submission
submission_ids = test_raw["PassengerId"]


# Unified core preprocessing
train_clean, test_clean = core_preprocess(train_raw, test_raw)


# KNN-specific encoding and standardization
X, y, X_test = encode_for_knn(train_clean, test_clean)

print(f"Processed training shape: {X.shape}")
print(f"Processed test shape: {X_test.shape}")

# PCA dimensionality reduction
print("\nApplying PCA...")

pca = PCA(n_components=0.95, random_state=42)

X_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)

print(f"Original feature dimension: {X.shape[1]}")
print(f"Reduced feature dimension: {X_pca.shape[1]}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nStarting Grid Search...")

param_grid = {
    'n_neighbors': [5,9,15,21,31,37],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=1,
    verbose=2
)

grid_search.fit(X_train, y_train)


# Output best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Score:")
print(grid_search.best_score_)

# Validation set evaluation
best_knn = grid_search.best_estimator_

y_val_pred = best_knn.predict(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"\nValidation Accuracy: {val_accuracy:.5f}")


# Retrain using the full training dataset
print("\nTraining final model on full dataset...")

best_knn.fit(X_pca, y)


# Generate test-set predictions
test_predictions = best_knn.predict(X_test_pca)

# Convert to True / False required by Kaggle
final_predictions = test_predictions.astype(bool)


# Create submission file
submission = pd.DataFrame({
    'PassengerId': submission_ids,
    'Transported': final_predictions
})

submission.to_csv('submission_knn.csv', index=False)

print("\nSubmission file saved as submission_knn.csv")



#Feature information
print("\nFinal Model Summary")
print("=" * 40)
print(f"Original Features : {X.shape[1]}")
print(f"PCA Features      : {X_pca.shape[1]}")
print(f"Best Validation   : {val_accuracy:.5f}")
print(f"Best Parameters   : {grid_search.best_params_}")



