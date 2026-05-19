"""
random_forest.py — applies the following optimizations based on random_forest2.py:
  1. Add new features CabinRegion(CabinNum // 300)+ SurnameFreq(family size)
  2. Change CV to GroupKFold (group by GroupId to avoid within-group leakage)
  3. Remove class_weight='balanced' (data is about 50/50) and increase n_estimators
  4. Use submission file name submission_rf3.csv for easier comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from preprocessing import encode_for_rf

# ============================================================
# 0. Read preprocessed data + raw IDs (for GroupKFold and submission)
# ============================================================
train_clean = pd.read_csv("/Users/cheeseowl/mlw/train_processed.csv")
test_clean  = pd.read_csv("/Users/cheeseowl/mlw/test_processed.csv")
train_ids   = pd.read_csv("/Users/cheeseowl/mlw/spaceship-titanic/train.csv", usecols=['PassengerId'])
test_ids    = pd.read_csv("/Users/cheeseowl/mlw/spaceship-titanic/test.csv",  usecols=['PassengerId'])

# core_preprocess keeps the original train/test order, so PassengerId can be aligned directly
groups = train_ids['PassengerId'].str.split('_').str[0].values

# ============================================================
# 1. Add file-level feature engineering (do not modify the shared preprocessing.py)
# ============================================================
def add_extra_features(train_c, test_c):
    train_c = train_c.copy(); test_c = test_c.copy()

    # CabinRegion: cabin-number bucketing (widely recognized as a strong feature)
    train_c['CabinRegion'] = (train_c['CabinNum'] // 300).fillna(-1).astype(int)
    test_c['CabinRegion']  = (test_c['CabinNum']  // 300).fillna(-1).astype(int)

    # SurnameFreq: family-size proxy (counted on train+test together without leakage)
    surname_counts = pd.concat([train_c['Surname'], test_c['Surname']]).value_counts()
    train_c['SurnameFreq'] = train_c['Surname'].map(surname_counts).fillna(1).astype(int)
    test_c['SurnameFreq']  = test_c['Surname'].map(surname_counts).fillna(1).astype(int)

    return train_c, test_c

train_clean, test_clean = add_extra_features(train_clean, test_clean)

# ============================================================
# 2. RF encoding (shared encode_for_rf automatically drops Surname)
# ============================================================
X, y, test = encode_for_rf(train_clean, test_clean)
print(f"Preprocessing done. Train: {X.shape}, Test: {test.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# ============================================================
# 3. Split into training / validation sets(still keep a single split to inspect the distribution)
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

# ============================================================
# 4. Baseline model
# ============================================================
print("\n--- Baseline Model (default params) ---")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
print(f"Validation Accuracy: {accuracy_score(y_val, rf_baseline.predict(X_val)):.4f}")

# ============================================================
# 5. Tuned model
# ============================================================
print("\n--- Tuned Model (rf3) ---")
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    max_features=0.35,
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred  = rf.predict(X_val)
y_proba = rf.predict_proba(X_val)[:, 1]
val_acc = accuracy_score(y_val, y_pred)
val_auc = roc_auc_score(y_val, y_proba)
print(f"Validation Accuracy : {val_acc:.4f}")
print(f"Validation AUC      : {val_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")

# ============================================================
# 6. GroupKFold cross-validation (5 folds, grouped by GroupId)
# ============================================================
print("\n--- 5-Fold GroupKFold Cross Validation ---")
gkf = GroupKFold(n_splits=5)
cv_scores = cross_val_score(rf, X, y, cv=gkf.split(X, y, groups=groups),
                            scoring='accuracy', n_jobs=-1)
print(f"CV Accuracy per fold : {[round(s,4) for s in cv_scores]}")
print(f"CV Mean Accuracy     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 6.5 Save OOF + test probabilities (for the ensemble model)
# ============================================================
oof_proba = np.zeros(len(y))
fold_test_probas = []
for tr_idx, va_idx in gkf.split(X, y, groups=groups):
    rf_oof = RandomForestClassifier(
        n_estimators=1000, max_depth=None, max_features=0.35,
        min_samples_leaf=2, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf_oof.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof_proba[va_idx] = rf_oof.predict_proba(X.iloc[va_idx])[:, 1]
    fold_test_probas.append(rf_oof.predict_proba(test)[:, 1])
test_proba = np.mean(fold_test_probas, axis=0)
np.save("/Users/cheeseowl/mlw/oof_rf3.npy",  oof_proba)
np.save("/Users/cheeseowl/mlw/test_rf3.npy", test_proba)
print(f"Saved: oof_rf3.npy ({oof_proba.shape}), test_rf3.npy ({test_proba.shape})")

# ============================================================
# 7. Visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Random Forest v3 — Model Evaluation', fontsize=13, fontweight='bold')

ax = axes[0]
cm = confusion_matrix(y_val, y_pred)
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Predicted 0','Predicted 1'])
ax.set_yticklabels(['Actual 0','Actual 1'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14, fontweight='bold',
                color='white' if cm[i,j] > cm.max()/2 else 'black')
ax.set_title(f'Confusion Matrix\nAccuracy: {val_acc:.4f}')

ax = axes[1]
fpr, tpr, _ = roc_curve(y_val, y_proba)
ax.plot(fpr, tpr, color='#0F6E56', lw=2, label=f'AUC = {val_auc:.4f}')
ax.plot([0,1],[0,1], color='gray', linestyle='--', lw=1, label='Random baseline')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve'); ax.legend()

ax = axes[2]
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top20 = feat_imp.sort_values(ascending=True).tail(20)
ax.barh(top20.index, top20.values, color='#AFA9EC', edgecolor='#534AB7', linewidth=0.5)
ax.set_title('Top 20 Feature Importances'); ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig("/Users/cheeseowl/mlw/spaceship-titanic/rf3_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: rf3_evaluation.png")

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([f'Fold {i+1}' for i in range(5)], cv_scores,
              color='#9FE1CB', edgecolor='#0F6E56', linewidth=0.8)
for bar, val in zip(bars, cv_scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'{val:.4f}', ha='center', fontsize=10)
ax.axhline(y=cv_scores.mean(), color='#D85A30', linestyle='--',
           linewidth=1.5, label=f'Mean = {cv_scores.mean():.4f}')
ax.set_title('5-Fold GroupKFold Accuracy — RF v3')
ax.set_ylabel('Accuracy')
ax.set_ylim(cv_scores.min()-0.02, cv_scores.max()+0.03)
ax.legend()
plt.tight_layout()
plt.savefig("/Users/cheeseowl/mlw/spaceship-titanic/rf3_cv_scores.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: rf3_cv_scores.png")

# ============================================================
# 8. Create submission file
# ============================================================
print("\n--- Generating Submission ---")
rf_final = RandomForestClassifier(
    n_estimators=1000, max_depth=None, max_features=0.35,
    min_samples_leaf=2, min_samples_split=5,
    random_state=42, n_jobs=-1
)
rf_final.fit(X, y)

test_pred = rf_final.predict(test)
submission = pd.DataFrame({
    'PassengerId': test_ids['PassengerId'],
    'Transported': test_pred.astype(bool)
})
submission.to_csv("/Users/cheeseowl/mlw/spaceship-titanic/submission_rf3.csv", index=False)
print(f"Saved: submission_rf3.csv")
print(f"Submission preview:\n{submission.head()}")
print(f"\nTransported distribution:\n{submission['Transported'].value_counts()}")
