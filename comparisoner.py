"""
This script compares the outputs produced by different Spaceship Titanic models.

Main features:
1. Load OOF probability files and evaluate them against the training labels.
2. Load test probability files and summarize prediction distributions.
3. Load submission files and compare final binary predictions.
4. Compute detailed metrics: accuracy, balanced accuracy, precision, recall, F1, ROC AUC,
   log loss, Brier score, confusion matrix values, and best threshold search.
5. Compare model agreement, disagreement, and probability correlations.
6. Evaluate a fixed weighted ensemble and optionally search a simple best-weight ensemble.
7. Generate visualizations for model metrics, confusion matrices, ROC curves, probability
   distributions, correlation matrices, agreement tables, and ensemble weight search.
8. Export all tables and figures into one output folder.

Expected project structure:
- Either spaceship-titanic/train.csv and spaceship-titanic/test.csv
- Or train.csv and test.csv in the current directory
- Model output files such as:
  oof_ct_v2.npy, test_ct_v2.npy
  oof_MLP-wide.npy, test_mlp.npy
  oof_lgb_base.npy, test_lgb_base.npy
  oof_rf3.npy, test_rf3.npy
  mlp_probs.csv
  submission_final.csv, submission_knn.csv, svm_submission.csv, etc.

Run:
python model_comparison_evaluator_with_visuals.py
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
)


RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.5
OUTPUT_DIR = Path("model_evaluation_outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"


TRAIN_CANDIDATES = [
    Path("spaceship-titanic/train.csv"),
    Path("train.csv"),
]

TEST_CANDIDATES = [
    Path("spaceship-titanic/test.csv"),
    Path("test.csv"),
]


MODEL_FILES = {
    "CatBoost_v2": {
        "oof": "oof_ct_v2.npy",
        "test_proba": "test_ct_v2.npy",
        "submission": None,
    },
    "MLP_sklearn": {
        "oof": "oof_MLP-wide.npy",
        "test_proba": "test_mlp.npy",
        "submission": None,
    },
    "LightGBM_base": {
        "oof": "oof_lgb_base.npy",
        "test_proba": "test_lgb_base.npy",
        "submission": None,
    },
    "RandomForest_rf3": {
        "oof": "oof_rf3.npy",
        "test_proba": "test_rf3.npy",
        "submission": "submission_rf3.csv",
    },
    "KNN": {
        "oof": None,
        "test_proba": None,
        "submission": "submission_knn.csv",
    },
    "LightGBM_single": {
        "oof": None,
        "test_proba": None,
        "submission": "submission.csv",
    },
    "SVM": {
        "oof": None,
        "test_proba": None,
        "submission": "svm_submission.csv",
    },
    "PyTorch_MLP_pseudo": {
        "oof": None,
        "test_proba_csv": "mlp_probs.csv",
        "submission": "submission_pseudo_label_blend.csv",
    },
    "Final_blend": {
        "oof": None,
        "test_proba": None,
        "submission": "submission_final.csv",
    },
}


FIXED_ENSEMBLES = {
    "CT65_MLP25_LGB9": {
        "CatBoost_v2": 65 / 99,
        "MLP_sklearn": 25 / 99,
        "LightGBM_base": 9 / 99,
    }
}


def print_section(title: str) -> None:
    line = "=" * 88
    print("\n" + line)
    print(title)
    print(line)


def save_current_figure(file_name: str) -> None:
    path = FIGURE_DIR / file_name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def find_existing_file(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def find_file_by_name(file_name: Optional[str]) -> Optional[Path]:
    if not file_name:
        return None

    candidates = [
        Path(file_name),
        Path("spaceship-titanic") / file_name,
        OUTPUT_DIR / file_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(Path(".").rglob(file_name))
    if matches:
        return matches[0]

    return None


def load_labels() -> Tuple[np.ndarray, Optional[pd.Series], Optional[pd.DataFrame]]:
    train_path = find_existing_file(TRAIN_CANDIDATES)
    if train_path is None:
        raise FileNotFoundError(
            "Training data was not found. Please place train.csv in the current directory "
            "or in the spaceship-titanic folder."
        )

    train_df = pd.read_csv(train_path)
    if "Transported" not in train_df.columns:
        raise ValueError("The training file does not contain the Transported column.")

    y_true = train_df["Transported"].astype(int).values
    passenger_ids = train_df["PassengerId"] if "PassengerId" in train_df.columns else None

    print(f"Training labels loaded from: {train_path}")
    print(f"Number of training rows: {len(y_true)}")
    print(f"Target distribution: {dict(pd.Series(y_true).value_counts().sort_index())}")

    return y_true, passenger_ids, train_df


def load_test_data() -> Optional[pd.DataFrame]:
    test_path = find_existing_file(TEST_CANDIDATES)
    if test_path is None:
        print("Test data was not found. Test-set summaries will still use available prediction files.")
        return None

    test_df = pd.read_csv(test_path)
    print(f"Test data loaded from: {test_path}")
    print(f"Number of test rows: {len(test_df)}")
    return test_df


def to_probability(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)

    if arr.dtype == bool:
        return arr.astype(float)

    if not np.issubdtype(arr.dtype, np.number):
        arr = pd.Series(arr).map({
            True: 1,
            False: 0,
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }).astype(float).values

    arr = arr.astype(float)

    if np.nanmin(arr) < 0 or np.nanmax(arr) > 1:
        print("Warning: values outside [0, 1] were clipped to the valid probability range.")
        arr = np.clip(arr, 0, 1)

    return arr


def load_npy_probability(file_name: Optional[str], expected_len: Optional[int] = None) -> Optional[np.ndarray]:
    path = find_file_by_name(file_name)
    if path is None:
        return None

    arr = to_probability(np.load(path))

    if expected_len is not None and len(arr) != expected_len:
        print(
            f"Warning: {path} has length {len(arr)}, but expected length is {expected_len}. "
            "This file will be skipped."
        )
        return None

    print(f"Loaded probability file: {path}")
    return arr


def load_csv_probability(file_name: Optional[str], expected_len: Optional[int] = None) -> Optional[np.ndarray]:
    path = find_file_by_name(file_name)
    if path is None:
        return None

    df = pd.read_csv(path)
    candidate_cols = [
        "Probability",
        "probability",
        "proba",
        "prob",
        "Transported_Probability",
        "TransportedProbability",
    ]

    selected_col = None
    for col in candidate_cols:
        if col in df.columns:
            selected_col = col
            break

    if selected_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        numeric_cols = [c for c in numeric_cols if c.lower() != "passengerid"]
        if numeric_cols:
            selected_col = numeric_cols[-1]

    if selected_col is None:
        if "Transported" in df.columns:
            selected_col = "Transported"
        else:
            print(f"Warning: no usable probability column was found in {path}.")
            return None

    arr = to_probability(df[selected_col].values)

    if expected_len is not None and len(arr) != expected_len:
        print(
            f"Warning: {path} has length {len(arr)}, but expected length is {expected_len}. "
            "This file will be skipped."
        )
        return None

    print(f"Loaded CSV probability file: {path}, column: {selected_col}")
    return arr


def load_submission_prediction(file_name: Optional[str], expected_len: Optional[int] = None) -> Optional[np.ndarray]:
    path = find_file_by_name(file_name)
    if path is None:
        return None

    df = pd.read_csv(path)
    if "Transported" not in df.columns:
        print(f"Warning: submission file {path} does not contain the Transported column.")
        return None

    pred = to_probability(df["Transported"].values)

    if expected_len is not None and len(pred) != expected_len:
        print(
            f"Warning: {path} has length {len(pred)}, but expected length is {expected_len}. "
            "This file will be skipped."
        )
        return None

    print(f"Loaded submission file: {path}")
    return pred


def threshold_predictions(proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    return (proba >= threshold).astype(int)


def search_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    thresholds = np.linspace(0.01, 0.99, 99)

    best_acc = {"threshold": DEFAULT_THRESHOLD, "score": -1.0}
    best_f1 = {"threshold": DEFAULT_THRESHOLD, "score": -1.0}

    for threshold in thresholds:
        pred = threshold_predictions(proba, threshold)
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)

        if acc > best_acc["score"]:
            best_acc = {"threshold": float(threshold), "score": float(acc)}

        if f1 > best_f1["score"]:
            best_f1 = {"threshold": float(threshold), "score": float(f1)}

    return {
        "best_accuracy_threshold": best_acc["threshold"],
        "best_accuracy": best_acc["score"],
        "best_f1_threshold": best_f1["threshold"],
        "best_f1": best_f1["score"],
    }


def evaluate_oof_model(name: str, y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    pred = threshold_predictions(proba, DEFAULT_THRESHOLD)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    row = {
        "model": name,
        "n_samples": len(y_true),
        "mean_probability": float(np.mean(proba)),
        "std_probability": float(np.std(proba)),
        "min_probability": float(np.min(proba)),
        "max_probability": float(np.max(proba)),
        "positive_rate_at_0_5": float(np.mean(pred)),
        "accuracy_at_0_5": float(accuracy_score(y_true, pred)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, pred)),
        "precision_at_0_5": float(precision_score(y_true, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7))),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    row.update(search_best_threshold(y_true, proba))
    return row


def summarize_test_predictions(name: str, proba_or_pred: np.ndarray, is_probability: bool) -> Dict[str, float]:
    arr = to_probability(proba_or_pred)
    pred = threshold_predictions(arr, DEFAULT_THRESHOLD)

    return {
        "model": name,
        "source_type": "probability" if is_probability else "binary_submission",
        "n_samples": len(arr),
        "mean_value": float(np.mean(arr)),
        "std_value": float(np.std(arr)),
        "min_value": float(np.min(arr)),
        "q05_value": float(np.quantile(arr, 0.05)),
        "q25_value": float(np.quantile(arr, 0.25)),
        "median_value": float(np.quantile(arr, 0.50)),
        "q75_value": float(np.quantile(arr, 0.75)),
        "q95_value": float(np.quantile(arr, 0.95)),
        "max_value": float(np.max(arr)),
        "predicted_true_count": int(np.sum(pred)),
        "predicted_false_count": int(len(pred) - np.sum(pred)),
        "predicted_true_rate": float(np.mean(pred)),
    }


def build_pairwise_agreement(pred_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []

    for left, right in itertools.combinations(pred_dict.keys(), 2):
        a = threshold_predictions(pred_dict[left])
        b = threshold_predictions(pred_dict[right])

        if len(a) != len(b):
            continue

        agreement = np.mean(a == b)
        disagreement = 1 - agreement
        both_true = np.mean((a == 1) & (b == 1))
        both_false = np.mean((a == 0) & (b == 0))
        left_only_true = np.mean((a == 1) & (b == 0))
        right_only_true = np.mean((a == 0) & (b == 1))

        rows.append({
            "model_a": left,
            "model_b": right,
            "agreement_rate": float(agreement),
            "disagreement_rate": float(disagreement),
            "both_true_rate": float(both_true),
            "both_false_rate": float(both_false),
            "model_a_true_model_b_false_rate": float(left_only_true),
            "model_a_false_model_b_true_rate": float(right_only_true),
        })

    return pd.DataFrame(rows).sort_values("disagreement_rate", ascending=False)


def build_probability_correlation(proba_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    valid = {name: arr for name, arr in proba_dict.items() if arr is not None}

    lengths = {}
    for name, arr in valid.items():
        lengths.setdefault(len(arr), []).append(name)

    if not lengths:
        return pd.DataFrame()

    main_length = max(lengths.keys(), key=lambda k: len(lengths[k]))
    selected_names = lengths[main_length]

    if len(selected_names) < 2:
        return pd.DataFrame()

    df = pd.DataFrame({name: valid[name] for name in selected_names})
    return df.corr()


def evaluate_fixed_ensembles(
    y_true: np.ndarray,
    oof_dict: Dict[str, np.ndarray],
    test_dict: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rows = []
    ensemble_test_outputs = {}
    ensemble_oof_outputs = {}

    for ensemble_name, weights in FIXED_ENSEMBLES.items():
        missing_oof = [name for name in weights if name not in oof_dict]
        if missing_oof:
            print(f"Skipping ensemble {ensemble_name}; missing OOF files: {missing_oof}")
            continue

        total_weight = sum(weights.values())
        oof_blend = np.zeros(len(y_true), dtype=float)

        for model_name, weight in weights.items():
            oof_blend += (weight / total_weight) * oof_dict[model_name]

        row = evaluate_oof_model(ensemble_name, y_true, oof_blend)
        rows.append(row)
        ensemble_oof_outputs[ensemble_name] = oof_blend

        missing_test = [name for name in weights if name not in test_dict]
        if not missing_test:
            first_len = len(next(iter(test_dict.values())))
            test_blend = np.zeros(first_len, dtype=float)

            for model_name, weight in weights.items():
                test_blend += (weight / total_weight) * test_dict[model_name]

            ensemble_test_outputs[ensemble_name] = test_blend

    return pd.DataFrame(rows), ensemble_test_outputs, ensemble_oof_outputs


def search_simple_weight_ensemble(
    y_true: np.ndarray,
    oof_dict: Dict[str, np.ndarray],
    max_models: int = 4,
    step: float = 0.05,
) -> pd.DataFrame:
    candidate_names = list(oof_dict.keys())[:max_models]

    if len(candidate_names) < 2:
        return pd.DataFrame()

    print(f"Searching simple ensemble weights for models: {candidate_names}")
    print(f"Weight step size: {step}")

    weight_values = np.arange(0, 1 + 1e-9, step)
    rows = []

    for weights in itertools.product(weight_values, repeat=len(candidate_names)):
        weight_sum = sum(weights)
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
            continue

        blend = np.zeros(len(y_true), dtype=float)
        for model_name, weight in zip(candidate_names, weights):
            blend += weight * oof_dict[model_name]

        pred = threshold_predictions(blend)
        acc = accuracy_score(y_true, pred)
        auc = roc_auc_score(y_true, blend)
        f1 = f1_score(y_true, pred, zero_division=0)

        row = {
            "accuracy_at_0_5": float(acc),
            "roc_auc": float(auc),
            "f1_at_0_5": float(f1),
        }
        for model_name, weight in zip(candidate_names, weights):
            row[f"weight_{model_name}"] = float(weight)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["accuracy_at_0_5", "roc_auc", "f1_at_0_5"],
        ascending=False,
    ).head(30)


def save_submission_from_probability(
    name: str,
    proba: np.ndarray,
    test_df: Optional[pd.DataFrame],
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    if test_df is None or "PassengerId" not in test_df.columns:
        return

    output_path = OUTPUT_DIR / f"submission_{name}.csv"
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": threshold_predictions(proba, threshold).astype(bool),
    })
    submission.to_csv(output_path, index=False)
    print(f"Saved generated ensemble submission: {output_path}")


def plot_metric_bar_chart(metrics_df: pd.DataFrame, metric: str, file_name: str, title: str) -> None:
    if metrics_df.empty or metric not in metrics_df.columns:
        return

    plot_df = metrics_df.sort_values(metric, ascending=True)

    plt.figure(figsize=(11, max(5, 0.45 * len(plot_df))))
    plt.barh(plot_df["model"], plot_df[metric])
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.title(title)

    for i, value in enumerate(plot_df[metric]):
        plt.text(value, i, f" {value:.4f}", va="center")

    save_current_figure(file_name)


def plot_oof_metrics(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        return

    metric_cols = [
        "accuracy_at_0_5",
        "balanced_accuracy_at_0_5",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
        "roc_auc",
    ]

    available_cols = [col for col in metric_cols if col in metrics_df.columns]
    plot_df = metrics_df.set_index("model")[available_cols].sort_values("accuracy_at_0_5", ascending=False)

    plt.figure(figsize=(13, max(5, 0.55 * len(plot_df))))
    x = np.arange(len(plot_df.index))
    width = 0.12

    for i, col in enumerate(available_cols):
        plt.bar(x + (i - len(available_cols) / 2) * width, plot_df[col].values, width=width, label=col)

    plt.xticks(x, plot_df.index, rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("OOF Model Metrics Comparison")
    plt.ylim(0, 1.05)
    plt.legend()
    save_current_figure("oof_metrics_grouped_bar.png")

    for metric in available_cols:
        plot_metric_bar_chart(
            metrics_df,
            metric,
            f"oof_{metric}_bar.png",
            f"OOF {metric} by Model",
        )


def plot_confusion_matrices(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        return

    required = ["true_negative", "false_positive", "false_negative", "true_positive"]
    if any(col not in metrics_df.columns for col in required):
        return

    for _, row in metrics_df.iterrows():
        matrix = np.array([
            [row["true_negative"], row["false_positive"]],
            [row["false_negative"], row["true_positive"]],
        ], dtype=float)

        plt.figure(figsize=(5, 4))
        plt.imshow(matrix)
        plt.title(f"Confusion Matrix: {row['model']}")
        plt.xticks([0, 1], ["Predicted False", "Predicted True"], rotation=20, ha="right")
        plt.yticks([0, 1], ["Actual False", "Actual True"])
        plt.colorbar()

        for i in range(2):
            for j in range(2):
                plt.text(j, i, int(matrix[i, j]), ha="center", va="center")

        safe_name = str(row["model"]).replace("/", "_").replace(" ", "_")
        save_current_figure(f"confusion_matrix_{safe_name}.png")


def plot_roc_curves(y_true: np.ndarray, oof_dict: Dict[str, np.ndarray]) -> None:
    if len(oof_dict) == 0:
        return

    plt.figure(figsize=(8, 7))

    for model_name, proba in oof_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_value = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc_value:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("OOF ROC Curves")
    plt.legend()
    save_current_figure("oof_roc_curves.png")


def plot_probability_distributions(prob_dict: Dict[str, np.ndarray], prefix: str) -> None:
    if len(prob_dict) == 0:
        return

    for model_name, proba in prob_dict.items():
        plt.figure(figsize=(8, 5))
        plt.hist(proba, bins=40)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.title(f"{prefix} Probability Distribution: {model_name}")
        safe_name = str(model_name).replace("/", "_").replace(" ", "_")
        save_current_figure(f"{prefix.lower()}_probability_distribution_{safe_name}.png")

    plt.figure(figsize=(9, 6))
    for model_name, proba in prob_dict.items():
        sorted_proba = np.sort(proba)
        y_axis = np.linspace(0, 1, len(sorted_proba))
        plt.plot(sorted_proba, y_axis, label=model_name)

    plt.xlabel("Predicted Probability")
    plt.ylabel("Cumulative Share")
    plt.title(f"{prefix} Probability Cumulative Distribution")
    plt.legend()
    save_current_figure(f"{prefix.lower()}_probability_cdf.png")


def plot_correlation_heatmap(corr_df: pd.DataFrame, file_name: str, title: str) -> None:
    if corr_df.empty:
        return

    labels = corr_df.columns.tolist()
    values = corr_df.values

    plt.figure(figsize=(max(7, 0.75 * len(labels)), max(6, 0.65 * len(labels))))
    plt.imshow(values, vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    save_current_figure(file_name)


def plot_test_summary(test_summary: pd.DataFrame) -> None:
    if test_summary.empty:
        return

    plot_metric_bar_chart(
        test_summary,
        "predicted_true_rate",
        "test_predicted_true_rate_bar.png",
        "Predicted True Rate on Test Set",
    )

    if "mean_value" in test_summary.columns:
        plot_metric_bar_chart(
            test_summary,
            "mean_value",
            "test_mean_prediction_value_bar.png",
            "Mean Prediction Value on Test Set",
        )


def plot_agreement_heatmap(pred_dict: Dict[str, np.ndarray]) -> None:
    if len(pred_dict) < 2:
        return

    names = list(pred_dict.keys())
    n = len(names)
    matrix = np.zeros((n, n), dtype=float)

    for i, left in enumerate(names):
        for j, right in enumerate(names):
            a = threshold_predictions(pred_dict[left])
            b = threshold_predictions(pred_dict[right])
            if len(a) == len(b):
                matrix[i, j] = np.mean(a == b)
            else:
                matrix[i, j] = np.nan

    plt.figure(figsize=(max(8, 0.65 * n), max(7, 0.55 * n)))
    plt.imshow(matrix, vmin=0, vmax=1)
    plt.colorbar(label="Agreement Rate")
    plt.xticks(range(n), names, rotation=45, ha="right")
    plt.yticks(range(n), names)
    plt.title("Pairwise Prediction Agreement Heatmap")

    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            if not np.isnan(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    save_current_figure("pairwise_prediction_agreement_heatmap.png")


def plot_weight_search(simple_search_df: pd.DataFrame) -> None:
    if simple_search_df.empty:
        return

    top_df = simple_search_df.head(15).copy()
    top_df["rank"] = np.arange(1, len(top_df) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(top_df["rank"], top_df["accuracy_at_0_5"], marker="o", label="Accuracy")
    plt.plot(top_df["rank"], top_df["roc_auc"], marker="o", label="ROC AUC")
    plt.plot(top_df["rank"], top_df["f1_at_0_5"], marker="o", label="F1")
    plt.xlabel("Rank")
    plt.ylabel("Score")
    plt.title("Top Ensemble Weight Search Results")
    plt.xticks(top_df["rank"])
    plt.legend()
    save_current_figure("simple_weight_search_top_scores.png")

    weight_cols = [col for col in simple_search_df.columns if col.startswith("weight_")]
    if not weight_cols:
        return

    best_row = simple_search_df.iloc[0]
    labels = [col.replace("weight_", "") for col in weight_cols]
    values = [best_row[col] for col in weight_cols]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylabel("Weight")
    plt.title("Best Simple Ensemble Weights")
    plt.xticks(rotation=30, ha="right")
    save_current_figure("best_simple_ensemble_weights.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print_section("Model Comparison Evaluator with Visualizations")
    print("All comments and output messages in this script are written in English.")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print(f"Figure folder: {FIGURE_DIR.resolve()}")

    y_true, _, train_df = load_labels()
    test_df = load_test_data()
    expected_test_len = len(test_df) if test_df is not None else None

    oof_probabilities: Dict[str, np.ndarray] = {}
    test_probabilities: Dict[str, np.ndarray] = {}
    test_predictions_for_agreement: Dict[str, np.ndarray] = {}

    oof_metric_rows = []
    test_summary_rows = []

    print_section("Loading Model Output Files")

    for model_name, files in MODEL_FILES.items():
        print(f"\nChecking model: {model_name}")

        oof = load_npy_probability(files.get("oof"), expected_len=len(y_true))
        if oof is not None:
            oof_probabilities[model_name] = oof
            oof_metric_rows.append(evaluate_oof_model(model_name, y_true, oof))

        test_proba = load_npy_probability(files.get("test_proba"), expected_len=expected_test_len)
        if test_proba is None:
            test_proba = load_csv_probability(files.get("test_proba_csv"), expected_len=expected_test_len)

        if test_proba is not None:
            test_probabilities[model_name] = test_proba
            test_predictions_for_agreement[model_name] = test_proba
            test_summary_rows.append(summarize_test_predictions(model_name, test_proba, is_probability=True))

        submission_pred = load_submission_prediction(files.get("submission"), expected_len=expected_test_len)
        if submission_pred is not None:
            test_predictions_for_agreement[f"{model_name}_submission"] = submission_pred
            test_summary_rows.append(summarize_test_predictions(f"{model_name}_submission", submission_pred, is_probability=False))

    print_section("OOF Evaluation Results")

    if oof_metric_rows:
        oof_metrics = pd.DataFrame(oof_metric_rows)
        oof_metrics = oof_metrics.sort_values(
            ["accuracy_at_0_5", "roc_auc", "f1_at_0_5"],
            ascending=False,
        )
        print(oof_metrics.to_string(index=False))
        oof_metrics.to_csv(OUTPUT_DIR / "oof_model_metrics.csv", index=False)
        print(f"Saved OOF metrics: {OUTPUT_DIR / 'oof_model_metrics.csv'}")
    else:
        oof_metrics = pd.DataFrame()
        print("No valid OOF probability files were found.")

    print_section("Fixed Ensemble Evaluation")

    fixed_ensemble_df, ensemble_test_outputs, ensemble_oof_outputs = evaluate_fixed_ensembles(
        y_true=y_true,
        oof_dict=oof_probabilities,
        test_dict=test_probabilities,
    )

    if not fixed_ensemble_df.empty:
        fixed_ensemble_df = fixed_ensemble_df.sort_values(
            ["accuracy_at_0_5", "roc_auc", "f1_at_0_5"],
            ascending=False,
        )
        print(fixed_ensemble_df.to_string(index=False))
        fixed_ensemble_df.to_csv(OUTPUT_DIR / "fixed_ensemble_metrics.csv", index=False)
        print(f"Saved fixed ensemble metrics: {OUTPUT_DIR / 'fixed_ensemble_metrics.csv'}")

        for ensemble_name, oof_proba in ensemble_oof_outputs.items():
            oof_probabilities[ensemble_name] = oof_proba

        for ensemble_name, test_proba in ensemble_test_outputs.items():
            test_probabilities[ensemble_name] = test_proba
            test_predictions_for_agreement[ensemble_name] = test_proba
            test_summary_rows.append(summarize_test_predictions(ensemble_name, test_proba, is_probability=True))
            save_submission_from_probability(ensemble_name, test_proba, test_df)
    else:
        print("No fixed ensemble could be evaluated.")

    print_section("Simple Weight Search")

    simple_search_df = search_simple_weight_ensemble(y_true, oof_probabilities, max_models=4, step=0.05)
    if not simple_search_df.empty:
        print(simple_search_df.head(15).to_string(index=False))
        simple_search_df.to_csv(OUTPUT_DIR / "simple_weight_search_top30.csv", index=False)
        print(f"Saved weight search results: {OUTPUT_DIR / 'simple_weight_search_top30.csv'}")
    else:
        print("Weight search was skipped because fewer than two OOF models were available.")

    print_section("Test Prediction Summary")

    if test_summary_rows:
        test_summary = pd.DataFrame(test_summary_rows)
        test_summary = test_summary.sort_values("predicted_true_rate", ascending=False)
        print(test_summary.to_string(index=False))
        test_summary.to_csv(OUTPUT_DIR / "test_prediction_summary.csv", index=False)
        print(f"Saved test prediction summary: {OUTPUT_DIR / 'test_prediction_summary.csv'}")
    else:
        test_summary = pd.DataFrame()
        print("No test prediction or submission files were found.")

    print_section("Model Agreement Analysis")

    if len(test_predictions_for_agreement) >= 2:
        agreement_df = build_pairwise_agreement(test_predictions_for_agreement)
        print(agreement_df.head(30).to_string(index=False))
        agreement_df.to_csv(OUTPUT_DIR / "pairwise_prediction_agreement.csv", index=False)
        print(f"Saved pairwise agreement table: {OUTPUT_DIR / 'pairwise_prediction_agreement.csv'}")
    else:
        agreement_df = pd.DataFrame()
        print("Agreement analysis requires at least two test prediction sources.")

    print_section("Probability Correlation Analysis")

    oof_corr = pd.DataFrame()
    test_corr = pd.DataFrame()

    if len(oof_probabilities) >= 2:
        oof_corr = build_probability_correlation(oof_probabilities)
        if not oof_corr.empty:
            print("OOF probability correlation matrix:")
            print(oof_corr.round(4).to_string())
            oof_corr.to_csv(OUTPUT_DIR / "oof_probability_correlation.csv")
            print(f"Saved OOF probability correlation matrix: {OUTPUT_DIR / 'oof_probability_correlation.csv'}")

    if len(test_probabilities) >= 2:
        test_corr = build_probability_correlation(test_probabilities)
        if not test_corr.empty:
            print("\nTest probability correlation matrix:")
            print(test_corr.round(4).to_string())
            test_corr.to_csv(OUTPUT_DIR / "test_probability_correlation.csv")
            print(f"Saved test probability correlation matrix: {OUTPUT_DIR / 'test_probability_correlation.csv'}")

    print_section("Generating Visualizations")

    plot_oof_metrics(oof_metrics)
    plot_confusion_matrices(oof_metrics)
    plot_roc_curves(y_true, oof_probabilities)
    plot_probability_distributions(oof_probabilities, prefix="OOF")
    plot_probability_distributions(test_probabilities, prefix="Test")
    plot_correlation_heatmap(oof_corr, "oof_probability_correlation_heatmap.png", "OOF Probability Correlation")
    plot_correlation_heatmap(test_corr, "test_probability_correlation_heatmap.png", "Test Probability Correlation")
    plot_test_summary(test_summary)
    plot_agreement_heatmap(test_predictions_for_agreement)
    plot_weight_search(simple_search_df)

    print_section("Evaluation Completed")
    print("The comparison tables and visualization figures have been saved successfully.")
    print(f"Please check this folder: {OUTPUT_DIR.resolve()}")
    print(f"Please check visualization images here: {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
