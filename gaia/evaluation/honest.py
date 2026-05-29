"""Honest evaluation helpers for shortcut-resistant model validation.

These utilities make the failure modes explicit:
- mean/majority baselines that every useful model must beat
- probes for shortcut variables such as lab, study, country, or protocol
- leave-one-group-out validation for OOD claims
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values)


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def mean_regression_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the train-mean predictor on a regression test split."""
    y_train = _as_array(y_train).astype(float)
    y_test = _as_array(y_test).astype(float)
    y_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)

    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }


def majority_classification_baseline(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | str | int]:
    """Evaluate the train-majority-class predictor on a classification split."""
    y_train = _as_array(y_train)
    y_test = _as_array(y_test)
    majority_label, majority_count = Counter(y_train).most_common(1)[0]
    y_pred = np.full(y_test.shape, fill_value=majority_label, dtype=y_test.dtype)

    return {
        "majority_label": majority_label.item()
        if hasattr(majority_label, "item")
        else majority_label,
        "majority_count": int(majority_count),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": _safe_balanced_accuracy(y_test, y_pred),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def regression_split_report(
    estimator: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Fit a regressor and report it next to the train-mean baseline."""
    model = clone(estimator)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": {
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        },
        "mean_baseline": mean_regression_baseline(y_train, y_test),
    }


def shortcut_probe(
    X: np.ndarray,
    labels: np.ndarray,
    estimator: BaseEstimator | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Measure how easily an embedding reveals a shortcut label.

    Use this for variables that should not dominate embeddings, such as lab,
    sequencing protocol, study accession, or country.
    """
    X = _as_array(X)
    labels = _as_array(labels)
    classes, counts = np.unique(labels, return_counts=True)

    if len(classes) < 2:
        raise ValueError("shortcut_probe requires at least two classes")

    max_splits = int(np.min(counts))
    if max_splits < 2:
        raise ValueError("each class needs at least two samples for CV probing")
    n_splits = min(n_splits, max_splits)

    estimator = estimator or make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=10000, class_weight="balanced"),
    )
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    accs = []
    baccs = []
    f1s = []
    majority_accs = []

    for train_idx, test_idx in skf.split(X, labels):
        model = clone(estimator)
        model.fit(X[train_idx], labels[train_idx])
        pred = model.predict(X[test_idx])

        accs.append(accuracy_score(labels[test_idx], pred))
        baccs.append(balanced_accuracy_score(labels[test_idx], pred))
        f1s.append(f1_score(labels[test_idx], pred, average="macro", zero_division=0))
        majority_accs.append(
            majority_classification_baseline(
                labels[train_idx],
                labels[test_idx],
            )["accuracy"]
        )

    return {
        "n_samples": int(len(labels)),
        "n_classes": int(len(classes)),
        "n_splits": int(n_splits),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(baccs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "majority_baseline_accuracy_mean": float(np.mean(majority_accs)),
        "accuracy_over_majority": float(np.mean(accs) - np.mean(majority_accs)),
    }


def leave_one_group_out_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator_factory: Callable[[], BaseEstimator] | None = None,
    min_test_size: int = 2,
) -> dict[str, Any]:
    """Evaluate target classification while holding out each group.

    Groups can be country, lab, study, sequencing platform, or any variable
    that should not be memorized. Groups that cannot form a valid train/test
    classification problem are skipped and listed in the report.
    """
    X = _as_array(X)
    y = _as_array(y)
    groups = _as_array(groups)
    estimator_factory = estimator_factory or (
        lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=10000, class_weight="balanced"),
        )
    )

    per_group = {}
    skipped = {}

    for group in np.unique(groups):
        test_mask = groups == group
        train_mask = ~test_mask
        y_train = y[train_mask]
        y_test = y[test_mask]

        if int(test_mask.sum()) < min_test_size:
            skipped[str(group)] = "too_few_test_samples"
            continue
        if len(np.unique(y_train)) < 2:
            skipped[str(group)] = "train_has_one_class"
            continue
        if len(np.unique(y_test)) < 1:
            skipped[str(group)] = "test_has_no_class"
            continue

        model = estimator_factory()
        model.fit(X[train_mask], y_train)
        pred = model.predict(X[test_mask])
        majority = majority_classification_baseline(y_train, y_test)

        per_group[str(group)] = {
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": _safe_balanced_accuracy(y_test, pred),
            "macro_f1": float(
                f1_score(y_test, pred, average="macro", zero_division=0)
            ),
            "majority_baseline_accuracy": float(majority["accuracy"]),
            "accuracy_over_majority": float(
                accuracy_score(y_test, pred) - majority["accuracy"]
            ),
        }

    if per_group:
        accuracies = [v["accuracy"] for v in per_group.values()]
        balanced = [
            v["balanced_accuracy"]
            for v in per_group.values()
            if not np.isnan(v["balanced_accuracy"])
        ]
        over_majority = [v["accuracy_over_majority"] for v in per_group.values()]
        summary = {
            "groups_evaluated": int(len(per_group)),
            "groups_skipped": int(len(skipped)),
            "accuracy_mean": float(np.mean(accuracies)),
            "balanced_accuracy_mean": float(np.mean(balanced))
            if balanced
            else float("nan"),
            "accuracy_over_majority_mean": float(np.mean(over_majority)),
        }
    else:
        summary = {
            "groups_evaluated": 0,
            "groups_skipped": int(len(skipped)),
            "accuracy_mean": float("nan"),
            "balanced_accuracy_mean": float("nan"),
            "accuracy_over_majority_mean": float("nan"),
        }

    return {
        "summary": summary,
        "per_group": per_group,
        "skipped": skipped,
    }


def leave_one_group_out_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator_factory: Callable[[], BaseEstimator] | None = None,
    min_test_size: int = 2,
) -> dict[str, Any]:
    """Evaluate target regression while holding out each group."""
    X = _as_array(X)
    y = _as_array(y).astype(float)
    groups = _as_array(groups)
    estimator_factory = estimator_factory or (lambda: Ridge(alpha=1.0))

    per_group = {}
    skipped = {}

    for group in np.unique(groups):
        test_mask = groups == group
        train_mask = ~test_mask

        if int(test_mask.sum()) < min_test_size:
            skipped[str(group)] = "too_few_test_samples"
            continue

        model = estimator_factory()
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        baseline = mean_regression_baseline(y[train_mask], y[test_mask])

        per_group[str(group)] = {
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "r2": float(r2_score(y[test_mask], pred)),
            "rmse": float(np.sqrt(mean_squared_error(y[test_mask], pred))),
            "mae": float(mean_absolute_error(y[test_mask], pred)),
            "mean_baseline_r2": float(baseline["r2"]),
            "rmse_improvement_over_mean": float(baseline["rmse"])
            - float(np.sqrt(mean_squared_error(y[test_mask], pred))),
        }

    if per_group:
        r2s = [v["r2"] for v in per_group.values()]
        improvements = [v["rmse_improvement_over_mean"] for v in per_group.values()]
        summary = {
            "groups_evaluated": int(len(per_group)),
            "groups_skipped": int(len(skipped)),
            "r2_mean": float(np.mean(r2s)),
            "rmse_improvement_over_mean": float(np.mean(improvements)),
        }
    else:
        summary = {
            "groups_evaluated": 0,
            "groups_skipped": int(len(skipped)),
            "r2_mean": float("nan"),
            "rmse_improvement_over_mean": float("nan"),
        }

    return {
        "summary": summary,
        "per_group": per_group,
        "skipped": skipped,
    }
