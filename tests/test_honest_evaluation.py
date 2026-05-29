"""Tests for shortcut-resistant evaluation helpers."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from gaia.evaluation.honest import (
    leave_one_group_out_classification,
    leave_one_group_out_regression,
    majority_classification_baseline,
    mean_regression_baseline,
    regression_split_report,
    shortcut_probe,
)


def test_mean_regression_baseline_reports_train_mean_performance():
    result = mean_regression_baseline(
        y_train=np.array([1.0, 3.0]),
        y_test=np.array([2.0, 4.0]),
    )

    assert result["rmse"] == pytest.approx(np.sqrt(2.0))
    assert result["mae"] == pytest.approx(1.0)
    assert "r2" in result


def test_majority_classification_baseline_uses_train_majority():
    result = majority_classification_baseline(
        y_train=np.array(["a", "a", "b"]),
        y_test=np.array(["a", "b", "b"]),
    )

    assert result["majority_label"] == "a"
    assert result["majority_count"] == 2
    assert result["accuracy"] == pytest.approx(1 / 3)


def test_shortcut_probe_detects_separable_shortcut_label():
    rng = np.random.default_rng(42)
    y = np.array([0] * 10 + [1] * 10)
    X = np.column_stack(
        [
            y.astype(float),
            rng.normal(scale=0.01, size=len(y)),
        ]
    )

    result = shortcut_probe(X, y, n_splits=5)

    assert result["n_classes"] == 2
    assert result["accuracy_mean"] > 0.9
    assert result["accuracy_over_majority"] > 0.3


def test_shortcut_probe_rejects_singleton_classes():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array(["a", "a", "b"])

    with pytest.raises(ValueError, match="at least two samples"):
        shortcut_probe(X, y)


def test_leave_one_group_out_classification_reports_per_group_gap():
    X = np.array(
        [
            [0.0],
            [0.1],
            [1.0],
            [1.1],
            [0.2],
            [0.3],
            [1.2],
            [1.3],
        ]
    )
    y = np.array(["low", "low", "high", "high", "low", "low", "high", "high"])
    groups = np.array(["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"])

    result = leave_one_group_out_classification(
        X,
        y,
        groups,
        estimator_factory=lambda: LogisticRegression(max_iter=500),
    )

    assert result["summary"]["groups_evaluated"] == 2
    assert set(result["per_group"]) == {"g1", "g2"}
    assert result["summary"]["accuracy_mean"] >= 0.75


def test_leave_one_group_out_regression_reports_mean_baseline_gap():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = X[:, 0] * 2.0
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)

    result = leave_one_group_out_regression(
        X,
        y,
        groups,
        estimator_factory=lambda: Ridge(alpha=0.0),
    )

    assert result["summary"]["groups_evaluated"] == 3
    assert result["summary"]["r2_mean"] > 0.9
    assert result["summary"]["rmse_improvement_over_mean"] > 0


def test_regression_split_report_includes_model_and_mean_baseline():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0.0, 2.0, 4.0, 6.0])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array([8.0, 10.0])

    result = regression_split_report(
        Ridge(alpha=0.0),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert set(result) == {"model", "mean_baseline"}
    assert result["model"]["r2"] > result["mean_baseline"]["r2"]
