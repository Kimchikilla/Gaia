"""Tests for benchmark baseline runners."""

import numpy as np

from benchmarks.baselines import (
    run_classification_baselines,
    run_regression_baselines,
)


def test_classification_baselines_include_majority_baseline():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array(["a", "a", "a", "b"])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array(["a", "b"])

    results = run_classification_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert "MajorityBaseline" in results
    assert results["MajorityBaseline"]["model"] is None
    assert results["MajorityBaseline"]["y_pred"].tolist() == ["a", "a"]


def test_regression_baselines_include_mean_baseline():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array([9.0, 11.0])

    results = run_regression_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert "MeanBaseline" in results
    assert results["MeanBaseline"]["model"] is None
    assert results["MeanBaseline"]["y_pred"].tolist() == [4.0, 4.0]
