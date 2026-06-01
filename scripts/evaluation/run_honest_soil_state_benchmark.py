"""Run honest soil-state benchmarks on cleaned public datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gaia.evaluation.criteria import (
    evaluate_diagnostic_gate,
    evaluate_prescription_gate,
)
from gaia.evaluation.honest import (
    leave_one_group_out_regression,
    mean_regression_baseline,
    shortcut_probe,
)
from gaia.preprocessing.soil_state import clr_matrix


OUT = Path("data/processed_real/honest_soil_state_benchmark.json")


def _feature_cols(df: pd.DataFrame) -> list[str]:
    id_like = {
        "sample_id",
        "dataset",
        "group_id",
        "site",
        "month",
        "target_source",
        "Plot_ID",
        "Experimental_Year",
        "Soil_Type",
        "Tillage_norm",
        "Fertilization_norm",
        "ph",
        "total_carbon",
        "total_nitrogen",
        "organic_matter",
    }
    return [col for col in df.columns if col not in id_like]


def _matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return clr_matrix(df[features].to_numpy(dtype=float))


def _group_cv_regression(
    df: pd.DataFrame,
    target: str,
    group_col: str,
    model_name: str,
    estimator,
) -> dict:
    features = _feature_cols(df)
    X = _matrix(df, features)
    y = df[target].to_numpy(dtype=float)
    groups = df[group_col].to_numpy()

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    fold_rows = []
    preds = np.zeros_like(y, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = estimator()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        preds[test_idx] = pred
        baseline = mean_regression_baseline(y[train_idx], y[test_idx])
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "held_out_groups": sorted(set(groups[test_idx].tolist())),
                "r2": float(r2_score(y[test_idx], pred)),
                "rmse": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
                "mae": float(mean_absolute_error(y[test_idx], pred)),
                "mean_baseline_r2": float(baseline["r2"]),
                "mean_baseline_rmse": float(baseline["rmse"]),
                "rmse_improvement_over_mean": float(baseline["rmse"])
                - float(np.sqrt(mean_squared_error(y[test_idx], pred))),
            }
        )

    return {
        "model": model_name,
        "target": target,
        "split": f"GroupKFold({group_col})",
        "n_samples": int(len(df)),
        "n_groups": int(df[group_col].nunique()),
        "n_features": int(len(features)),
        "r2_oof": float(r2_score(y, preds)),
        "rmse_oof": float(np.sqrt(mean_squared_error(y, preds))),
        "mae_oof": float(mean_absolute_error(y, preds)),
        "folds": fold_rows,
        "group_r2_mean": float(np.mean([row["r2"] for row in fold_rows])),
        "rmse_improvement_over_mean": float(
            np.mean([row["rmse_improvement_over_mean"] for row in fold_rows])
        ),
    }


def _cross_dataset_regression(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    model_name: str,
    estimator,
) -> dict:
    features = sorted(set(_feature_cols(train)) | set(_feature_cols(test)))
    train_aligned = train.reindex(columns=features, fill_value=0.0)
    test_aligned = test.reindex(columns=features, fill_value=0.0)

    X_train = _matrix(train_aligned, features)
    y_train = train[target].to_numpy(dtype=float)
    X_test = _matrix(test_aligned, features)
    y_test = test[target].to_numpy(dtype=float)

    model = estimator()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = mean_regression_baseline(y_train, y_test)

    return {
        "model": model_name,
        "target": target,
        "split": f"{train['dataset'].iloc[0]} -> {test['dataset'].iloc[0]}",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(features)),
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "mean_baseline_r2": float(baseline["r2"]),
        "mean_baseline_rmse": float(baseline["rmse"]),
        "rmse_improvement_over_mean": float(baseline["rmse"])
        - float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def main() -> None:
    neon = pd.read_csv("data/processed_real/soil_state_neon_ph.csv")
    westerfeld = pd.read_csv("data/processed_real/soil_state_westerfeld.csv")
    bernburg = pd.read_csv("data/processed_real/soil_state_bernburg.csv")

    models = {
        "Ridge_CLR": lambda: Ridge(alpha=10.0),
        "RandomForest_CLR": lambda: RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    neon_reports = []
    for name, factory in models.items():
        neon_reports.append(
            _group_cv_regression(neon, "ph", "group_id", name, factory)
        )

    cross_reports = []
    for name, factory in models.items():
        cross_reports.append(
            _cross_dataset_regression(westerfeld, bernburg, "ph", name, factory)
        )

    # Shortcut probe: can the cleaned taxa matrix recover NEON site?
    neon_features = _feature_cols(neon)
    site_probe = shortcut_probe(
        _matrix(neon, neon_features),
        neon["group_id"].to_numpy(),
        n_splits=3,
    )

    best_neon = max(
        neon_reports,
        key=lambda item: item["rmse_improvement_over_mean"],
    )
    diagnostic_report = {
        "n_samples": int(len(neon)),
        "n_groups": int(neon["group_id"].nunique()),
        "group_r2_mean": float(best_neon["group_r2_mean"]),
        "rmse_improvement_over_mean": float(best_neon["rmse_improvement_over_mean"]),
        "shortcut_accuracy_over_majority": float(
            site_probe["accuracy_over_majority"]
        ),
    }

    result = {
        "criteria": {
            "diagnostic_gate": evaluate_diagnostic_gate(diagnostic_report),
            "prescription_gate": evaluate_prescription_gate(
                {
                    "n_intervention_records": 0,
                    "n_intervention_types": 0,
                    "n_sites": 0,
                    "max_followup_months": 0,
                    "has_control_plots": False,
                }
            ),
        },
        "diagnostic_report_for_gate": diagnostic_report,
        "neon_leave_one_site_out": neon_reports,
        "westerfeld_to_bernburg": cross_reports,
        "shortcut_probe_neon_site": site_probe,
        "interpretation": (
            "This is a cleaned taxonomy-only diagnostic benchmark. "
            "Passing prescription is impossible without intervention/outcome data."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
