"""Train public-data soil diagnostic and prescription-candidate baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gaia.evaluation.honest import (  # noqa: E402
    majority_classification_baseline,
    mean_regression_baseline,
    shortcut_probe,
)
from gaia.preprocessing.soil_state import clr_matrix  # noqa: E402


OUT_DIR = Path("data/processed_real")
RESULT_JSON = OUT_DIR / "public_soil_baseline_results.json"
RESULT_CSV = OUT_DIR / "public_soil_baseline_summary.csv"

ID_LIKE = {
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


def _safe_float(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) == 0:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _taxa_features(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in ID_LIKE]


def _prefixed_features(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [col for col in df.columns if col.startswith(prefixes)]


def _clr_from_frame(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return clr_matrix(df[features].to_numpy(dtype=float))


def _one_hot_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    arr = encoder.fit_transform(df[columns].fillna("missing").astype(str))
    return pd.DataFrame(
        arr,
        columns=encoder.get_feature_names_out(columns),
        index=df.index,
    )


def _regression_models() -> dict[str, Any]:
    return {
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=80,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }


def _classification_models() -> dict[str, Any]:
    return {
        "Logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=10000,
                class_weight="balanced",
                n_jobs=None,
            ),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=80,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }


def group_cv_regression(
    task: str,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    valid = np.isfinite(y)
    X = X[valid]
    y = y[valid]
    groups = groups[valid]

    unique_groups = np.unique(groups)
    if len(y) < 5 or len(unique_groups) < 2:
        return []

    n_splits = min(5, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)
    reports = []
    for model_name, estimator in (model_map or _regression_models()).items():
        preds = np.full(len(y), np.nan, dtype=float)
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            model = clone(estimator)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            preds[test_idx] = pred
            baseline = mean_regression_baseline(y[train_idx], y[test_idx])
            rmse = float(np.sqrt(mean_squared_error(y[test_idx], pred)))
            fold_rows.append(
                {
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "held_out_groups": sorted(map(str, np.unique(groups[test_idx]))),
                    "r2": _safe_float(_safe_r2(y[test_idx], pred)),
                    "rmse": rmse,
                    "mae": float(mean_absolute_error(y[test_idx], pred)),
                    "mean_baseline_rmse": float(baseline["rmse"]),
                    "rmse_improvement_over_mean": float(baseline["rmse"]) - rmse,
                }
            )

        reports.append(
            {
                "task_type": "regression",
                "task": task,
                "target": target,
                "model": model_name,
                "split": "GroupKFold",
                "n_samples": int(len(y)),
                "n_groups": int(len(unique_groups)),
                "r2_oof": _safe_float(_safe_r2(y, preds)),
                "rmse_oof": float(np.sqrt(mean_squared_error(y, preds))),
                "mae_oof": float(mean_absolute_error(y, preds)),
                "rmse_improvement_over_mean": float(
                    np.nanmean([row["rmse_improvement_over_mean"] for row in fold_rows])
                ),
                "fold_r2_mean": _safe_float(
                    np.nanmean(
                        [
                            row["r2"]
                            for row in fold_rows
                            if row["r2"] is not None and np.isfinite(row["r2"])
                        ]
                    )
                ),
                "folds": fold_rows,
            }
        )
    return reports


def cross_dataset_regression(
    task: str,
    target: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    train_features = set(_taxa_features(train))
    test_features = set(_taxa_features(test))
    features = sorted(train_features | test_features)
    X_train = _clr_from_frame(train.reindex(columns=features, fill_value=0.0), features)
    X_test = _clr_from_frame(test.reindex(columns=features, fill_value=0.0), features)
    y_train = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    valid_train = np.isfinite(y_train)
    valid_test = np.isfinite(y_test)
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    X_test = X_test[valid_test]
    y_test = y_test[valid_test]

    reports = []
    for model_name, estimator in (model_map or _regression_models()).items():
        model = clone(estimator)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        baseline = mean_regression_baseline(y_train, y_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        reports.append(
            {
                "task_type": "regression",
                "task": task,
                "target": target,
                "model": model_name,
                "split": f"{train['dataset'].iloc[0]} -> {test['dataset'].iloc[0]}",
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_features": int(len(features)),
                "r2": _safe_float(_safe_r2(y_test, pred)),
                "rmse": rmse,
                "mae": float(mean_absolute_error(y_test, pred)),
                "mean_baseline_rmse": float(baseline["rmse"]),
                "rmse_improvement_over_mean": float(baseline["rmse"]) - rmse,
            }
        )
    return reports


def group_cv_classification(
    task: str,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    y = np.asarray(y)
    groups = np.asarray(groups)
    valid = pd.notna(y)
    X = X[valid]
    y = y[valid]
    groups = groups[valid]
    unique_groups = np.unique(groups)
    classes = np.unique(y)
    if len(y) < 5 or len(unique_groups) < 2 or len(classes) < 2:
        return []

    reports = []
    n_splits = min(5, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)
    for model_name, estimator in (model_map or _classification_models()).items():
        preds = np.full(y.shape, None, dtype=object)
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            if len(np.unique(y[train_idx])) < 2:
                continue
            model = clone(estimator)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            preds[test_idx] = pred
            baseline = majority_classification_baseline(y[train_idx], y[test_idx])
            acc = float(accuracy_score(y[test_idx], pred))
            fold_rows.append(
                {
                    "fold": fold,
                    "held_out_groups": sorted(map(str, np.unique(groups[test_idx]))),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "accuracy": acc,
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y[test_idx], pred)
                    )
                    if len(np.unique(y[test_idx])) > 1
                    else None,
                    "macro_f1": float(
                        f1_score(y[test_idx], pred, average="macro", zero_division=0)
                    ),
                    "majority_baseline_accuracy": float(baseline["accuracy"]),
                    "accuracy_over_majority": acc - float(baseline["accuracy"]),
                }
            )

        if len(fold_rows) == 0:
            continue
        valid_pred = pd.notna(preds)
        reports.append(
            {
                "task_type": "classification",
                "task": task,
                "target": target,
                "model": model_name,
                "split": "GroupKFold",
                "n_samples": int(valid_pred.sum()),
                "n_groups": int(len(unique_groups)),
                "accuracy": float(accuracy_score(y[valid_pred], preds[valid_pred])),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y[valid_pred], preds[valid_pred])
                ),
                "macro_f1": float(
                    f1_score(
                        y[valid_pred],
                        preds[valid_pred],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "accuracy_over_majority": float(
                    np.nanmean([row["accuracy_over_majority"] for row in fold_rows])
                ),
                "folds": fold_rows,
            }
        )
    return reports


def stratified_classification(
    task: str,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    model_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    y = np.asarray(y)
    valid = pd.notna(y)
    X = X[valid]
    y = y[valid]
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return []

    n_splits = min(5, int(counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    reports = []
    for model_name, estimator in (model_map or _classification_models()).items():
        preds = np.empty(y.shape, dtype=object)
        majority_gaps = []
        for train_idx, test_idx in cv.split(X, y):
            model = clone(estimator)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            preds[test_idx] = pred
            baseline = majority_classification_baseline(y[train_idx], y[test_idx])
            majority_gaps.append(
                float(accuracy_score(y[test_idx], pred)) - float(baseline["accuracy"])
            )
        reports.append(
            {
                "task_type": "classification",
                "task": task,
                "target": target,
                "model": model_name,
                "split": f"StratifiedKFold({n_splits})",
                "n_samples": int(len(y)),
                "n_groups": None,
                "accuracy": float(accuracy_score(y, preds)),
                "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
                "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
                "accuracy_over_majority": float(np.mean(majority_gaps)),
            }
        )
    return reports


def safe_shortcut_probe(
    task: str,
    X: np.ndarray,
    labels: np.ndarray,
    min_count: int = 2,
) -> dict[str, Any] | None:
    labels = np.asarray(labels).astype(str)
    counts = pd.Series(labels).value_counts()
    keep = np.isin(labels, counts[counts >= min_count].index)
    if keep.sum() < 5 or len(np.unique(labels[keep])) < 2:
        return None
    result = shortcut_probe(X[keep], labels[keep], n_splits=3)
    result["task"] = task
    result["n_filtered_out"] = int((~keep).sum())
    return result


def build_usda_feature_sets() -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    df = pd.read_csv("data/raw/tillage/usda_potato.csv", low_memory=False)
    microbe_cols = _prefixed_features(df, ("BF_g_", "FF_g_"))
    microbe = clr_matrix(df[microbe_cols].fillna(0.0).to_numpy(dtype=float))
    soil_cols = [
        "pH_1_1",
        "CEC",
        "OM_percent",
        "P_ppm",
        "K_ppm",
        "Mg_ppm",
        "Ca_ppm",
        "K_Sat_percent",
        "Mg_Sat_percent",
        "Ca_Sat_percent",
    ]
    soil = df[soil_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    management = _one_hot_frame(
        df,
        ["Fumigation1", "Rotation length", "Rotation diversity", "Year"],
    ).to_numpy(dtype=float)
    Xs = {
        "microbiome_only": microbe,
        "microbiome_soil_management": np.hstack([microbe, soil, management]),
    }
    site_groups = (df["State"].astype(str) + "_" + df["Field1"].astype(str)).to_numpy()
    return df, Xs, site_groups


def build_westerfeld_yield_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    state = pd.read_csv("data/processed_real/soil_state_westerfeld.csv")
    index = pd.read_csv("data/processed_real/public_soil_prescription_candidate_index.csv")
    index = index[index["dataset_id"] == "bonares_westerfeld"]
    df = state.merge(
        index[["sample_id", "yield_value", "intervention", "year"]],
        on="sample_id",
        how="inner",
    )
    features = _taxa_features(state)
    taxa = _clr_from_frame(df, features)
    intervention = _one_hot_frame(df, ["intervention"]).to_numpy(dtype=float)
    X = np.hstack([taxa, intervention])
    groups = df["year"].astype(str).to_numpy()
    return df, X, groups


def build_naylor_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    genus = pd.read_csv("data/raw/naylor/naylor_genus_with_labels.csv")
    meta = pd.read_csv("data/raw/naylor/naylor_metadata.csv")
    df = genus.merge(meta, on="run_id", how="left", suffixes=("", "_meta"))
    feature_cols = [
        col
        for col in genus.columns
        if col not in {"sample_id", "run_id", "treatment", "host"}
    ]
    X = clr_matrix(df[feature_cols].fillna(0.0).to_numpy(dtype=float))
    groups = df["host_genotype"].fillna(df["plant_body_site"]).astype(str).to_numpy()
    return df, X, groups


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "diagnostic_regression": [],
        "prescription_candidate_regression": [],
        "management_and_stress_classification": [],
        "cross_dataset_transfer": [],
        "shortcut_probes": [],
    }

    neon = pd.read_csv("data/processed_real/soil_state_neon_ph.csv")
    neon_features = _taxa_features(neon)
    neon_X = _clr_from_frame(neon, neon_features)
    results["diagnostic_regression"].extend(
        group_cv_regression(
            "neon_site_level_ph_from_microbiome",
            "ph",
            neon_X,
            neon["ph"].to_numpy(),
            neon["group_id"].to_numpy(),
        )
    )
    neon_probe = safe_shortcut_probe("neon_site_shortcut", neon_X, neon["group_id"])
    if neon_probe:
        results["shortcut_probes"].append(neon_probe)

    westerfeld = pd.read_csv("data/processed_real/soil_state_westerfeld.csv")
    bernburg = pd.read_csv("data/processed_real/soil_state_bernburg.csv")
    for target in ["ph", "total_carbon", "total_nitrogen"]:
        results["cross_dataset_transfer"].extend(
            cross_dataset_regression(
                f"westerfeld_to_bernburg_{target}",
                target,
                westerfeld,
                bernburg,
            )
        )
        results["cross_dataset_transfer"].extend(
            cross_dataset_regression(
                f"bernburg_to_westerfeld_{target}",
                target,
                bernburg,
                westerfeld,
            )
        )

    usda, usda_feature_sets, usda_groups = build_usda_feature_sets()
    for feature_set_name, X in usda_feature_sets.items():
        for target in ["pH_1_1", "OM_percent", "CEC", "P_ppm", "K_ppm"]:
            results["diagnostic_regression"].extend(
                group_cv_regression(
                    f"usda_{feature_set_name}_{target}",
                    target,
                    X,
                    pd.to_numeric(usda[target], errors="coerce").to_numpy(),
                    usda_groups,
                )
            )
        results["prescription_candidate_regression"].extend(
            group_cv_regression(
                f"usda_{feature_set_name}_yield",
                "Yield_per_meter",
                X,
                pd.to_numeric(usda["Yield_per_meter"], errors="coerce").to_numpy(),
                usda_groups,
            )
        )
    results["management_and_stress_classification"].extend(
        group_cv_classification(
            "usda_microbiome_fumigation_detection",
            "Fumigation1",
            usda_feature_sets["microbiome_only"],
            usda["Fumigation1"].astype(str).to_numpy(),
            usda_groups,
        )
    )
    usda_probe = safe_shortcut_probe(
        "usda_field_shortcut",
        usda_feature_sets["microbiome_only"],
        usda_groups,
    )
    if usda_probe:
        results["shortcut_probes"].append(usda_probe)

    west_yield, west_X, west_groups = build_westerfeld_yield_features()
    results["prescription_candidate_regression"].extend(
        group_cv_regression(
            "bonares_westerfeld_microbiome_management_yield",
            "yield_value",
            west_X,
            pd.to_numeric(west_yield["yield_value"], errors="coerce").to_numpy(),
            west_groups,
        )
    )

    naylor, naylor_X, naylor_groups = build_naylor_features()
    results["management_and_stress_classification"].extend(
        group_cv_classification(
            "naylor_microbiome_drought_detection_leave_genotype",
            "treatment",
            naylor_X,
            naylor["treatment"].astype(str).to_numpy(),
            naylor_groups,
        )
    )
    results["management_and_stress_classification"].extend(
        stratified_classification(
            "naylor_microbiome_drought_detection_random_cv",
            "treatment",
            naylor_X,
            naylor["treatment"].astype(str).to_numpy(),
        )
    )

    RESULT_JSON.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = []
    for section, reports in results.items():
        if section == "shortcut_probes":
            for report in reports:
                rows.append(
                    {
                        "section": section,
                        "task": report["task"],
                        "model": "probe",
                        "target": "shortcut_label",
                        "split": f"StratifiedKFold({report['n_splits']})",
                        "n_samples": report["n_samples"],
                        "metric_primary": report["accuracy_over_majority"],
                        "metric_name": "accuracy_over_majority",
                    }
                )
            continue
        for report in reports:
            if report["task_type"] == "regression":
                rows.append(
                    {
                        "section": section,
                        "task": report["task"],
                        "model": report["model"],
                        "target": report["target"],
                        "split": report["split"],
                        "n_samples": report.get("n_samples")
                        or report.get("n_test"),
                        "metric_primary": report.get("r2_oof", report.get("r2")),
                        "metric_name": "r2",
                        "rmse": report.get("rmse_oof", report.get("rmse")),
                        "rmse_improvement_over_mean": report.get(
                            "rmse_improvement_over_mean"
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "section": section,
                        "task": report["task"],
                        "model": report["model"],
                        "target": report["target"],
                        "split": report["split"],
                        "n_samples": report["n_samples"],
                        "metric_primary": report["balanced_accuracy"],
                        "metric_name": "balanced_accuracy",
                        "accuracy": report["accuracy"],
                        "accuracy_over_majority": report["accuracy_over_majority"],
                    }
                )
    pd.DataFrame(rows).to_csv(RESULT_CSV, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
