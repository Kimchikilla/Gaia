"""Cleaning utilities for soil-state microbiome tables."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


DEFAULT_TAXON_BLOCKLIST = (
    "ambiguous",
    "metagenome",
    "uncultured",
    "unclassified",
    "unknown",
    "organism",
    "bacterium",
    "archaeon",
    "eukaryote",
    "candidate division",
)


@dataclass
class CleaningReport:
    n_rows: int
    n_original_features: int
    n_numeric_features: int
    n_valid_taxa: int
    n_prevalence_features: int
    min_prevalence: float
    dropped_invalid_taxa: int
    dropped_low_prevalence: int

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_taxon_name(name: str) -> str:
    """Normalize a taxon column name without inventing taxonomy."""
    value = str(name).strip()
    value = re.sub(r"^[a-z]__", "", value)
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_valid_taxon(
    name: str,
    blocklist: tuple[str, ...] = DEFAULT_TAXON_BLOCKLIST,
) -> bool:
    """Reject vague pseudo-taxa that usually represent annotation noise."""
    value = canonical_taxon_name(name).lower()
    if not value:
        return False
    if value in {"na", "nan", "none"}:
        return False
    return not any(term in value for term in blocklist)


def clean_abundance_table(
    df: pd.DataFrame,
    id_cols: list[str],
    min_prevalence: float = 0.02,
    drop_invalid_taxa: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Coerce, aggregate, and prevalence-filter a wide abundance table.

    Returns a table with id columns first and numeric taxon columns after them.
    Duplicate taxa after canonicalization are summed.
    """
    missing = [col for col in id_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing id columns: {missing}")

    feature_cols = [col for col in df.columns if col not in id_cols]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    numeric[numeric < 0] = 0.0

    renamed = {}
    valid_cols = []
    for col in numeric.columns:
        canonical = canonical_taxon_name(col)
        if drop_invalid_taxa and not is_valid_taxon(canonical):
            continue
        renamed[col] = canonical
        valid_cols.append(col)

    valid_numeric = numeric[valid_cols].rename(columns=renamed)
    if not valid_numeric.empty:
        valid_numeric = valid_numeric.T.groupby(level=0).sum().T

    if len(valid_numeric) == 0:
        keep_cols = []
    else:
        prevalence = (valid_numeric > 0).mean(axis=0)
        keep_cols = prevalence[prevalence >= min_prevalence].index.tolist()

    cleaned = pd.concat(
        [
            df[id_cols].reset_index(drop=True),
            valid_numeric[keep_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    report = CleaningReport(
        n_rows=int(len(df)),
        n_original_features=int(len(feature_cols)),
        n_numeric_features=int(numeric.shape[1]),
        n_valid_taxa=int(valid_numeric.shape[1]),
        n_prevalence_features=int(len(keep_cols)),
        min_prevalence=float(min_prevalence),
        dropped_invalid_taxa=int(numeric.shape[1] - valid_numeric.shape[1]),
        dropped_low_prevalence=int(valid_numeric.shape[1] - len(keep_cols)),
    )
    return cleaned, report


def relative_abundance(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Convert feature columns to per-row relative abundance."""
    out = df.copy()
    values = out[feature_cols].to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    out[feature_cols] = values / row_sums
    return out


def clr_matrix(values: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """Centered log-ratio transform for compositional features."""
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    comp = arr / row_sums
    logged = np.log(comp + pseudocount)
    return logged - logged.mean(axis=1, keepdims=True)


def align_feature_tables(
    frames: list[pd.DataFrame],
    id_cols: list[str],
) -> tuple[list[pd.DataFrame], list[str]]:
    """Align cleaned wide tables to the union of taxon feature columns."""
    features = sorted(
        {
            col
            for frame in frames
            for col in frame.columns
            if col not in id_cols
        }
    )

    aligned = []
    for frame in frames:
        missing_features = [
            feature for feature in features if feature not in frame.columns
        ]
        if missing_features:
            zeros = pd.DataFrame(0.0, index=frame.index, columns=missing_features)
            out = pd.concat([frame, zeros], axis=1)
        else:
            out = frame.copy()

        present_id_cols = [col for col in id_cols if col in out.columns]
        aligned.append(out[present_id_cols + features].copy())

    return aligned, features
