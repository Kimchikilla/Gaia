"""Tests for soil-state preprocessing."""

import numpy as np
import pandas as pd

from gaia.preprocessing.soil_state import (
    align_feature_tables,
    canonical_taxon_name,
    clean_abundance_table,
    clr_matrix,
    is_valid_taxon,
    relative_abundance,
)


def test_canonical_taxon_name_strips_prefix_and_underscores():
    assert canonical_taxon_name("g__Bacillus_subtilis") == "Bacillus subtilis"


def test_is_valid_taxon_rejects_ambiguous_entries():
    assert is_valid_taxon("Bacillus")
    assert not is_valid_taxon("uncultured bacterium")
    assert not is_valid_taxon("metagenome")


def test_clean_abundance_table_aggregates_and_filters():
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "g__Bacillus": [1, 2, 0],
            "Bacillus": [1, 0, 0],
            "uncultured bacterium": [10, 10, 10],
            "Rare": [0, 0, 1],
        }
    )

    cleaned, report = clean_abundance_table(
        df,
        id_cols=["sample_id"],
        min_prevalence=0.5,
    )

    assert "Bacillus" in cleaned.columns
    assert "uncultured bacterium" not in cleaned.columns
    assert "Rare" not in cleaned.columns
    assert cleaned["Bacillus"].tolist() == [2, 2, 0]
    assert report.n_prevalence_features == 1


def test_relative_abundance_and_clr_matrix():
    df = pd.DataFrame({"sample_id": ["s1", "s2"], "A": [1.0, 0.0], "B": [1.0, 2.0]})
    rel = relative_abundance(df, ["A", "B"])

    assert rel.loc[0, "A"] == 0.5
    assert rel.loc[1, "B"] == 1.0

    clr = clr_matrix(rel[["A", "B"]].to_numpy())
    np.testing.assert_allclose(clr.mean(axis=1), 0.0, atol=1e-10)


def test_align_feature_tables_tolerates_dataset_specific_metadata():
    left = pd.DataFrame({"sample_id": ["s1"], "site": ["a"], "A": [1.0]})
    right = pd.DataFrame({"sample_id": ["s2"], "year": [2020], "B": [2.0]})

    aligned, features = align_feature_tables(
        [left, right],
        id_cols=["sample_id", "site", "year"],
    )

    assert features == ["A", "B"]
    assert aligned[0][["A", "B"]].iloc[0].tolist() == [1.0, 0.0]
    assert aligned[1][["A", "B"]].iloc[0].tolist() == [0.0, 2.0]
    assert "site" in aligned[0].columns
    assert "site" not in aligned[1].columns
