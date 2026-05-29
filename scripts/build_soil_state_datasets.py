"""Build cleaned soil-state benchmark datasets.

Outputs:
  data/processed_real/soil_state_neon_ph.csv
  data/processed_real/soil_state_westerfeld.csv
  data/processed_real/soil_state_bernburg.csv
  data/processed_real/soil_state_data_manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaia.preprocessing.soil_state import clean_abundance_table, relative_abundance


OUT_DIR = Path("data/processed_real")


def build_neon() -> tuple[pd.DataFrame, dict]:
    abundance = pd.read_csv("data/raw/neon/neon_microbe_abundance.csv")
    site_ph = pd.read_csv("data/raw/neon/neon_site_ph.csv")

    abundance = abundance.merge(site_ph[["site", "ph"]], on="site", how="inner")
    abundance["dataset"] = "NEON"
    abundance["group_id"] = abundance["site"]
    abundance["target_source"] = "site_mean_ph"

    id_cols = ["sample_id", "dataset", "group_id", "site", "month", "target_source", "ph"]
    cleaned, report = clean_abundance_table(
        abundance,
        id_cols=id_cols,
        min_prevalence=0.02,
    )
    feature_cols = [col for col in cleaned.columns if col not in id_cols]
    cleaned = relative_abundance(cleaned, feature_cols)
    cleaned.to_csv(OUT_DIR / "soil_state_neon_ph.csv", index=False)
    return cleaned, report.to_dict()


def _build_westerfeld_raw() -> pd.DataFrame:
    base = Path("data/raw/longterm/bonares_data")
    bacteria = pd.read_csv(base / "lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
    genus = pd.read_csv(base / "lte_westerfeld.V1_0_GENUS.csv")
    name_map = dict(zip(genus["Genus_ID"], genus["Name"]))

    bacteria["Genus_Name"] = bacteria["Genus_ID"].map(name_map)
    grouped = (
        bacteria.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"]
        .sum()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["Plot_ID", "Experimental_Year"],
        columns="Genus_Name",
        values="Value",
        fill_value=0,
    ).reset_index()

    soil = pd.read_csv(base / "lte_westerfeld.V1_0_SOIL_LAB.csv")
    sampling = pd.read_csv(base / "lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
    soil = soil.merge(
        sampling[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]],
        on="Soil_Sampling_ID",
        how="left",
    )
    chem_cols = ["pH", "Total_Carbon", "Total_Nitrogen"]
    chemistry = (
        soil.dropna(subset=["Plot_ID"])
        .groupby(["Plot_ID", "Experimental_Year"])[chem_cols]
        .mean()
        .reset_index()
    )

    paired = pivot.merge(chemistry, on=["Plot_ID", "Experimental_Year"], how="inner")
    paired = paired.dropna(subset=chem_cols).reset_index(drop=True)
    paired["sample_id"] = paired.apply(
        lambda row: f"WESTERFELD_{int(row['Plot_ID'])}_{int(row['Experimental_Year'])}",
        axis=1,
    )
    paired["dataset"] = "Westerfeld"
    paired["group_id"] = paired["Experimental_Year"].astype(str)
    paired = paired.rename(
        columns={
            "pH": "ph",
            "Total_Carbon": "total_carbon",
            "Total_Nitrogen": "total_nitrogen",
        }
    )
    return paired


def build_westerfeld() -> tuple[pd.DataFrame, dict]:
    paired = _build_westerfeld_raw()
    id_cols = [
        "sample_id",
        "dataset",
        "group_id",
        "Plot_ID",
        "Experimental_Year",
        "ph",
        "total_carbon",
        "total_nitrogen",
    ]
    cleaned, report = clean_abundance_table(
        paired,
        id_cols=id_cols,
        min_prevalence=0.02,
    )
    feature_cols = [col for col in cleaned.columns if col not in id_cols]
    cleaned = relative_abundance(cleaned, feature_cols)
    cleaned.to_csv(OUT_DIR / "soil_state_westerfeld.csv", index=False)
    return cleaned, report.to_dict()


def build_bernburg() -> tuple[pd.DataFrame, dict]:
    abundance = pd.read_csv("data/processed_real/bernburg_abundance.csv")
    metadata = pd.read_csv("data/processed_real/bernburg_metadata.csv")
    paired = abundance.merge(metadata, left_on="sample_id", right_on="Sample", how="inner")
    paired["dataset"] = "Bernburg"
    paired["group_id"] = paired["Experimental_Year"].astype(str)
    paired = paired.rename(
        columns={
            "pH": "ph",
            "C[%]": "total_carbon",
            "N[%]": "total_nitrogen",
            "OM[%]": "organic_matter",
        }
    )

    id_cols = [
        "sample_id",
        "dataset",
        "group_id",
        "Soil_Type",
        "Experimental_Year",
        "Tillage_norm",
        "Fertilization_norm",
        "ph",
        "total_carbon",
        "total_nitrogen",
        "organic_matter",
    ]
    cleaned, report = clean_abundance_table(
        paired,
        id_cols=id_cols,
        min_prevalence=0.02,
    )
    feature_cols = [col for col in cleaned.columns if col not in id_cols]
    cleaned = relative_abundance(cleaned, feature_cols)
    cleaned.to_csv(OUT_DIR / "soil_state_bernburg.csv", index=False)
    return cleaned, report.to_dict()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    neon, neon_report = build_neon()
    westerfeld, westerfeld_report = build_westerfeld()
    bernburg, bernburg_report = build_bernburg()

    manifest = {
        "datasets": {
            "NEON": {
                "path": str(OUT_DIR / "soil_state_neon_ph.csv"),
                "n_samples": int(len(neon)),
                "n_groups": int(neon["group_id"].nunique()),
                "target": "ph",
                "target_source": "site-level mean pH joined by site",
                "cleaning": neon_report,
            },
            "Westerfeld": {
                "path": str(OUT_DIR / "soil_state_westerfeld.csv"),
                "n_samples": int(len(westerfeld)),
                "n_groups": int(westerfeld["group_id"].nunique()),
                "targets": ["ph", "total_carbon", "total_nitrogen"],
                "target_source": "paired long-term field-trial soil lab data",
                "cleaning": westerfeld_report,
            },
            "Bernburg": {
                "path": str(OUT_DIR / "soil_state_bernburg.csv"),
                "n_samples": int(len(bernburg)),
                "n_groups": int(bernburg["group_id"].nunique()),
                "targets": ["ph", "total_carbon", "total_nitrogen", "organic_matter"],
                "target_source": "paired long-term field-trial soil lab data",
                "cleaning": bernburg_report,
            },
        },
        "acceptance_note": (
            "NEON pH is site-level and useful for cross-site stress testing, "
            "not calibrated per-sample chemistry. Westerfeld/Bernburg are true "
            "paired chemistry datasets but remain small and European."
        ),
    }
    (OUT_DIR / "soil_state_data_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
