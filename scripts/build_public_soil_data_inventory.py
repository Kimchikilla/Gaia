"""Build an inventory of public soil microbiome datasets against Gaia criteria."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaia.evaluation.criteria import evaluate_prescription_gate


OUT_DIR = Path("data/processed_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _csv_exists(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() and path.stat().st_size > 0


def _read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def _gate_report(
    n_intervention_records: int,
    n_intervention_types: int,
    n_sites: int,
    max_followup_months: int,
    has_control_plots: bool,
) -> dict[str, Any]:
    report = {
        "n_intervention_records": int(n_intervention_records),
        "n_intervention_types": int(n_intervention_types),
        "n_sites": int(n_sites),
        "max_followup_months": int(max_followup_months),
        "has_control_plots": bool(has_control_plots),
    }
    return {
        "report": report,
        "gate": evaluate_prescription_gate(report),
    }


def _inventory_row(
    dataset_id: str,
    source: str,
    status: str,
    tier: str,
    n_samples: int = 0,
    n_sites: int = 0,
    n_groups: int = 0,
    has_microbiome: bool = False,
    has_soil_chemistry: bool = False,
    has_intervention: bool = False,
    has_outcome: bool = False,
    has_control_plots: bool = False,
    max_followup_months: int = 0,
    n_intervention_records: int = 0,
    n_intervention_types: int = 0,
    usable_for: str = "",
    blocker: str = "",
) -> dict[str, Any]:
    gate = _gate_report(
        n_intervention_records=n_intervention_records,
        n_intervention_types=n_intervention_types,
        n_sites=n_sites,
        max_followup_months=max_followup_months,
        has_control_plots=has_control_plots,
    )
    return {
        "dataset_id": dataset_id,
        "source": source,
        "status": status,
        "tier": tier,
        "n_samples": int(n_samples),
        "n_sites": int(n_sites),
        "n_groups": int(n_groups),
        "has_microbiome": bool(has_microbiome),
        "has_soil_chemistry": bool(has_soil_chemistry),
        "has_intervention": bool(has_intervention),
        "has_outcome": bool(has_outcome),
        "has_control_plots": bool(has_control_plots),
        "max_followup_months": int(max_followup_months),
        "n_intervention_records": int(n_intervention_records),
        "n_intervention_types": int(n_intervention_types),
        "prescription_gate_passed": bool(gate["gate"]["passed"]),
        "usable_for": usable_for,
        "blocker": blocker,
        "gate_checks": gate["gate"]["checks"],
    }


def build_westerfeld() -> tuple[dict[str, Any], pd.DataFrame]:
    base = Path("data/raw/longterm/bonares_data")
    bacteria = _read_csv(
        base / "lte_westerfeld.V1_0_BACTERIA.csv",
        usecols=["Plot_ID", "Experimental_Year", "Genus_ID", "Value"],
    )
    microbe_samples = (
        bacteria.groupby(["Plot_ID", "Experimental_Year"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "microbiome_read_count"})
    )

    soil = _read_csv(base / "lte_westerfeld.V1_0_SOIL_LAB.csv")
    sampling = _read_csv(base / "lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
    soil = soil.merge(
        sampling[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]],
        on="Soil_Sampling_ID",
        how="left",
    )
    chemistry_cols = [
        "pH",
        "Total_Carbon",
        "Total_Organic_Carbon",
        "Total_Nitrogen",
        "Organic_Matter",
        "CN_Ratio",
        "Potassium_Oxide",
        "Diphosphorus_Pentoxide",
    ]
    chemistry = (
        soil.dropna(subset=["Plot_ID"])
        .groupby(["Plot_ID", "Experimental_Year"], as_index=False)[chemistry_cols]
        .mean()
    )

    plot = _read_csv(base / "lte_westerfeld.V1_0_PLOT.csv")
    treatment = _read_csv(base / "lte_westerfeld.V1_0_TREATMENT.csv")
    f1 = _read_csv(base / "lte_westerfeld.V1_0_FACTOR_1_LEVEL.csv")
    f2 = _read_csv(base / "lte_westerfeld.V1_0_FACTOR_2_LEVEL.csv")
    plot_treatment = (
        plot[["Plot_ID", "Treatment_ID"]]
        .merge(treatment, on="Treatment_ID", how="left")
        .merge(
            f1[["Factor_1_Level_ID", "Name_EN"]].rename(
                columns={"Name_EN": "tillage_system"}
            ),
            on="Factor_1_Level_ID",
            how="left",
        )
        .merge(
            f2[["Factor_2_Level_ID", "Name_EN"]].rename(
                columns={"Name_EN": "fertilization_system"}
            ),
            on="Factor_2_Level_ID",
            how="left",
        )
    )
    plot_treatment["intervention"] = (
        plot_treatment["tillage_system"].astype(str)
        + "+"
        + plot_treatment["fertilization_system"].astype(str)
    )

    fertilization = _read_csv(base / "lte_westerfeld.V1_0_FERTILIZATION.csv")
    fert_summary = (
        fertilization.groupby(["Plot_ID", "Experimental_Year"], as_index=False)
        .agg(
            fertilization_events=("Fertilization_ID", "count"),
            nitrogen_kg=("Nitrogen", "sum"),
            phosphorus_kg=("Phosphorus", "sum"),
            potassium_kg=("Potassium", "sum"),
            fertilizer_types=("Fertilizer_ID", pd.Series.nunique),
        )
    )

    tillage = _read_csv(base / "lte_westerfeld.V1_0_TILLAGE.csv")
    tillage_summary = (
        tillage.groupby(["Plot_ID", "Experimental_Year"], as_index=False)
        .agg(
            tillage_events=("Tillage_ID", "count"),
            mean_tillage_depth=("Depth", "mean"),
            tillage_measure_types=("Tillage_Measure_ID", pd.Series.nunique),
        )
    )

    harvest = _read_csv(base / "lte_westerfeld.V1_0_HARVEST.csv")
    yld = _read_csv(base / "lte_westerfeld.V1_0_YIELD.csv")
    yld = yld.merge(
        harvest[["Harvest_ID", "Plot_ID", "Experimental_Year"]],
        on="Harvest_ID",
        how="left",
    )
    yield_summary = yld.groupby(["Plot_ID", "Experimental_Year"], as_index=False).agg(
        yield_total=("Yield_Total", "mean"),
        crude_protein=("Crude_Protein", "mean"),
    )

    paired = (
        microbe_samples.merge(chemistry, on=["Plot_ID", "Experimental_Year"], how="inner")
        .merge(plot_treatment[["Plot_ID", "intervention"]], on="Plot_ID", how="left")
        .merge(fert_summary, on=["Plot_ID", "Experimental_Year"], how="left")
        .merge(tillage_summary, on=["Plot_ID", "Experimental_Year"], how="left")
        .merge(yield_summary, on=["Plot_ID", "Experimental_Year"], how="left")
    )
    paired["sample_id"] = paired.apply(
        lambda row: f"WESTERFELD_{int(row['Plot_ID'])}_{int(row['Experimental_Year'])}",
        axis=1,
    )

    years = paired["Experimental_Year"].dropna().astype(int)
    max_followup = int((years.max() - years.min()) * 12) if len(years) else 0
    intervention_types = int(
        paired["intervention"].nunique(dropna=True)
        + fertilization["Fertilizer_ID"].nunique(dropna=True)
        + tillage["Tillage_Measure_ID"].nunique(dropna=True)
    )

    row = _inventory_row(
        dataset_id="bonares_westerfeld",
        source="BonaRes long-term field trial Westerfeld",
        status="collected",
        tier="best_public_prescription_candidate",
        n_samples=len(paired),
        n_sites=1,
        n_groups=paired["Experimental_Year"].nunique(),
        has_microbiome=True,
        has_soil_chemistry=True,
        has_intervention=True,
        has_outcome=paired["yield_total"].notna().any(),
        has_control_plots=False,
        max_followup_months=max_followup,
        n_intervention_records=paired["intervention"].notna().sum(),
        n_intervention_types=intervention_types,
        usable_for="management contrast, soil chemistry, yield, diagnostic and weak prescription-candidate modeling",
        blocker="single site and no true untreated control plot; microbiome only measured for limited years",
    )
    candidate = pd.DataFrame(
        {
            "dataset_id": "bonares_westerfeld",
            "sample_id": paired["sample_id"],
            "site_id": "Westerfeld",
            "year": paired["Experimental_Year"],
            "intervention": paired["intervention"],
            "control_flag": False,
            "microbiome_available": True,
            "soil_chemistry_available": True,
            "outcome_available": paired["yield_total"].notna(),
            "soil_ph": paired["pH"],
            "total_carbon": paired["Total_Carbon"],
            "total_nitrogen": paired["Total_Nitrogen"],
            "organic_matter": paired["Organic_Matter"],
            "cec": pd.NA,
            "phosphorus": paired["Diphosphorus_Pentoxide"],
            "potassium": paired["Potassium_Oxide"],
            "yield_value": paired["yield_total"],
            "outcome_type": "yield_total",
            "notes": "single-site factorial tillage/fertilization long-term trial",
        }
    )
    return row, candidate


def build_bernburg() -> tuple[dict[str, Any], pd.DataFrame]:
    path = Path("data/processed_real/soil_state_bernburg.csv")
    df = _read_csv(path)
    source = df[
        [
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
    ].copy()
    source["intervention"] = (
        source["Tillage_norm"].astype(str)
        + "+"
        + source["Fertilization_norm"].astype(str)
    )
    years = source["Experimental_Year"].dropna().astype(int)
    max_followup = int((years.max() - years.min()) * 12) if len(years) else 0

    row = _inventory_row(
        dataset_id="bernburg_three_years",
        source="Bernburg three-year long-term field trial repo",
        status="collected",
        tier="diagnostic_and_management_contrast",
        n_samples=len(source),
        n_sites=1,
        n_groups=source["Experimental_Year"].nunique(),
        has_microbiome=True,
        has_soil_chemistry=True,
        has_intervention=True,
        has_outcome=False,
        has_control_plots=False,
        max_followup_months=max_followup,
        n_intervention_records=len(source),
        n_intervention_types=source["intervention"].nunique(),
        usable_for="soil chemistry diagnosis and tillage/fertilization contrast checks",
        blocker="small, single site, no yield/outcome table in the processed Gaia copy, no untreated control",
    )
    candidate = pd.DataFrame(
        {
            "dataset_id": "bernburg_three_years",
            "sample_id": source["sample_id"],
            "site_id": "Bernburg",
            "year": source["Experimental_Year"],
            "intervention": source["intervention"],
            "control_flag": False,
            "microbiome_available": True,
            "soil_chemistry_available": True,
            "outcome_available": False,
            "soil_ph": source["ph"],
            "total_carbon": source["total_carbon"],
            "total_nitrogen": source["total_nitrogen"],
            "organic_matter": source["organic_matter"],
            "cec": pd.NA,
            "phosphorus": pd.NA,
            "potassium": pd.NA,
            "yield_value": pd.NA,
            "outcome_type": pd.NA,
            "notes": "single-site tillage/fertilization soil chemistry contrast",
        }
    )
    return row, candidate


def build_usda_potato() -> tuple[dict[str, Any], pd.DataFrame]:
    path = Path("data/raw/tillage/usda_potato.csv")
    df = _read_csv(path)
    microbe_cols = [
        col for col in df.columns if col.startswith("BF_g_") or col.startswith("FF_g_")
    ]
    source = df[
        [
            "Sample",
            "State",
            "Field1",
            "Plot",
            "Year",
            "Yield_per_meter",
            "Fumigation1",
            "Rotation length",
            "Rotation diversity",
            "Soil_type",
            "pH_1_1",
            "CEC",
            "OM_percent",
            "P_ppm",
            "K_ppm",
        ]
    ].copy()
    source = source.rename(columns={"Sample": "sample_id"})
    source["site_id"] = (
        source["State"].astype(str) + "_" + source["Field1"].astype(str)
    )
    source["intervention"] = (
        "fumigation="
        + source["Fumigation1"].astype(str)
        + ";rotation_length="
        + source["Rotation length"].astype(str)
        + ";rotation_diversity="
        + source["Rotation diversity"].astype(str)
    )
    has_control = source["Fumigation1"].astype(str).str.lower().eq("no").any()

    row = _inventory_row(
        dataset_id="usda_potato_rotation",
        source="USDA potato rotation/tillage microbiome table already present locally",
        status="collected",
        tier="best_public_prescription_candidate",
        n_samples=len(source),
        n_sites=source["site_id"].nunique(),
        n_groups=source["Year"].nunique(),
        has_microbiome=bool(microbe_cols),
        has_soil_chemistry=True,
        has_intervention=True,
        has_outcome=True,
        has_control_plots=has_control,
        max_followup_months=36,
        n_intervention_records=len(source),
        n_intervention_types=source["intervention"].nunique(),
        usable_for="yield, soil chemistry, rotation/fumigation candidate modeling",
        blocker="only 423 records; intervention timing and before/after soil-state pairing are incomplete",
    )
    candidate = pd.DataFrame(
        {
            "dataset_id": "usda_potato_rotation",
            "sample_id": source["sample_id"],
            "site_id": source["site_id"],
            "year": source["Year"],
            "intervention": source["intervention"],
            "control_flag": source["Fumigation1"].astype(str).str.lower().eq("no"),
            "microbiome_available": True,
            "soil_chemistry_available": True,
            "outcome_available": True,
            "soil_ph": source["pH_1_1"],
            "total_carbon": pd.NA,
            "total_nitrogen": pd.NA,
            "organic_matter": source["OM_percent"],
            "cec": source["CEC"],
            "phosphorus": source["P_ppm"],
            "potassium": source["K_ppm"],
            "yield_value": source["Yield_per_meter"],
            "outcome_type": "yield_per_meter",
            "notes": "multi-field potato rotation/fumigation table with soil chemistry and yield",
        }
    )
    return row, candidate


def build_naylor() -> tuple[dict[str, Any], pd.DataFrame]:
    meta = _read_csv("data/raw/naylor/naylor_metadata.csv")
    genus = _read_csv("data/raw/naylor/naylor_genus_with_labels.csv", nrows=1)
    source = meta[
        [
            "run_id",
            "collection_date",
            "geo_loc_name",
            "host_genotype",
            "host_life_stage",
            "isolation_source",
            "plant_body_site",
            "watering_regm",
            "replicate",
            "treatment",
        ]
    ].copy()
    source = source.rename(columns={"run_id": "sample_id"})

    row = _inventory_row(
        dataset_id="naylor_sorghum_drought",
        source="Naylor sorghum drought microbiome data",
        status="collected",
        tier="stress_response_diagnostic",
        n_samples=len(source),
        n_sites=source["geo_loc_name"].nunique(),
        n_groups=source["treatment"].nunique(),
        has_microbiome=len(genus.columns) > 10,
        has_soil_chemistry=False,
        has_intervention=True,
        has_outcome=False,
        has_control_plots=source["treatment"].astype(str).str.lower().eq("control").any(),
        max_followup_months=0,
        n_intervention_records=len(source),
        n_intervention_types=source["treatment"].nunique(),
        usable_for="drought/control stress signature and OOD probe",
        blocker="no paired soil chemistry, no treatment response outcome",
    )
    candidate = pd.DataFrame(
        {
            "dataset_id": "naylor_sorghum_drought",
            "sample_id": source["sample_id"],
            "site_id": source["geo_loc_name"],
            "year": pd.to_datetime(source["collection_date"], errors="coerce").dt.year,
            "intervention": source["treatment"],
            "control_flag": source["treatment"].astype(str).str.lower().eq("control"),
            "microbiome_available": True,
            "soil_chemistry_available": False,
            "outcome_available": False,
            "soil_ph": pd.NA,
            "total_carbon": pd.NA,
            "total_nitrogen": pd.NA,
            "organic_matter": pd.NA,
            "cec": pd.NA,
            "phosphorus": pd.NA,
            "potassium": pd.NA,
            "yield_value": pd.NA,
            "outcome_type": pd.NA,
            "notes": (
                "drought/control stress dataset; plant body site="
                + source["plant_body_site"].astype(str)
            ),
        }
    )
    return row, candidate


def build_neon() -> dict[str, Any]:
    path = Path("data/processed_real/soil_state_neon_ph.csv")
    df = _read_csv(path)
    return _inventory_row(
        dataset_id="neon_soil_microbe_ph",
        source="NEON soil microbe community composition plus site-level pH",
        status="collected",
        tier="diagnostic_only",
        n_samples=len(df),
        n_sites=df["site"].nunique(),
        n_groups=df["group_id"].nunique(),
        has_microbiome=True,
        has_soil_chemistry=True,
        has_intervention=False,
        has_outcome=False,
        has_control_plots=False,
        max_followup_months=0,
        n_intervention_records=0,
        n_intervention_types=0,
        usable_for="large cross-site diagnostic stress test",
        blocker="pH is site-level mean, not sample-level chemistry; no intervention/outcome layer",
    )


def dryad_blocked_rows() -> list[dict[str, Any]]:
    return [
        _inventory_row(
            dataset_id="dryad_organic_amendments",
            source="Dryad 10.5061/dryad.4qrfj6q9n",
            status="download_blocked",
            tier="high_value_candidate_not_collected",
            has_microbiome=True,
            has_soil_chemistry=True,
            has_intervention=True,
            has_outcome=True,
            has_control_plots=True,
            usable_for="organic amendment response if downloaded",
            blocker="Dryad file download is behind Anubis/AWS WAF in this environment; metadata is visible but file download is blocked",
        ),
        _inventory_row(
            dataset_id="dryad_vanadium",
            source="Dryad 10.5061/dryad.6wwpzgn52",
            status="download_blocked",
            tier="diagnostic_candidate_not_collected",
            has_microbiome=True,
            has_soil_chemistry=True,
            has_intervention=False,
            has_outcome=False,
            usable_for="contamination/functional diagnostic signatures if downloaded",
            blocker="Dryad file download is behind Anubis/AWS WAF in this environment",
        ),
    ]


def main() -> None:
    inventory = []
    candidate_frames = []

    for builder in [build_westerfeld, build_bernburg, build_usda_potato, build_naylor]:
        row, candidate = builder()
        inventory.append(row)
        candidate_frames.append(candidate)

    inventory.append(build_neon())
    inventory.extend(dryad_blocked_rows())

    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(OUT_DIR / "public_soil_dataset_inventory.csv", index=False)
    (OUT_DIR / "public_soil_dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    index = pd.concat(candidate_frames, ignore_index=True, sort=False)
    index.to_csv(OUT_DIR / "public_soil_prescription_candidate_index.csv", index=False)

    print(inventory_df[[
        "dataset_id",
        "status",
        "tier",
        "n_samples",
        "n_sites",
        "n_intervention_records",
        "n_intervention_types",
        "prescription_gate_passed",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
