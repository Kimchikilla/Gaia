"""
MGnify Data Collector for Gaia Project.

Collects genus-level taxonomic abundance tables from soil biome samples
via the MGnify REST API. Uses JSON BIOM download files to extract
genus-level OTU counts.

Source: https://www.ebi.ac.uk/metagenomics/api/v1
Target: 5,000-15,000 soil-related samples
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "data/configs/mgnify.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_soil_studies(config: dict) -> list[dict]:
    """Fetch studies associated with soil biomes."""
    base_url = config["api_base_url"]
    studies = []

    for lineage in config["biome_lineages"]:
        url = f"{base_url}/biomes/{lineage}/studies"
        while url:
            try:
                resp = requests.get(
                    url, params={"page_size": config["page_size"]}, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()

                for item in data.get("data", []):
                    studies.append(
                        {
                            "study_id": item["id"],
                            "biome_lineage": lineage,
                            "study_name": item["attributes"].get("study-name", ""),
                            "samples_count": item["attributes"].get("samples-count", 0),
                        }
                    )

                url = data.get("links", {}).get("next")
                time.sleep(0.5)
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                break

    # Deduplicate by study_id
    seen = set()
    unique = []
    for s in studies:
        if s["study_id"] not in seen:
            seen.add(s["study_id"])
            unique.append(s)

    logger.info(f"Found {len(unique)} soil-related studies")
    return unique


def fetch_analyses_for_study(study_id: str, config: dict) -> list[dict]:
    """Fetch analyses (runs) for a given study."""
    base_url = config["api_base_url"]
    url = f"{base_url}/studies/{study_id}/analyses"
    analyses = []

    while url:
        try:
            resp = requests.get(
                url, params={"page_size": config["page_size"]}, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                sample_data = (
                    item.get("relationships", {})
                    .get("sample", {})
                    .get("data", {})
                )
                analyses.append(
                    {
                        "analysis_id": item["id"],
                        "study_id": study_id,
                        "sample_id": sample_data.get("id", "") if sample_data else "",
                        "pipeline_version": item["attributes"].get(
                            "pipeline-version", ""
                        ),
                    }
                )

            url = data.get("links", {}).get("next")
            time.sleep(0.3)
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch analyses for {study_id}: {e}")
            break

    return analyses


def fetch_biom_taxonomy(analysis_id: str, config: dict) -> dict[str, float]:
    """
    Download the JSON BIOM file for an analysis and extract
    genus-level taxonomy with counts.
    """
    base_url = config["api_base_url"]

    # Step 1: Find the JSON BIOM download URL
    try:
        resp = requests.get(
            f"{base_url}/analyses/{analysis_id}/downloads", timeout=30
        )
        resp.raise_for_status()
        downloads = resp.json().get("data", [])
    except requests.RequestException as e:
        logger.debug(f"Failed to get downloads for {analysis_id}: {e}")
        return {}

    biom_url = None
    for dl in downloads:
        alias = dl.get("attributes", {}).get("alias", "")
        if "SSU_OTU_TABLE_JSON" in alias:
            biom_url = dl["links"]["self"]
            break

    if not biom_url:
        return {}

    # Step 2: Download and parse BIOM JSON
    try:
        resp = requests.get(biom_url, timeout=60)
        resp.raise_for_status()
        biom_data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.debug(f"Failed to download BIOM for {analysis_id}: {e}")
        return {}

    rows = biom_data.get("rows", [])
    data_entries = biom_data.get("data", [])

    # Build count map: row_index -> count
    row_counts = defaultdict(float)
    for entry in data_entries:
        if len(entry) >= 3:
            row_idx, col_idx, count = entry[0], entry[1], entry[2]
            row_counts[row_idx] += count

    # Step 3: Extract genus-level taxonomy
    genus_counts = {}
    for i, row in enumerate(rows):
        raw_taxonomy = row.get("metadata", {}).get("taxonomy", [])
        count = row_counts.get(i, 0)
        if count <= 0:
            continue

        # taxonomy can be a string ("sk__X;k__Y;...") or a list
        if isinstance(raw_taxonomy, str):
            levels = [t.strip() for t in raw_taxonomy.split(";")]
        else:
            levels = list(raw_taxonomy)

        # Standard 7-level: sk__, k__, p__, c__, o__, f__, g__
        genus = None

        # Look for g__ (genus) level
        for level in levels:
            if level and level.startswith("g__") and len(level) > 3:
                genus = level[3:]
                break

        # If no g__ found, try f__ (family) as fallback
        if genus is None:
            for level in levels:
                if level and level.startswith("f__") and len(level) > 3:
                    genus = level[3:]
                    break

        if genus and genus.strip():
            genus = genus.strip()
            genus_counts[genus] = genus_counts.get(genus, 0) + count

    return genus_counts


def fetch_sample_metadata(sample_id: str, config: dict) -> dict:
    """Fetch metadata for a sample."""
    base_url = config["api_base_url"]

    try:
        resp = requests.get(f"{base_url}/samples/{sample_id}", timeout=30)
        resp.raise_for_status()
        attrs = resp.json().get("data", {}).get("attributes", {})
        return {
            "sample_id": sample_id,
            "sample_name": attrs.get("sample-name", ""),
            "latitude": attrs.get("latitude"),
            "longitude": attrs.get("longitude"),
            "collection_date": attrs.get("collection-date", ""),
            "biome": attrs.get("environment-biome", ""),
            "feature": attrs.get("environment-feature", ""),
            "material": attrs.get("environment-material", ""),
        }
    except requests.RequestException as e:
        logger.debug(f"Failed to fetch metadata for {sample_id}: {e}")
        return {"sample_id": sample_id}


def collect_all(config: dict, output_dir: Path, max_samples: int | None = None):
    """Main collection pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get all soil studies
    logger.info("Step 1: Finding soil studies...")
    studies = fetch_soil_studies(config)
    pd.DataFrame(studies).to_csv(output_dir / "studies.csv", index=False)

    # Step 2: Get analyses for each study (limit to max_samples)
    logger.info("Step 2: Finding analyses...")
    all_analyses = []
    for study in tqdm(studies, desc="Studies"):
        analyses = fetch_analyses_for_study(study["study_id"], config)
        all_analyses.extend(analyses)
        if max_samples and len(all_analyses) >= max_samples:
            all_analyses = all_analyses[:max_samples]
            break

    logger.info(f"Found {len(all_analyses)} analyses")

    # Step 3: Download BIOM files and extract genus-level taxonomy
    logger.info("Step 3: Downloading genus-level data from BIOM files...")
    abundance_records = []
    metadata_records = []
    seen_samples = set()
    n_success = 0

    for analysis in tqdm(all_analyses, desc="Downloading"):
        analysis_id = analysis["analysis_id"]
        sample_id = analysis["sample_id"]

        # Get genus-level taxonomy from BIOM
        genus_counts = fetch_biom_taxonomy(analysis_id, config)

        if genus_counts:
            record = {"sample_id": sample_id, "analysis_id": analysis_id}
            record.update(genus_counts)
            abundance_records.append(record)
            n_success += 1

        # Get metadata (deduplicate)
        if sample_id and sample_id not in seen_samples:
            metadata = fetch_sample_metadata(sample_id, config)
            metadata["study_id"] = analysis["study_id"]
            metadata["pipeline_version"] = analysis["pipeline_version"]
            metadata_records.append(metadata)
            seen_samples.add(sample_id)

        time.sleep(0.2)

    # Step 4: Save results
    logger.info("Step 4: Saving results...")

    if abundance_records:
        abundance_df = pd.DataFrame(abundance_records).fillna(0)
        abundance_df.to_csv(output_dir / config["abundance_file"], index=False)
        n_genera = abundance_df.shape[1] - 2  # Minus sample_id, analysis_id
        logger.info(f"Saved: {abundance_df.shape[0]} samples, {n_genera} genera")
    else:
        logger.warning("No genus-level abundance data collected!")

    if metadata_records:
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_csv(output_dir / config["metadata_file"], index=False)
        logger.info(f"Saved metadata: {metadata_df.shape[0]} samples")

    logger.info(f"Done! {n_success}/{len(all_analyses)} analyses had genus-level data")


def main():
    parser = argparse.ArgumentParser(
        description="Collect soil microbiome data from MGnify"
    )
    parser.add_argument(
        "--config", default="data/configs/mgnify.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to collect (for testing)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    collect_all(config, output_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
