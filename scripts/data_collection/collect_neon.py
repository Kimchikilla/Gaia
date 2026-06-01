"""
NEON Data Collector for Gaia Project.

Collects paired soil microbiome + environmental data from the
National Ecological Observatory Network (NEON).

Data Products:
  - DP1.10107.001: Soil microbe metagenome sequencing
  - DP1.10086.001: Soil chemical properties (pH, organic C, total N)
  - DP1.00094.001: Soil temperature and moisture
  - DP1.00006.001: Precipitation and air temperature

Value: Only large-scale public source with paired microbiome + environmental data.
"""

import argparse
import logging
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


def load_config(config_path: str = "data/configs/neon.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_neon_sites(config: dict) -> list[dict]:
    """Get all terrestrial NEON sites."""
    url = f"{config['api_base_url']}/sites"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    sites = []
    allowed_types = config.get("site_types", ["CORE", "GRADIENT"])
    for site in data.get("data", []):
        if site.get("siteType") in allowed_types:
            sites.append(
                {
                    "site_code": site["siteCode"],
                    "site_name": site.get("siteName", ""),
                    "state": site.get("stateCode", ""),
                    "latitude": site.get("siteLatitude"),
                    "longitude": site.get("siteLongitude"),
                    "domain": site.get("domainCode", ""),
                }
            )

    logger.info(f"Found {len(sites)} terrestrial NEON sites")
    return sites


def get_available_months(product_id: str, config: dict) -> dict[str, list[str]]:
    """Get available months per site for a data product."""
    url = f"{config['api_base_url']}/products/{product_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = {}
        for site_info in data.get("data", {}).get("siteCodes", []):
            result[site_info["siteCode"]] = site_info.get("availableMonths", [])
        return result
    except requests.RequestException as e:
        logger.warning(f"Failed to get product info for {product_id}: {e}")
        return {}


def get_available_data(
    site_code: str, product_id: str, month: str, config: dict
) -> list[dict]:
    """Get available data files for a site, product, and month."""
    url = f"{config['api_base_url']}/data/{product_id}/{site_code}/{month}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        files = []
        if data.get("data") and data["data"].get("files"):
            for file_info in data["data"]["files"]:
                files.append(
                    {
                        "name": file_info.get("name", ""),
                        "url": file_info.get("url", ""),
                        "size": file_info.get("size", 0),
                    }
                )
        return files

    except requests.RequestException as e:
        logger.debug(f"No data for {site_code}/{product_id}/{month}: {e}")
        return []


def download_product_data(
    sites: list[dict],
    product_id: str,
    product_name: str,
    config: dict,
    output_dir: Path,
    file_filter: str | None = None,
    max_months_per_site: int | None = None,
) -> pd.DataFrame:
    """Download and combine data for a product across all sites."""
    # Get available months per site
    available = get_available_months(product_id, config)
    all_frames = []

    for site in tqdm(sites, desc=f"Downloading {product_name}"):
        site_code = site["site_code"]
        months = available.get(site_code, [])
        if max_months_per_site:
            months = months[:max_months_per_site]

        for month in months:
            files = get_available_data(site_code, product_id, month, config)

            for file_info in files:
                file_url = file_info["url"]
                name = file_info["name"]
                if not file_url or not name.endswith(".csv"):
                    continue
                # Skip metadata/variable/validation files
                if any(skip in name for skip in ["variables.", "validation.", "categoricalCodes."]):
                    continue
                # Apply file filter if specified
                if file_filter and file_filter not in name:
                    continue

                try:
                    df = pd.read_csv(file_url)
                    df["siteID"] = site_code
                    all_frames.append(df)
                except Exception as e:
                    logger.debug(f"Failed to read {file_url}: {e}")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        output_path = output_dir / f"neon_{product_name}.csv"
        combined.to_csv(output_path, index=False)
        logger.info(
            f"Saved {product_name}: {combined.shape[0]} rows, "
            f"{combined.shape[1]} columns"
        )
        return combined

    logger.warning(f"No data collected for {product_name}")
    return pd.DataFrame()


def create_paired_dataset(
    microbe_df: pd.DataFrame,
    chemical_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Create paired dataset by matching microbiome data with
    environmental measurements by site and date.
    """
    if microbe_df.empty or chemical_df.empty:
        logger.warning("Cannot create paired dataset: missing data")
        return pd.DataFrame()

    # Standardize date columns for joining
    for df in [microbe_df, chemical_df, physical_df]:
        if "collectDate" in df.columns:
            df["collect_month"] = pd.to_datetime(
                df["collectDate"]
            ).dt.to_period("M")

    # Join on site + month
    paired = microbe_df.merge(
        chemical_df[
            [
                "siteID",
                "collect_month",
                "soilInWaterpH",
                "organicCPercent",
                "nitrogenPercent",
            ]
        ].drop_duplicates(),
        on=["siteID", "collect_month"],
        how="inner",
    )

    if not physical_df.empty and "collect_month" in physical_df.columns:
        paired = paired.merge(
            physical_df[
                [
                    "siteID",
                    "collect_month",
                    "soilMoisture",
                    "soilTemp",
                ]
            ].drop_duplicates(),
            on=["siteID", "collect_month"],
            how="left",
        )

    output_path = output_dir / "neon_paired.csv"
    paired.to_csv(output_path, index=False)
    logger.info(f"Created paired dataset: {paired.shape[0]} samples")
    return paired


def collect_all(config: dict, output_dir: Path, max_sites: int | None = None):
    """Main NEON collection pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get NEON sites
    logger.info("Step 1: Fetching NEON sites...")
    sites = get_neon_sites(config)
    if max_sites:
        # Filter to sites that have soil data products
        sites_with_data = []
        available = get_available_months(
            config["data_products"]["soil_chemical"]["id"], config
        )
        for s in sites:
            if s["site_code"] in available:
                sites_with_data.append(s)
            if len(sites_with_data) >= max_sites:
                break
        sites = sites_with_data
        logger.info(f"Limited to {len(sites)} sites with soil data")

    sites_df = pd.DataFrame(sites)
    sites_df.to_csv(output_dir / "neon_sites.csv", index=False)

    products = config["data_products"]

    # Step 2: Download soil pH data
    logger.info("Step 2: Downloading soil pH data...")
    chemical_df = download_product_data(
        sites,
        products["soil_chemical"]["id"],
        "soil_chemical",
        config,
        output_dir,
        file_filter="soilpH",
        max_months_per_site=2,
    )

    # Step 3: Download soil moisture data
    logger.info("Step 3: Downloading soil moisture data...")
    physical_df = download_product_data(
        sites,
        products["soil_physical"]["id"],
        "soil_physical",
        config,
        output_dir,
        file_filter="soilMoisture",
        max_months_per_site=2,
    )

    logger.info("NEON data collection complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Collect paired soil data from NEON"
    )
    parser.add_argument(
        "--config",
        default="data/configs/neon.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help="Max number of sites to collect (for testing)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    collect_all(config, output_dir, max_sites=args.max_sites)


if __name__ == "__main__":
    main()
