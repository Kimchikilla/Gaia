"""NEON DP1.10081.001 — soil microbe community composition (16S only).

각 site/month 폴더에서 sample 단위 16S CSV 를 모아 abundance 표로 합친다.
Genus 단위로 individualCount 를 sum.

출력:
  data/raw/neon/neon_microbe_abundance.csv  (sample_id × genera)
  data/raw/neon/neon_microbe_metadata.csv   (sample_id, site, month, ...)
"""
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API = "https://data.neonscience.org/api/v0"
PRODUCT = "DP1.10081.001"
OUT = Path("data/raw/neon")
OUT.mkdir(parents=True, exist_ok=True)


def list_site_months():
    r = requests.get(f"{API}/products/{PRODUCT}", timeout=60)
    r.raise_for_status()
    out = []
    for s in r.json()["data"]["siteCodes"]:
        site = s["siteCode"]
        for m in s.get("availableMonths", []):
            out.append((site, m))
    return out


def list_sample_files(site, month):
    """Return [(name, url), ...] for 16S per-sample CSVs in one site/month."""
    try:
        r = requests.get(f"{API}/data/{PRODUCT}/{site}/{month}", timeout=60)
        r.raise_for_status()
        files = r.json()["data"]["files"]
    except requests.RequestException as e:
        logger.warning(f"{site}/{month}: list failed: {e}")
        return []
    out = []
    for f in files:
        name = f.get("name", "")
        # per-sample 16S CSV; skip NEON.* aggregate files
        if "_16S__" in name and not name.startswith("NEON"):
            out.append((name, f.get("url"), f.get("size", 0)))
    return out


def fetch_one(name, url, site, month):
    """Download + aggregate per-genus counts for one sample CSV."""
    try:
        df = pd.read_csv(url, usecols=["dnaSampleID", "genus", "individualCount"])
    except Exception as e:
        return None
    if df.empty:
        return None
    df = df.dropna(subset=["genus"])
    if df.empty:
        return None
    sample_id = df["dnaSampleID"].iloc[0]
    grp = df.groupby("genus")["individualCount"].sum()
    rec = {"sample_id": str(sample_id), "site": site, "month": month}
    rec.update(grp.to_dict())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=3000)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    args = ap.parse_args()

    logger.info("Listing site/months...")
    sm = list_site_months()
    logger.info(f"site-months: {len(sm)}")

    logger.info("Enumerating per-sample 16S files...")
    file_jobs = []
    for site, month in tqdm(sm, desc="Index"):
        for name, url, _ in list_sample_files(site, month):
            file_jobs.append((name, url, site, month))
    logger.info(f"Total 16S sample files: {len(file_jobs)}")

    file_jobs = file_jobs[:args.max_samples]

    records = []
    meta = []
    n_ok, n_fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, n, u, s, m): n for (n, u, s, m) in file_jobs}
        with tqdm(total=len(futs), desc="Download") as pb:
            for fu in as_completed(futs):
                try:
                    rec = fu.result(timeout=120)
                except Exception:
                    rec = None
                if rec:
                    records.append(rec)
                    meta.append({"sample_id": rec["sample_id"], "site": rec["site"], "month": rec["month"]})
                    n_ok += 1
                else:
                    n_fail += 1
                pb.update(1); pb.set_postfix(ok=n_ok, fail=n_fail)
                if n_ok > 0 and n_ok % args.checkpoint_every == 0:
                    pd.DataFrame(records).fillna(0).to_csv(OUT / "neon_microbe_abundance.csv", index=False)
                    pd.DataFrame(meta).to_csv(OUT / "neon_microbe_metadata.csv", index=False)
                    logger.info(f"checkpoint at {n_ok}")

    pd.DataFrame(records).fillna(0).to_csv(OUT / "neon_microbe_abundance.csv", index=False)
    pd.DataFrame(meta).to_csv(OUT / "neon_microbe_metadata.csv", index=False)
    logger.info(f"Saved: ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
