"""MGnify /samples API 직접 — study 안 거치고 sample 단위 페이지네이션.

study_id → analyses → BIOM 보다 빠를 수 있음.
저장: data/raw/mgnify_v5/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from tqdm import tqdm

from collect_mgnify_parallel import fetch_biom_taxonomy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("data/raw/mgnify_v5")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://www.ebi.ac.uk/metagenomics/api/v1"


def list_samples_by_biome(biome_lineage, max_pages=100):
    """Sample 단위 페이지 — analysis 정보까지 포함."""
    samples = []
    url = f"{BASE}/biomes/{biome_lineage}/samples"
    params = {"page_size": 100}
    page = 0
    while url and page < max_pages:
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            logger.warning(f"{biome_lineage} page {page}: {e}")
            time.sleep(15); continue
        for it in d.get("data", []):
            attrs = it.get("attributes", {})
            samples.append({
                "sample_id": it.get("id"),
                "biome_lineage": biome_lineage,
                "sample_name": attrs.get("sample-name", ""),
                "latitude": attrs.get("latitude"),
                "longitude": attrs.get("longitude"),
                "biome": attrs.get("environment-biome", ""),
                "feature": attrs.get("environment-feature", ""),
            })
        url = d.get("links", {}).get("next")
        params = {}  # next URL has it baked in
        page += 1
        time.sleep(0.5)
    return samples


def fetch_sample_analyses(sample_id):
    """One sample → analyses (each analysis has its own BIOM)."""
    try:
        r = requests.get(f"{BASE}/samples/{sample_id}/analyses",
                         params={"page_size": 50}, timeout=30)
        r.raise_for_status()
        d = r.json()
        return [it["id"] for it in d.get("data", [])]
    except Exception:
        return []


def process_sample(sample_id):
    """Fetch first available analysis BIOM."""
    aids = fetch_sample_analyses(sample_id)
    for aid in aids:
        gc = fetch_biom_taxonomy(aid, BASE)
        if gc:
            return {"sample_id": sample_id, "analysis_id": aid, **gc}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biome", default="root:Environmental:Terrestrial:Soil")
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # already-fetched analysis IDs
    existing = set()
    for p in [Path("data/raw/mgnify/mgnify_abundance.csv"),
              Path("data/raw/mgnify_v3/mgnify_v3_abundance.csv"),
              Path("data/raw/mgnify_v4/mgnify_v4_abundance.csv")]:
        if p.exists():
            try:
                ex = pd.read_csv(p)
                if "analysis_id" in ex.columns:
                    existing |= set(ex["analysis_id"].astype(str))
            except Exception: pass
    logger.info(f"existing analyses: {len(existing)}")

    logger.info(f"Listing samples for biome: {args.biome}")
    samples = list_samples_by_biome(args.biome, max_pages=200)
    logger.info(f"found {len(samples)} samples")
    pd.DataFrame(samples).to_csv(OUT / "samples_v5.csv", index=False)

    samples = samples[:args.max_samples]
    abundance = []
    n_ok, n_fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_sample, s["sample_id"]): s for s in samples}
        with tqdm(total=len(futs), desc="Sample") as pb:
            for fu in as_completed(futs):
                try:
                    res = fu.result(timeout=120)
                except Exception:
                    res = None
                if res and str(res.get("analysis_id")) not in existing:
                    abundance.append(res)
                    n_ok += 1
                else:
                    n_fail += 1
                pb.update(1); pb.set_postfix(ok=n_ok, fail=n_fail)
                if n_ok and n_ok % 100 == 0:
                    pd.DataFrame(abundance).fillna(0).to_csv(
                        OUT / "mgnify_v5_abundance.csv", index=False)

    pd.DataFrame(abundance).fillna(0).to_csv(OUT / "mgnify_v5_abundance.csv", index=False)
    logger.info(f"Done. ok={n_ok} fail={n_fail}, saved {OUT}")


if __name__ == "__main__":
    main()
