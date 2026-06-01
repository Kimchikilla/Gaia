"""MGnify 인접 lineage 추가 수집 — soil 외에 plant rhizosphere, marine sediment 등.

기존 v1, v3 받은 analysis_id 와 중복 안되게 skip.
저장: data/raw/mgnify_v4/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import yaml
from tqdm import tqdm

from collect_mgnify_parallel import fetch_analyses_for_study, process_one_analysis, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("data/raw/mgnify_v4")
OUT.mkdir(parents=True, exist_ok=True)

# 기존 코퍼스 외 토양·미생물 인접 lineage
EXTRA_LINEAGES = [
    "root:Host-associated:Plants:Rhizosphere",
    "root:Host-associated:Plants:Roots",
    "root:Host-associated:Plants:Phylloplane",
    "root:Environmental:Aquatic:Marine:Sediment",
    "root:Environmental:Aquatic:Freshwater:Sediment",
    "root:Environmental:Terrestrial:Volcanic",
    "root:Environmental:Terrestrial:Cave",
    "root:Environmental:Terrestrial:Tundra",
    "root:Environmental:Air",
]


def fetch_studies_safe(lineage, base_url, page_size=100, max_retries=3):
    studies = []
    url = f"{base_url}/biomes/{lineage}/studies"
    for attempt in range(max_retries):
        try:
            while url:
                r = requests.get(url, params={"page_size": page_size}, timeout=60)
                if r.status_code == 404:
                    return studies
                r.raise_for_status()
                d = r.json()
                for it in d.get("data", []):
                    studies.append({
                        "study_id": it["id"], "biome_lineage": lineage,
                        "samples_count": it["attributes"].get("samples-count", 0),
                    })
                url = d.get("links", {}).get("next")
                time.sleep(0.5)
            return studies
        except requests.RequestException as e:
            logger.warning(f"{lineage} attempt {attempt+1}: {e}")
            time.sleep(10)
    return studies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/configs/mgnify.yaml")
    ap.add_argument("--max-extra", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    base_url = config["api_base_url"]

    # combined existing analysis_id from v1 and v3
    existing = set()
    for p in [Path("data/raw/mgnify/mgnify_abundance.csv"),
              Path("data/raw/mgnify_v3/mgnify_v3_abundance.csv")]:
        if p.exists():
            try:
                ex = pd.read_csv(p)
                if "analysis_id" in ex.columns:
                    existing |= set(ex["analysis_id"].astype(str))
                logger.info(f"existing analysis_ids from {p}: total {len(existing)}")
            except Exception as e:
                logger.warning(f"could not read {p}: {e}")

    logger.info(f"Lineages: {len(EXTRA_LINEAGES)}")
    studies = []
    for lin in EXTRA_LINEAGES:
        s = fetch_studies_safe(lin, base_url)
        logger.info(f"  {lin:60s} -> {len(s)} studies")
        studies.extend(s)

    seen = set(); unique_studies = []
    for s in studies:
        if s["study_id"] not in seen:
            seen.add(s["study_id"]); unique_studies.append(s)
    logger.info(f"unique studies: {len(unique_studies)}")
    pd.DataFrame(unique_studies).to_csv(OUT / "studies_v4.csv", index=False)

    # analyses
    logger.info("Fetching analyses per study...")
    new_analyses = []
    for st in tqdm(unique_studies, desc="Studies"):
        try:
            ans = fetch_analyses_for_study(st["study_id"], config)
        except Exception as e:
            logger.warning(f"{st['study_id']}: {e}")
            continue
        for a in ans:
            if str(a["analysis_id"]) in existing:
                continue
            new_analyses.append(a)
        if len(new_analyses) >= args.max_extra * 2:
            break
    logger.info(f"NEW analyses: {len(new_analyses)} (target {args.max_extra})")
    new_analyses = new_analyses[:args.max_extra * 2]

    abundance, metadata = [], []
    seen_samples = set()
    n_ok, n_fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one_analysis, a, base_url): a for a in new_analyses}
        with tqdm(total=len(futs), desc="Download") as pb:
            for fu in as_completed(futs):
                try:
                    res = fu.result(timeout=120)
                except Exception:
                    res = None
                if res:
                    abundance.append(res["abundance"])
                    sid = res["metadata"]["sample_id"]
                    if sid and sid not in seen_samples:
                        metadata.append(res["metadata"]); seen_samples.add(sid)
                    n_ok += 1
                else:
                    n_fail += 1
                pb.update(1); pb.set_postfix(ok=n_ok, fail=n_fail)
                if n_ok and n_ok % 100 == 0:
                    cfg2 = {"abundance_file": "mgnify_v4_abundance.csv",
                            "metadata_file": "mgnify_v4_metadata.csv"}
                    save_checkpoint(abundance, metadata, OUT, cfg2)
                if n_ok >= args.max_extra:
                    break

    cfg2 = {"abundance_file": "mgnify_v4_abundance.csv",
            "metadata_file": "mgnify_v4_metadata.csv"}
    save_checkpoint(abundance, metadata, OUT, cfg2)
    logger.info(f"Done. ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
