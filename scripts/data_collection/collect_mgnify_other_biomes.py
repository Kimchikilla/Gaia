"""MGnify 또 다른 토양 인접 biome — 더 광범위.

토양 미생물과 분포가 비슷한 인접 환경:
  - Engineered 농업 토양 처리장
  - Mountain / High altitude soil
  - Mangrove / Coastal sediment
  - Anthropogenic terrestrial 세부

각각의 biome 에서 받을 수 있는 최대 시도.
저장: data/raw/mgnify_v6/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from tqdm import tqdm

from collect_mgnify_parallel import fetch_analyses_for_study, process_one_analysis, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
OUT = Path("data/raw/mgnify_v6")
OUT.mkdir(parents=True, exist_ok=True)

BIOMES = [
    "root:Engineered:Bioremediation:Soil",
    "root:Engineered:Bioremediation:Hydrocarbon",
    "root:Engineered:Bioreactor",
    "root:Engineered:Wastewater",
    "root:Environmental:Terrestrial:Anthropogenic",
    "root:Environmental:Terrestrial:Mining",
]


def fetch_studies(lineage, base_url, retries=3):
    studies = []
    url = f"{base_url}/biomes/{lineage}/studies"
    for attempt in range(retries):
        try:
            while url:
                r = requests.get(url, params={"page_size": 100}, timeout=60)
                if r.status_code == 404: return studies
                r.raise_for_status()
                d = r.json()
                for it in d.get("data", []):
                    studies.append({"study_id": it["id"], "biome_lineage": lineage})
                url = d.get("links", {}).get("next")
                time.sleep(0.5)
            return studies
        except Exception as e:
            logger.warning(f"{lineage} attempt {attempt+1}: {e}"); time.sleep(15)
    return studies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-extra", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    base = "https://www.ebi.ac.uk/metagenomics/api/v1"
    config = {"api_base_url": base, "page_size": 100}

    existing = set()
    for p in [Path("data/raw/mgnify/mgnify_abundance.csv"),
              Path("data/raw/mgnify_v3/mgnify_v3_abundance.csv"),
              Path("data/raw/mgnify_v4/mgnify_v4_abundance.csv"),
              Path("data/raw/mgnify_v5/mgnify_v5_abundance.csv")]:
        if p.exists():
            try:
                ex = pd.read_csv(p)
                if "analysis_id" in ex.columns:
                    existing |= set(ex["analysis_id"].astype(str))
            except Exception: pass
    logger.info(f"existing analyses: {len(existing)}")

    studies = []
    for lin in BIOMES:
        s = fetch_studies(lin, base)
        logger.info(f"  {lin:55s} -> {len(s)}")
        studies.extend(s)
    seen = set(); unique = []
    for s in studies:
        if s["study_id"] not in seen:
            seen.add(s["study_id"]); unique.append(s)
    pd.DataFrame(unique).to_csv(OUT / "studies_v6.csv", index=False)
    logger.info(f"unique studies: {len(unique)}")

    new_analyses = []
    for st in tqdm(unique, desc="Studies"):
        try:
            ans = fetch_analyses_for_study(st["study_id"], config)
        except Exception: continue
        for a in ans:
            if str(a["analysis_id"]) in existing: continue
            new_analyses.append(a)
        if len(new_analyses) >= args.max_extra * 2:
            break
    logger.info(f"new analyses: {len(new_analyses)}")
    new_analyses = new_analyses[:args.max_extra * 2]

    abundance, metadata = [], []; seen_s = set(); n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one_analysis, a, base): a for a in new_analyses}
        with tqdm(total=len(futs), desc="Download") as pb:
            for fu in as_completed(futs):
                try: res = fu.result(timeout=120)
                except Exception: res = None
                if res:
                    abundance.append(res["abundance"])
                    sid = res["metadata"]["sample_id"]
                    if sid and sid not in seen_s:
                        metadata.append(res["metadata"]); seen_s.add(sid)
                    n_ok += 1
                else: n_fail += 1
                pb.update(1); pb.set_postfix(ok=n_ok, fail=n_fail)
                if n_ok and n_ok % 100 == 0:
                    save_checkpoint(abundance, metadata, OUT,
                        {"abundance_file": "mgnify_v6_abundance.csv",
                         "metadata_file": "mgnify_v6_metadata.csv"})
                if n_ok >= args.max_extra: break

    save_checkpoint(abundance, metadata, OUT,
        {"abundance_file": "mgnify_v6_abundance.csv",
         "metadata_file": "mgnify_v6_metadata.csv"})
    logger.info(f"Done. ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
