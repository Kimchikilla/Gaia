"""MGnify 추가 수집 — resumable, 기존 v1 analysis_id 는 건너뜀.

기존 collect_mgnify_parallel.py 의 핵심 함수를 재사용하면서:
  1) 이미 다운받은 analysis_id 셋 로드 → 그 외만 시도
  2) Rhizosphere lineage 추가 (Naylor 와 같은 매트릭스)
  3) 한 lineage 가 404/timeout 떠도 다음으로 진행
  4) 출력 디렉터리 분리 — data/raw/mgnify_v3/ (v1 무손실)
  5) 진행상황을 200 샘플마다 체크포인트
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import yaml
from tqdm import tqdm

# Reuse the existing functions
from collect_mgnify_parallel import (
    fetch_analyses_for_study,
    process_one_analysis,
    save_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("data/raw/mgnify_v3")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_studies_safe(lineage: str, base_url: str, page_size: int = 100, max_retries: int = 2):
    studies = []
    url = f"{base_url}/biomes/{lineage}/studies"
    for attempt in range(max_retries):
        try:
            while url:
                r = requests.get(url, params={"page_size": page_size}, timeout=45)
                if r.status_code == 404:
                    logger.warning(f"404 {lineage}")
                    return studies
                r.raise_for_status()
                d = r.json()
                for it in d.get("data", []):
                    studies.append({
                        "study_id": it["id"],
                        "biome_lineage": lineage,
                        "study_name": it["attributes"].get("study-name", ""),
                        "samples_count": it["attributes"].get("samples-count", 0),
                    })
                url = d.get("links", {}).get("next")
                time.sleep(0.5)
            return studies
        except requests.RequestException as e:
            logger.warning(f"{lineage} attempt {attempt+1}: {e}")
            time.sleep(5)
    return studies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/configs/mgnify.yaml")
    ap.add_argument("--max-extra", type=int, default=4000,
                    help="Max NEW samples to collect (on top of existing v1)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--existing", default="data/raw/mgnify/mgnify_abundance.csv",
                    help="Path to existing abundance CSV — analysis_id col used to skip")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    base_url = config["api_base_url"]

    # 1. Load existing analysis_ids to skip
    existing_aids = set()
    if Path(args.existing).exists():
        ex = pd.read_csv(args.existing)
        if "analysis_id" in ex.columns:
            existing_aids = set(ex["analysis_id"].astype(str))
        logger.info(f"Existing analysis_ids: {len(existing_aids)}")

    # 2. Lineages — config + extras
    base_lineages = config.get("biome_lineages", [])
    extra_lineages = [
        "root:Host-associated:Plants:Rhizosphere",
        "root:Environmental:Terrestrial:Soil:Wetlands",
    ]
    all_lineages = list(dict.fromkeys(base_lineages + extra_lineages))
    logger.info(f"Lineages to scan: {len(all_lineages)}")

    # 3. Find studies
    studies = []
    for lin in all_lineages:
        s = fetch_studies_safe(lin, base_url, page_size=config.get("page_size", 100))
        logger.info(f"  {lin:60s} -> {len(s)} studies")
        studies.extend(s)

    seen = set()
    unique_studies = []
    for s in studies:
        if s["study_id"] not in seen:
            seen.add(s["study_id"])
            unique_studies.append(s)
    logger.info(f"Unique studies: {len(unique_studies)}")
    pd.DataFrame(unique_studies).to_csv(OUT / "studies_v3.csv", index=False)

    # 4. Find analyses (skip already-done)
    logger.info("Fetching analyses per study...")
    new_analyses = []
    for st in tqdm(unique_studies, desc="Studies"):
        try:
            ans = fetch_analyses_for_study(st["study_id"], config)
        except Exception as e:
            logger.warning(f"study {st['study_id']}: {e}")
            continue
        for a in ans:
            if str(a["analysis_id"]) in existing_aids:
                continue
            new_analyses.append(a)
        if len(new_analyses) >= args.max_extra * 2:  # over-collect, downloads will fail some
            break
    logger.info(f"NEW analyses found: {len(new_analyses)} (target: {args.max_extra})")
    new_analyses = new_analyses[:args.max_extra * 2]

    # 5. Parallel BIOM download
    abundance, metadata = [], []
    seen_samples = set()
    n_ok, n_fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one_analysis, a, base_url): a for a in new_analyses}
        with tqdm(total=len(futs), desc="Download") as pb:
            for fu in as_completed(futs):
                try:
                    res = fu.result(timeout=120)
                except Exception as e:
                    res = None
                    logger.debug(f"future error: {e}")
                if res:
                    abundance.append(res["abundance"])
                    sid = res["metadata"]["sample_id"]
                    if sid and sid not in seen_samples:
                        metadata.append(res["metadata"])
                        seen_samples.add(sid)
                    n_ok += 1
                else:
                    n_fail += 1
                pb.update(1); pb.set_postfix(ok=n_ok, fail=n_fail)
                if n_ok > 0 and n_ok % args.checkpoint_every == 0:
                    cfg2 = {"abundance_file": "mgnify_v3_abundance.csv",
                            "metadata_file":  "mgnify_v3_metadata.csv"}
                    save_checkpoint(abundance, metadata, OUT, cfg2)
                    logger.info(f"checkpoint at {n_ok} ok")
                if n_ok >= args.max_extra:
                    logger.info(f"reached target {args.max_extra}, stopping")
                    break

    cfg2 = {"abundance_file": "mgnify_v3_abundance.csv",
            "metadata_file":  "mgnify_v3_metadata.csv"}
    save_checkpoint(abundance, metadata, OUT, cfg2)
    logger.info(f"Done. ok={n_ok} fail={n_fail}")
    logger.info(f"Saved to {OUT}/")


if __name__ == "__main__":
    main()
