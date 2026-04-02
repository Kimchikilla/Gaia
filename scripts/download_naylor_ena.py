"""
Download Naylor fastq files from ENA (European Nucleotide Archive).
Much simpler than SRA Toolkit — direct HTTP download.
"""

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import urllib.request
import time

OUTPUT_DIR = Path("data/raw/naylor/fastq")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata
meta = pd.read_csv("data/raw/naylor/naylor_metadata.csv")
run_ids = meta["run_id"].tolist()
print(f"Samples to download: {len(run_ids)}")


def get_fastq_url(run_id):
    """Get fastq download URL from ENA."""
    url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={run_id}&result=read_run&fields=fastq_ftp&format=json"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if data and data[0].get("fastq_ftp"):
            ftp_paths = data[0]["fastq_ftp"].split(";")
            # Convert FTP to HTTP
            return [f"http://{p}" for p in ftp_paths]
    except Exception:
        pass
    return None


def download_one(run_id):
    """Download fastq for one run."""
    outfile = OUTPUT_DIR / f"{run_id}.fastq.gz"
    if outfile.exists() and outfile.stat().st_size > 1000:
        return run_id, "skip"

    urls = get_fastq_url(run_id)
    if not urls:
        return run_id, "no_url"

    try:
        urllib.request.urlretrieve(urls[0], str(outfile))
        return run_id, "ok"
    except Exception as e:
        return run_id, f"error"


# Parallel download (10 workers)
n_ok, n_fail, n_skip = 0, 0, 0

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_one, rid): rid for rid in run_ids}

    with tqdm(total=len(futures), desc="Downloading") as pbar:
        for future in as_completed(futures):
            rid, status = future.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
            pbar.update(1)
            pbar.set_postfix(ok=n_ok, fail=n_fail, skip=n_skip)

print(f"\nDone! ok={n_ok}, fail={n_fail}, skip={n_skip}")

# Check total size
total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.fastq.gz"))
print(f"Total downloaded: {total_size / 1024 / 1024 / 1024:.1f} GB")
