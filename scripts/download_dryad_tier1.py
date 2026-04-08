"""
Download Tier 1 Dryad soil microbiome datasets (processed tables only).
- Dryad organic amendments: 16S/ITS/N-cycle OTU tables + soil chem metadata
- Dryad vanadium: KO functional + carbon fixation + sample metadata (skip raw fastq)
"""

import requests
from pathlib import Path
from tqdm import tqdm

DATASETS = {
    "dryad_amendments": {
        "doi": "10.5061/dryad.4qrfj6q9n",
        "skip_patterns": [],  # take all (small)
    },
    "dryad_vanadium": {
        "doi": "10.5061/dryad.6wwpzgn52",
        "skip_patterns": [".fastq.gz"],  # skip 16GB raw seqs
    },
}


def get_files(doi):
    enc = "doi%3A" + doi.replace("/", "%2F")
    url = f"https://datadryad.org/api/v2/datasets/{enc}/versions"
    r = requests.get(url, timeout=30).json()
    ver = r["_embedded"]["stash:versions"][-1]
    files_url = "https://datadryad.org" + ver["_links"]["stash:files"]["href"]
    fr = requests.get(files_url, timeout=30).json()
    return fr["_embedded"]["stash:files"]


def download(url, outpath):
    r = requests.get(url, stream=True, timeout=60)
    total = int(r.headers.get("content-length", 0))
    with open(outpath, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=outpath.name
    ) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


for name, cfg in DATASETS.items():
    out_dir = Path(f"data/raw/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ({cfg['doi']}) ===")
    files = get_files(cfg["doi"])
    for f in files:
        path = f["path"]
        if any(p in path for p in cfg["skip_patterns"]):
            print(f"  SKIP {path}")
            continue
        outfile = out_dir / path
        if outfile.exists() and outfile.stat().st_size > 0:
            print(f"  EXISTS {path}")
            continue
        dl = "https://datadryad.org" + f["_links"]["stash:download"]["href"]
        try:
            download(dl, outfile)
        except Exception as e:
            print(f"  FAIL {path}: {e}")

print("\nDone.")
