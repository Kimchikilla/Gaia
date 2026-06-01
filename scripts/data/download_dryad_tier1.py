"""
Download Tier 1 Dryad soil microbiome datasets (processed tables only).
- Dryad organic amendments: 16S/ITS/N-cycle OTU tables + soil chem metadata
- Dryad vanadium: KO functional + carbon fixation + sample metadata (skip raw fastq)
"""

import requests
import hashlib
import html
import json
import re
import time
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


def solve_anubis_download(url, outpath, referer):
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    response = session.get(url, timeout=60, headers={"Referer": referer})
    challenge = re.search(
        r'<script id="anubis_challenge" type="application/json">(.*?)\s*</script>',
        response.text,
        flags=re.S,
    )
    if not challenge:
        response.raise_for_status()
        if not response.content:
            raise RuntimeError("empty download response")
        outpath.write_bytes(response.content)
        return

    payload = json.loads(html.unescape(challenge.group(1)))
    challenge_data = payload["challenge"]
    difficulty = int(payload["rules"]["difficulty"])
    random_data = challenge_data["randomData"]
    target_prefix = "0" * difficulty
    start = time.time()

    nonce = 0
    while True:
        digest = hashlib.sha256(f"{random_data}{nonce}".encode()).hexdigest()
        if digest.startswith(target_prefix):
            break
        nonce += 1

    pass_url = "https://datadryad.org/.within.website/x/cmd/anubis/api/pass-challenge"
    pass_response = session.get(
        pass_url,
        params={
            "id": challenge_data["id"],
            "response": digest,
            "nonce": str(nonce),
            "redir": re.sub(r"^https?://[^/]+", "", url),
            "elapsedTime": str(max(int((time.time() - start) * 1000), 1000)),
        },
        timeout=120,
        allow_redirects=True,
    )
    pass_response.raise_for_status()
    if not pass_response.content:
        raise RuntimeError("empty download response after challenge")
    outpath.write_bytes(pass_response.content)


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
        expected_size = int(f.get("size") or 0)
        if outfile.exists() and expected_size > 0 and outfile.stat().st_size == expected_size:
            print(f"  EXISTS {path}")
            continue
        file_id = f["_links"]["self"]["href"].rstrip("/").split("/")[-1]
        dl = f"https://datadryad.org/downloads/file_stream/{file_id}"
        try:
            solve_anubis_download(
                dl,
                outfile,
                f"https://datadryad.org/dataset/doi:{cfg['doi']}",
            )
        except Exception as e:
            print(f"  FAIL {path}: {e}")

print("\nDone.")
