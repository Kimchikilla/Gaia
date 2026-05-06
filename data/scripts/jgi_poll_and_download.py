"""JGI 복원 요청 폴링 + 다운로드 + 파싱.

흐름:
  1. request_archived_files/requests/{id} 폴링 — pending → ready
  2. ready 되면 download_files POST 로 zip/tar 받음
  3. 받은 파일을 data/raw/jgi_manual/ 에 저장
  4. .ko.txt / .cog.txt / .pfam.txt 파싱

사용:
  python data/scripts/jgi_poll_and_download.py --request-id 587420
  python data/scripts/jgi_poll_and_download.py --request-id 587420 --poll-interval 600  # 10분마다
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path

import requests


def _load_dotenv(p=".env"):
    p = Path(p)
    if not p.exists(): return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
TOKEN = os.environ.get("JGI_TOKEN")
H = {"Authorization": TOKEN, "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
BASE = "https://files.jgi.doe.gov"
OUT = Path("data/raw/jgi_manual")
OUT.mkdir(parents=True, exist_ok=True)


def get_request_status(rid):
    r = requests.get(f"{BASE}/request_archived_files/requests/{rid}", headers=H, timeout=60)
    r.raise_for_status()
    return r.json()


def download_files(es_ids):
    """POST download_files/ with es _id list. Returns response (streamed)."""
    H2 = dict(H); H2["Accept"] = "*/*"
    r = requests.post(f"{BASE}/download_files/", headers=H2,
                      json={"ids": list(es_ids)}, timeout=300, stream=True,
                      allow_redirects=False)
    return r


def save_response(r, fname):
    out = OUT / fname
    sz = 0
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                sz += len(chunk)
    return out, sz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request-id", type=str, required=True)
    ap.add_argument("--poll-interval", type=int, default=600,
                    help="seconds between status polls (default 600 = 10 min)")
    ap.add_argument("--max-wait", type=int, default=24*3600,
                    help="give up after N seconds (default 24 hours)")
    args = ap.parse_args()

    if not TOKEN:
        print("ERR: JGI_TOKEN not set in .env", file=sys.stderr); sys.exit(1)

    rid = args.request_id
    start = time.time()
    print(f"[poll] request_id={rid}, interval={args.poll_interval}s, max_wait={args.max_wait}s")

    while True:
        elapsed = int(time.time() - start)
        try:
            st = get_request_status(rid)
        except Exception as e:
            print(f"[{elapsed}s] poll error: {e}")
            time.sleep(args.poll_interval); continue
        s = st.get("status", "?")
        print(f"[{elapsed}s] status={s}")
        if s.lower() in ("ready", "complete", "done", "completed"):
            print("[poll] ready! starting download")
            break
        if elapsed > args.max_wait:
            print("[poll] timeout"); sys.exit(2)
        time.sleep(args.poll_interval)

    es_ids = st.get("file_ids") or []
    print(f"[dl] {len(es_ids)} file(s) to download")

    # Try downloading all in one batch
    r = download_files(es_ids)
    print(f"[dl] status:{r.status_code}  ct={r.headers.get('content-type','?')[:40]}  cd={r.headers.get('content-disposition','?')[:80]}")
    if r.status_code != 200:
        body = b"".join(r.iter_content(2048))
        print(f"[dl] body: {body[:500].decode('utf-8','ignore')}")
        sys.exit(3)

    # Determine filename from Content-Disposition or fallback
    cd = r.headers.get("content-disposition", "")
    import re
    m = re.search(r'filename[^=]*=("?)([^";]+)\1', cd)
    fname = m.group(2) if m else f"jgi_request_{rid}.zip"
    out, sz = save_response(r, fname)
    print(f"[dl] saved {out} ({sz/1024/1024:.1f} MB)")

    # Try to extract if zip/tar
    if fname.endswith((".zip", ".tar.gz", ".tar")):
        extract_dir = OUT / f"request_{rid}"
        extract_dir.mkdir(exist_ok=True)
        if fname.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(out) as z:
                z.extractall(extract_dir)
        else:
            import tarfile
            with tarfile.open(out) as t:
                t.extractall(extract_dir)
        print(f"[dl] extracted -> {extract_dir}")
        for f in extract_dir.rglob("*"):
            if f.is_file():
                print(f"    {f.relative_to(extract_dir)}  {f.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
