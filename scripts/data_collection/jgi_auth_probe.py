"""JGI 인증 + 토양 metagenome 카탈로그 점검.

선결 조건 — 사용자가 환경변수 설정:
  $env:JGI_USER = "your_email_or_username"
  $env:JGI_PASS = "your_password"

본 스크립트:
  1. signon.jgi.doe.gov 에 로그인 → 쿠키 받음
  2. 그 쿠키로 토양 metagenome 카탈로그 다운 시도
  3. 첫 metagenome 의 파일 목록 확인 → KEGG/COG annotation 있는지 확인

비밀번호는 절대 코드에 박지 말 것. 환경변수 또는 ~/.netrc 만.
"""
import os
import sys
import json
import requests
from pathlib import Path

# auto-load .env at repo root
def _load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists(): return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)
_load_dotenv()

OUT = Path("data/raw/jgi_probe")
OUT.mkdir(parents=True, exist_ok=True)

SIGNON_URL = "https://signon.jgi.doe.gov/signon/create"
PORTAL_BASE = "https://genome.jgi.doe.gov/portal"


def login():
    user = os.environ.get("JGI_USER")
    pw = os.environ.get("JGI_PASS")
    if not user or not pw:
        print("ERROR: set JGI_USER and JGI_PASS env vars first", file=sys.stderr)
        print("  PowerShell:  $env:JGI_USER='you@example.com'; $env:JGI_PASS='yourpw'", file=sys.stderr)
        print("  bash:        export JGI_USER=...; export JGI_PASS=...", file=sys.stderr)
        sys.exit(2)

    s = requests.Session()
    print(f"[jgi] signing in as {user[:3]}...{user[-3:] if '@' in user else ''}")
    r = s.post(SIGNON_URL, data={"login": user, "password": pw}, timeout=30,
               allow_redirects=True)
    if r.status_code not in (200, 302):
        print(f"  signon failed: HTTP {r.status_code}")
        print(r.text[:500])
        sys.exit(1)
    if "JGI_SESSION" not in [c.name for c in s.cookies]:
        # JGI sets a cookie like "jgi_session" or "JGI_SESSION"
        print(f"  cookies set: {[c.name for c in s.cookies]}")
        # accept whatever cookies got set; many JGI flows use signon cookie
        if not s.cookies:
            print("  no cookies — login likely failed")
            sys.exit(1)
    print(f"  signon OK. cookies: {[c.name for c in s.cookies]}")
    return s


def list_soil_metagenomes(s, organism_query="soil"):
    """Try the search-service API."""
    print()
    print(f"[probe] searching '{organism_query}' on JGI portal...")
    r = s.get(f"{PORTAL_BASE}/ext-api/search-service/search",
              params={"q": organism_query, "p": 1, "x": 25}, timeout=30,
              headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    print(f"  search status: {r.status_code}, content-type: {r.headers.get('content-type', '?')[:30]}")
    try:
        d = r.json()
        # JGI sometimes wraps in different keys
        for k in ("hits", "data", "results", "organism_list"):
            if k in d:
                print(f"  found {len(d[k])} results in '{k}'")
                if d[k]:
                    print(f"  first: {json.dumps(d[k][0], indent=2)[:400]}")
                break
        else:
            print(f"  json keys: {list(d.keys())[:10]}")
            print(json.dumps(d, indent=2)[:500])
    except Exception as e:
        print(f"  parse failed: {e}")
        print(r.text[:500])


def list_organism_files(s, organism="soil_metagenomes"):
    """get-directory endpoint."""
    print()
    print(f"[probe] get-directory for organism={organism}")
    r = s.get(f"{PORTAL_BASE}/ext-api/downloads/get-directory",
              params={"organism": organism}, timeout=30,
              headers={"User-Agent": "Mozilla/5.0"})
    print(f"  status: {r.status_code}, len: {len(r.content)}")
    if "<?xml" in r.text[:50] or "<organismDownloads" in r.text:
        # XML response — JGI common format
        OUT.joinpath(f"directory_{organism}.xml").write_text(r.text, encoding="utf-8")
        print(f"  saved → {OUT}/directory_{organism}.xml")
        # crude file count
        import re
        files = re.findall(r"<file\s+", r.text)
        print(f"  approx files in listing: {len(files)}")
    else:
        print(f"  preview: {r.text[:500]}")


def main():
    s = login()
    # Try a known JGI portal organism
    for org in ["soil_metagenomes", "soil_metagenome", "Soil_FD", "EarthMicrobiome"]:
        try:
            list_organism_files(s, org)
        except Exception as e:
            print(f"  err {org}: {e}")

    # Also try search
    list_soil_metagenomes(s, "soil metagenome")
    list_soil_metagenomes(s, "soil 16S")


if __name__ == "__main__":
    main()
