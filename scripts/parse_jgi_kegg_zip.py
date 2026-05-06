"""JGI 다운 zip 안의 .a.ko.txt 들을 파싱해 sample × KO_id 카운트 행렬로 변환.

각 .a.ko.txt 라인:
  gene_id  is_top_hit  KO:Kxxxxx  ident  qstart qend  sstart send  evalue  bits  aln_len

처리:
  - is_top_hit == 'Yes' 만 카운트
  - sample_id = 파일명 prefix (예 3300004081)
  - 결과: data/raw/jgi_manual/jgi_kegg_counts.csv (sample_id × Kxxxxx)

사용:
  python scripts/parse_jgi_kegg_zip.py data/raw/jgi_manual/IMG_SP-1003787_587420.zip
  # 또는 폴더 모드 (여러 zip 합치기):
  python scripts/parse_jgi_kegg_zip.py data/raw/jgi_manual/
"""
import sys
import zipfile
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_ko_stream(stream, sample_id):
    """yield (sample_id, KO_id) for top-hit lines in the file."""
    counts = Counter()
    n_lines = 0
    n_hits = 0
    for raw in stream:
        n_lines += 1
        line = raw.decode("utf-8", errors="replace")
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3: continue
        if parts[1] != "Yes": continue
        ko = parts[2]
        if ko.startswith("KO:"):
            ko = ko[3:]
        counts[ko] += 1
        n_hits += 1
    return counts, n_lines, n_hits


def process_zip(zp: Path):
    """Iterate through .a.ko.txt files in a zip, return {sample_id: Counter}."""
    samples = {}
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            name = info.filename
            if not name.endswith("a.ko.txt"):
                continue
            # extract sample_id (e.g., 3300004081 from path)
            base = Path(name).name
            m = re.match(r"^(\d+)\.a\.ko\.txt$", base)
            sid = m.group(1) if m else base
            print(f"  parsing {name} -> sample_id={sid} ({info.file_size/1024/1024:.1f} MB)")
            with z.open(info) as f:
                counts, n_lines, n_hits = parse_ko_stream(f, sid)
            print(f"    lines={n_lines:,}  top-hits={n_hits:,}  unique_KOs={len(counts):,}")
            samples[sid] = counts
    return samples


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    target = Path(sys.argv[1])
    out_dir = Path("data/raw/jgi_manual")

    if target.is_dir():
        zips = list(target.glob("*.zip"))
    else:
        zips = [target]
    print(f"processing {len(zips)} zip file(s)")

    all_samples = {}
    for zp in zips:
        print(f"\n=== {zp.name} ===")
        s = process_zip(zp)
        all_samples.update(s)

    if not all_samples:
        print("no .a.ko.txt found")
        return

    # union of KOs
    all_kos = sorted(set().union(*(c.keys() for c in all_samples.values())))
    print(f"\ntotal samples: {len(all_samples)}, unique KOs: {len(all_kos)}")

    # Build wide DataFrame
    rows = []
    for sid, counts in all_samples.items():
        row = {"sample_id": sid}
        row.update({k: counts.get(k, 0) for k in all_kos})
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = out_dir / "jgi_kegg_counts.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({df.shape})")

    # Summary stats
    sums = df.iloc[:, 1:].sum(axis=1)
    print(f"\nper-sample total KO hits: min={sums.min():.0f}, max={sums.max():.0f}, mean={sums.mean():.0f}")


if __name__ == "__main__":
    main()
