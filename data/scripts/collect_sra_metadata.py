"""SRA / ENA 토양 16S 메타데이터 카탈로그 수집.

abundance 표는 raw FASTQ 처리 (QIIME2) 가 필요해서 여기선 메타데이터만 모음.
어떤 study/sample 이 있는지 카탈로그 만들어 놓으면 다음 단계에 처리 가능.

저장:
  data/raw/sra_catalog/sra_soil_16S.tsv  — accession, study, country, lat/lon 등
"""
import argparse, time, json, requests, csv
from pathlib import Path

OUT = Path("data/raw/sra_catalog")
OUT.mkdir(parents=True, exist_ok=True)

ENA_FIELDS = [
    "accession", "study_accession", "sample_accession", "experiment_accession",
    "scientific_name", "library_strategy", "library_source", "library_selection",
    "country", "geographic_location_(country_and/or_sea)",
    "latitude", "longitude",
    "environment_biome", "environment_feature", "environment_material",
    "collection_date",
    "instrument_platform", "instrument_model",
    "read_count", "base_count",
]


def fetch_ena_search(query, fields, limit=10000, offset=0):
    url = "https://www.ebi.ac.uk/ena/portal/api/search"
    params = {
        "result": "read_run",
        "query": query,
        "fields": ",".join(fields),
        "format": "tsv",
        "limit": limit,
        "offset": offset,
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=50000)
    ap.add_argument("--page-size", type=int, default=10000)
    args = ap.parse_args()

    query = '(library_strategy="AMPLICON" OR library_source="METAGENOMIC") AND environment_biome="soil"'
    print(f"ENA query: {query}")

    out_tsv = OUT / "ena_soil_16S.tsv"
    print(f"Writing to {out_tsv}")
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        offset = 0; total = 0
        while total < args.max_records:
            try:
                txt = fetch_ena_search(query, ENA_FIELDS,
                                       limit=args.page_size, offset=offset)
            except Exception as e:
                print(f"page offset={offset}: {e}, retry in 30s")
                time.sleep(30); continue
            lines = txt.strip().split("\n")
            if len(lines) <= 1 and offset > 0:
                print("no more pages")
                break
            if offset == 0:
                f.write(txt)
                total += len(lines) - 1
            else:
                # skip header
                if len(lines) > 1:
                    f.write("\n" + "\n".join(lines[1:]))
                    total += len(lines) - 1
            print(f"  cumulative records: {total}")
            if len(lines) - 1 < args.page_size:
                break
            offset += args.page_size
            time.sleep(2)

    print(f"Done. Total ENA soil records: {total}")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
