"""
Process Naylor fastq files into OTU table.

Steps:
  1. Quality filter (cutadapt)
  2. Dereplicate (vsearch)
  3. Cluster OTUs at 97% (vsearch)
  4. Map reads to OTUs (vsearch)
  5. Assign taxonomy (vsearch + SILVA)
  6. Build OTU table

Uses 8 CPU cores, leaves 12 for other work.
"""

import gzip
import os
import subprocess
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

VSEARCH = "C:/Users/User/desktop/gaia/vsearch.exe"
BASE = Path("C:/Users/User/desktop/gaia")
FASTQ_DIR = BASE / "data/raw/naylor/fastq"
WORK_DIR = BASE / "data/raw/naylor/processing"
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE / "data/raw/naylor"

N_THREADS = 8


def decompress_fastq(gz_path, out_path):
    """Decompress .fastq.gz to .fastq"""
    if out_path.exists():
        return
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def quality_filter_one(fastq_path, filtered_path):
    """Quality filter one fastq file using vsearch."""
    if filtered_path.exists() and filtered_path.stat().st_size > 0:
        return True
    try:
        result = subprocess.run(
            [
                VSEARCH,
                "--fastq_filter", str(fastq_path),
                "--fastq_maxee", "1.0",
                "--fastq_minlen", "200",
                "--fastaout", str(filtered_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    # Step 0: Decompress all fastq.gz files
    gz_files = sorted(FASTQ_DIR.glob("*.fastq.gz"))
    print(f"Step 0: Decompressing {len(gz_files)} files...")

    fastq_files = []
    for gz in tqdm(gz_files, desc="Decompress"):
        fq = WORK_DIR / gz.name.replace(".fastq.gz", ".fastq")
        decompress_fastq(gz, fq)
        if fq.exists():
            fastq_files.append(fq)

    print(f"Decompressed: {len(fastq_files)} files")

    # Step 1: Quality filter each file
    print(f"\nStep 1: Quality filtering {len(fastq_files)} files...")
    filtered_files = []
    for fq in tqdm(fastq_files, desc="Filter"):
        filtered = WORK_DIR / fq.name.replace(".fastq", ".filtered.fasta")
        if quality_filter_one(fq, filtered):
            if filtered.exists() and filtered.stat().st_size > 0:
                filtered_files.append(filtered)

    print(f"Filtered: {len(filtered_files)} files passed")

    # Step 2: Merge all filtered reads into one file
    print("\nStep 2: Merging all filtered reads...")
    merged_fasta = WORK_DIR / "all_filtered.fasta"
    with open(merged_fasta, "w") as out:
        for i, fasta in enumerate(filtered_files):
            run_id = fasta.stem.replace(".filtered", "")
            with open(fasta) as f:
                seq_num = 0
                for line in f:
                    if line.startswith(">"):
                        # Add sample label to sequence header
                        out.write(f">{run_id}_seq{seq_num};sample={run_id}\n")
                        seq_num += 1
                    else:
                        out.write(line)

    total_seqs = sum(1 for line in open(merged_fasta) if line.startswith(">"))
    print(f"Total sequences: {total_seqs:,}")

    # Step 3: Dereplicate
    print("\nStep 3: Dereplicating...")
    derep_fasta = WORK_DIR / "derep.fasta"
    subprocess.run(
        [
            VSEARCH,
            "--derep_fulllength", str(merged_fasta),
            "--output", str(derep_fasta),
            "--sizeout",
            "--minuniquesize", "2",
        ],
        capture_output=True,
    )
    derep_count = sum(1 for line in open(derep_fasta) if line.startswith(">"))
    print(f"Unique sequences: {derep_count:,}")

    # Step 4: Cluster OTUs at 97%
    print("\nStep 4: Clustering OTUs (97% identity)...")
    otus_fasta = WORK_DIR / "otus.fasta"
    subprocess.run(
        [
            VSEARCH,
            "--cluster_size", str(derep_fasta),
            "--id", "0.97",
            "--centroids", str(otus_fasta),
            "--relabel", "OTU_",
            "--threads", str(N_THREADS),
        ],
        capture_output=True,
    )
    n_otus = sum(1 for line in open(otus_fasta) if line.startswith(">"))
    print(f"OTUs found: {n_otus:,}")

    # Step 5: Map all reads back to OTUs
    print("\nStep 5: Mapping reads to OTUs...")
    uc_file = WORK_DIR / "map.uc"
    subprocess.run(
        [
            VSEARCH,
            "--usearch_global", str(merged_fasta),
            "--db", str(otus_fasta),
            "--id", "0.97",
            "--uc", str(uc_file),
            "--threads", str(N_THREADS),
        ],
        capture_output=True,
    )

    # Step 6: Build OTU table from UC file
    print("\nStep 6: Building OTU table...")
    otu_counts = defaultdict(lambda: defaultdict(int))
    with open(uc_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] == "H":  # Hit
                query = parts[8]  # query sequence name
                target = parts[9]  # OTU name
                # Extract sample ID from query name
                sample = query.split(";sample=")[-1] if ";sample=" in query else query.split("_seq")[0]
                otu_counts[sample][target] += 1

    # Convert to DataFrame
    otu_df = pd.DataFrame(otu_counts).fillna(0).astype(int).T
    otu_df.index.name = "sample_id"
    otu_df = otu_df.reset_index()

    # Merge with drought/control labels
    meta = pd.read_csv(OUTPUT_DIR / "naylor_metadata.csv")
    meta_slim = meta[["run_id", "treatment", "host"]].rename(columns={"run_id": "sample_id"})
    result = otu_df.merge(meta_slim, on="sample_id", how="inner")

    # Save
    result.to_csv(OUTPUT_DIR / "naylor_otu_table.csv", index=False)

    otu_cols = [c for c in result.columns if c.startswith("OTU_")]
    print(f"\n=== DONE ===")
    print(f"Samples: {len(result)}")
    print(f"OTUs: {len(otu_cols)}")
    print(f"Drought: {(result['treatment'] == 'drought').sum()}")
    print(f"Control: {(result['treatment'] == 'control').sum()}")
    print(f"Saved: {OUTPUT_DIR / 'naylor_otu_table.csv'}")

    # Cleanup: remove decompressed fastq to save space
    print("\nCleaning up decompressed files...")
    for fq in WORK_DIR.glob("*.fastq"):
        fq.unlink()
    for fa in WORK_DIR.glob("*.filtered.fasta"):
        fa.unlink()
    print("Done!")


if __name__ == "__main__":
    main()
