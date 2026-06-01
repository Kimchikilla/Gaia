"""기존 코퍼스에 EMP 4628 샘플을 합쳐 v2 코퍼스 생성.

기존 파일은 절대 덮어쓰지 않음. 새 파일에 저장.
출력:
  data/processed_real/gaia-abundance-v2.csv
  data/processed_real/gaia-metadata-v2.csv
  data/processed_real/gaia-corpus-v2.pkl  (토큰화된 시퀀스, 학습용)
"""
import torch  # torch first
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/processed_real")
EMP_AB = "data/raw/emp/emp_soil_genus_20260403_181037.csv"
EMP_META = "data/raw/emp/emp_soil_metadata_20260403_180909.csv"
TOK = "checkpoints/gaia_v4/tokenizer.pkl"


def main():
    print("Loading existing v1 corpus...")
    ab1 = pd.read_csv(OUT / "gaia-abundance-v1.csv")
    md1 = pd.read_csv(OUT / "gaia-metadata-v1.csv")
    print(f"  v1 abundance: {ab1.shape}, metadata: {md1.shape}")

    print("Loading EMP...")
    emp_ab = pd.read_csv(EMP_AB)
    emp_md_path = Path(EMP_META)
    if emp_md_path.exists():
        emp_md = pd.read_csv(emp_md_path)
        print(f"  EMP metadata: {emp_md.shape}")
    else:
        emp_md = pd.DataFrame({"sample_id": emp_ab["sample_id"]})
    print(f"  EMP abundance: {emp_ab.shape}")

    # union of genus columns (sample_id stays as id)
    v1_cols = [c for c in ab1.columns if c not in ("sample_id", "analysis_id")]
    emp_cols = [c for c in emp_ab.columns if c != "sample_id"]
    all_cols = sorted(set(v1_cols) | set(emp_cols))
    print(f"  union genera: {len(all_cols)} (v1={len(v1_cols)} EMP={len(emp_cols)} overlap={len(set(v1_cols)&set(emp_cols))})")

    # expand each to common columns
    ab1_id = ab1["sample_id"].astype(str)
    ab1_x = ab1.reindex(columns=all_cols, fill_value=0).astype(float)
    ab1_x.insert(0, "sample_id", ab1_id.values)
    ab1_x.insert(1, "source", "v1")

    emp_id = emp_ab["sample_id"].astype(str)
    emp_x = emp_ab.reindex(columns=all_cols, fill_value=0).astype(float)
    emp_x.insert(0, "sample_id", emp_id.values)
    emp_x.insert(1, "source", "emp")

    merged = pd.concat([ab1_x, emp_x], ignore_index=True)
    print(f"  merged abundance: {merged.shape}")

    out_ab = OUT / "gaia-abundance-v2.csv"
    if out_ab.exists():
        print(f"REFUSE to overwrite {out_ab}")
        return
    merged.to_csv(out_ab, index=False)
    print(f"  wrote {out_ab}")

    # metadata: merge but keep schemas distinct (just concat with source col)
    md1_x = md1.copy()
    md1_x["source"] = "v1"
    if "biome" not in md1_x.columns:
        md1_x["biome"] = "soil"
    emp_md_min = emp_md.copy() if "sample_id" in emp_md.columns else pd.DataFrame({"sample_id": emp_id.values})
    emp_md_min["source"] = "emp"
    if "biome" not in emp_md_min.columns:
        emp_md_min["biome"] = "soil"
    common_md_cols = ["sample_id", "source", "biome"]
    md_merged = pd.concat(
        [md1_x.reindex(columns=common_md_cols, fill_value=None),
         emp_md_min.reindex(columns=common_md_cols, fill_value=None)],
        ignore_index=True,
    )
    out_md = OUT / "gaia-metadata-v2.csv"
    if out_md.exists():
        print(f"REFUSE to overwrite {out_md}")
        return
    md_merged.to_csv(out_md, index=False)
    print(f"  wrote {out_md} ({len(md_merged)} rows)")

    # tokenize for pretraining
    print("Tokenizing for continual pretrain...")
    with open(TOK, "rb") as f:
        tok = pickle.load(f)
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    genus_cols = [c for c in merged.columns if c not in ("sample_id", "source")]

    sequences = []
    skipped = 0
    for _, row in merged.iterrows():
        nz = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
        tokens = [bos]
        for g in nz.index:
            for cand in (f"g__{g}", g):
                if cand in tok.vocab:
                    tokens.append(tok.vocab[cand])
                    break
            if len(tokens) >= 511:
                break
        tokens.append(eos)
        if sum(1 for t in tokens if t not in (bos, eos, pad)) >= 5:
            while len(tokens) < 512:
                tokens.append(pad)
            sequences.append(np.array(tokens[:512], dtype=np.int32))
        else:
            skipped += 1

    print(f"  tokenized: {len(sequences)} sequences, skipped: {skipped}")
    arr = np.stack(sequences)
    out_pkl = OUT / "gaia-corpus-v2.pkl"
    if out_pkl.exists():
        print(f"REFUSE to overwrite {out_pkl}")
        return
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "sequences": arr,
            "v1_n": len(ab1_x),
            "emp_n": len(emp_x),
            "vocab_size": len(tok.vocab),
            "bos": bos, "eos": eos, "pad": pad,
        }, f)
    print(f"  wrote {out_pkl}, shape={arr.shape}")


if __name__ == "__main__":
    main()
