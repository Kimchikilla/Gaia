"""v3 코퍼스 = v2 (MGnify v1 + EMP) + MGnify v3 추가 + NEON 미생물.

기존 파일은 절대 안 건드림. 새로 저장:
  data/processed_real/gaia-abundance-v3.csv
  data/processed_real/gaia-metadata-v3.csv
  data/processed_real/gaia-corpus-v3.pkl
"""
import torch  # torch first
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/processed_real")
TOK = "checkpoints/gaia_v4/tokenizer.pkl"


def load_partition(label, ab_path, sample_col="sample_id", drop_cols=None, source_tag=None):
    """Load one abundance source, normalize column names."""
    p = Path(ab_path)
    if not p.exists():
        print(f"[skip] {label}: {ab_path} missing")
        return pd.DataFrame()
    df = pd.read_csv(p)
    drop_cols = drop_cols or []
    keep = [c for c in df.columns if c not in drop_cols]
    df = df[keep].copy()
    if sample_col not in df.columns:
        df.insert(0, "sample_id", [f"{source_tag or label}_{i:05d}" for i in range(len(df))])
    elif sample_col != "sample_id":
        df = df.rename(columns={sample_col: "sample_id"})
    df["source"] = source_tag or label
    print(f"[ok] {label}: {len(df)} samples × {df.shape[1]-2} genus cols")
    return df


def main():
    parts = []

    # v2 (already merged in expand_corpus_emp.py output)
    v2_path = OUT / "gaia-abundance-v2.csv"
    if v2_path.exists():
        v2 = pd.read_csv(v2_path)
        # v2 already has source col
        print(f"[ok] v2: {len(v2)} samples × {v2.shape[1]-2} cols")
        parts.append(v2)
    else:
        # fallback to v1
        v1 = load_partition("v1", OUT / "gaia-abundance-v1.csv",
                            drop_cols=["analysis_id"], source_tag="v1")
        parts.append(v1)
        emp = load_partition("emp", "data/raw/emp/emp_soil_genus_20260403_181037.csv",
                             source_tag="emp")
        parts.append(emp)

    # MGnify v3 extra
    mgv3_path = Path("data/raw/mgnify_v3/mgnify_v3_abundance.csv")
    mgv3 = load_partition("mgnify_v3", mgv3_path,
                          drop_cols=["analysis_id"], source_tag="mgnify_v3")
    if not mgv3.empty:
        parts.append(mgv3)

    # NEON microbe
    neon_path = Path("data/raw/neon/neon_microbe_abundance.csv")
    neon = load_partition("neon", neon_path,
                          drop_cols=["site", "month"], source_tag="neon")
    if not neon.empty:
        parts.append(neon)

    if not parts:
        print("nothing to merge")
        return

    # union of genus columns
    all_genus = sorted(
        set().union(*[set(p.columns) - {"sample_id", "source"} for p in parts])
    )
    print(f"union genera: {len(all_genus)}")

    aligned = []
    for p in parts:
        x = p.reindex(columns=["sample_id", "source"] + all_genus, fill_value=0)
        aligned.append(x)
    merged = pd.concat(aligned, ignore_index=True)
    print(f"merged: {merged.shape}")
    print("source breakdown:", merged["source"].value_counts().to_dict())

    out_ab = OUT / "gaia-abundance-v3.csv"
    if out_ab.exists():
        print(f"REFUSE to overwrite {out_ab}")
        return
    # Big file — write only metadata to v3 csv? Actually skip v3 abundance CSV
    # to avoid 100MB+ git issue. Tokenized pkl is what we use for training.
    md = merged[["sample_id", "source"]].copy()
    md.to_csv(OUT / "gaia-metadata-v3.csv", index=False)
    print(f"wrote {OUT/'gaia-metadata-v3.csv'} ({len(md)} rows)")

    # Tokenize for pretraining
    print("Tokenizing...")
    with open(TOK, "rb") as f:
        tok = pickle.load(f)
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id

    sequences = []
    skipped = 0
    for _, row in merged.iterrows():
        nz = row[all_genus][row[all_genus] > 0].sort_values(ascending=False)
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

    arr = np.stack(sequences)
    out_pkl = OUT / "gaia-corpus-v3.pkl"
    if out_pkl.exists():
        print(f"REFUSE to overwrite {out_pkl}")
        return
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "sequences": arr,
            "source_breakdown": merged["source"].value_counts().to_dict(),
            "vocab_size": len(tok.vocab),
            "bos": bos, "eos": eos, "pad": pad,
        }, f)
    print(f"wrote {out_pkl} shape={arr.shape}, skipped={skipped}")


if __name__ == "__main__":
    main()
