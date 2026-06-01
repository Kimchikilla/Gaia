"""v7 코퍼스 = v3 (10,514 샘플) + CLR + per-source batch 보정.

처리 순서:
  1) v3 wide abundance 행렬 (samples × 2,478 genera + source 컬럼) 로딩
     - v2 abundance CSV 가 너무 큼(>78MB)이므로 v1/EMP/NEON 부분을 다시 합쳐서 만듦
  2) 샘플별 TSS (총합 정규화) — 시퀀싱 깊이 차이 제거
  3) 샘플별 CLR — log(x_i / geomean(x))
  4) source(v1/emp/neon) 별 평균 빼기 — 명시적 batch 보정
  5) 결과 행렬 → 보정 후 값 내림차순 sort → top genera 토큰화

출력:
  data/processed_real/gaia-corpus-v7-clr.pkl
"""
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/processed_real")
TOK = "checkpoints/gaia_v6/tokenizer.pkl"
EPSILON = 1e-9


def clr_per_sample(x):
    """x: 1D array of non-negative abundances. Returns CLR-transformed."""
    x = x + EPSILON
    log_x = np.log(x)
    return log_x - log_x.mean()


def load_v1():
    df = pd.read_csv(OUT / "gaia-abundance-v1.csv")
    drop = [c for c in ("sample_id", "analysis_id") if c in df.columns]
    sid = df["sample_id"].astype(str).values
    g = df.drop(columns=drop)
    return sid, g


def load_emp():
    df = pd.read_csv("data/raw/emp/emp_soil_genus_20260403_181037.csv")
    sid = df["sample_id"].astype(str).values
    g = df.drop(columns=["sample_id"])
    return sid, g


def load_neon():
    df = pd.read_csv("data/raw/neon/neon_microbe_abundance.csv")
    drop = [c for c in ("sample_id", "site", "month") if c in df.columns]
    sid = df["sample_id"].astype(str).values
    g = df.drop(columns=drop)
    return sid, g


def main():
    print("Loading v1 / EMP / NEON ...")
    s1, g1 = load_v1()
    s2, g2 = load_emp()
    s3, g3 = load_neon()
    print(f"  v1: {g1.shape}  EMP: {g2.shape}  NEON: {g3.shape}")

    # union of genus columns
    all_genera = sorted(set(g1.columns) | set(g2.columns) | set(g3.columns))
    print(f"union genera: {len(all_genera)}")

    def align(g):
        return g.reindex(columns=all_genera, fill_value=0).astype(np.float32).values

    X1, X2, X3 = align(g1), align(g2), align(g3)
    sources = (["v1"] * len(s1)) + (["emp"] * len(s2)) + (["neon"] * len(s3))
    sample_ids = list(s1) + list(s2) + list(s3)
    X = np.concatenate([X1, X2, X3], axis=0)
    sources = np.array(sources)
    print(f"merged: {X.shape}, sources: {pd.Series(sources).value_counts().to_dict()}")

    # 1) TSS — divide each row by its sum (avoid /0)
    print("Step 1: TSS (per-sample relative abundance)...")
    rowsum = X.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum > 0, rowsum, 1.0)
    Xn = X / rowsum

    # 2) CLR per sample
    print("Step 2: CLR per sample...")
    Xc = np.empty_like(Xn)
    for i in range(Xn.shape[0]):
        Xc[i] = clr_per_sample(Xn[i])
    print(f"  CLR matrix: mean={Xc.mean():.3f} std={Xc.std():.3f}")

    # 3) per-source mean subtraction (explicit batch correction)
    print("Step 3: subtract per-source mean (batch correction)...")
    for src in np.unique(sources):
        mask = sources == src
        m = Xc[mask].mean(axis=0)
        Xc[mask] = Xc[mask] - m
        print(f"  {src}: subtracted mean (n={mask.sum()})")

    # 4) Tokenize: sort by adjusted CLR value descending, keep top genera that exist in vocab
    print("Step 4: tokenize (sort by adjusted CLR desc)...")
    with open(TOK, "rb") as f:
        tok = pickle.load(f)
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id

    sequences, kept_meta = [], []
    skipped = 0
    for i, row in enumerate(Xc):
        # sort indices by row value descending
        order = np.argsort(-row)
        tokens = [bos]
        for idx in order:
            g = all_genera[idx]
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
            kept_meta.append({"sample_id": sample_ids[i], "source": sources[i]})
        else:
            skipped += 1

    arr = np.stack(sequences)
    out_pkl = OUT / "gaia-corpus-v7-clr.pkl"
    if out_pkl.exists():
        print(f"REFUSE to overwrite {out_pkl}")
        return
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "sequences": arr,
            "source_breakdown": pd.Series([m["source"] for m in kept_meta]).value_counts().to_dict(),
            "vocab_size": len(tok.vocab),
            "bos": bos, "eos": eos, "pad": pad,
            "preprocessing": "TSS + CLR + per-source mean subtraction",
        }, f)
    md = pd.DataFrame(kept_meta)
    md.to_csv(OUT / "gaia-metadata-v7.csv", index=False)
    print(f"Wrote {out_pkl}: shape={arr.shape}, skipped={skipped}")
    print(f"Wrote metadata: {OUT / 'gaia-metadata-v7.csv'}")


if __name__ == "__main__":
    main()
