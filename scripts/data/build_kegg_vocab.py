"""KEGG KO id 기반 vocab + 토크나이저 빌드.

vocab 형식:
  0: <pad>
  1: <mask>
  2: <bos>
  3: <eos>
  4: K00001
  5: K00002
  ...

샘플 별 토큰화 흐름:
  1) (sample_id, KO_id) counts → CLR 정규화
  2) CLR 값 내림차순 정렬
  3) 상위 510 KO → token id 시퀀스
  4) [BOS] tok1 tok2 ... tok510 [EOS] [PAD]*

출력:
  checkpoints/gaia_v10/tokenizer.pkl
  data/processed_real/gaia-corpus-v10-kegg.pkl   (시퀀스 행렬)
  data/processed_real/gaia-metadata-v10.csv      (sample_id, source 정보)
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

OUT_CKPT = Path("checkpoints/gaia_v10")
OUT_CKPT.mkdir(parents=True, exist_ok=True)
OUT_CORPUS = Path("data/processed_real")
INPUT_CSV = Path("data/raw/jgi_manual/jgi_kegg_counts.csv")


def make_tokenizer(vocab):
    """Plain-dict tokenizer (no custom class — picklable everywhere)."""
    return {
        "vocab": vocab,
        "id_to_token": {i: t for t, i in vocab.items()},
        "pad_token_id": vocab["<pad>"],
        "mask_token_id": vocab["<mask>"],
        "bos_token_id": vocab["<bos>"],
        "eos_token_id": vocab["<eos>"],
    }


def tss_clr(x):
    s = x.sum()
    if s <= 0:
        return np.zeros_like(x, dtype=np.float64)
    xn = (x / s) + 1e-9
    log_x = np.log(xn)
    return log_x - log_x.mean()


def main():
    print(f"reading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  shape: {df.shape}")

    ko_cols = [c for c in df.columns if c.startswith("K") and len(c) == 6]
    print(f"  KO cols: {len(ko_cols)}")

    # Build vocab: special tokens + sorted KOs
    vocab = {"<pad>": 0, "<mask>": 1, "<bos>": 2, "<eos>": 3}
    for i, ko in enumerate(sorted(ko_cols)):
        vocab[ko] = 4 + i
    print(f"  vocab size: {len(vocab)}")

    tokenizer = make_tokenizer(vocab)
    with open(OUT_CKPT / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"  wrote {OUT_CKPT/'tokenizer.pkl'}")

    # Tokenize each sample
    bos, eos, pad = vocab["<bos>"], vocab["<eos>"], vocab["<pad>"]
    sequences = []
    skipped = 0
    metadata = []

    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        counts = row[ko_cols].values.astype(np.float32)
        # TSS + CLR + sort desc by CLR
        clr = tss_clr(counts.astype(np.float64))
        order = np.argsort(-clr)
        tokens = [bos]
        for idx in order:
            ko = ko_cols[idx]
            tid = vocab.get(ko)
            if tid is not None:
                tokens.append(tid)
            if len(tokens) >= 511:
                break
        tokens.append(eos)
        n_real = sum(1 for t in tokens if t not in (bos, eos, pad))
        if n_real >= 5:
            while len(tokens) < 512:
                tokens.append(pad)
            sequences.append(np.array(tokens[:512], dtype=np.int32))
            metadata.append({"sample_id": sid, "source": "jgi_kegg"})
        else:
            skipped += 1

    arr = np.stack(sequences) if sequences else np.zeros((0, 512), dtype=np.int32)
    print(f"  tokenized: {len(sequences)} samples (skipped {skipped})")

    pkl_path = OUT_CORPUS / "gaia-corpus-v10-kegg.pkl"
    if pkl_path.exists():
        print(f"  REFUSE to overwrite {pkl_path}")
        return
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "sequences": arr,
            "vocab_size": len(vocab),
            "bos": bos, "eos": eos, "pad": pad,
            "preprocessing": "JGI KEGG KO + TSS + CLR + sort desc",
            "n_samples": len(metadata),
        }, f)
    print(f"  wrote {pkl_path} (shape {arr.shape})")

    md = pd.DataFrame(metadata)
    md_path = OUT_CORPUS / "gaia-metadata-v10.csv"
    if md_path.exists():
        print(f"  REFUSE to overwrite {md_path}")
        return
    md.to_csv(md_path, index=False)
    print(f"  wrote {md_path} ({len(md)} rows)")


if __name__ == "__main__":
    main()
