"""Leave-one-country-out (LOCO) — batch shortcut 검증.

가설: 모델이 country/lab fingerprint 외워서 ENVO biome 이나 다른 라벨을 잘 맞히는 거.

검증:
  1. EMP 임베딩 + envo_biome 라벨
  2. country 단위로 leave-one-out — 한 나라 빼고 학습 → 그 나라에서 biome 분류 acc
  3. 비교: 같은 split 으로 random leave-out (non-LOCO) acc

LOCO acc 가 random acc 보다 많이 떨어지면 → batch shortcut. 비슷하면 → 진짜 신호.
"""
import torch
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

CKPT = Path("checkpoints/gaia_v6")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMP_AB = "data/raw/emp/emp_soil_genus_20260403_181037.csv"
EMP_MD = "data/raw/emp/emp_soil_metadata_20260403_180909.csv"
OUT_JSON = Path("docs/benchmark_loco.json")


def encode_row(row, gcols, tok, max_len=512):
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    nz = row[gcols][row[gcols] > 0].sort_values(ascending=False)
    t = [bos]
    for g in nz.index:
        for cand in (f"g__{g}", g):
            if cand in tok.vocab:
                t.append(tok.vocab[cand]); break
        if len(t) >= max_len - 1: break
    t.append(eos)
    while len(t) < max_len: t.append(pad)
    return torch.tensor(t[:max_len], dtype=torch.long)


@torch.no_grad()
def embed(df, gcols, tok, gpt):
    out = []
    for i, (_, row) in enumerate(df.iterrows()):
        x = encode_row(row, gcols, tok).unsqueeze(0).to(DEVICE)
        h = gpt(x, output_hidden_states=True).hidden_states[-1]
        m = (x != 0).unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(p.squeeze(0).cpu().numpy())
        if (i + 1) % 200 == 0:
            print(f"  embedded {i+1}/{len(df)}")
    return np.stack(out)


def main():
    print(f"Backbone: {CKPT}")
    gpt = GPT2LMHeadModel.from_pretrained(str(CKPT / "best")).to(DEVICE).eval()
    with open(CKPT / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    md = pd.read_csv(EMP_MD, low_memory=False)
    ab = pd.read_csv(EMP_AB)
    join_key = None
    for c in ["sample_id", "#SampleID", "SampleID", "X.SampleID"]:
        if c in md.columns:
            join_key = c; break
    if join_key and join_key != "sample_id":
        md = md.rename(columns={join_key: "sample_id"})
    df = ab.merge(md[["sample_id", "country", "envo_biome_2"]], on="sample_id", how="inner")
    df["country_short"] = df["country"].str.replace("GAZ:", "", regex=False).fillna("Unknown")
    df = df[df["country_short"] != "Unknown"]
    df = df.dropna(subset=["envo_biome_2"]).reset_index(drop=True)
    print(f"merged: {len(df)} samples")

    # subsample for speed: keep same as before
    df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
    gcols = [c for c in ab.columns if c != "sample_id"]
    print(f"genus cols: {len(gcols)}")

    print("Embedding...")
    X = embed(df, gcols, tok, gpt)
    print(f"X shape: {X.shape}")

    countries = df["country_short"].values
    biomes = df["envo_biome_2"].values

    # Only countries with >=50 samples to be a meaningful test set
    counts = pd.Series(countries).value_counts()
    test_countries = counts[counts >= 50].index.tolist()
    print(f"countries to LOCO: {test_countries} (>=50 samples each)")

    # Filter biomes with >=10 samples globally
    keep_biomes = pd.Series(biomes).value_counts()
    keep_biomes = keep_biomes[keep_biomes >= 10].index
    mask = pd.Series(biomes).isin(keep_biomes).values
    Xf = X[mask]; cf = countries[mask]; bf = biomes[mask]
    print(f"after biome >=10 filter: {len(Xf)}, biomes: {len(keep_biomes)}")

    results = {"backbone": CKPT.name, "n_total": int(len(Xf)),
               "loco_per_country": {}, "random_baseline": {}}

    # Random baseline: 80/20 split, all countries
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(Xf, bf, test_size=0.2, random_state=42, stratify=bf)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                             class_weight="balanced", n_jobs=-1).fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    rand_acc = accuracy_score(y_te, pred)
    rand_bal = balanced_accuracy_score(y_te, pred)
    rand_f1 = f1_score(y_te, pred, average="macro")
    print(f"\nRandom 80/20 (no LOCO):  acc={rand_acc:.3f}  bal_acc={rand_bal:.3f}  macro_F1={rand_f1:.3f}")
    results["random_baseline"] = {"acc": float(rand_acc), "bal_acc": float(rand_bal), "macro_f1": float(rand_f1)}

    # LOCO: train on all countries except one, test on that one
    print("\n=== Leave-one-country-out ===")
    for tc in test_countries:
        tr_mask = cf != tc
        te_mask = cf == tc
        # Need both classes present in test set
        if len(np.unique(bf[te_mask])) < 2:
            print(f"  {tc:20s}: only 1 biome class in test, skip")
            continue
        # Filter biomes that exist in both train and test
        common_biomes = set(np.unique(bf[tr_mask])) & set(np.unique(bf[te_mask]))
        if len(common_biomes) < 2:
            print(f"  {tc:20s}: <2 common biomes, skip")
            continue
        m_tr = tr_mask & np.isin(bf, list(common_biomes))
        m_te = te_mask & np.isin(bf, list(common_biomes))

        clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                 class_weight="balanced", n_jobs=-1)
        clf.fit(Xf[m_tr], bf[m_tr])
        pr = clf.predict(Xf[m_te])
        acc = accuracy_score(bf[m_te], pr)
        bal = balanced_accuracy_score(bf[m_te], pr)
        f1 = f1_score(bf[m_te], pr, average="macro")
        n_te = int(m_te.sum())

        # Per-country majority baseline (predict most common biome IN TRAIN)
        from collections import Counter
        most_train = Counter(bf[m_tr]).most_common(1)[0][0]
        maj_acc = (bf[m_te] == most_train).mean()

        print(f"  {tc:20s}: acc={acc:.3f}  bal_acc={bal:.3f}  F1={f1:.3f}  (n={n_te}, biomes={len(common_biomes)}, majority_baseline={maj_acc:.3f})")
        results["loco_per_country"][tc] = {
            "n_test": n_te, "n_biomes": len(common_biomes),
            "acc": float(acc), "bal_acc": float(bal), "macro_f1": float(f1),
            "majority_baseline_acc": float(maj_acc),
        }

    # Aggregate LOCO mean
    if results["loco_per_country"]:
        accs = [v["acc"] for v in results["loco_per_country"].values()]
        bals = [v["bal_acc"] for v in results["loco_per_country"].values()]
        f1s  = [v["macro_f1"] for v in results["loco_per_country"].values()]
        print(f"\nLOCO mean acc:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"LOCO mean bal:  {np.mean(bals):.3f} ± {np.std(bals):.3f}")
        print(f"LOCO mean F1:   {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        results["loco_mean"] = {"acc": float(np.mean(accs)), "bal_acc": float(np.mean(bals)),
                                "macro_f1": float(np.mean(f1s))}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_JSON}")
    print("\n=== 해석 ===")
    print(f"Random 80/20 acc:  {rand_acc:.3f}")
    if results.get("loco_mean"):
        loco_acc = results["loco_mean"]["acc"]
        gap = rand_acc - loco_acc
        print(f"LOCO mean acc:     {loco_acc:.3f}")
        print(f"Gap (Random - LOCO): {gap:.3f}")
        if gap > 0.15:
            print("→ 큰 갭. batch/country 효과가 임베딩에 강하게 들어가 있음 (shortcut 가능성)")
        elif gap > 0.05:
            print("→ 중간 갭. 어느 정도 batch 효과 있지만 미생물 신호도 살아있음")
        else:
            print("→ 작은 갭. 임베딩이 진짜 cross-country 일반화 가능 (좋은 신호)")


if __name__ == "__main__":
    main()
