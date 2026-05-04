"""임베딩의 지리 편향(batch effect) 정도 측정.

논리: 모델이 진짜 미생물 의미를 학습했다면 country/region 분류 능력은 '꽤'
있어야 하지만 압도적이면 안 됨. country 분류 acc 가 너무 높으면(예: 0.95+)
batch/locale 효과를 외운 거고, 너무 낮으면(0.3) 신호가 없는 거.

EMP 부분집합(country 가 있는 샘플)에 대해:
  1. v6 백본으로 임베딩 추출
  2. country (10+ 샘플 있는 나라만) 분류 — Logistic Regression / RF
  3. 같은 임베딩으로 envo_biome 분류
  4. UMAP 으로 2D 투영해서 PNG 저장

결과 파일:
  docs/benchmark_geo_signal.json
  docs/geo_umap.png
"""
import torch
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

CKPT = Path("checkpoints/gaia_v6")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMP_AB = "data/raw/emp/emp_soil_genus_20260403_181037.csv"
EMP_MD = "data/raw/emp/emp_soil_metadata_20260403_180909.csv"
OUT_JSON = Path("docs/benchmark_geo_signal.json")
OUT_PNG  = Path("docs/geo_umap.png")


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
def embed(df, gcols, tok, gpt, max_n=2000):
    """Subsample to max_n for speed."""
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=42).reset_index(drop=True)
    out = []
    for i, (_, row) in enumerate(df.iterrows()):
        x = encode_row(row, gcols, tok).unsqueeze(0).to(DEVICE)
        h = gpt(x, output_hidden_states=True).hidden_states[-1]
        m = (x != 0).unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(p.squeeze(0).cpu().numpy())
        if (i + 1) % 200 == 0:
            print(f"  embedded {i+1}/{len(df)}")
    return np.stack(out), df


def cv_classify(X, y, label):
    keep = pd.Series(y).value_counts()
    keep = keep[keep >= 10].index
    mask = pd.Series(y).isin(keep).values
    X, y = X[mask], np.array(y)[mask]
    print(f"  {label}: n={len(y)} classes={len(keep)} (>=10 samples each)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, baccs, f1s = [], [], []
    for tr_i, te_i in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                 class_weight="balanced", n_jobs=-1)
        clf.fit(X[tr_i], y[tr_i])
        pr = clf.predict(X[te_i])
        accs.append(accuracy_score(y[te_i], pr))
        baccs.append(balanced_accuracy_score(y[te_i], pr))
        f1s.append(f1_score(y[te_i], pr, average="macro"))
    print(f"    LR 5CV  acc={np.mean(accs):.3f}±{np.std(accs):.3f}  bal_acc={np.mean(baccs):.3f}  macro_F1={np.mean(f1s):.3f}")

    rf = RandomForestClassifier(200, random_state=42, n_jobs=-1, class_weight="balanced")
    rf_accs, rf_baccs, rf_f1s = [], [], []
    for tr_i, te_i in skf.split(X, y):
        rf.fit(X[tr_i], y[tr_i])
        pr = rf.predict(X[te_i])
        rf_accs.append(accuracy_score(y[te_i], pr))
        rf_baccs.append(balanced_accuracy_score(y[te_i], pr))
        rf_f1s.append(f1_score(y[te_i], pr, average="macro"))
    print(f"    RF 5CV  acc={np.mean(rf_accs):.3f}±{np.std(rf_accs):.3f}  bal_acc={np.mean(rf_baccs):.3f}  macro_F1={np.mean(rf_f1s):.3f}")

    # majority baseline
    from collections import Counter
    most = Counter(y).most_common(1)[0][0]
    maj_acc = (y == most).mean()
    print(f"    Majority baseline acc={maj_acc:.3f}")
    return {
        "n": int(len(y)), "n_classes": int(len(keep)),
        "lr_acc": float(np.mean(accs)), "lr_bal_acc": float(np.mean(baccs)),
        "lr_f1": float(np.mean(f1s)),
        "rf_acc": float(np.mean(rf_accs)), "rf_bal_acc": float(np.mean(rf_baccs)),
        "rf_f1": float(np.mean(rf_f1s)),
        "majority_baseline_acc": float(maj_acc),
    }


def umap_plot(X, labels, out_png):
    try:
        import umap
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (UMAP/matplotlib 없음 — UMAP 스킵)")
        return
    print("  UMAP fitting...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    Z = reducer.fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    keep = pd.Series(labels).value_counts()
    keep = keep[keep >= 30].index
    palette = plt.cm.tab20(np.linspace(0, 1, len(keep)))
    for i, lab in enumerate(keep):
        m = pd.Series(labels) == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, label=str(lab), c=[palette[i]])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("Gaia v6 embeddings — colored by country (EMP, n>=30)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"  saved {out_png}")


def main():
    print(f"Backbone: {CKPT}, device: {DEVICE}")
    gpt = GPT2LMHeadModel.from_pretrained(str(CKPT / "best")).to(DEVICE).eval()
    with open(CKPT / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    md = pd.read_csv(EMP_MD, low_memory=False)
    ab = pd.read_csv(EMP_AB)
    sample_col = "sample_id"
    join_key = None
    for c in ["sample_id", "#SampleID", "SampleID", "X.SampleID"]:
        if c in md.columns:
            join_key = c; break
    if join_key is None:
        # last resort: try first column
        join_key = md.columns[0]
    print(f"metadata join key: {join_key}")
    md = md.rename(columns={join_key: "sample_id"}) if join_key != "sample_id" else md
    df = ab.merge(md[["sample_id", "country", "envo_biome_2", "latitude_deg", "longitude_deg"]],
                  on="sample_id", how="inner")
    print(f"merged rows with country: {len(df)}")
    df["country_short"] = df["country"].str.replace("GAZ:", "", regex=False).fillna("Unknown")

    gcols = [c for c in ab.columns if c != "sample_id"]
    print(f"genus cols: {len(gcols)}")

    # subsample for speed
    df_sub = df[df["country_short"] != "Unknown"]
    print(f"after country filter: {len(df_sub)}")
    X, df_sub = embed(df_sub, gcols, tok, gpt, max_n=2000)
    print(f"embeddings: {X.shape}")

    print()
    print("=== Classification probes ===")
    res = {"backbone": CKPT.name, "n_embedded": int(X.shape[0])}
    res["country"] = cv_classify(X, df_sub["country_short"].values, "country")
    res["envo_biome_2"] = cv_classify(X, df_sub["envo_biome_2"].fillna("Unknown").values, "envo_biome_2")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"\nSaved {OUT_JSON}")

    umap_plot(X, df_sub["country_short"].values, OUT_PNG)


if __name__ == "__main__":
    main()
