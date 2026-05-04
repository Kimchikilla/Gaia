"""연속 라벨(pH/C/N) 을 4-class binning 후 분류로 평가.

회귀 R² 만으로 본 모습과 분류 F1 / balanced accuracy 의 차이를 보면
"분포 좁아서 잘 나오는 거 vs 진짜 신호" 구분이 더 선명해진다.

각 작업에 대해:
  1) Westerfeld 임베딩 추출
  2) 라벨 quartile 4-bin 으로 변환
  3) 분류 헤드 학습 (stratified split)
  4) accuracy / balanced accuracy / macro-F1 / 혼동행렬 보고
  5) 같은 split 으로 RF 분류기 vs majority-class baseline 비교
"""
import torch
import torch.nn as nn
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE = Path("data/raw/longterm/bonares_data")
CKPT = Path("checkpoints/gaia_v6")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_JSON = Path("docs/benchmark_diagnostic_classification.json")


def build_pivot():
    bac = pd.read_csv(BASE / "lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
    g = pd.read_csv(BASE / "lte_westerfeld.V1_0_GENUS.csv")
    nm = dict(zip(g["Genus_ID"], g["Name"]))
    bac["Genus_Name"] = bac["Genus_ID"].map(nm)
    grp = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
    return grp.pivot_table(index=["Plot_ID", "Experimental_Year"],
                           columns="Genus_Name", values="Value", fill_value=0).reset_index()


def build_pairs(pivot, col):
    soil = pd.read_csv(BASE / "lte_westerfeld.V1_0_SOIL_LAB.csv")
    samp = pd.read_csv(BASE / "lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
    soil = soil.merge(samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]],
                      on="Soil_Sampling_ID", how="left")
    sub = soil.dropna(subset=[col])
    if len(sub) == 0: return None
    agg = sub.groupby(["Plot_ID", "Experimental_Year"])[col].mean().reset_index()
    return pivot.merge(agg, on=["Plot_ID", "Experimental_Year"],
                       how="inner").dropna(subset=[col]).reset_index(drop=True)


def encode_row(row, gcols, tok, max_len=512):
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    nz = row[gcols][row[gcols] > 0].sort_values(ascending=False)
    t = [bos]
    for g in nz.index:
        tid = tok.vocab.get(f"g__{g}")
        if tid: t.append(tid)
        if len(t) >= max_len - 1: break
    t.append(eos)
    while len(t) < max_len: t.append(pad)
    return torch.tensor(t[:max_len], dtype=torch.long)


@torch.no_grad()
def get_emb_matrix(df, gcols, tok, gpt):
    out = []
    for _, row in df.iterrows():
        x = encode_row(row, gcols, tok).unsqueeze(0).to(DEVICE)
        h = gpt(x, output_hidden_states=True).hidden_states[-1]
        m = (x != 0).unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(p.squeeze(0).cpu().numpy())
    return np.stack(out)


class ClsHead(nn.Module):
    def __init__(self, in_dim, n_class):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, n_class))
    def forward(self, x): return self.net(x)


def train_eval_cls(Xtr, Xte, ytr, yte, n_class, epochs=80):
    Xt = torch.tensor(Xtr, dtype=torch.float).to(DEVICE)
    yt = torch.tensor(ytr, dtype=torch.long).to(DEVICE)
    Xv = torch.tensor(Xte, dtype=torch.float).to(DEVICE)

    head = ClsHead(Xtr.shape[1], n_class).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    bs = 16
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            logits = head(Xt[idx])
            loss = nn.CrossEntropyLoss()(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(Xv).argmax(-1).cpu().numpy()
    return pred


def majority_baseline(ytr, yte):
    """Always predict the most common class in training."""
    from collections import Counter
    most = Counter(ytr).most_common(1)[0][0]
    return np.full_like(yte, most)


def main():
    print(f"Device: {DEVICE}, backbone: {CKPT.name}")
    gpt = GPT2LMHeadModel.from_pretrained(str(CKPT / "best")).to(DEVICE).eval()
    with open(CKPT / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    pivot = build_pivot()
    gcols = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]
    targets = {"pH": "pH", "Total_Carbon": "Total_Carbon", "Total_Nitrogen": "Total_Nitrogen"}

    results = {"backbone": CKPT.name, "n_classes": 4, "tasks": {}}
    for label, col in targets.items():
        df = build_pairs(pivot, col)
        if df is None or len(df) < 50: continue
        y_cont = df[col].values.astype(float)
        # 4-bin quantiles
        bins = pd.qcut(y_cont, q=4, labels=False, duplicates="drop")
        n_class = int(bins.max() + 1)
        print(f"\n=== {label}: n={len(df)} → {n_class}-class quartiles ===")
        # bin edges for context
        edges = pd.qcut(y_cont, q=4, retbins=True, duplicates="drop")[1]
        print(f"   bin edges: {[round(e, 4) for e in edges]}")
        print(f"   class counts: {np.bincount(bins).tolist()}")

        emb = get_emb_matrix(df, gcols, tok, gpt)
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2,
                                          random_state=42, stratify=bins)

        # Gaia
        pred_g = train_eval_cls(emb[idx_tr], emb[idx_te], bins[idx_tr], bins[idx_te], n_class)
        # RF
        rf = RandomForestClassifier(200, random_state=42, n_jobs=-1, class_weight="balanced") \
            .fit(emb[idx_tr], bins[idx_tr])
        pred_rf = rf.predict(emb[idx_te])
        # Majority baseline
        pred_maj = majority_baseline(bins[idx_tr], bins[idx_te])

        def score(name, pr):
            acc = accuracy_score(bins[idx_te], pr)
            bal = balanced_accuracy_score(bins[idx_te], pr)
            f1 = f1_score(bins[idx_te], pr, average="macro")
            print(f"   {name:18s}: acc={acc:.3f}  balanced_acc={bal:.3f}  macro_F1={f1:.3f}")
            return {"acc": float(acc), "balanced_acc": float(bal), "macro_f1": float(f1)}

        gaia = score("Gaia(MLP)", pred_g)
        rfm  = score("RandomForest", pred_rf)
        maj  = score("Majority class", pred_maj)

        cm = confusion_matrix(bins[idx_te], pred_g, labels=list(range(n_class))).tolist()
        print(f"   Gaia confusion matrix (rows=true, cols=pred): {cm}")

        results["tasks"][label] = {
            "n": int(len(df)),
            "n_class": int(n_class),
            "class_counts": np.bincount(bins).tolist(),
            "bin_edges": [float(e) for e in edges],
            "gaia": gaia,
            "rf": rfm,
            "majority_baseline": maj,
            "gaia_confusion_matrix": cm,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
