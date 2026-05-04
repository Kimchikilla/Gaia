"""진단 R² 가 부풀려졌는지 체크.

세 가지 비교를 한 표에 박는다:
  (a) Random split + Gaia    ← 현재 보고된 수치
  (b) Stratified split + Gaia ← 라벨 bin 별 균등 분할
  (c) Predict-training-mean baseline R²  ← 어떤 모델도 이건 쉽게 이겨야 함
  (d) RandomForest 비교 (같은 split)

이렇게 보면 "0.95" 가 진짜 신호인지, 아니면 라벨 좁아서 베이스라인이 잘하는 건지 보임.
"""
import torch
import torch.nn as nn
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor

BASE = Path("data/raw/longterm/bonares_data")
CKPT = Path("checkpoints/gaia_v6")  # use latest backbone
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_JSON = Path("docs/benchmark_validation.json")


def build_pivot():
    bac = pd.read_csv(BASE / "lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
    g = pd.read_csv(BASE / "lte_westerfeld.V1_0_GENUS.csv")
    nm = dict(zip(g["Genus_ID"], g["Name"]))
    bac["Genus_Name"] = bac["Genus_ID"].map(nm)
    grp = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
    return grp.pivot_table(index=["Plot_ID", "Experimental_Year"],
                           columns="Genus_Name", values="Value", fill_value=0).reset_index()


def build_pairs(pivot, label_col):
    soil = pd.read_csv(BASE / "lte_westerfeld.V1_0_SOIL_LAB.csv")
    samp = pd.read_csv(BASE / "lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
    soil = soil.merge(samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]],
                      on="Soil_Sampling_ID", how="left")
    sub = soil.dropna(subset=[label_col])
    if len(sub) == 0:
        return None
    agg = sub.groupby(["Plot_ID", "Experimental_Year"])[label_col].mean().reset_index()
    return pivot.merge(agg, on=["Plot_ID", "Experimental_Year"],
                       how="inner").dropna(subset=[label_col]).reset_index(drop=True)


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
    embs = []
    for _, row in df.iterrows():
        x = encode_row(row, gcols, tok).unsqueeze(0).to(DEVICE)
        h = gpt(x, output_hidden_states=True).hidden_states[-1]
        m = (x != 0).unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        embs.append(p.squeeze(0).cpu().numpy())
    return np.stack(embs)


class MLP(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def train_eval_mlp(Xtr, Xte, ytr, yte, epochs=100):
    ym, ys = ytr.mean(), ytr.std() + 1e-9
    ytr_n = (ytr - ym) / ys
    Xt = torch.tensor(Xtr, dtype=torch.float).to(DEVICE)
    yt = torch.tensor(ytr_n, dtype=torch.float).to(DEVICE)
    Xv = torch.tensor(Xte, dtype=torch.float).to(DEVICE)

    mlp = MLP(in_dim=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    bs = 16
    n = len(Xt)
    for ep in range(epochs):
        mlp.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            pr = mlp(Xt[idx])
            loss = nn.MSELoss()(pr, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        pred_n = mlp(Xv).cpu().numpy()
    pred = pred_n * ys + ym
    return pred


def evaluate_split(emb, y, idx_tr, idx_te):
    """Return dict of R² values for: gaia(MLP), RF, mean-baseline."""
    Xtr, Xte = emb[idx_tr], emb[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]

    # Gaia head
    pred_g = train_eval_mlp(Xtr, Xte, ytr, yte)
    r2_g = r2_score(yte, pred_g)

    # RF on raw embeddings
    rf = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    r2_rf = r2_score(yte, rf.predict(Xte))

    # Mean baseline (always predict training mean)
    pred_mean = np.full_like(yte, ytr.mean())
    r2_mean = r2_score(yte, pred_mean)

    rmse_g = float(np.sqrt(mean_squared_error(yte, pred_g)))
    rmse_mean = float(np.sqrt(mean_squared_error(yte, pred_mean)))

    return {
        "gaia_r2": float(r2_g),
        "rf_r2": float(r2_rf),
        "mean_baseline_r2": float(r2_mean),
        "gaia_rmse": rmse_g,
        "mean_baseline_rmse": rmse_mean,
        "improvement_over_mean_rmse_pct": float((rmse_mean - rmse_g) / rmse_mean * 100),
    }


def main():
    print(f"Device: {DEVICE}, backbone: {CKPT.name}")
    gpt = GPT2LMHeadModel.from_pretrained(str(CKPT / "best")).to(DEVICE).eval()
    with open(CKPT / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    pivot = build_pivot()
    gcols = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]

    targets = {"pH": "pH", "Total_Carbon": "Total_Carbon", "Total_Nitrogen": "Total_Nitrogen"}
    results = {"backbone": CKPT.name, "tasks": {}}

    for label, col in targets.items():
        df = build_pairs(pivot, col)
        if df is None or len(df) < 30:
            continue
        y = df[col].values.astype(float)
        print(f"\n=== {label}: n={len(df)}, label range [{y.min():.3f}, {y.max():.3f}] std={y.std():.3f} ===")

        emb = get_emb_matrix(df, gcols, tok, gpt)
        print(f"emb shape: {emb.shape}")

        # (a) Random split
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
        rand = evaluate_split(emb, y, idx_tr, idx_te)
        print(f"  Random split:    gaia R2={rand['gaia_r2']:.3f}  RF R2={rand['rf_r2']:.3f}  mean R2={rand['mean_baseline_r2']:.3f}  RMSE imp {rand['improvement_over_mean_rmse_pct']:.1f}%")

        # (b) Stratified split — bin label into quartiles and stratify
        bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
        idx_tr_s, idx_te_s = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=42, stratify=bins
        )
        strat = evaluate_split(emb, y, idx_tr_s, idx_te_s)
        print(f"  Stratified:      gaia R2={strat['gaia_r2']:.3f}  RF R2={strat['rf_r2']:.3f}  mean R2={strat['mean_baseline_r2']:.3f}  RMSE imp {strat['improvement_over_mean_rmse_pct']:.1f}%")

        # (c) 5-fold CV for robustness
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_g, cv_rf, cv_mean = [], [], []
        for tr_i, te_i in skf.split(np.arange(len(df)), bins):
            r = evaluate_split(emb, y, tr_i, te_i)
            cv_g.append(r["gaia_r2"]); cv_rf.append(r["rf_r2"]); cv_mean.append(r["mean_baseline_r2"])
        print(f"  5-fold CV mean:  gaia R2={np.mean(cv_g):.3f}±{np.std(cv_g):.3f}  RF R2={np.mean(cv_rf):.3f}±{np.std(cv_rf):.3f}  mean R2={np.mean(cv_mean):.3f}±{np.std(cv_mean):.3f}")

        results["tasks"][label] = {
            "label_col": col, "n": int(len(df)),
            "label_min": float(y.min()), "label_max": float(y.max()),
            "label_mean": float(y.mean()), "label_std": float(y.std()),
            "random_split":     rand,
            "stratified_split": strat,
            "cv5_gaia_r2_mean": float(np.mean(cv_g)),
            "cv5_gaia_r2_std":  float(np.std(cv_g)),
            "cv5_rf_r2_mean":   float(np.mean(cv_rf)),
            "cv5_mean_baseline_r2_mean": float(np.mean(cv_mean)),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
