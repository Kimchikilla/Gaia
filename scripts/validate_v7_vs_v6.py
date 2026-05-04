"""v6 vs v7 비교 검증 — CLR 가 batch shortcut 줄였는지 한 번에 본다.

세 가지 지표 한 표로:
  A. country probe acc (낮을수록 좋음 — batch fingerprint 약해진다는 뜻)
  B. LOCO acc gap (작을수록 좋음 — cross-country 일반화 잘 된다는 뜻)
  C. Westerfeld 진단 R² (유지되어야 함 — 실제 신호는 살아남아야 함)

같은 EMP 임베딩 시드, 같은 Westerfeld 시드 사용해 직접 비교.
"""
import torch
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMP_AB = "data/raw/emp/emp_soil_genus_20260403_181037.csv"
EMP_MD = "data/raw/emp/emp_soil_metadata_20260403_180909.csv"
WEST_BASE = Path("data/raw/longterm/bonares_data")
OUT_JSON = Path("docs/benchmark_v6_vs_v7.json")


# ============ Tokenization ============
# v6: abundance descending sort, raw counts
# v7: CLR + batch-corrected, sort by adjusted CLR descending — BUT for evaluation
# we need to use the SAME tokenization the model was trained on.
# So when evaluating v7, we apply CLR to the EMP/Westerfeld inputs first.

def tss_clr(x):
    """x: 1D abundance row. Returns (TSS-normalized, CLR-transformed) value array."""
    s = x.sum()
    if s <= 0:
        return np.zeros_like(x, dtype=np.float64)
    xn = (x / s) + 1e-9
    log_x = np.log(xn)
    return log_x - log_x.mean()


def encode_v6(row_values, gcols, tok, max_len=512):
    """v6 토큰화: raw abundance descending."""
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    nz_mask = row_values > 0
    order = np.argsort(-row_values)
    t = [bos]
    for i in order:
        if not nz_mask[i]: break
        g = gcols[i]
        for cand in (f"g__{g}", g):
            if cand in tok.vocab:
                t.append(tok.vocab[cand]); break
        if len(t) >= max_len - 1: break
    t.append(eos)
    while len(t) < max_len: t.append(pad)
    return torch.tensor(t[:max_len], dtype=torch.long)


def encode_v7(row_values, gcols, tok, max_len=512):
    """v7 토큰화: CLR sort descending."""
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    clr = tss_clr(row_values.astype(np.float64))
    order = np.argsort(-clr)
    t = [bos]
    for i in order:
        g = gcols[i]
        for cand in (f"g__{g}", g):
            if cand in tok.vocab:
                t.append(tok.vocab[cand]); break
        if len(t) >= max_len - 1: break
    t.append(eos)
    while len(t) < max_len: t.append(pad)
    return torch.tensor(t[:max_len], dtype=torch.long)


@torch.no_grad()
def get_embeddings(values, gcols, tok, gpt, encode_fn, max_n=2000):
    """values: (N, G) numpy. Return (N', 256) embeddings + selected indices."""
    n = values.shape[0]
    if n > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_n, replace=False)
    else:
        idx = np.arange(n)
    out = []
    for k, i in enumerate(idx):
        x = encode_fn(values[i], gcols, tok).unsqueeze(0).to(DEVICE)
        h = gpt(x, output_hidden_states=True).hidden_states[-1]
        m = (x != 0).unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(p.squeeze(0).cpu().numpy())
        if (k + 1) % 200 == 0:
            print(f"    embedded {k+1}/{len(idx)}")
    return np.stack(out), idx


# ============ Country probe + LOCO ============
def country_probes(X, countries, biomes):
    """Return (random_acc, loco_mean_acc, loco_per_country, country_probe_acc)."""
    # country probe
    keep_c = pd.Series(countries).value_counts()
    keep_c = keep_c[keep_c >= 10].index
    mask_c = pd.Series(countries).isin(keep_c).values
    Xc, yc = X[mask_c], np.array(countries)[mask_c]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cp_accs = []
    for tr, te in skf.split(Xc, yc):
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                 class_weight="balanced", n_jobs=-1).fit(Xc[tr], yc[tr])
        cp_accs.append(accuracy_score(yc[te], clf.predict(Xc[te])))
    country_acc = float(np.mean(cp_accs))
    print(f"  country probe (5CV): {country_acc:.3f}")

    # LOCO biome
    keep_b = pd.Series(biomes).value_counts()
    keep_b = keep_b[keep_b >= 10].index
    mask_b = pd.Series(biomes).isin(keep_b).values
    Xf, cf, bf = X[mask_b], np.array(countries)[mask_b], np.array(biomes)[mask_b]

    Xtr, Xte, ytr, yte = train_test_split(Xf, bf, test_size=0.2, random_state=42, stratify=bf)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                             class_weight="balanced", n_jobs=-1).fit(Xtr, ytr)
    rand_acc = accuracy_score(yte, clf.predict(Xte))

    counts = pd.Series(cf).value_counts()
    test_countries = counts[counts >= 50].index.tolist()
    loco = {}
    for tc in test_countries:
        tr_m = cf != tc; te_m = cf == tc
        common = set(np.unique(bf[tr_m])) & set(np.unique(bf[te_m]))
        if len(common) < 2: continue
        m_tr = tr_m & np.isin(bf, list(common))
        m_te = te_m & np.isin(bf, list(common))
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                 class_weight="balanced", n_jobs=-1)
        clf.fit(Xf[m_tr], bf[m_tr])
        acc = accuracy_score(bf[m_te], clf.predict(Xf[m_te]))
        loco[tc] = float(acc)

    loco_mean = float(np.mean(list(loco.values()))) if loco else None
    print(f"  random 80/20 biome acc: {rand_acc:.3f}")
    print(f"  LOCO mean (cross-country): {loco_mean:.3f}")
    return {"country_probe_acc": country_acc, "biome_random_acc": float(rand_acc),
            "loco_mean_acc": loco_mean, "loco_per_country": loco}


# ============ Westerfeld diagnostic R² ============
def build_west_pivot():
    bac = pd.read_csv(WEST_BASE / "lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
    g = pd.read_csv(WEST_BASE / "lte_westerfeld.V1_0_GENUS.csv")
    nm = dict(zip(g["Genus_ID"], g["Name"]))
    bac["Genus_Name"] = bac["Genus_ID"].map(nm)
    grp = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
    return grp.pivot_table(index=["Plot_ID", "Experimental_Year"],
                           columns="Genus_Name", values="Value", fill_value=0).reset_index()


def build_pairs(pivot, col):
    soil = pd.read_csv(WEST_BASE / "lte_westerfeld.V1_0_SOIL_LAB.csv")
    samp = pd.read_csv(WEST_BASE / "lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
    soil = soil.merge(samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]],
                      on="Soil_Sampling_ID", how="left")
    sub = soil.dropna(subset=[col])
    if len(sub) == 0: return None
    agg = sub.groupby(["Plot_ID", "Experimental_Year"])[col].mean().reset_index()
    return pivot.merge(agg, on=["Plot_ID", "Experimental_Year"],
                       how="inner").dropna(subset=[col]).reset_index(drop=True)


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
    bs = 16; n = len(Xt)
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
    return pred_n * ys + ym


def diag_r2(emb, y, n_splits=5):
    bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, te in skf.split(np.arange(len(y)), bins):
        pred = train_eval_mlp(emb[tr], emb[te], y[tr], y[te])
        scores.append(r2_score(y[te], pred))
    return float(np.mean(scores)), float(np.std(scores))


# ============ Main comparison ============
def evaluate_backbone(ckpt_dir: Path, encode_fn):
    print(f"\n##### {ckpt_dir.name} #####")
    gpt = GPT2LMHeadModel.from_pretrained(str(ckpt_dir / "best")).to(DEVICE).eval()
    with open(ckpt_dir / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    # ------ EMP probes ------
    print("Loading EMP...")
    md = pd.read_csv(EMP_MD, low_memory=False)
    ab = pd.read_csv(EMP_AB)
    join = next((c for c in ["sample_id", "#SampleID", "SampleID"] if c in md.columns), None)
    if join != "sample_id":
        md = md.rename(columns={join: "sample_id"})
    df = ab.merge(md[["sample_id", "country", "envo_biome_2"]], on="sample_id", how="inner")
    df["country_short"] = df["country"].str.replace("GAZ:", "", regex=False).fillna("Unknown")
    df = df[df["country_short"] != "Unknown"].dropna(subset=["envo_biome_2"]).reset_index(drop=True)
    print(f"  EMP rows: {len(df)}")
    gcols = [c for c in ab.columns if c != "sample_id"]
    values = df[gcols].values.astype(np.float32)

    print("  Embedding EMP (max 2000)...")
    emp_X, idx = get_embeddings(values, gcols, tok, gpt, encode_fn, max_n=2000)
    countries = df["country_short"].values[idx]
    biomes = df["envo_biome_2"].values[idx]

    emp_res = country_probes(emp_X, countries, biomes)

    # ------ Westerfeld diagnostic ------
    print("\n  Westerfeld diagnostic R² (5-fold CV)...")
    pivot = build_west_pivot()
    g_west = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]
    diag = {}
    for col in ["pH", "Total_Carbon", "Total_Nitrogen"]:
        pairs = build_pairs(pivot, col)
        if pairs is None or len(pairs) < 50: continue
        X_w = pairs[g_west].values.astype(np.float32)
        emb_w, _ = get_embeddings(X_w, g_west, tok, gpt, encode_fn, max_n=len(pairs))
        y_w = pairs[col].values.astype(float)
        r2_m, r2_s = diag_r2(emb_w, y_w)
        diag[col] = {"r2_mean": r2_m, "r2_std": r2_s, "n": len(pairs)}
        print(f"    {col}: R²={r2_m:.3f}±{r2_s:.3f}")

    return {"emp": emp_res, "diagnostic": diag}


def main():
    res = {}
    res["v6"] = evaluate_backbone(Path("checkpoints/gaia_v6"), encode_v6)
    res["v7"] = evaluate_backbone(Path("checkpoints/gaia_v7"), encode_v7)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"\nSaved {OUT_JSON}")

    # Summary
    v6, v7 = res["v6"], res["v7"]
    print("\n" + "="*70)
    print(f"{'metric':<35} {'v6':>15} {'v7 (CLR)':>15}  {'change':>8}")
    print("="*70)

    def row(name, a, b):
        chg = b - a if (a is not None and b is not None) else None
        print(f"{name:<35} {a:>15.3f} {b:>15.3f}  {chg:>+8.3f}")

    row("country probe acc (lower=better)",
        v6["emp"]["country_probe_acc"], v7["emp"]["country_probe_acc"])
    row("biome random acc",
        v6["emp"]["biome_random_acc"], v7["emp"]["biome_random_acc"])
    row("LOCO mean (higher=better)",
        v6["emp"]["loco_mean_acc"], v7["emp"]["loco_mean_acc"])
    for c in ["pH", "Total_Carbon", "Total_Nitrogen"]:
        if c in v6["diagnostic"] and c in v7["diagnostic"]:
            row(f"diag R² {c}",
                v6["diagnostic"][c]["r2_mean"],
                v7["diagnostic"][c]["r2_mean"])

    print("="*70)
    print("\n해석:")
    cp_drop = v6["emp"]["country_probe_acc"] - v7["emp"]["country_probe_acc"]
    if cp_drop > 0.10:
        print(f"  country probe {cp_drop:.2f} 만큼 떨어짐 → CLR + batch 보정이 효과 있음")
    else:
        print(f"  country probe {cp_drop:.2f} 만 떨어짐 → 토큰화 자체가 batch 보존 → 다음 단계 필요")


if __name__ == "__main__":
    main()
