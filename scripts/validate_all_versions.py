"""v6 / v7 / v8 / v9 통합 비교.

각 모델로:
  A. country probe acc (낮을수록 좋음)
  B. LOCO mean acc (높을수록 좋음, 또는 random gap 작을수록 좋음)
  C. Westerfeld 진단 R² (유지)

모델별 토큰화 차이 처리:
  v6: GPT2, abundance descending sort
  v7: GPT2, CLR descending sort
  v8: BERT, CLR descending sort
  v9: BERT (encoder), CLR descending sort
"""
import torch
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel, BertModel, BertConfig
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMP_AB = "data/raw/emp/emp_soil_genus_20260403_181037.csv"
EMP_MD = "data/raw/emp/emp_soil_metadata_20260403_180909.csv"
WEST_BASE = Path("data/raw/longterm/bonares_data")
OUT_JSON = Path("docs/benchmark_v6_v7_v8_v9.json")


def tss_clr(x):
    s = x.sum()
    if s <= 0: return np.zeros_like(x, dtype=np.float64)
    xn = (x / s) + 1e-9
    log_x = np.log(xn)
    return log_x - log_x.mean()


def encode_abundance_sort(row, gcols, tok, max_len=512):
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    nz = row > 0
    order = np.argsort(-row)
    t = [bos]
    for i in order:
        if not nz[i]: break
        g = gcols[i]
        for cand in (f"g__{g}", g):
            if cand in tok.vocab:
                t.append(tok.vocab[cand]); break
        if len(t) >= max_len - 1: break
    t.append(eos)
    while len(t) < max_len: t.append(pad)
    return torch.tensor(t[:max_len], dtype=torch.long)


def encode_clr_sort(row, gcols, tok, max_len=512):
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    clr = tss_clr(row.astype(np.float64))
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


def load_model(ckpt_dir, model_kind):
    if model_kind == "gpt2":
        return GPT2LMHeadModel.from_pretrained(str(ckpt_dir / "best")).to(DEVICE).eval()
    elif model_kind == "bert":
        return BertModel.from_pretrained(str(ckpt_dir / "best")).to(DEVICE).eval()
    raise ValueError(model_kind)


@torch.no_grad()
def embed(values, gcols, tok, model, encode_fn, model_kind, max_n=2000):
    n = values.shape[0]
    if n > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_n, replace=False)
    else:
        idx = np.arange(n)
    out = []
    for k, i in enumerate(idx):
        x = encode_fn(values[i], gcols, tok).unsqueeze(0).to(DEVICE)
        if model_kind == "gpt2":
            h = model(x, output_hidden_states=True).hidden_states[-1]
        else:
            attn = (x != tok.pad_token_id).long()
            h = model(input_ids=x, attention_mask=attn).last_hidden_state
        m = (x != tok.pad_token_id).unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(p.squeeze(0).cpu().numpy())
        if (k + 1) % 200 == 0:
            print(f"      embedded {k+1}/{len(idx)}")
    return np.stack(out), idx


def country_probes(X, countries, biomes):
    keep_c = pd.Series(countries).value_counts()
    keep_c = keep_c[keep_c >= 10].index
    mask_c = pd.Series(countries).isin(keep_c).values
    Xc, yc = X[mask_c], np.array(countries)[mask_c]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cp = []
    for tr, te in skf.split(Xc, yc):
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                 class_weight="balanced", n_jobs=-1).fit(Xc[tr], yc[tr])
        cp.append(accuracy_score(yc[te], clf.predict(Xc[te])))
    country_acc = float(np.mean(cp))

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
    loco = []
    for tc in test_countries:
        tr_m = cf != tc; te_m = cf == tc
        common = set(np.unique(bf[tr_m])) & set(np.unique(bf[te_m]))
        if len(common) < 2: continue
        m_tr = tr_m & np.isin(bf, list(common))
        m_te = te_m & np.isin(bf, list(common))
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                 class_weight="balanced", n_jobs=-1).fit(Xf[m_tr], bf[m_tr])
        loco.append(accuracy_score(bf[m_te], clf.predict(Xf[m_te])))
    return country_acc, float(rand_acc), float(np.mean(loco)) if loco else None


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


def train_eval_mlp(Xtr, Xte, ytr, yte, epochs=80):
    ym, ys = ytr.mean(), ytr.std() + 1e-9
    ytr_n = (ytr - ym) / ys
    Xt = torch.tensor(Xtr, dtype=torch.float).to(DEVICE)
    yt = torch.tensor(ytr_n, dtype=torch.float).to(DEVICE)
    Xv = torch.tensor(Xte, dtype=torch.float).to(DEVICE)
    mlp = MLP(in_dim=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    bs = 16; n = len(Xt)
    for _ in range(epochs):
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


def evaluate(name, ckpt_dir, kind, encode_fn):
    print(f"\n##### {name} #####")
    with open(ckpt_dir / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    if not hasattr(tok, "pad_token_id"):
        tok.pad_token_id = 0
    model = load_model(ckpt_dir, kind)

    print("  EMP probes...")
    md = pd.read_csv(EMP_MD, low_memory=False)
    ab = pd.read_csv(EMP_AB)
    join = next((c for c in ["sample_id", "#SampleID", "SampleID"] if c in md.columns), None)
    if join != "sample_id":
        md = md.rename(columns={join: "sample_id"})
    df = ab.merge(md[["sample_id", "country", "envo_biome_2"]], on="sample_id", how="inner")
    df["country_short"] = df["country"].str.replace("GAZ:", "", regex=False).fillna("Unknown")
    df = df[df["country_short"] != "Unknown"].dropna(subset=["envo_biome_2"]).reset_index(drop=True)
    gcols = [c for c in ab.columns if c != "sample_id"]
    values = df[gcols].values.astype(np.float32)
    X, idx = embed(values, gcols, tok, model, encode_fn, kind, max_n=2000)
    countries = df["country_short"].values[idx]
    biomes = df["envo_biome_2"].values[idx]
    country_acc, rand_acc, loco_mean = country_probes(X, countries, biomes)
    print(f"    country probe: {country_acc:.3f}  random biome: {rand_acc:.3f}  LOCO: {loco_mean}")

    print("  Westerfeld R² 5CV...")
    pivot = build_west_pivot()
    g_w = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]
    diag = {}
    for col in ["pH", "Total_Carbon", "Total_Nitrogen"]:
        pairs = build_pairs(pivot, col)
        if pairs is None or len(pairs) < 50: continue
        Xw = pairs[g_w].values.astype(np.float32)
        emb_w, _ = embed(Xw, g_w, tok, model, encode_fn, kind, max_n=len(pairs))
        y_w = pairs[col].values.astype(float)
        m, s = diag_r2(emb_w, y_w)
        diag[col] = {"r2_mean": m, "r2_std": s}
        print(f"    {col}: {m:.3f}±{s:.3f}")

    return {"country_probe_acc": country_acc, "biome_random_acc": rand_acc,
            "loco_mean_acc": loco_mean, "diagnostic": diag}


def main():
    res = {}
    res["v6"] = evaluate("gaia_v6 (GPT2, abund-sort)", Path("checkpoints/gaia_v6"), "gpt2", encode_abundance_sort)
    res["v7"] = evaluate("gaia_v7 (GPT2, CLR-sort)",   Path("checkpoints/gaia_v7"), "gpt2", encode_clr_sort)
    if Path("checkpoints/gaia_v8/best").exists():
        res["v8"] = evaluate("gaia_v8 (BERT, CLR-sort)", Path("checkpoints/gaia_v8"), "bert", encode_clr_sort)
    if Path("checkpoints/gaia_v9/best").exists():
        res["v9"] = evaluate("gaia_v9 (BERT+adv, CLR-sort)", Path("checkpoints/gaia_v9"), "bert", encode_clr_sort)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2))

    print("\n" + "="*90)
    print(f"{'metric':<35} " + "  ".join([f"{k:>10}" for k in res.keys()]))
    print("="*90)
    def row(name, getter):
        vals = []
        for k in res:
            v = getter(res[k])
            vals.append(f"{v:>10.3f}" if v is not None else "       N/A")
        print(f"{name:<35} " + "  ".join(vals))
    row("country probe (lower=better)", lambda r: r["country_probe_acc"])
    row("biome random acc",            lambda r: r["biome_random_acc"])
    row("LOCO mean (higher=better)",   lambda r: r["loco_mean_acc"])
    for c in ["pH", "Total_Carbon", "Total_Nitrogen"]:
        row(f"diag R² {c}", lambda r: r["diagnostic"].get(c, {}).get("r2_mean"))
    print("="*90)


if __name__ == "__main__":
    main()
