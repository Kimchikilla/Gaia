"""v6 백본으로 진단 헤드(pH/C/N) 재학습. v4 헤드와 비교용."""
import torch  # torch first
import torch.nn as nn
import pickle, json, numpy as np, pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score

BASE = Path("data/raw/longterm/bonares_data")
CKPT = Path("checkpoints/gaia_v6")
OUT = CKPT / "heads"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pivot():
    bac = pd.read_csv(BASE / "lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
    g = pd.read_csv(BASE / "lte_westerfeld.V1_0_GENUS.csv")
    name_map = dict(zip(g["Genus_ID"], g["Name"]))
    bac["Genus_Name"] = bac["Genus_ID"].map(name_map)
    grp = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
    return grp.pivot_table(index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0).reset_index()


def build_pairs(pivot, label_col):
    soil = pd.read_csv(BASE / "lte_westerfeld.V1_0_SOIL_LAB.csv")
    samp = pd.read_csv(BASE / "lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
    soil = soil.merge(samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]], on="Soil_Sampling_ID", how="left")
    sub = soil.dropna(subset=[label_col])
    if len(sub) == 0: return None
    agg = sub.groupby(["Plot_ID", "Experimental_Year"])[label_col].mean().reset_index()
    return pivot.merge(agg, on=["Plot_ID", "Experimental_Year"], how="inner").dropna(subset=[label_col]).reset_index(drop=True)


class DS(Dataset):
    def __init__(self, df, gcols, tok, labels):
        self.s, self.l = [], []
        bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nz = row[gcols][row[gcols] > 0].sort_values(ascending=False)
            t = [bos]
            for g in nz.index:
                tid = tok.vocab.get(f"g__{g}")
                if tid: t.append(tid)
                if len(t) >= 511: break
            t.append(eos)
            while len(t) < 512: t.append(pad)
            if sum(1 for x in t if x not in (bos, eos, pad)) >= 3:
                self.s.append(torch.tensor(t[:512], dtype=torch.long))
                self.l.append(labels[i])
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i], self.l[i]


class Head(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters(): p.requires_grad = False
        self.mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        m = (x != 0).unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.mlp(p).squeeze(-1)


def train(name, df, gcols, label, tok, gpt, epochs=40):
    ym, ys = float(df[label].mean()), float(df[label].std() or 1.0)
    labels_n = ((df[label] - ym) / ys).tolist()
    ds = DS(df, gcols, tok, labels_n)
    if len(ds) < 20: return None
    n_tr = int(0.8 * len(ds))
    tr, te = random_split(ds, [n_tr, len(ds) - n_tr], generator=torch.Generator().manual_seed(42))
    head = Head(gpt).to(DEVICE)
    opt = torch.optim.Adam(head.mlp.parameters(), lr=1e-3)
    best, bs = -1e9, None
    for ep in range(epochs):
        head.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            pr = head(bx.to(DEVICE))
            loss = nn.MSELoss()(pr, torch.tensor(by, dtype=torch.float).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        ap, al = [], []
        with torch.no_grad():
            for bx, by in DataLoader(te, batch_size=16):
                ap.extend(head(bx.to(DEVICE)).cpu().tolist()); al.extend(by)
        po = [v * ys + ym for v in ap]; lo = [v * ys + ym for v in al]
        r2 = r2_score(lo, po)
        if r2 > best:
            best = r2
            bs = {k: v.detach().cpu().clone() for k, v in head.mlp.state_dict().items()}
    print(f"  {name} ({label}): best R2={best:.4f}")
    torch.save({"state_dict": bs, "label_col": label, "y_mean": ym, "y_std": ys, "best_r2": best, "n_samples": len(ds), "hidden_size": 256}, OUT / f"{name}.pt")
    return best


def main():
    print(f"Device: {DEVICE}")
    gpt = GPT2LMHeadModel.from_pretrained(str(CKPT / "best")).to(DEVICE).eval()
    with open(CKPT / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    pivot = build_pivot()
    gcols = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]
    print(f"Westerfeld: {len(pivot)} × {len(gcols)}")
    targets = {"ph": "pH", "total_carbon": "Total_Carbon", "total_n": "Total_Nitrogen"}
    summ = {}
    for n, c in targets.items():
        p = build_pairs(pivot, c)
        if p is None: continue
        r2 = train(n, p, gcols, c, tok, gpt, epochs=40)
        if r2 is not None:
            summ[n] = {"label_col": c, "best_r2": float(r2), "n": int(len(p))}
    Path(OUT / "manifest.json").write_text(json.dumps({"backbone": "gaia_v6", "heads": summ}, indent=2))
    print("done. heads saved at", OUT)


if __name__ == "__main__":
    main()
