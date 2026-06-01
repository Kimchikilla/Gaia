"""Westerfeld 데이터로 pH/C/N/OM 진단 헤드를 학습하고 저장.

CLI 도구(gaia diagnose)가 로드해서 새 abundance.csv에 바로 예측을 줄 수 있게
헤드 가중치 + 정규화 통계 + 메타데이터를 한 디렉터리에 모아둔다.
"""

import torch  # torch first (Windows c10.dll workaround)
import torch.nn as nn
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error

BASE = Path("data/raw/longterm/bonares_data")
CKPT = Path("checkpoints/gaia_v4")
OUT = CKPT / "heads"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def build_pivot():
    bac = pd.read_csv(BASE / "lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
    genus_ref = pd.read_csv(BASE / "lte_westerfeld.V1_0_GENUS.csv")
    genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
    bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)
    g = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
    pivot = g.pivot_table(
        index=["Plot_ID", "Experimental_Year"],
        columns="Genus_Name",
        values="Value",
        fill_value=0,
    ).reset_index()
    return pivot


def build_pairs(pivot, label_col):
    soil = pd.read_csv(BASE / "lte_westerfeld.V1_0_SOIL_LAB.csv")
    samp = pd.read_csv(BASE / "lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
    soil = soil.merge(
        samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]],
        on="Soil_Sampling_ID",
        how="left",
    )
    sub = soil.dropna(subset=[label_col])
    if len(sub) == 0:
        return None
    agg = sub.groupby(["Plot_ID", "Experimental_Year"])[label_col].mean().reset_index()
    paired = pivot.merge(agg, on=["Plot_ID", "Experimental_Year"], how="inner")
    paired = paired.dropna(subset=[label_col]).reset_index(drop=True)
    return paired


class GenusDataset(Dataset):
    def __init__(self, df, genus_cols, tokenizer, labels):
        self.samples, self.labels = [], []
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nz = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
            tokens = [bos]
            for genus in nz.index:
                tid = tokenizer.vocab.get(f"g__{genus}")
                if tid is not None:
                    tokens.append(tid)
                if len(tokens) >= 511:
                    break
            tokens.append(eos)
            while len(tokens) < 512:
                tokens.append(pad)
            if sum(1 for t in tokens if t not in [bos, eos, pad]) >= 3:
                self.samples.append(torch.tensor(tokens[:512], dtype=torch.long))
                self.labels.append(labels[i])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i], self.labels[i]


class Head(nn.Module):
    def __init__(self, gpt, hidden=256):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.mlp(pooled).squeeze(-1)


def train_head(name, df, genus_cols, label_col, tokenizer, gpt, epochs=40):
    print(f"\n{'='*50}\nTraining head: {name} ({label_col})\n{'='*50}")
    ym, ys = float(df[label_col].mean()), float(df[label_col].std() or 1.0)
    labels_norm = ((df[label_col] - ym) / ys).tolist()

    ds = GenusDataset(df, genus_cols, tokenizer, labels_norm)
    print(f"Samples: {len(ds)}")
    if len(ds) < 20:
        print("Skipping — too few samples")
        return None

    n_tr = int(0.8 * len(ds))
    tr, te = random_split(ds, [n_tr, len(ds) - n_tr],
                          generator=torch.Generator().manual_seed(42))

    head = Head(gpt).to(DEVICE)
    opt = torch.optim.Adam(head.mlp.parameters(), lr=1e-3)
    best_r2, best_state = -1e9, None

    for ep in range(epochs):
        head.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            pred = head(bx.to(DEVICE))
            loss = nn.MSELoss()(pred, torch.tensor(by, dtype=torch.float).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        p, l = [], []
        with torch.no_grad():
            for bx, by in DataLoader(te, batch_size=16):
                p.extend(head(bx.to(DEVICE)).cpu().tolist())
                l.extend(by)
        po = [v * ys + ym for v in p]
        lo = [v * ys + ym for v in l]
        r2 = r2_score(lo, po)

        if r2 > best_r2:
            best_r2 = r2
            best_state = {k: v.detach().cpu().clone() for k, v in head.mlp.state_dict().items()}

        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1}: R2={r2:.4f} (best={best_r2:.4f})")

    print(f"  → best R2={best_r2:.4f}")
    torch.save({
        "state_dict": best_state,
        "label_col": label_col,
        "y_mean": ym,
        "y_std": ys,
        "best_r2": best_r2,
        "n_samples": len(ds),
        "hidden_size": 256,
    }, OUT / f"{name}.pt")
    return best_r2


def main():
    print("Loading model + tokenizer...")
    gpt = GPT2LMHeadModel.from_pretrained(str(CKPT / "best")).to(DEVICE)
    gpt.eval()
    with open(CKPT / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Vocab: {len(tokenizer.vocab)}")

    pivot = build_pivot()
    genus_cols = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]
    print(f"Westerfeld pivot: {len(pivot)} samples × {len(genus_cols)} genera")

    targets = {
        "ph":           "pH",
        "total_carbon": "Total_Carbon",
        "total_n":      "Total_Nitrogen",
    }
    summary = {}
    for name, col in targets.items():
        paired = build_pairs(pivot, col)
        if paired is None or len(paired) < 20:
            print(f"Skip {name}: no data for {col}")
            continue
        r2 = train_head(name, paired, genus_cols, col, tokenizer, gpt, epochs=40)
        if r2 is not None:
            summary[name] = {"label_col": col, "best_r2": r2, "n": len(paired)}

    with open(OUT / "manifest.json", "w") as f:
        json.dump({"backbone": "gaia_v4", "heads": summary}, f, indent=2)
    print("\nSaved heads:", list(summary.keys()))
    print(f"Manifest: {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
