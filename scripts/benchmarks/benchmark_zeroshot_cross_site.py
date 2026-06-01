"""
True zero-shot cross-site test:
  Train head on Westerfeld -> evaluate on Bernburg (no Bernburg training).
Tasks: pH, Total Carbon, Total Nitrogen.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# 1. Load Gaia v4
print("=== Loading Gaia v4 ===")
model = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_v4/best")
model.eval().cuda()
with open("checkpoints/gaia_v4/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 2. Build Westerfeld (train site)
print("\n=== Building Westerfeld ===")
BASE = "data/raw/longterm/bonares_data"
bac = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
genus_ref = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_GENUS.csv")
genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)
grouped = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
w_pivot = grouped.pivot_table(
    index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0
).reset_index()
w_genera = [c for c in w_pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]

# Westerfeld chemistry
soil = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_SOIL_LAB.csv")
samp = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
soil = soil.merge(samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]], on="Soil_Sampling_ID")
chem = soil.groupby(["Plot_ID", "Experimental_Year"])[["Total_Carbon", "pH", "Total_Nitrogen"]].mean().reset_index()
westerfeld = w_pivot.merge(chem, on=["Plot_ID", "Experimental_Year"], how="inner")
print(f"  {len(westerfeld)} samples, {len(w_genera)} genera")

# 3. Load Bernburg (test site)
print("\n=== Building Bernburg ===")
b_abund = pd.read_csv("data/processed_real/bernburg_abundance.csv", index_col=0)
b_meta = pd.read_csv("data/processed_real/bernburg_metadata.csv")
bernburg = b_abund.merge(b_meta, left_index=True, right_on="Sample", how="inner")
b_genera = list(b_abund.columns)
print(f"  {len(bernburg)} samples, {len(b_genera)} genera")

# 4. Use shared genera (intersection)
shared = sorted(set(w_genera) & set(b_genera))
print(f"\nShared genera: {len(shared)} (Westerfeld {len(w_genera)} ∩ Bernburg {len(b_genera)})")


class GenusDataset(Dataset):
    def __init__(self, df, genus_cols, tokenizer, labels):
        self.samples, self.labels = [], []
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nonzero = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
            tokens = [bos]
            for g in nonzero.index:
                tid = tokenizer.vocab.get(f"g__{g}")
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

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class Head(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def zero_shot(train_df, test_df, train_label, test_label, name):
    print(f"\n--- {name} ---")
    train_df = train_df.dropna(subset=[train_label]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[test_label]).reset_index(drop=True)
    print(f"  Train (Westerfeld): n={len(train_df)}, range={train_df[train_label].min():.2f}~{train_df[train_label].max():.2f}")
    print(f"  Test  (Bernburg):   n={len(test_df)}, range={test_df[test_label].min():.2f}~{test_df[test_label].max():.2f}")

    # Normalize using train stats
    ym, ys = train_df[train_label].mean(), train_df[train_label].std()
    tr_labels = ((train_df[train_label] - ym) / ys).tolist()
    te_labels = ((test_df[test_label] - ym) / ys).tolist()

    # Datasets use shared genera only
    tr_ds = GenusDataset(train_df, shared, tokenizer, tr_labels)
    te_ds = GenusDataset(test_df, shared, tokenizer, te_labels)

    # Train head on Westerfeld
    reg = Head(model).cuda()
    opt = torch.optim.Adam(reg.head.parameters(), lr=1e-3)
    for ep in range(40):
        reg.train()
        for bx, by in DataLoader(tr_ds, batch_size=16, shuffle=True):
            loss = nn.MSELoss()(reg(bx.cuda()), torch.tensor(by, dtype=torch.float).cuda())
            opt.zero_grad(); loss.backward(); opt.step()

    # Predict on Bernburg
    reg.eval()
    p, l = [], []
    with torch.no_grad():
        for bx, by in DataLoader(te_ds, batch_size=16):
            p.extend(reg(bx.cuda()).cpu().tolist())
            l.extend(by)
    po = [v * ys + ym for v in p]
    lo = [v * ys + ym for v in l]
    g_r2 = r2_score(lo, po)

    # RF zero-shot baseline
    Xtr = train_df[shared].values
    ytr = train_df[train_label].values
    Xte = test_df[shared].values
    yte = test_df[test_label].values
    rf = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    rf_r2 = r2_score(yte, rf.predict(Xte))

    print(f"  Gaia (zero-shot): R2={g_r2:.3f}")
    print(f"  RF   (zero-shot): R2={rf_r2:.3f}")


zero_shot(westerfeld, bernburg, "Total_Carbon", "C[%]",        "Total Carbon (Westerfeld -> Bernburg)")
zero_shot(westerfeld, bernburg, "pH",           "pH",          "pH (Westerfeld -> Bernburg)")
zero_shot(westerfeld, bernburg, "Total_Nitrogen", "N[%]",       "Total Nitrogen (Westerfeld -> Bernburg)")

print("\n=== Zero-shot benchmark complete ===")
