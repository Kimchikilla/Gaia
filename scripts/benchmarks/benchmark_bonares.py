"""BonaRes 20-year data: tillage classification + soil chemistry prediction"""

import pkg_resources
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from mgm.src.utils import CustomUnpickler
from mgm.CLI.CLI_utils import find_pkg_resource
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE = "data/raw/longterm/bonares_data"

# 1. Build genus-level abundance table
print("=== Building genus abundance table ===")

bac = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
genus_ref = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_GENUS.csv")
plot = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_PLOT.csv")
treatment = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_TREATMENT.csv")
f1_level = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_FACTOR_1_LEVEL.csv")

# Genus name mapping
genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)

# Group by Plot + Year → genus abundance
print("Aggregating by plot and year...")
grouped = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
pivot = grouped.pivot_table(index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0)
pivot = pivot.reset_index()
print(f"Abundance table: {pivot.shape[0]} samples x {pivot.shape[1]-2} genera")

# Add tillage label via Plot → Treatment → Factor_1
plot_treat = plot[["Plot_ID", "Treatment_ID"]].merge(treatment, on="Treatment_ID")
plot_treat = plot_treat.merge(f1_level[["Factor_1_Level_ID", "Name_EN"]], on="Factor_1_Level_ID")
plot_treat = plot_treat.rename(columns={"Name_EN": "tillage"})

data = pivot.merge(plot_treat[["Plot_ID", "tillage"]], on="Plot_ID", how="left")
data = data.dropna(subset=["tillage"]).reset_index(drop=True)
print(f"With tillage label: {data.shape[0]} samples")
print(f"Cultivator: {(data['tillage']=='Cultivator').sum()}, Plough: {(data['tillage']=='Plough').sum()}")

genus_cols = [c for c in data.columns if c not in ["Plot_ID", "Experimental_Year", "tillage"]]

# 2. Load MGM model
print("\n=== Loading MGM model ===")
model = GPT2LMHeadModel.from_pretrained("checkpoints/mgm_soil_3k/best")
with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
    tokenizer = CustomUnpickler(f).load()

matched = sum(1 for g in genus_cols if f"g__{g}" in tokenizer.vocab)
print(f"Genera: {len(genus_cols)}, MGM matched: {matched} ({matched/len(genus_cols)*100:.0f}%)")


class GenusDataset(Dataset):
    def __init__(self, df, genus_cols, tokenizer, labels):
        self.samples, self.labels = [], []
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nonzero = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
            tokens = [bos]
            for genus in nonzero.index:
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

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class Classifier(nn.Module):
    def __init__(self, gpt, n):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n)
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)


# 3. Tillage Classification
print("\n" + "=" * 50)
print("Task: Tillage Classification (Cultivator vs Plough)")
print("=" * 50)

data["tillage_label"] = (data["tillage"] == "Plough").astype(int)
ds = GenusDataset(data, genus_cols, tokenizer, data["tillage_label"].tolist())
print(f"Tokenized: {len(ds)} samples")

clf = Classifier(model, 2)
tr_size = int(0.8 * len(ds))
tr, te = random_split(ds, [tr_size, len(ds) - tr_size], generator=torch.Generator().manual_seed(42))
opt = torch.optim.Adam(clf.head.parameters(), lr=1e-3)

for ep in range(20):
    clf.train()
    for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
        loss = nn.CrossEntropyLoss()(clf(bx), torch.tensor(by))
        opt.zero_grad()
        loss.backward()
        opt.step()

clf.eval()
preds, labels = [], []
with torch.no_grad():
    for bx, by in DataLoader(te, batch_size=16):
        preds.extend(clf(bx).argmax(1).tolist())
        labels.extend(by)

g_acc = accuracy_score(labels, preds)
g_f1 = f1_score(labels, preds, average="weighted")

# RF
X = data[genus_cols].values
y = data["tillage_label"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
r_acc = accuracy_score(yte, rf.predict(Xte))
r_f1 = f1_score(yte, rf.predict(Xte), average="weighted")

print(f"Gaia: accuracy={g_acc:.4f}, f1={g_f1:.4f}")
print(f"RF:   accuracy={r_acc:.4f}, f1={r_f1:.4f}")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Tillage Classification: Gaia {g_acc:.1%} vs RF {r_acc:.1%}")
