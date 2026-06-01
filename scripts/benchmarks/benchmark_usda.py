"""USDA 데이터로 3개 벤치마크: 토양유형, pH, 훈증처리"""

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

# Model
model = GPT2LMHeadModel.from_pretrained("checkpoints/mgm_soil_3k/best")
with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
    tokenizer = CustomUnpickler(f).load()
print("Model loaded")

# Data
df = pd.read_csv("data/raw/tillage/usda_potato.csv")
genus_cols = [c for c in df.columns if c.startswith("BF_g_") or c.startswith("FF_g_")]
genus_name_map = {}
for col in genus_cols:
    parts = col.split("_", 3)
    genus_name_map[col] = parts[3].split("_")[0] if len(parts) >= 4 else col

matched = sum(1 for g in genus_name_map.values() if f"g__{g}" in tokenizer.vocab)
print(f"Samples: {len(df)}, Genera: {len(genus_cols)}, MGM matched: {matched}")


class USDADataset(Dataset):
    def __init__(self, df, genus_cols, genus_name_map, tokenizer, labels):
        self.samples, self.labels = [], []
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nonzero = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
            tokens = [bos]
            for col in nonzero.index:
                genus = genus_name_map.get(col, "")
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


class GaiaClassifier(nn.Module):
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


class GaiaRegressor(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


# === Task 1: Soil Type ===
print("\n" + "=" * 50)
print("Task 1: Soil Type Classification (Sand vs Loam)")
print("=" * 50)

df["soil_label"] = df["Region"].map({"Sand": 0, "Loam": 1})
v1 = df.dropna(subset=["soil_label"]).reset_index(drop=True)
ds1 = USDADataset(v1, genus_cols, genus_name_map, tokenizer, v1["soil_label"].astype(int).tolist())
print(f"Samples: {len(ds1)}")

clf1 = GaiaClassifier(model, 2)
tr, te = random_split(ds1, [int(0.8 * len(ds1)), len(ds1) - int(0.8 * len(ds1))], generator=torch.Generator().manual_seed(42))
opt = torch.optim.Adam(clf1.head.parameters(), lr=1e-3)
for ep in range(20):
    clf1.train()
    for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
        loss = nn.CrossEntropyLoss()(clf1(bx), torch.tensor(by))
        opt.zero_grad()
        loss.backward()
        opt.step()

clf1.eval()
preds, labels = [], []
with torch.no_grad():
    for bx, by in DataLoader(te, batch_size=16):
        preds.extend(clf1(bx).argmax(1).tolist())
        labels.extend(by)
g_acc1 = accuracy_score(labels, preds)
g_f1_1 = f1_score(labels, preds, average="weighted")

X, y = v1[genus_cols].values, v1["soil_label"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf1 = RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
r_acc1 = accuracy_score(yte, rf1.predict(Xte))
r_f1_1 = f1_score(yte, rf1.predict(Xte), average="weighted")
print(f"Gaia: acc={g_acc1:.4f}, f1={g_f1_1:.4f}")
print(f"RF:   acc={r_acc1:.4f}, f1={r_f1_1:.4f}")

# === Task 2: pH Prediction ===
print("\n" + "=" * 50)
print("Task 2: pH Prediction (Real Data)")
print("=" * 50)

ds2 = USDADataset(df, genus_cols, genus_name_map, tokenizer, df["pH_1_1"].tolist())
print(f"Samples: {len(ds2)}")

reg = GaiaRegressor(model)
tr2, te2 = random_split(ds2, [int(0.8 * len(ds2)), len(ds2) - int(0.8 * len(ds2))], generator=torch.Generator().manual_seed(42))
opt2 = torch.optim.Adam(reg.head.parameters(), lr=1e-3)
for ep in range(30):
    reg.train()
    for bx, by in DataLoader(tr2, batch_size=16, shuffle=True):
        loss = nn.MSELoss()(reg(bx), torch.tensor(by, dtype=torch.float))
        opt2.zero_grad()
        loss.backward()
        opt2.step()

reg.eval()
preds2, labels2 = [], []
with torch.no_grad():
    for bx, by in DataLoader(te2, batch_size=16):
        preds2.extend(reg(bx).tolist())
        labels2.extend(by)
g_r2 = r2_score(labels2, preds2)
g_rmse = np.sqrt(mean_squared_error(labels2, preds2))

X, y = df[genus_cols].values, df["pH_1_1"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
rf2 = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
r_r2 = r2_score(yte, rf2.predict(Xte))
r_rmse = np.sqrt(mean_squared_error(yte, rf2.predict(Xte)))
print(f"Gaia: R2={g_r2:.4f}, RMSE={g_rmse:.4f}")
print(f"RF:   R2={r_r2:.4f}, RMSE={r_rmse:.4f}")

# === Task 3: Fumigation ===
print("\n" + "=" * 50)
print("Task 3: Fumigation Classification")
print("=" * 50)

df["fum_label"] = df["Fumigation1"].map({"Yes": 1, "No": 0})
v3 = df.dropna(subset=["fum_label"]).reset_index(drop=True)
ds3 = USDADataset(v3, genus_cols, genus_name_map, tokenizer, v3["fum_label"].astype(int).tolist())
print(f"Samples: {len(ds3)}")

clf3 = GaiaClassifier(model, 2)
tr3, te3 = random_split(ds3, [int(0.8 * len(ds3)), len(ds3) - int(0.8 * len(ds3))], generator=torch.Generator().manual_seed(42))
opt3 = torch.optim.Adam(clf3.head.parameters(), lr=1e-3)
for ep in range(20):
    clf3.train()
    for bx, by in DataLoader(tr3, batch_size=16, shuffle=True):
        loss = nn.CrossEntropyLoss()(clf3(bx), torch.tensor(by))
        opt3.zero_grad()
        loss.backward()
        opt3.step()

clf3.eval()
preds3, labels3 = [], []
with torch.no_grad():
    for bx, by in DataLoader(te3, batch_size=16):
        preds3.extend(clf3(bx).argmax(1).tolist())
        labels3.extend(by)
g_acc3 = accuracy_score(labels3, preds3)
g_f1_3 = f1_score(labels3, preds3, average="weighted")

X, y = v3[genus_cols].values, v3["fum_label"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf3 = RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
r_acc3 = accuracy_score(yte, rf3.predict(Xte))
r_f1_3 = f1_score(yte, rf3.predict(Xte), average="weighted")
print(f"Gaia: acc={g_acc3:.4f}, f1={g_f1_3:.4f}")
print(f"RF:   acc={r_acc3:.4f}, f1={r_f1_3:.4f}")

# Summary
print("\n" + "=" * 50)
print("ALL BENCHMARKS SUMMARY")
print("=" * 50)
print(f"Soil Type:   Gaia {g_acc1:.1%} vs RF {r_acc1:.1%}")
print(f"pH Predict:  Gaia R2={g_r2:.3f} vs RF R2={r_r2:.3f}")
print(f"Fumigation:  Gaia {g_acc3:.1%} vs RF {r_acc3:.1%}")
