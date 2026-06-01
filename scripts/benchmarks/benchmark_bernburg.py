"""
Benchmark Bernburg long-term trial as out-of-distribution test for Gaia v4.
Tasks:
  1. Tillage classification (Cultivator vs Plough)
  2. Fertilization classification (extensive vs intensive)
  3. pH regression
  4. Total carbon (C[%]) regression
  5. Total nitrogen (N[%]) regression
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Load Bernburg processed data
print("=== Loading Bernburg ===")
abundance = pd.read_csv("data/processed_real/bernburg_abundance.csv", index_col=0)
metadata = pd.read_csv("data/processed_real/bernburg_metadata.csv")

# Join: drop samples missing metadata
data = abundance.merge(metadata, left_index=True, right_on="Sample", how="inner")
print(f"Joined samples: {len(data)}")

genus_cols = list(abundance.columns)
print(f"Genera: {len(genus_cols)}")

# 2. Load Gaia v4
print("\n=== Loading Gaia v4 ===")
model = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_v4/best")
model.eval()
model.cuda()
with open("checkpoints/gaia_v4/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print(f"Vocab: {len(tokenizer.vocab)}")

matched = sum(1 for g in genus_cols if f"g__{g}" in tokenizer.vocab)
print(f"Matched genera: {matched}/{len(genus_cols)} ({matched/len(genus_cols)*100:.0f}%)")


class GenusDataset(Dataset):
    def __init__(self, df, genus_cols, tokenizer, labels):
        self.samples, self.labels = [], []
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
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


class Head(nn.Module):
    def __init__(self, gpt, n_out=1):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def run_classification(df, label_col, task_name):
    print(f"\n--- {task_name} ---")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    classes = sorted(df[label_col].unique())
    cls_map = {c: i for i, c in enumerate(classes)}
    print(f"  Classes: {classes}, n={len(df)}")

    labels = [cls_map[v] for v in df[label_col]]
    ds = GenusDataset(df, genus_cols, tokenizer, labels)
    if len(ds) < 20:
        print("  Too few samples")
        return
    tr_n = int(0.8 * len(ds))
    tr, te = random_split(ds, [tr_n, len(ds) - tr_n], generator=torch.Generator().manual_seed(42))

    reg = Head(model, n_out=len(classes)).cuda()
    opt = torch.optim.Adam(reg.head.parameters(), lr=1e-3)
    for ep in range(30):
        reg.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            logits = reg(bx.cuda())
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            loss = nn.CrossEntropyLoss()(logits, torch.tensor(by, dtype=torch.long).cuda())
            opt.zero_grad(); loss.backward(); opt.step()

    reg.eval()
    p, l = [], []
    with torch.no_grad():
        for bx, by in DataLoader(te, batch_size=16):
            out = reg(bx.cuda())
            if out.dim() == 1:
                out = out.unsqueeze(0)
            p.extend(out.argmax(-1).cpu().tolist())
            l.extend(by)
    g_acc = accuracy_score(l, p)

    # RF baseline
    X = df[genus_cols].values
    y = np.array(labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    rf_acc = accuracy_score(yte, rf.predict(Xte))
    print(f"  Gaia: {g_acc*100:.1f}%  |  RF: {rf_acc*100:.1f}%")


def run_regression(df, label_col, task_name):
    print(f"\n--- {task_name} ---")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    print(f"  n={len(df)}, range={df[label_col].min():.2f}~{df[label_col].max():.2f}")
    if len(df) < 20:
        print("  Too few samples")
        return

    ym, ys = df[label_col].mean(), df[label_col].std()
    if ys == 0:
        print("  Zero variance, skip")
        return
    labels_norm = ((df[label_col] - ym) / ys).tolist()
    ds = GenusDataset(df, genus_cols, tokenizer, labels_norm)

    tr_n = int(0.8 * len(ds))
    tr, te = random_split(ds, [tr_n, len(ds) - tr_n], generator=torch.Generator().manual_seed(42))

    reg = Head(model, n_out=1).cuda()
    opt = torch.optim.Adam(reg.head.parameters(), lr=1e-3)
    for ep in range(30):
        reg.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            loss = nn.MSELoss()(reg(bx.cuda()), torch.tensor(by, dtype=torch.float).cuda())
            opt.zero_grad(); loss.backward(); opt.step()

    reg.eval()
    p, l = [], []
    with torch.no_grad():
        for bx, by in DataLoader(te, batch_size=16):
            p.extend(reg(bx.cuda()).cpu().tolist())
            l.extend(by)
    po = [v * ys + ym for v in p]
    lo = [v * ys + ym for v in l]
    g_r2 = r2_score(lo, po)

    # RF
    X = df[genus_cols].values
    y = df[label_col].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    rf_r2 = r2_score(yte, rf.predict(Xte))
    print(f"  Gaia R2={g_r2:.3f}  |  RF R2={rf_r2:.3f}")


# 3. Run benchmarks
run_classification(data, "Tillage_norm", "Tillage classification")
run_classification(data, "Fertilization_norm", "Fertilization classification")
run_regression(data, "pH", "pH prediction")
run_regression(data, "C[%]", "Total carbon (C[%]) prediction")
run_regression(data, "N[%]", "Total nitrogen (N[%]) prediction")
run_regression(data, "OM[%]", "Organic matter (OM[%]) prediction")

print("\n=== Bernburg benchmark complete ===")
