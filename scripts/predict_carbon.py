"""Carbon prediction: microbiome → soil organic carbon (current + future)"""

import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE = "data/raw/longterm/bonares_data"

# 1. Build data
print("=== Building microbiome + carbon pairs ===")
bac = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
genus_ref = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_GENUS.csv")
genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)
grouped = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
pivot = grouped.pivot_table(index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0).reset_index()
genus_cols = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]

# Carbon data
soil = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_SOIL_LAB.csv")
soil_samp = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
soil = soil.merge(soil_samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]], on="Soil_Sampling_ID", how="left")

# Current carbon
carbon = soil.dropna(subset=["Total_Carbon"]).groupby(["Plot_ID", "Experimental_Year"])["Total_Carbon"].mean().reset_index()
carbon_lookup = dict(zip(zip(carbon["Plot_ID"], carbon["Experimental_Year"]), carbon["Total_Carbon"]))

# Current pairs
data_current = pivot.merge(carbon, on=["Plot_ID", "Experimental_Year"], how="inner").dropna(subset=["Total_Carbon"]).reset_index(drop=True)
print(f"Current carbon pairs: {len(data_current)}")
print(f"Carbon range: {data_current['Total_Carbon'].min():.2f} ~ {data_current['Total_Carbon'].max():.2f}")

# Future pairs (this year microbiome → next year carbon)
future_pairs = []
for _, row in pivot.iterrows():
    plot, year = row["Plot_ID"], row["Experimental_Year"]
    future_c = carbon_lookup.get((plot, year + 1))
    if future_c is not None:
        r = row.to_dict()
        r["Future_Carbon"] = future_c
        future_pairs.append(r)
data_future = pd.DataFrame(future_pairs)
print(f"Future carbon pairs: {len(data_future)}")

# 2. Load model
model = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_v4/best")
model.eval()
model.cuda()
with open("checkpoints/gaia_v4/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print(f"Model: vocab={len(tokenizer.vocab)}")


class CarbonDataset(Dataset):
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


class CarbonRegressor(nn.Module):
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


def run_prediction(data, genus_cols, label_col, task_name, tokenizer, model):
    print(f"\n{'='*50}")
    print(f"{task_name}")
    print(f"{'='*50}")

    ym, ys = data[label_col].mean(), data[label_col].std()
    labels_norm = ((data[label_col] - ym) / ys).tolist()
    ds = CarbonDataset(data, genus_cols, tokenizer, labels_norm)
    print(f"Samples: {len(ds)}")

    tr_n = int(0.8 * len(ds))
    tr, te = random_split(ds, [tr_n, len(ds) - tr_n], generator=torch.Generator().manual_seed(42))

    reg = CarbonRegressor(model).cuda()
    opt = torch.optim.Adam(reg.head.parameters(), lr=1e-3)

    for ep in range(30):
        reg.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            loss = nn.MSELoss()(reg(bx.cuda()), torch.tensor(by, dtype=torch.float).cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (ep + 1) % 10 == 0:
            reg.eval()
            p, l = [], []
            with torch.no_grad():
                for bx, by in DataLoader(te, batch_size=16):
                    p.extend(reg(bx.cuda()).cpu().tolist())
                    l.extend(by)
            po = [v * ys + ym for v in p]
            lo = [v * ys + ym for v in l]
            r2 = r2_score(lo, po)
            print(f"  Epoch {ep+1}: R2={r2:.4f}")

    # Final
    reg.eval()
    p, l = [], []
    with torch.no_grad():
        for bx, by in DataLoader(te, batch_size=16):
            p.extend(reg(bx.cuda()).cpu().tolist())
            l.extend(by)
    po = [v * ys + ym for v in p]
    lo = [v * ys + ym for v in l]
    g_r2 = r2_score(lo, po)
    g_rmse = np.sqrt(mean_squared_error(lo, po))

    # RF
    X = data[genus_cols].values
    y = data[label_col].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    rf_r2 = r2_score(yte, rf.predict(Xte))

    print(f"\nGaia: R2={g_r2:.4f}, RMSE={g_rmse:.4f}")
    print(f"RF:   R2={rf_r2:.4f}")

    print(f"\nSample predictions:")
    for i in range(min(8, len(lo))):
        print(f"  Actual: {lo[i]:.3f}%  Predicted: {po[i]:.3f}%  Error: {abs(lo[i]-po[i]):.3f}%")

    return g_r2, rf_r2


# 3. Run predictions
r1_g, r1_r = run_prediction(data_current, genus_cols, "Total_Carbon",
                             "Current Soil Organic Carbon Prediction", tokenizer, model)

if len(data_future) > 20:
    r2_g, r2_r = run_prediction(data_future, genus_cols, "Future_Carbon",
                                 "Future Soil Organic Carbon Prediction (next year)", tokenizer, model)
else:
    print(f"\nFuture carbon: not enough pairs ({len(data_future)})")
    r2_g, r2_r = None, None

# Summary
print(f"\n{'='*50}")
print("CARBON PREDICTION SUMMARY")
print(f"{'='*50}")
print(f"Current Carbon: Gaia R2={r1_g:.3f} vs RF R2={r1_r:.3f}")
if r2_g is not None:
    print(f"Future Carbon:  Gaia R2={r2_g:.3f} vs RF R2={r2_r:.3f}")
