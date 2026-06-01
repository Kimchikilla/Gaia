"""BonaRes: fertilization, soil chemistry, yield benchmarks"""

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

# Build genus abundance table (same as before)
print("Building genus abundance table...")
bac = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
genus_ref = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_GENUS.csv")
plot_df = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_PLOT.csv")
treatment = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_TREATMENT.csv")
f1_level = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_FACTOR_1_LEVEL.csv")
f2_level = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_FACTOR_2_LEVEL.csv")

genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)

grouped = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
pivot = grouped.pivot_table(index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0)
pivot = pivot.reset_index()
genus_cols = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]
print(f"Abundance: {pivot.shape[0]} samples x {len(genus_cols)} genera")

# Add treatment labels
plot_treat = plot_df[["Plot_ID", "Treatment_ID"]].merge(treatment, on="Treatment_ID")
plot_treat = plot_treat.merge(f1_level[["Factor_1_Level_ID", "Name_EN"]].rename(columns={"Name_EN": "tillage"}), on="Factor_1_Level_ID")
plot_treat = plot_treat.merge(f2_level[["Factor_2_Level_ID", "Name_EN"]].rename(columns={"Name_EN": "fertilization"}), on="Factor_2_Level_ID")
data = pivot.merge(plot_treat[["Plot_ID", "tillage", "fertilization"]], on="Plot_ID", how="left")
data = data.dropna(subset=["tillage"]).reset_index(drop=True)

# Load model
model = GPT2LMHeadModel.from_pretrained("checkpoints/mgm_soil_3k/best")
with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
    tokenizer = CustomUnpickler(f).load()
print("Model loaded")


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


class Clf(nn.Module):
    def __init__(self, gpt, n):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n))

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        return self.head((h * mask).sum(1) / mask.sum(1).clamp(min=1))


class Reg(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        return self.head((h * mask).sum(1) / mask.sum(1).clamp(min=1)).squeeze(-1)


def run_classification(data, genus_cols, label_col, task_name, model, tokenizer):
    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"{'='*50}")
    ds = GenusDataset(data, genus_cols, tokenizer, data[label_col].tolist())
    print(f"Samples: {len(ds)}")
    clf = Clf(model, 2)
    tr_n = int(0.8 * len(ds))
    tr, te = random_split(ds, [tr_n, len(ds) - tr_n], generator=torch.Generator().manual_seed(42))
    opt = torch.optim.Adam(clf.head.parameters(), lr=1e-3)
    for ep in range(20):
        clf.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            loss = nn.CrossEntropyLoss()(clf(bx), torch.tensor(by))
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    preds, labels = [], []
    with torch.no_grad():
        for bx, by in DataLoader(te, batch_size=16):
            preds.extend(clf(bx).argmax(1).tolist()); labels.extend(by)
    g_acc = accuracy_score(labels, preds)
    g_f1 = f1_score(labels, preds, average="weighted")
    X, y = data[genus_cols].values, data[label_col].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    r_acc = accuracy_score(yte, rf.predict(Xte))
    print(f"Gaia: {g_acc:.1%}  RF: {r_acc:.1%}")
    return g_acc, r_acc


def run_regression(data, genus_cols, label_col, task_name, model, tokenizer):
    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"{'='*50}")
    y_mean, y_std = data[label_col].mean(), data[label_col].std()
    labels_norm = ((data[label_col] - y_mean) / y_std).tolist()
    ds = GenusDataset(data, genus_cols, tokenizer, labels_norm)
    print(f"Samples: {len(ds)}")
    reg = Reg(model)
    tr_n = int(0.8 * len(ds))
    tr, te = random_split(ds, [tr_n, len(ds) - tr_n], generator=torch.Generator().manual_seed(42))
    opt = torch.optim.Adam(reg.head.parameters(), lr=1e-3)
    for ep in range(30):
        reg.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            loss = nn.MSELoss()(reg(bx), torch.tensor(by, dtype=torch.float))
            opt.zero_grad(); loss.backward(); opt.step()
    reg.eval()
    preds, labels = [], []
    with torch.no_grad():
        for bx, by in DataLoader(te, batch_size=16):
            preds.extend(reg(bx).tolist()); labels.extend(by)
    pred_orig = [p * y_std + y_mean for p in preds]
    true_orig = [l * y_std + y_mean for l in labels]
    g_r2 = r2_score(true_orig, pred_orig)
    g_rmse = np.sqrt(mean_squared_error(true_orig, pred_orig))
    X, y = data[genus_cols].values, data[label_col].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    r_r2 = r2_score(yte, rf.predict(Xte))
    print(f"Gaia: R2={g_r2:.4f}  RF: R2={r_r2:.4f}")
    return g_r2, r_r2


# === Task 1: Fertilization Classification ===
data["fert_label"] = (data["fertilization"] == "intensive").astype(int)
f_g, f_r = run_classification(data, genus_cols, "fert_label", "Fertilization (extensive vs intensive)", model, tokenizer)

# === Task 2: Soil pH Prediction ===
soil = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_SOIL_LAB.csv")
soil_samp = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
soil = soil.merge(soil_samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]], on="Soil_Sampling_ID", how="left")
soil_ph = soil.dropna(subset=["pH"]).groupby(["Plot_ID", "Experimental_Year"])["pH"].mean().reset_index()
data_ph = pivot.merge(soil_ph, on=["Plot_ID", "Experimental_Year"], how="inner")
data_ph = data_ph.dropna(subset=["pH"]).reset_index(drop=True)

if len(data_ph) > 10:
    ph_g, ph_r = run_regression(data_ph, genus_cols, "pH", "pH Prediction (BonaRes real)", model, tokenizer)
else:
    print(f"\npH: not enough paired samples ({len(data_ph)})")
    ph_g, ph_r = None, None

# === Task 3: Yield Prediction ===
harvest = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_HARVEST.csv")
yld = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_YIELD.csv")
harvest_yld = harvest.merge(yld[["Harvest_ID", "Yield_Total"]], on="Harvest_ID", how="inner")
harvest_yld = harvest_yld.dropna(subset=["Yield_Total"])
harvest_yld = harvest_yld.groupby(["Plot_ID", "Experimental_Year"])["Yield_Total"].mean().reset_index()
data_yld = pivot.merge(harvest_yld, on=["Plot_ID", "Experimental_Year"], how="inner")
data_yld = data_yld.dropna(subset=["Yield_Total"]).reset_index(drop=True)

if len(data_yld) > 10:
    y_g, y_r = run_regression(data_yld, genus_cols, "Yield_Total", "Yield Prediction (BonaRes)", model, tokenizer)
else:
    print(f"\nYield: not enough paired samples ({len(data_yld)})")
    y_g, y_r = None, None

# Summary
print("\n" + "=" * 50)
print("BONARES BENCHMARKS SUMMARY")
print("=" * 50)
print(f"Tillage (prev):    Gaia 81.0% vs RF 100.0%")
print(f"Fertilization:     Gaia {f_g:.1%} vs RF {f_r:.1%}")
if ph_g is not None:
    print(f"pH Prediction:     Gaia R2={ph_g:.3f} vs RF R2={ph_r:.3f}")
if y_g is not None:
    print(f"Yield Prediction:  Gaia R2={y_g:.3f} vs RF R2={y_r:.3f}")
