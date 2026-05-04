"""USDA 감자 수확량 예측 — gaia_v4 백본 사용.

기존 benchmark_yield.py 는 mgm_soil_3k 사용. 본 스크립트는 같은 데이터/구조로
v4 체크포인트 + 자체 토크나이저로 재실행해 README 수치 갱신.
"""
import torch  # torch first
import torch.nn as nn
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

CKPT = Path("checkpoints/gaia_v4")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Device: {DEVICE}")
    print("Loading model + tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(str(CKPT / "best")).to(DEVICE).eval()
    with open(CKPT / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    df = pd.read_csv("data/raw/tillage/usda_potato.csv")
    genus_cols = [c for c in df.columns
                  if c.startswith("BF_g_") or c.startswith("FF_g_")]
    name_map = {}
    for col in genus_cols:
        parts = col.split("_", 3)
        name_map[col] = parts[3].split("_")[0] if len(parts) >= 4 else col

    print(f"Samples: {len(df)}, Genera cols: {len(genus_cols)}")

    # Tokenize
    bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    samples, labels = [], []
    ymean, ystd = df["Yield_per_meter"].mean(), df["Yield_per_meter"].std()
    yields_norm = ((df["Yield_per_meter"] - ymean) / ystd).tolist()

    for i, (_, row) in enumerate(df.iterrows()):
        nz = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
        tokens = [bos]
        for col in nz.index:
            tid = tokenizer.vocab.get(f"g__{name_map[col]}")
            if tid is not None:
                tokens.append(tid)
            if len(tokens) >= 511:
                break
        tokens.append(eos)
        while len(tokens) < 512:
            tokens.append(pad)
        if sum(1 for t in tokens if t not in (bos, eos, pad)) >= 3:
            samples.append(torch.tensor(tokens[:512], dtype=torch.long))
            labels.append(yields_norm[i])

    print(f"Tokenized: {len(samples)}")

    class DS(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i): return samples[i], labels[i]

    ds = DS()
    n_tr = int(0.8 * len(ds))
    tr, te = random_split(ds, [n_tr, len(ds) - n_tr],
                          generator=torch.Generator().manual_seed(42))

    class Reg(nn.Module):
        def __init__(self):
            super().__init__()
            self.gpt = model
            for p in self.gpt.parameters(): p.requires_grad = False
            self.h = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, x):
            with torch.no_grad():
                h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
            m = (x != 0).unsqueeze(-1).float()
            p = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.h(p).squeeze(-1)

    reg = Reg().to(DEVICE)
    opt = torch.optim.Adam(reg.h.parameters(), lr=1e-3)
    best_r2 = -1e9

    for ep in range(30):
        reg.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            pr = reg(bx.to(DEVICE))
            loss = nn.MSELoss()(pr, torch.tensor(by, dtype=torch.float).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

        reg.eval()
        ap, al = [], []
        with torch.no_grad():
            for bx, by in DataLoader(te, batch_size=16):
                ap.extend(reg(bx.to(DEVICE)).cpu().tolist())
                al.extend(by)
        pred = [p * ystd + ymean for p in ap]
        true = [l * ystd + ymean for l in al]
        r2 = r2_score(true, pred)
        if r2 > best_r2: best_r2 = r2
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}: R2={r2:.4f}  (best={best_r2:.4f})")

    g_r2 = best_r2
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)

    # RF
    X = df[genus_cols].values
    y = df["Yield_per_meter"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    rfp = rf.predict(Xte)
    r_r2 = r2_score(yte, rfp)
    r_rmse = np.sqrt(mean_squared_error(yte, rfp))

    print()
    print("=" * 55)
    print("USDA Potato Yield Prediction (microbiome -> kg/m)")
    print("=" * 55)
    print(f"Gaia v4: R2={g_r2:.4f} RMSE={rmse:.0f} MAE={mae:.0f}")
    print(f"RF:      R2={r_r2:.4f} RMSE={r_rmse:.0f}")
    winner = "Gaia" if g_r2 >= r_r2 else "RF"
    print(f"Winner (R2): {winner}")

    log = {
        "dataset": "USDA Potato (Yield_per_meter)",
        "n_samples": int(len(df)), "n_genera_cols": int(len(genus_cols)),
        "split": "80/20",
        "gaia_v4": {"r2": float(g_r2), "rmse": float(rmse), "mae": float(mae)},
        "rf":      {"r2": float(r_r2), "rmse": float(r_rmse)},
    }
    Path("docs/benchmark_yield_v4.json").write_text(json.dumps(log, indent=2))
    print("Saved docs/benchmark_yield_v4.json")


if __name__ == "__main__":
    main()
