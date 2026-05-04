"""Naylor (미국 캘리포니아, Sorghum 가뭄 실험) 가뭄 스트레스 분류 OOD 벤치마크.

Westerfeld/Bernburg 는 둘 다 독일 농경지였음. Naylor 는 다른 대륙 + 다른 작물(수수)
+ 다른 매트릭스(rhizosphere) 라 Gaia 표현의 진짜 cross-context 일반화를 시험.

Linear probe: 백본 동결, head만 학습. RF 와 비교.
"""
import torch  # torch first
import torch.nn as nn
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

CKPT = Path("checkpoints/gaia_v4")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "data/raw/naylor/naylor_genus_with_labels.csv"


class NaylorDS(Dataset):
    def __init__(self, df, genus_cols, tokenizer, labels):
        self.s, self.y = [], []
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nz = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
            tokens = [bos]
            for g in nz.index:
                for cand in (f"g__{g}", g):
                    if cand in tokenizer.vocab:
                        tokens.append(tokenizer.vocab[cand]); break
                if len(tokens) >= 511:
                    break
            tokens.append(eos)
            while len(tokens) < 512:
                tokens.append(pad)
            if sum(1 for t in tokens if t not in (bos, eos, pad)) >= 3:
                self.s.append(torch.tensor(tokens[:512], dtype=torch.long))
                self.y.append(labels[i])

    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i], self.y[i]


class Cls(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters(): p.requires_grad = False
        self.h = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        m = (x != 0).unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.h(p)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA)
    label_col = "treatment"
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = (df[label_col].str.lower() == "drought").astype(int).values
    print(f"Samples: {len(df)}  drought={y.sum()}  control={(1-y).sum()}")

    drop_cols = ["sample_id", "run_id", "treatment", "host"]
    genus_cols = [c for c in df.columns if c not in drop_cols]
    print(f"Genus cols: {len(genus_cols)}")

    print("Loading model...")
    gpt = GPT2LMHeadModel.from_pretrained(str(CKPT / "best")).to(DEVICE).eval()
    with open(CKPT / "tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    ds = NaylorDS(df, genus_cols, tok, y.tolist())
    print(f"Tokenized: {len(ds)}")
    n_tr = int(0.8 * len(ds))
    tr, te = random_split(ds, [n_tr, len(ds) - n_tr],
                          generator=torch.Generator().manual_seed(42))

    cls = Cls(gpt).to(DEVICE)
    opt = torch.optim.Adam(cls.h.parameters(), lr=1e-3)

    best_acc = 0.0
    for ep in range(30):
        cls.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            logits = cls(bx.to(DEVICE))
            loss = nn.CrossEntropyLoss()(logits, torch.tensor(by, dtype=torch.long).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

        cls.eval()
        with torch.no_grad():
            preds, probs, true = [], [], []
            for bx, by in DataLoader(te, batch_size=16):
                lo = cls(bx.to(DEVICE))
                pr = torch.softmax(lo, dim=-1)[:, 1].cpu().tolist()
                p = lo.argmax(-1).cpu().tolist()
                preds.extend(p); probs.extend(pr); true.extend(by)
            acc = accuracy_score(true, preds)
            best_acc = max(best_acc, acc)

        if (ep + 1) % 5 == 0:
            f1 = f1_score(true, preds)
            try:
                auc = roc_auc_score(true, probs)
            except ValueError:
                auc = float("nan")
            print(f"  ep {ep+1}: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}  best_acc={best_acc:.3f}")

    # final
    cls.eval()
    with torch.no_grad():
        preds, probs, true = [], [], []
        for bx, by in DataLoader(te, batch_size=16):
            lo = cls(bx.to(DEVICE))
            pr = torch.softmax(lo, dim=-1)[:, 1].cpu().tolist()
            preds.extend(lo.argmax(-1).cpu().tolist())
            probs.extend(pr); true.extend(by)
    g_acc = accuracy_score(true, preds)
    g_f1 = f1_score(true, preds)
    g_auc = roc_auc_score(true, probs)

    # RF baseline
    X = df[genus_cols].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    rf_prob = rf.predict_proba(Xte)[:, 1]
    r_acc = accuracy_score(yte, rf_pred); r_f1 = f1_score(yte, rf_pred)
    r_auc = roc_auc_score(yte, rf_prob)

    print()
    print(f"{'='*55}")
    print(f"Naylor drought classification - OOD (USA Sorghum)")
    print(f"{'='*55}")
    print(f"Gaia: acc={g_acc:.3f}  f1={g_f1:.3f}  auc={g_auc:.3f}")
    print(f"RF:   acc={r_acc:.3f}  f1={r_f1:.3f}  auc={r_auc:.3f}")
    winner = "Gaia" if g_acc >= r_acc else "RF"
    print(f"Winner (acc): {winner}")

    log = {
        "dataset": "Naylor (USA Sorghum drought)",
        "n_samples": len(df), "n_genera": len(genus_cols), "split": "80/20",
        "gaia": {"acc": g_acc, "f1": g_f1, "auc": g_auc},
        "rf":   {"acc": r_acc, "f1": r_f1, "auc": r_auc},
    }
    Path("docs/benchmark_naylor.json").write_text(json.dumps(log, indent=2))
    print("Saved docs/benchmark_naylor.json")


if __name__ == "__main__":
    main()
