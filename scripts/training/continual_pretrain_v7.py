"""v6 가중치 → v7 코퍼스(CLR+batch 보정) 로 continual pretrain.

CLR 후엔 토큰 순서가 v6 학습 데이터랑 다름 — 즉 모델 입장에서 살짝 OOD.
fine-tune 으로 적응시킨다.
"""
import torch
import pickle
from pathlib import Path
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, random_split

CORPUS = Path("data/processed_real/gaia-corpus-v7-clr.pkl")
SRC = Path("checkpoints/gaia_v6/best")
DST = Path("checkpoints/gaia_v7")
DST.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 1500
BATCH = 16
LR = 5e-5    # CLR 가 분포를 바꿔서 lr 조금 낮춤
WARMUP = 100


class CorpusDS(Dataset):
    def __init__(self, seqs):
        self.x = torch.tensor(seqs, dtype=torch.long)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i]


def main():
    print(f"Device: {DEVICE}")
    with open(CORPUS, "rb") as f:
        data = pickle.load(f)
    seqs = data["sequences"]
    print(f"corpus v7: {seqs.shape}, vocab={data['vocab_size']}")
    print("preprocessing:", data.get("preprocessing"))
    print("source breakdown:", data["source_breakdown"])

    ds = CorpusDS(seqs)
    n_val = max(50, int(0.05 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=BATCH, shuffle=False)
    print(f"train: {len(tr)}  val: {len(va)}")

    model = GPT2LMHeadModel.from_pretrained(str(SRC)).to(DEVICE)
    pad = data["pad"]

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, (s + 1) / WARMUP),
    )

    @torch.no_grad()
    def eval_loss():
        model.eval()
        tot, n = 0.0, 0
        for x in va_dl:
            x = x.to(DEVICE)
            mask = (x != pad).long()
            out = model(input_ids=x, attention_mask=mask, labels=x)
            tot += float(out.loss) * x.size(0)
            n += x.size(0)
        return tot / max(n, 1)

    print(f"Initial val loss (v6 weights on v7 corpus): {eval_loss():.4f}")

    step, best = 0, float("inf")
    while step < MAX_STEPS:
        model.train()
        for x in tr_dl:
            x = x.to(DEVICE)
            mask = (x != pad).long()
            out = model(input_ids=x, attention_mask=mask, labels=x)
            loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            if step % 100 == 0:
                vl = eval_loss()
                print(f"step {step}: train_loss={float(loss):.4f}  val_loss={vl:.4f}")
                if vl < best:
                    best = vl
                    model.save_pretrained(DST / "best")
                    print(f"  saved best (val={best:.4f})")
            if step >= MAX_STEPS:
                break

    print(f"Final best val loss: {best:.4f}")
    print(f"Checkpoint: {DST / 'best'}")
    import shutil
    shutil.copy("checkpoints/gaia_v6/tokenizer.pkl", DST / "tokenizer.pkl")


if __name__ == "__main__":
    main()
