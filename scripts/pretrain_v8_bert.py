"""v8 = BERT (encoder-only) + MLM + v7-CLR 코퍼스, 처음부터 학습.

GPT-2 causal mask 버림 — 모든 토큰이 양방향으로 봄.
미생물 set 의 본질에 더 맞음.

학습 목표: 마스킹된 속(genus) 을 양방향 컨텍스트로 맞히기 (MLM).
"""
import torch  # torch first
import torch.nn as nn
import pickle
import numpy as np
from pathlib import Path
from transformers import BertConfig, BertForMaskedLM
from torch.utils.data import Dataset, DataLoader, random_split

CORPUS = Path("data/processed_real/gaia-corpus-v7-clr.pkl")
DST = Path("checkpoints/gaia_v8")
DST.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 1500
BATCH = 16
LR = 1e-4
WARMUP = 200
MASK_PROB = 0.15


class CorpusDS(Dataset):
    def __init__(self, seqs):
        self.x = torch.tensor(seqs, dtype=torch.long)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i]


def mask_tokens(input_ids, mask_token_id, vocab_size, pad_id, special_ids,
                mask_prob=MASK_PROB):
    """BERT MLM masking — 15% of non-special tokens."""
    device = input_ids.device
    labels = input_ids.clone()
    masked_indices = torch.bernoulli(torch.full(input_ids.shape, mask_prob, device=device)).bool()
    for sid in special_ids + [pad_id]:
        masked_indices &= input_ids != sid

    labels[~masked_indices] = -100  # only compute loss on masked

    # 80% [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    # 10% random
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    # 10% unchanged
    return input_ids, labels


def main():
    print(f"Device: {DEVICE}")
    with open(CORPUS, "rb") as f:
        data = pickle.load(f)
    seqs = data["sequences"]
    print(f"corpus v7 (CLR): {seqs.shape}, vocab={data['vocab_size']}")

    # tokenizer info from v6
    with open("checkpoints/gaia_v6/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    vocab = tok.vocab
    print(f"vocab keys sample: {list(vocab.items())[:6]}")
    mask_id = vocab.get("<mask>", 1)
    pad_id = data["pad"]
    bos_id = data["bos"]
    eos_id = data["eos"]
    special_ids = [bos_id, eos_id, mask_id]
    print(f"BOS={bos_id}, EOS={eos_id}, PAD={pad_id}, MASK={mask_id}")

    # BERT 작은 모델 — GPT-2 v6 와 같은 사이즈 (8L, 256d, 8H)
    config = BertConfig(
        vocab_size=data["vocab_size"],
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        pad_token_id=pad_id,
        type_vocab_size=1,
    )
    model = BertForMaskedLM(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BERT params: {n_params:,}")

    ds = CorpusDS(seqs)
    n_val = max(50, int(0.05 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=BATCH, shuffle=False)
    print(f"train: {len(tr)}  val: {len(va)}")

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
            inp, lbl = mask_tokens(x.clone(), mask_id, data["vocab_size"],
                                    pad_id, special_ids)
            attn = (inp != pad_id).long()
            out = model(input_ids=inp, attention_mask=attn, labels=lbl)
            tot += float(out.loss) * x.size(0); n += x.size(0)
        return tot / max(n, 1)

    print(f"Initial val MLM loss (random init): {eval_loss():.4f}")

    step, best = 0, float("inf")
    while step < MAX_STEPS:
        model.train()
        for x in tr_dl:
            x = x.to(DEVICE)
            inp, lbl = mask_tokens(x.clone(), mask_id, data["vocab_size"],
                                    pad_id, special_ids)
            attn = (inp != pad_id).long()
            out = model(input_ids=inp, attention_mask=attn, labels=lbl)
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
            if step >= MAX_STEPS: break

    print(f"Final best val MLM loss: {best:.4f}")
    print(f"Checkpoint: {DST / 'best'}")
    import shutil
    shutil.copy("checkpoints/gaia_v6/tokenizer.pkl", DST / "tokenizer.pkl")


if __name__ == "__main__":
    main()
