"""v10 = BERT (encoder) + KEGG vocab + MLM, 처음부터 학습.

샘플 수가 적으면 sanity check 모드 (50 step, overfit 확인).
샘플 수 많아지면 (--steps 1500+) 진짜 학습.
"""
import argparse
import torch
import torch.nn as nn
import pickle
import numpy as np
from pathlib import Path
from transformers import BertConfig, BertForMaskedLM
from torch.utils.data import Dataset, DataLoader

CORPUS = Path("data/processed_real/gaia-corpus-v10-kegg.pkl")
DST = Path("checkpoints/gaia_v10")
DST.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASK_PROB = 0.15


class CorpusDS(Dataset):
    def __init__(self, seqs):
        self.x = torch.tensor(seqs, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]


def mask_tokens(input_ids, mask_id, vocab_size, pad_id, special_ids):
    device = input_ids.device
    labels = input_ids.clone()
    masked = torch.bernoulli(torch.full(input_ids.shape, MASK_PROB, device=device)).bool()
    for sid in special_ids + [pad_id]:
        masked &= input_ids != sid
    labels[~masked] = -100
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & masked
    input_ids[indices_replaced] = mask_id
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & masked & ~indices_replaced
    rand_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = rand_words[indices_random]
    return input_ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50,
                    help="for 1-sample sanity 50, for real training 1500+")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4,
                    help="if only 1 sample available, batch=1")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    with open(CORPUS, "rb") as f:
        data = pickle.load(f)
    seqs = data["sequences"]
    print(f"corpus: {seqs.shape}, vocab={data['vocab_size']}")
    print(f"preprocessing: {data.get('preprocessing')}")

    # adjust batch if only 1 sample
    batch_size = min(args.batch, max(seqs.shape[0], 1))
    if seqs.shape[0] < 2:
        print("WARNING: <2 samples - sanity mode (model will overfit)")

    pad_id = data["pad"]
    bos_id = data["bos"]
    eos_id = data["eos"]
    with open(DST / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    mask_id = tokenizer["mask_token_id"]
    special_ids = [bos_id, eos_id, mask_id]

    config = BertConfig(
        vocab_size=data["vocab_size"], hidden_size=256, num_hidden_layers=8,
        num_attention_heads=8, intermediate_size=1024,
        max_position_embeddings=512, pad_token_id=pad_id, type_vocab_size=1,
    )
    model = BertForMaskedLM(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BERT params: {n_params:,}")

    ds = CorpusDS(seqs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"\n=== training {args.steps} steps ===")
    step = 0
    while step < args.steps:
        model.train()
        for x in dl:
            x = x.to(DEVICE)
            inp, lbl = mask_tokens(x.clone(), mask_id, data["vocab_size"], pad_id, special_ids)
            attn = (inp != pad_id).long()
            out = model(input_ids=inp, attention_mask=attn, labels=lbl)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"  step {step:4d}: loss={float(loss):.4f}")
            if step >= args.steps:
                break

    print(f"\nfinal loss: {float(loss):.4f}")
    model.save_pretrained(DST / "best")
    print(f"saved {DST/'best'}")
    print()
    print("NOTE: with 1 sample this only verifies pipeline works.")
    print("      Real training requires 30+ samples.")



if __name__ == "__main__":
    main()
