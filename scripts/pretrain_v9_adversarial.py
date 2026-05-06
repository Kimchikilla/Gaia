"""v9 = v8 BERT + adversarial source 분류 discriminator.

학습 목표 두 개 동시:
  1. MLM (마스킹된 속 맞히기) — 진짜 미생물 신호
  2. Adversarial: source(v1/emp/neon) 분류 head 가 임베딩에서 source 를 못 맞히게 함

Gradient Reversal Layer (GRL):
  forward: identity
  backward: gradient * -lambda
이걸 source 분류 head 앞에 끼우면 — encoder 는 source 정보 임베딩에서 빼려 함.
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import BertConfig, BertForMaskedLM
from torch.utils.data import Dataset, DataLoader, random_split

CORPUS = Path("data/processed_real/gaia-corpus-v7-clr.pkl")
META = Path("data/processed_real/gaia-metadata-v7.csv")
DST = Path("checkpoints/gaia_v9")
DST.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 1500
BATCH = 16
LR = 1e-4
WARMUP = 200
MASK_PROB = 0.15
ADV_LAMBDA = 0.5  # adversarial loss 가중치


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


class CorpusWithSourceDS(Dataset):
    def __init__(self, seqs, sources):
        self.x = torch.tensor(seqs, dtype=torch.long)
        self.s = torch.tensor(sources, dtype=torch.long)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.s[i]


def mask_tokens(input_ids, mask_token_id, vocab_size, pad_id, special_ids,
                mask_prob=MASK_PROB):
    device = input_ids.device
    labels = input_ids.clone()
    masked_indices = torch.bernoulli(torch.full(input_ids.shape, mask_prob, device=device)).bool()
    for sid in special_ids + [pad_id]:
        masked_indices &= input_ids != sid
    labels[~masked_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    return input_ids, labels


class V9Model(nn.Module):
    def __init__(self, bert_config, n_sources):
        super().__init__()
        self.bert = BertForMaskedLM(bert_config)
        h = bert_config.hidden_size
        # source discriminator head — sees gradient-reversed mean embedding
        self.source_head = nn.Sequential(
            nn.Linear(h, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_sources))

    def forward(self, input_ids, attention_mask, mlm_labels, source_labels, lambda_adv):
        bert_out = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = bert_out.last_hidden_state
        # MLM logits
        mlm_logits = self.bert.cls(seq)
        mlm_loss = None
        if mlm_labels is not None:
            mlm_loss = nn.CrossEntropyLoss()(mlm_logits.view(-1, mlm_logits.size(-1)),
                                              mlm_labels.view(-1))
        # mean pooled embedding (mask out PAD)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (seq * m).sum(1) / m.sum(1).clamp(min=1)
        # adversarial source head with gradient reversal
        rev = grad_reverse(pooled, lambda_adv)
        src_logits = self.source_head(rev)
        src_loss = nn.CrossEntropyLoss()(src_logits, source_labels)
        return mlm_loss, src_loss, src_logits


def main():
    print(f"Device: {DEVICE}")
    with open(CORPUS, "rb") as f:
        data = pickle.load(f)
    seqs = data["sequences"]

    md = pd.read_csv(META)
    src_map = {s: i for i, s in enumerate(sorted(md["source"].unique()))}
    sources = md["source"].map(src_map).values
    print(f"corpus: {seqs.shape}, sources: {src_map}")
    assert len(sources) == len(seqs), f"size mismatch: meta={len(sources)} corpus={len(seqs)}"

    with open("checkpoints/gaia_v6/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    mask_id = tok.vocab.get("<mask>", 1)
    pad_id, bos_id, eos_id = data["pad"], data["bos"], data["eos"]
    special_ids = [bos_id, eos_id, mask_id]

    config = BertConfig(
        vocab_size=data["vocab_size"], hidden_size=256, num_hidden_layers=8,
        num_attention_heads=8, intermediate_size=1024,
        max_position_embeddings=512, pad_token_id=pad_id, type_vocab_size=1,
    )
    model = V9Model(config, n_sources=len(src_map)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"v9 params: {n_params:,}")

    ds = CorpusWithSourceDS(seqs, sources)
    n_val = max(50, int(0.05 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=BATCH, shuffle=False)
    print(f"train: {len(tr)}  val: {len(va)}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / WARMUP))

    @torch.no_grad()
    def eval_step():
        model.eval()
        tot_mlm, tot_src, n, src_correct = 0.0, 0.0, 0, 0
        for x, s in va_dl:
            x = x.to(DEVICE); s = s.to(DEVICE)
            inp, lbl = mask_tokens(x.clone(), mask_id, data["vocab_size"], pad_id, special_ids)
            attn = (inp != pad_id).long()
            mlm_l, src_l, src_logits = model(inp, attn, lbl, s, lambda_adv=ADV_LAMBDA)
            tot_mlm += float(mlm_l) * x.size(0)
            tot_src += float(src_l) * x.size(0)
            src_correct += (src_logits.argmax(-1) == s).float().sum().item()
            n += x.size(0)
        return tot_mlm/n, tot_src/n, src_correct/n

    mlm_l, src_l, src_acc = eval_step()
    print(f"Init: MLM={mlm_l:.3f}  src_loss={src_l:.3f}  src_acc={src_acc:.3f}")
    print(f"  (random src_acc with {len(src_map)} classes ~= {1/len(src_map):.3f}; majority = {pd.Series(sources).value_counts().iloc[0]/len(sources):.3f})")

    step, best = 0, float("inf")
    while step < MAX_STEPS:
        model.train()
        for x, s in tr_dl:
            x = x.to(DEVICE); s = s.to(DEVICE)
            inp, lbl = mask_tokens(x.clone(), mask_id, data["vocab_size"], pad_id, special_ids)
            attn = (inp != pad_id).long()
            mlm_l, src_l, _ = model(inp, attn, lbl, s, lambda_adv=ADV_LAMBDA)
            # total loss: MLM helps; source head backward through GRL pushes encoder
            # to remove source info. Source head itself trains to be good (gets correct grad
            # because GRL flips sign only for input feature, not its own params).
            loss = mlm_l + src_l
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            if step % 100 == 0:
                mlm_v, src_v, src_a = eval_step()
                print(f"step {step}: tr_mlm={float(mlm_l):.3f} tr_src={float(src_l):.3f}  "
                      f"val_mlm={mlm_v:.3f} val_src_acc={src_a:.3f}")
                if mlm_v < best:
                    best = mlm_v
                    model.bert.save_pretrained(DST / "best")
                    torch.save(model.source_head.state_dict(), DST / "source_head.pt")
                    print(f"  saved best (val_mlm={best:.4f})")
            if step >= MAX_STEPS: break

    print(f"Final best val MLM: {best:.4f}")
    import shutil
    shutil.copy("checkpoints/gaia_v6/tokenizer.pkl", DST / "tokenizer.pkl")


if __name__ == "__main__":
    main()
