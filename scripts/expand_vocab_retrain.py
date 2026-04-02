"""MGM 어휘 확장 + 토양 미생물 임베딩 재학습"""

import pkg_resources
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import copy
from transformers import GPT2LMHeadModel, GPT2Config
from mgm.src.utils import CustomUnpickler
from mgm.CLI.CLI_utils import find_pkg_resource
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Trainer, TrainingArguments

# 1. Load original MGM
print("=== Step 1: Load MGM ===")
model_path = pkg_resources.resource_filename("mgm", "resources/general_model")
model = GPT2LMHeadModel.from_pretrained(model_path)
with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
    tokenizer = CustomUnpickler(f).load()

original_vocab_size = len(tokenizer.vocab)
print(f"Original vocab: {original_vocab_size}")

# 2. Collect all soil genera from our data
print("\n=== Step 2: Collect soil genera ===")
all_genera = set()

# MGnify
mgnify = pd.read_csv("data/raw/mgnify/mgnify_abundance.csv")
for c in mgnify.columns:
    if c not in ["sample_id", "analysis_id"]:
        all_genera.add(c)

# Naylor
naylor = pd.read_csv("data/raw/naylor/naylor_genus_with_labels.csv")
for c in naylor.columns:
    if c not in ["sample_id", "run_id", "treatment", "host"]:
        all_genera.add(c)

# BonaRes
genus_ref = pd.read_csv("data/raw/longterm/bonares_data/lte_westerfeld.V1_0_GENUS.csv")
for g in genus_ref["Name"].dropna().unique():
    all_genera.add(g)

# USDA
usda = pd.read_csv("data/raw/tillage/usda_potato.csv")
for c in usda.columns:
    if c.startswith("BF_g_") or c.startswith("FF_g_"):
        parts = c.split("_", 3)
        if len(parts) >= 4:
            all_genera.add(parts[3].split("_")[0])

# Find new genera
existing = set(k for k in tokenizer.vocab.keys())
new_genera = sorted([g for g in all_genera if f"g__{g}" not in existing])
print(f"Total soil genera: {len(all_genera)}")
print(f"New genera to add: {len(new_genera)}")

# 3. Expand tokenizer
print("\n=== Step 3: Expand tokenizer ===")
new_tokens = [f"g__{g}" for g in new_genera]
tokenizer.add_tokens(new_tokens)
new_vocab_size = len(tokenizer.vocab)
print(f"Expanded vocab: {original_vocab_size} → {new_vocab_size} (+{new_vocab_size - original_vocab_size})")

# Save expanded tokenizer
import os
os.makedirs("checkpoints/gaia_expanded", exist_ok=True)
with open("checkpoints/gaia_expanded/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# 4. Resize model embeddings
print("\n=== Step 4: Resize model ===")
model.resize_token_embeddings(new_vocab_size)
params = sum(p.numel() for p in model.parameters())
print(f"Model params: {params:,}")

# 5. Prepare training data (all soil data combined)
print("\n=== Step 5: Prepare training data ===")


class SoilCorpus(Dataset):
    def __init__(self, dataframes, tokenizer, max_len=512):
        """dataframes: list of (df, genus_cols) tuples"""
        self.samples = []
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        for df, genus_cols in dataframes:
            for _, row in df.iterrows():
                nonzero = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
                tokens = [bos]
                for genus in nonzero.index:
                    tid = tokenizer.vocab.get(f"g__{genus}")
                    if tid is not None:
                        tokens.append(tid)
                    if len(tokens) >= max_len - 1:
                        break
                tokens.append(eos)
                while len(tokens) < max_len:
                    tokens.append(pad)
                if sum(1 for t in tokens if t not in [bos, eos, pad]) >= 5:
                    self.samples.append(torch.tensor(tokens[:max_len], dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"input_ids": self.samples[idx], "labels": self.samples[idx].clone()}


# Prepare all datasets
mgnify_genus = [c for c in mgnify.columns if c not in ["sample_id", "analysis_id"]]
naylor_genus = [c for c in naylor.columns if c not in ["sample_id", "run_id", "treatment", "host"]]

# BonaRes: build genus abundance
bac = pd.read_csv("data/raw/longterm/bonares_data/lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)
grouped = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
bonares_pivot = grouped.pivot_table(
    index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0
).reset_index()
bonares_genus = [c for c in bonares_pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]

all_data = [
    (mgnify, mgnify_genus),
    (naylor, naylor_genus),
    (bonares_pivot, bonares_genus),
]

corpus = SoilCorpus(all_data, tokenizer)
print(f"Total corpus: {len(corpus)} samples")

# 6. Train
print("\n=== Step 6: Training ===")
train_size = int(0.9 * len(corpus))
train_set, val_set = random_split(
    corpus, [train_size, len(corpus) - train_size], generator=torch.Generator().manual_seed(42)
)

training_args = TrainingArguments(
    output_dir="checkpoints/gaia_expanded",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=200,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    lr_scheduler_type="cosine",
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
)

trainer.train()
trainer.save_model("checkpoints/gaia_expanded/best")

results = trainer.evaluate()
print(f"\nFinal eval loss: {results['eval_loss']:.4f}")
print("Done! Model saved to checkpoints/gaia_expanded/best")

# Check new matching rate
matched = sum(1 for g in all_genera if f"g__{g}" in tokenizer.vocab)
print(f"\nNew matching rate: {matched}/{len(all_genera)} ({matched/len(all_genera)*100:.1f}%)")
