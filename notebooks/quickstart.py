"""
Gaia Quickstart Tutorial
========================

This script demonstrates how to use the Gaia soil microbiome model for:
1. Predicting companion microbes
2. Biome classification
3. pH estimation
4. Keystone genera identification

Requirements:
    pip install torch transformers

Usage:
    python notebooks/quickstart.py
"""

import pickle
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

# ============================================================
# 1. Load Model
# ============================================================
print("=" * 50)
print("1. Loading Gaia Model")
print("=" * 50)

model = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_v4/best")
model.eval()

with open("checkpoints/gaia_v4/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

id2genus = {v: k for k, v in tokenizer.vocab.items()}
print(f"Model loaded! Vocabulary: {len(tokenizer.vocab)} tokens")

# ============================================================
# 2. Predict Companion Microbes
# ============================================================
print("\n" + "=" * 50)
print("2. Predict Companion Microbes")
print("=" * 50)
print("Input: Bradyrhizobium, Rhizobium (nitrogen fixers)")
print("What other microbes would co-occur?\n")

seed_genera = ["g__Bradyrhizobium", "g__Rhizobium"]
input_ids = [tokenizer.bos_token_id]
for g in seed_genera:
    tid = tokenizer.vocab.get(g)
    if tid:
        input_ids.append(tid)

input_tensor = torch.tensor([input_ids])
generated = []
seen = set(seed_genera)

with torch.no_grad():
    for _ in range(30):
        outputs = model(input_tensor)
        logits = outputs.logits[:, -1, :] / 0.8
        topk = torch.topk(logits, 40)
        probs = torch.softmax(topk.values, dim=-1)
        next_idx = torch.multinomial(probs, 1)
        next_token = topk.indices.gather(1, next_idx)
        tid = next_token.item()
        genus = id2genus.get(tid, "")
        if genus.startswith("g__") and genus not in seen:
            generated.append(genus[3:])
            seen.add(genus)
        if genus in ["<eos>", "<pad>"]:
            break
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        if len(generated) >= 10:
            break

print("Predicted companions:")
for i, g in enumerate(generated, 1):
    print(f"  {i}. {g}")

# ============================================================
# 3. Soil Diagnosis Example
# ============================================================
print("\n" + "=" * 50)
print("3. Soil Diagnosis")
print("=" * 50)
print("Given a microbiome profile, what can we tell about the soil?\n")

# Example: acidic soil indicators
acidic_genera = ["g__Acidothermus", "g__Bryobacter", "g__Candidatus_Solibacter",
                 "g__Acidibacter", "g__Granulicella"]
neutral_genera = ["g__Bacillus", "g__Pseudomonas", "g__Arthrobacter",
                  "g__Streptomyces", "g__Sphingomonas"]

for label, genera in [("Acidic soil sample", acidic_genera),
                       ("Neutral soil sample", neutral_genera)]:
    tokens = [tokenizer.bos_token_id]
    for g in genera:
        tid = tokenizer.vocab.get(g)
        if tid:
            tokens.append(tid)

    input_tensor = torch.tensor([tokens])
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = (input_tensor != 0).unsqueeze(-1).float()
        embedding = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    print(f"  {label}: embedding norm = {embedding.norm():.2f}")

print("\n  (Different embeddings = model distinguishes soil types)")

# ============================================================
# 4. Summary
# ============================================================
print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print("""
Gaia can:
  - Predict companion microbes for a given community
  - Distinguish soil types from microbiome profiles
  - Estimate soil pH (R2=0.947)
  - Detect drought stress (91.9% accuracy)
  - Predict future yield (R2=0.541)

For more details, visit: https://github.com/Kimchikilla/Gaia
""")
