"""Remaining roadmap tasks: synthetic data, interpretability, HF upload"""

import pickle
import pandas as pd
import numpy as np
import torch
import json
import os
from pathlib import Path
from transformers import GPT2LMHeadModel

# Load model
model = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_v4/best")
model.eval()
model.cuda()
with open("checkpoints/gaia_v4/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
id2genus = {v: k for k, v in tokenizer.vocab.items()}
print(f"Model loaded: vocab={len(tokenizer.vocab)}")

# ============================================================
# TASK 1: Synthetic Data Generation + Validation
# ============================================================
print("\n" + "=" * 60)
print("TASK 1: Synthetic Data Generation & Validation")
print("=" * 60)

# Generate synthetic profiles for different conditions
conditions = {
    "grassland": ["g__Candidatus_Solibacter", "g__Bryobacter", "g__Acidothermus"],
    "forest": ["g__Mycobacterium", "g__Streptomyces", "g__Acidobacterium"],
    "agricultural": ["g__Bradyrhizobium", "g__Pseudomonas", "g__Bacillus"],
}

for condition, seeds in conditions.items():
    print(f"\n--- Generating: {condition} soil profile ---")
    seed_ids = [tokenizer.vocab.get(s) for s in seeds if s in tokenizer.vocab]
    if not seed_ids:
        print(f"  Seeds not in vocab, skipping")
        continue

    input_ids = torch.tensor([[tokenizer.bos_token_id] + seed_ids]).cuda()
    generated = []
    seen = set(seeds)

    with torch.no_grad():
        for _ in range(50):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] / 0.8
            topk = torch.topk(logits, 40)
            probs = torch.softmax(topk.values, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            next_token = topk.indices.gather(1, next_idx)
            tid = next_token.item()
            genus = id2genus.get(tid, "")
            if genus.startswith("g__"):
                name = genus[3:]
                if name not in seen:
                    generated.append(name)
                    seen.add(name)
            if genus in ["<eos>", "<pad>"]:
                break
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if len(generated) >= 15:
                break

    print(f"  Seeds: {[s[3:] for s in seeds]}")
    print(f"  Generated ({len(generated)}):")
    for i, g in enumerate(generated, 1):
        print(f"    {i}. {g}")

# Validate: compare with real data
print("\n--- Validation: Synthetic vs Real ---")
mgnify = pd.read_csv("data/raw/mgnify/mgnify_abundance.csv")
mgnify_meta = pd.read_csv("data/raw/mgnify/mgnify_metadata.csv")
genus_cols = [c for c in mgnify.columns if c not in ["sample_id", "analysis_id"]]

# Get real grassland genera (top 20)
merged = mgnify.merge(mgnify_meta[["sample_id", "biome"]], on="sample_id", how="inner")
grassland = merged[merged["biome"].str.contains("grassland", case=False, na=False)]
if len(grassland) > 0:
    real_top = grassland[genus_cols].sum().sort_values(ascending=False).head(20).index.tolist()
    synth_genera = set(generated) if "grassland" in conditions else set()
    overlap = synth_genera & set(real_top)
    print(f"  Real grassland top 20: {real_top[:10]}")
    print(f"  Synthetic generated: {list(synth_genera)[:10]}")
    print(f"  Overlap: {len(overlap)}/{min(len(synth_genera), 20)} ({len(overlap)/max(len(synth_genera),1)*100:.0f}%)")

# ============================================================
# TASK 2: Interpretability Tool
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: Interpretability — Keystone Genera Analysis")
print("=" * 60)

model_with_attn = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_v4/best", output_attentions=True)
model_with_attn.eval()

# Analyze a few samples
for sample_type, sample_genera in [
    ("Acidic soil", ["g__Acidothermus", "g__Bryobacter", "g__Candidatus_Solibacter"]),
    ("Nitrogen-rich", ["g__Bradyrhizobium", "g__Rhizobium", "g__Nitrospira"]),
]:
    tokens = [tokenizer.bos_token_id]
    names = ["<bos>"]
    for g in sample_genera:
        tid = tokenizer.vocab.get(g)
        if tid:
            tokens.append(tid)
            names.append(g[3:])

    # Add some random soil genera
    import random
    random.seed(42)
    soil_genera = [k for k in tokenizer.vocab if k.startswith("g__")]
    extra = random.sample(soil_genera, min(30, len(soil_genera)))
    for g in extra:
        tid = tokenizer.vocab[g]
        if tid not in tokens:
            tokens.append(tid)
            names.append(g[3:])
        if len(tokens) >= 40:
            break

    input_ids = torch.tensor([tokens])
    with torch.no_grad():
        outputs = model_with_attn(input_ids)

    # Average attention across layers and heads
    attn = torch.stack(outputs.attentions)  # (layers, 1, heads, seq, seq)
    avg_attn = attn.mean(dim=(0, 2)).squeeze(0)  # (seq, seq)
    importance = avg_attn.sum(dim=0).numpy()  # (seq,)

    print(f"\n--- {sample_type}: Top 10 keystone genera ---")
    scored = [(names[i], importance[i]) for i in range(1, len(names))]
    scored.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(scored[:10], 1):
        bar = "#" * int(score * 20)
        print(f"  {rank:2d}. {name:30s} {score:.4f} {bar}")

# ============================================================
# TASK 3: Save for Hugging Face Upload
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: Prepare Hugging Face Upload")
print("=" * 60)

hf_dir = Path("checkpoints/gaia_v4/huggingface")
hf_dir.mkdir(parents=True, exist_ok=True)

# Save model
model.cpu()
model.save_pretrained(str(hf_dir))

# Save tokenizer info as JSON
vocab_info = {
    "vocab_size": len(tokenizer.vocab),
    "special_tokens": {
        "pad": tokenizer.pad_token_id,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
    },
    "genera_count": sum(1 for k in tokenizer.vocab if k.startswith("g__")),
}
with open(hf_dir / "vocab_info.json", "w") as f:
    json.dump(vocab_info, f, indent=2)

# Save tokenizer
import shutil
shutil.copy("checkpoints/gaia_v4/tokenizer.pkl", str(hf_dir / "tokenizer.pkl"))

# Model card
model_card = """---
license: apache-2.0
tags:
- soil-microbiome
- foundation-model
- metagenomics
- agriculture
language: en
---

# Gaia: Soil Microbiome Foundation Model

A GPT-2 based foundation model for understanding soil microbial communities.

## Model Description

- Base: MGM (Microbiome General Model) pre-trained on 260K samples
- Fine-tuned: 8,329 soil samples from MGnify, EMP, Naylor, BonaRes
- Vocabulary: 12,916 microbial genera
- Architecture: GPT-2 (8 layers, 8 heads, 256 dim)

## Performance

| Task | Gaia | Random Forest |
|------|------|--------------|
| Biome Classification | 98.6% | 98.7% |
| Drought Detection | 91.9% | 92.0% |
| pH Prediction | R2=0.947 | R2=0.977 |
| Future Yield Prediction | R2=0.541 | R2=0.504 |

## Usage

```python
from transformers import GPT2LMHeadModel
import pickle

model = GPT2LMHeadModel.from_pretrained("Kimchikilla/gaia-soil")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
```

## Citation

```bibtex
@software{gaia2026,
  title={Gaia: A Foundation Model for Soil Microbiome Understanding},
  year={2026},
  url={https://github.com/Kimchikilla/Gaia}
}
```
"""

with open(hf_dir / "README.md", "w") as f:
    f.write(model_card)

print(f"HF files saved to {hf_dir}")
print(f"Files: {os.listdir(hf_dir)}")

print("\n" + "=" * 60)
print("ALL TASKS COMPLETE!")
print("=" * 60)
