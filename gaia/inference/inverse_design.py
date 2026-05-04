"""목표 토양 상태 → 미생물 컨소시엄 설계 (역설계).

전략 — 검색 기반(k-NN over latent):
  1. 학습 데이터(Westerfeld 등) 모든 샘플의 임베딩 + 측정 토양화학을 미리 계산.
  2. 사용자가 (pH, total_C, total_N) 목표를 주면 — 그 화학값과 가장 가까운
     k개의 실제 샘플을 찾는다.
  3. 그 샘플들의 abundance를 가까움 가중평균해서 추천 속(genus) 순위를 낸다.

생성 모델로 "뽑아내는" 게 아니라, 실제 데이터에 grounded된 추천이라
해석가능하고 실험실에서 바로 검증해볼 수 있다.
"""
from __future__ import annotations

import torch
import pickle
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel


@dataclass
class DesignTarget:
    ph: float | None = None
    total_carbon: float | None = None
    total_nitrogen: float | None = None

    def as_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ConsortiumRecommendation:
    target: dict
    n_neighbors: int
    genera: list[tuple[str, float]]   # (genus, weight) sorted descending
    reference_samples: list[str]
    achieved_chemistry: dict          # mean of neighbors' actual chemistry

    def to_text(self, top_n=15):
        lines = ["=== Inverse-design recommendation ==="]
        lines.append(f"Target: {self.target}")
        lines.append(f"Neighbors: {self.n_neighbors}")
        lines.append(f"Achieved (mean of neighbors): {self.achieved_chemistry}")
        lines.append("")
        lines.append("Recommended consortium (relative weight):")
        total = sum(w for _, w in self.genera[:top_n]) or 1.0
        for genus, w in self.genera[:top_n]:
            lines.append(f"  - {genus:30s} {w/total:6.4f}")
        return "\n".join(lines)


def _build_pivot_westerfeld(base="data/raw/longterm/bonares_data"):
    base = Path(base)
    bac = pd.read_csv(base / "lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
    g = pd.read_csv(base / "lte_westerfeld.V1_0_GENUS.csv")
    name_map = dict(zip(g["Genus_ID"], g["Name"]))
    bac["Genus_Name"] = bac["Genus_ID"].map(name_map)
    grp = (bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"]
              .sum().reset_index())
    pivot = grp.pivot_table(
        index=["Plot_ID", "Experimental_Year"],
        columns="Genus_Name", values="Value", fill_value=0,
    ).reset_index()

    soil = pd.read_csv(base / "lte_westerfeld.V1_0_SOIL_LAB.csv")
    samp = pd.read_csv(base / "lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
    soil = soil.merge(samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]],
                      on="Soil_Sampling_ID", how="left")
    chem_cols = ["pH", "Total_Carbon", "Total_Nitrogen"]
    chem = (soil.dropna(subset=["Plot_ID"])
                .groupby(["Plot_ID", "Experimental_Year"])[chem_cols]
                .mean().reset_index())
    paired = pivot.merge(chem, on=["Plot_ID", "Experimental_Year"], how="inner")
    paired = paired.dropna(subset=chem_cols).reset_index(drop=True)
    return paired


def _encode(row, genus_cols, tokenizer, max_len=512):
    bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    nz = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
    tokens = [bos]
    for g in nz.index:
        tid = tokenizer.vocab.get(f"g__{g}")
        if tid is not None:
            tokens.append(tid)
        if len(tokens) >= max_len - 1:
            break
    tokens.append(eos)
    while len(tokens) < max_len:
        tokens.append(pad)
    return torch.tensor(tokens[:max_len], dtype=torch.long)


def build_reference_index(
    ckpt_dir="checkpoints/gaia_v4",
    cache_path="checkpoints/gaia_v4/inverse_index.npz",
    device=None,
):
    """Westerfeld의 임베딩 + 화학 행렬을 만들어 디스크에 캐시한다."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {
            "embeddings": data["embeddings"],
            "chemistry":  data["chemistry"],
            "abundance":  data["abundance"],
            "genus_cols": list(data["genus_cols"]),
            "sample_ids": list(data["sample_ids"]),
            "chem_cols":  list(data["chem_cols"]),
        }

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    gpt = GPT2LMHeadModel.from_pretrained(str(ckpt_dir / "best")).to(device)
    gpt.eval()
    with open(ckpt_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    paired = _build_pivot_westerfeld()
    chem_cols = ["pH", "Total_Carbon", "Total_Nitrogen"]
    genus_cols = [c for c in paired.columns
                  if c not in ["Plot_ID", "Experimental_Year"] + chem_cols]
    print(f"[invdesign] {len(paired)} samples × {len(genus_cols)} genera")

    embs, chems, abus, ids = [], [], [], []
    with torch.no_grad():
        for _, row in paired.iterrows():
            x = _encode(row, genus_cols, tokenizer).unsqueeze(0).to(device)
            h = gpt(x, output_hidden_states=True).hidden_states[-1]
            mask = (x != 0).unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(pooled.squeeze(0).cpu().numpy())
            chems.append(row[chem_cols].values.astype(float))
            abus.append(row[genus_cols].values.astype(float))
            ids.append(f"{row['Plot_ID']}_y{int(row['Experimental_Year'])}")

    cache = {
        "embeddings": np.stack(embs),
        "chemistry":  np.stack(chems),
        "abundance":  np.stack(abus),
        "genus_cols": np.array(genus_cols),
        "sample_ids": np.array(ids),
        "chem_cols":  np.array(chem_cols),
    }
    np.savez_compressed(cache_path, **cache)
    print(f"[invdesign] cached index → {cache_path}")
    return {k: (list(v) if k in ("genus_cols", "sample_ids", "chem_cols") else v)
            for k, v in cache.items()}


def design_consortium(
    target: DesignTarget,
    index=None,
    k=8,
    ckpt_dir="checkpoints/gaia_v4",
) -> ConsortiumRecommendation:
    if index is None:
        index = build_reference_index(ckpt_dir=ckpt_dir)

    chem_cols = index["chem_cols"]
    chem = index["chemistry"]                       # (N, 3)
    means = chem.mean(axis=0)
    stds = chem.std(axis=0) + 1e-9

    # build target vector — fill missing with reference mean (no penalty)
    tgt = []
    weights = []
    for i, c in enumerate(chem_cols):
        v = getattr(target, {"pH": "ph",
                              "Total_Carbon": "total_carbon",
                              "Total_Nitrogen": "total_nitrogen"}[c])
        if v is None:
            tgt.append(means[i]); weights.append(0.0)
        else:
            tgt.append(float(v)); weights.append(1.0)
    tgt = np.array(tgt)
    w = np.array(weights)

    # weighted normalized distance
    dz = (chem - tgt) / stds
    dist = np.sqrt(((dz * w) ** 2).sum(axis=1))
    order = np.argsort(dist)[:k]

    # weight neighbors by inverse distance (epsilon-stabilized)
    nbr_w = 1.0 / (dist[order] + 1e-3)
    nbr_w /= nbr_w.sum()

    # aggregate genus abundances
    abu = index["abundance"][order]                 # (k, G)
    agg = (abu * nbr_w[:, None]).sum(axis=0)
    agg = agg / max(agg.sum(), 1e-12)
    genus_cols = list(index["genus_cols"])
    ranked = sorted(zip(genus_cols, agg), key=lambda kv: -kv[1])
    ranked = [(g, float(v)) for g, v in ranked if v > 0]

    achieved = {chem_cols[i]: float((chem[order, i] * nbr_w).sum())
                for i in range(len(chem_cols))}

    return ConsortiumRecommendation(
        target=target.as_dict(),
        n_neighbors=k,
        genera=ranked,
        reference_samples=[index["sample_ids"][int(i)] for i in order],
        achieved_chemistry=achieved,
    )
