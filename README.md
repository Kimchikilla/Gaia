# Gaia: Soil Microbiome Foundation Model

A transformer-based foundation model trained on public soil-microbiome data, with linear-probe heads for soil chemistry prediction and a CLI for diagnosis and consortium design.

**English** | [한국어](README_KO.md)

[![HF Model](https://img.shields.io/badge/HuggingFace-Kimchikilla%2Fgaia-yellow)](https://huggingface.co/Kimchikilla/gaia)
[![HF Dataset](https://img.shields.io/badge/HuggingFace_Dataset-Kimchikilla%2Fgaia--corpus-yellow)](https://huggingface.co/datasets/Kimchikilla/gaia-corpus)
[![Dataset Downloads](https://img.shields.io/badge/dataset_downloads-2.7k%2F30d-brightgreen)](https://huggingface.co/datasets/Kimchikilla/gaia-corpus)

Gaia is pre-trained on public metagenomic data (MGnify + EMP), and supports soil chemistry prediction, drought-stress classification, and consortium recommendation. See the benchmark table below for what it does and does not do — and **[ROADMAP.md](ROADMAP.md)** for the honest direction the project moved to after 2026-05-06.

The `gaia-corpus` dataset has been downloaded ~2,770 times in the last 30 days on the Hugging Face Hub (snapshot: 2026-05-04).

---

## Key Features

- **Pre-trained Foundation Model**: 8-layer GPT-style transformer pre-trained on **7,170 soil microbiome sequences** from MGnify and EMP (v2 corpus); v1 (~2k) corpus also available
- **Soil Health Diagnosis**: Predict soil chemical properties (pH R²=0.95, total carbon R²=0.88, total nitrogen R²=0.88 on Westerfeld in-distribution; pH R²=0.59, C R²=0.72, N R²=0.73 OOD on Bernburg) from microbial profiles
- **Cross-Site OOD**: Outperforms RandomForest on 5/6 linear-probe tasks and 3/3 zero-shot Westerfeld→Bernburg tasks; loses on yield regression
- **Drought Stress Detection**: Binary classification benchmarked on Naylor (USA Sorghum) — see [docs/benchmark_naylor.json](docs/benchmark_naylor.json)
- **Inverse Design (consortium recommendation)**: Given a target (pH, C, N), retrieve and aggregate microbial profiles from reference samples whose embeddings best match the target
- **CLI Tool**: `gaia diagnose abundance.csv` produces JSON or Markdown soil-health reports; `gaia design --ph 6.5 --carbon 1.8 --nitrogen 0.18` recommends a consortium

## Quick Start

### Installation

```bash
pip install gaia-soil
```

Or install from source:

```bash
git clone https://github.com/Kimchikilla/ProjectGaia.git
cd ProjectGaia
pip install -e ".[dev]"
```

### Basic Usage

CLI:

```bash
# Soil health diagnosis (predicts pH, total C, total N + lists keystone genera)
gaia diagnose path/to/abundance.csv --markdown report.md

# Inverse design — recommend a microbial consortium for a target soil state
gaia design --ph 6.5 --carbon 1.8 --nitrogen 0.18 --top-n 15
```

Python API (legacy, in-process):

```python
from gaia.cli import diagnose_file
from gaia.inference.inverse_design import DesignTarget, design_consortium

reports = diagnose_file("path/to/abundance.csv")
for r in reports:
    print(r.to_text())

rec = design_consortium(DesignTarget(ph=6.5, total_carbon=1.8, total_nitrogen=0.18))
print(rec.to_text(top_n=15))
```

## Project Structure

```
gaia/
├── README.md
├── LICENSE                    # Apache 2.0
├── CONTRIBUTING.md
├── docs/
│   ├── roadmap.md
│   ├── data_standard.md       # Data standardization guide
│   └── tutorials/
├── data/
│   ├── scripts/               # Data collection & preprocessing scripts
│   ├── configs/               # Data source configurations
│   └── README.md              # Data catalog
├── gaia/
│   ├── preprocessing/         # Preprocessing modules
│   ├── models/                # Model architectures
│   ├── training/              # Training scripts
│   ├── evaluation/            # Evaluation modules
│   └── inference/             # Inference modules
├── benchmarks/                # Benchmark datasets & evaluation criteria
├── notebooks/                 # Tutorial Jupyter notebooks
└── tests/
```

## Data Sources

Actual processed corpora used for pretraining and benchmarks:

| Source | Description | Used in v2 corpus |
|--------|------------|-----:|
| [MGnify](https://www.ebi.ac.uk/metagenomics/) | Taxonomic abundance tables (genus level) | 2,887 |
| [Earth Microbiome Project](https://earthmicrobiome.org/) | Soil samples filtered from EMP release1 | 4,628 |
| **Pre-training total (v2)** | After tokenization filter (≥5 valid genera) | **7,170 sequences** |
| [Westerfeld LTE](https://www.bonares.de/) | Paired microbiome + soil chemistry (Germany) | held out — fine-tune & benchmark |
| [Bernburg LTE](https://github.com/raabmarie/Synthesis_Three_Years_Bernburg) | Paired microbiome + soil chemistry (Germany) | held out — OOD benchmark |
| [Naylor 2017](https://github.com/raabmarie/Naylor) | Sorghum drought (USA, California) | held out — OOD drought benchmark |

NEON paired-microbiome data is queued for ingestion (currently only chemistry & site metadata are downloaded).

## Benchmarks

> **READ THIS FIRST.** The headline R² figures from earlier (v6 / v7) are *partly* the model recognising lab/country fingerprints, not real microbial signal. We trained an adversarial debiased model (v9) to quantify this. **The honest, batch-effect-free numbers are the v9 column. Treat that as the model's true diagnostic capability on a brand-new lab.**

Concrete results, grouped by backbone — frozen + linear-probe MLP head, 5-fold CV:

| Westerfeld (in-dist.) | v6 (GPT2, raw counts) | v7 (GPT2, CLR) | v8 (BERT, CLR) | **v9 (BERT + adversarial)** |
|---|---|---|---|---|
| pH R² | 0.962 | 0.761 | 0.070 | **0.108** |
| Total Carbon R² | 0.932 | 0.781 | −0.008 | **−0.017** |
| Total Nitrogen R² | 0.905 | 0.683 | 0.097 | **0.084** |

| EMP probes | v6 | v7 | v8 | **v9** |
|---|---|---|---|---|
| country probe acc (lower = less batch shortcut) | 0.941 | 0.870 | 0.427 | **0.188** |
| biome random-split acc | 0.935 | 0.897 | 0.632 | **0.305** |
| LOCO mean (cross-country biome acc) | 0.562 | 0.488 | 0.305 | **0.263** |

Other tasks (still on v6/v7 — pending re-run on v9):

| Task | Dataset | v6 score | RF |
|---|---|---|---|
| Drought classification (cross-continent) | Naylor (USA Sorghum) 623 | acc 0.944 / AUC 0.970 | acc 0.920 / AUC 0.951 |
| Yield regression | USDA Potato 423 | R² 0.05 | R² 0.26 |
| pH OOD | Westerfeld → Bernburg 96 | R² 0.39 | R² −0.52 |
| Total Carbon OOD | Westerfeld → Bernburg 96 | R² 0.29 | R² 0.20 |
| Total Nitrogen OOD | Westerfeld → Bernburg 96 | R² 0.52 | R² 0.31 |

What v9 tells us:

- The v9 model has a country probe accuracy of **0.19** vs **0.94** for v6 — adversarial training plus CLR + BERT removed nearly all the lab/country fingerprint from the embeddings.
- The price of removing that shortcut is severe: Westerfeld pH R² drops from **0.962 → 0.108**, Total Carbon goes to **near zero**, Total Nitrogen to **0.08**.
- This means the v6 R² ≈ 0.95 numbers were carrying a large amount of "I recognise this is a Westerfeld / German agricultural sample → predict the typical Westerfeld pH." When the model is no longer allowed to use that shortcut, the genuine microbiome → soil-chemistry signal it has learned is small.
- The Naylor cross-continent drought result and the Westerfeld→Bernburg OOD numbers were generated on v6 and very likely contain some of the same shortcut. They need to be re-run on v9 to be trusted as out-of-distribution claims.

What this implies for the README's mission:
- The model **does** carry genuine signal — country probe is far above majority baseline (~0.06 for 16 classes), and v9 R² is positive for pH and N — but the magnitude of that genuine signal is much smaller than v6's headline numbers suggested.
- Predictions on a soil sample from a lab the model has not seen (e.g. a Korean vineyard, a Brazilian Cerrado plot) should currently be expected to behave more like the v9 column than the v6 column.
- Closing this gap requires more lab diversity in the corpus, abundance-aware tokenisation (current vocab is presence-only), and probably a larger model trained for longer than 1500 steps.

## Known Limitations

These are honest caveats from internal validation work (`scripts/validate_diagnostic_heads.py`, `scripts/leave_one_country_out.py`, `scripts/geo_embedding_analysis.py`).

1. **Strong batch / country signal in embeddings.** A logistic-regression probe over `gaia_v6` embeddings recovers the country of an EMP soil sample at 5-fold CV accuracy **0.94** (16 countries, majority baseline 0.44). UMAP shows nearly perfect country clusters (`docs/geo_umap.png`). This very likely reflects laboratory / sequencing-platform fingerprints, not just real biological geography.

2. **Leave-one-country-out (LOCO) confirms batch shortcut.** When the same biome classifier is asked to predict on a country that was held out from training:
   - Random in-distribution split: acc **0.96**
   - LOCO mean: acc **0.63**, balanced acc **0.49**
   - On the largest held-out country (USA, n=834), the model performs at **0.54 acc, worse than the majority-class baseline (0.58)**.
   - Conclusion: a meaningful share of in-distribution accuracy comes from the model recognising "this is a sample from country X" rather than from generalisable microbial signal.

3. **Westerfeld diagnostic R² (0.95 / 0.88 / 0.88) cannot be tested for batch shortcut.** Westerfeld is a single site, so there is no cross-site holdout within that dataset. Bernburg (also Germany, same data provider/protocol) is the closest available OOD test, and the OOD R² there drops to 0.59 / 0.72 / 0.73 — consistent with both genuine generalisation and partial batch shortcut.

4. **Geographic and biome coverage of the pretraining corpus is uneven.** v3 corpus = MGnify v1 (2,887) + EMP (4,628) + NEON (2,999). NEON is 24 sites, all in the USA. EMP covers 21 countries but is dominated by the USA (43%) and Europe. Korean, tropical (sub-Saharan Africa, Southeast Asia, Latin America), and arid soils are sparsely represented.

5. **Country and biome are partially confounded in EMP.** Several countries (Japan, Australia, Mongolia, Tanzania) only have a single ENVO biome class in the dataset. This makes "country generalisation" and "biome generalisation" hard to separate without more data.

What this means for Gaia in practice:
- Strong on tasks where the inference-time soil sample comes from a population the model has seen (same lab, same continent, same biome distribution).
- Substantially weaker on truly novel geographies — a Korean vineyard or a Brazilian Cerrado sample is currently expected to be partially out-of-distribution at the **batch-effect** level, not just the biological-signal level.
- Mitigations to try next: (a) batch-correction normalisation (CLR / log-ratio) before tokenisation, (b) leave-one-country-out fine-tuning, (c) actually adding non-Western datasets (Korean RDA, Brazilian agricultural soil studies on MGnify).

## Model Architecture

- **Base**: GPT-2 style Transformer Decoder (Hugging Face `GPT2LMHeadModel`)
- **Layers**: 8
- **Attention Heads**: 8
- **Embedding Dim**: 256
- **Context length**: 512 tokens
- **Vocabulary**: 12,916 (BOS / EOS / PAD / MASK + ~12.9k `g__<Genus>` tokens)
- **Pre-training**: Continual pre-training from [MGM](https://github.com/HUST-NingKang-Lab/MGM) weights → `gaia_v4` (current public). `gaia_v5` is a continual-pretrain on the v2 (EMP-expanded) corpus.

## Tech Stack

| Area | Tool |
|------|------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.x |
| Transformers | Hugging Face Transformers |
| Data | Pandas, AnnData, Biom-format |
| Bioinformatics | QIIME2, Kraken2, MetaPhlAn |
| Visualization | Matplotlib, Seaborn, UMAP |
| Experiment Tracking | Weights & Biases |
| Model Hosting | Hugging Face Hub |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Useful directions: code fixes, additional standardized soil-microbiome datasets, new benchmark tasks, and ecological validation.

## Citation

```bibtex
@software{gaia2026,
  title={Gaia: A Foundation Model for Soil Microbiome Understanding},
  year={2026},
  url={https://github.com/Kimchikilla/ProjectGaia}
}
```

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

