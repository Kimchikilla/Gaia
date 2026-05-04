# Gaia: Soil Microbiome Foundation Model

> *Gaia — the Greek goddess of Earth. Decoding the hidden language of soil microbiomes.*

**"The AlphaFold of Soil Microbiomes, built open-source."**

**English** | [한국어](README_KO.md)

[![HF Model](https://img.shields.io/badge/HuggingFace-Kimchikilla%2Fgaia-yellow)](https://huggingface.co/Kimchikilla/gaia)
[![HF Dataset](https://img.shields.io/badge/HuggingFace_Dataset-Kimchikilla%2Fgaia--corpus-yellow)](https://huggingface.co/datasets/Kimchikilla/gaia-corpus)
[![Dataset Downloads](https://img.shields.io/badge/dataset_downloads-2.7k%2F30d-brightgreen)](https://huggingface.co/datasets/Kimchikilla/gaia-corpus)

Gaia is a foundation model that understands the "language" of soil microbial communities. Pre-trained on public metagenomic data, it enables soil health diagnosis, yield prediction, and microbial consortium design.

> Community pickup — `gaia-corpus` has been downloaded **~2,770 times in the last 30 days** on the Hugging Face Hub (snapshot: 2026-05-04). The dataset link is the [community-canonical entry point](https://huggingface.co/datasets/Kimchikilla/gaia-corpus); please cite both the dataset and this repo if you use Gaia in research.

---

## Key Features

- **Pre-trained Foundation Model**: 8-layer GPT-style transformer pre-trained on **7,170 soil microbiome sequences** from MGnify and EMP (v2 corpus); v1 (~2k) corpus also available
- **Soil Health Diagnosis**: Predict soil chemical properties (pH R²=0.95, total carbon R²=0.88, total nitrogen R²=0.88 on Westerfeld in-distribution; pH R²=0.59, C R²=0.72, N R²=0.73 OOD on Bernburg) from microbial profiles
- **Cross-Site OOD Generalization**: Beats RandomForest on 5/6 linear-probe tasks and 3/3 zero-shot Westerfeld→Bernburg tasks
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

Concrete results (v4 backbone, frozen — linear probe MLP head):

| Task | Dataset | Gaia | RF | Winner |
|---|---|---|---|---|
| pH prediction (in-dist.) | Westerfeld 192 | R² = **0.95** | — | — |
| Total Carbon (in-dist.) | Westerfeld 192 | R² = **0.88** | — | — |
| Total Nitrogen (in-dist.) | Westerfeld 192 | R² = **0.88** | — | — |
| pH prediction (OOD) | Bernburg 96 | R² = **0.59** | 0.55 | Gaia |
| Total Carbon (OOD) | Bernburg 96 | R² = **0.72** | 0.36 | Gaia (~2×) |
| Total Nitrogen (OOD) | Bernburg 96 | R² = **0.73** | 0.73 | tie |
| pH zero-shot (Westerfeld→Bernburg) | 96 | R² = **0.39** | −0.52 | Gaia |
| Total Carbon zero-shot | 96 | R² = **0.29** | 0.20 | Gaia |
| Total Nitrogen zero-shot | 96 | R² = **0.52** | 0.31 | Gaia |
| Drought classification (cross-continent) | Naylor (USA Sorghum) 623 | acc **0.944** / AUC **0.970** | acc 0.920 / AUC 0.951 | Gaia |
| Yield regression | USDA Potato 423 | R² 0.05 | **R² 0.26** | RF |

Honest read: Gaia dominates soil-chemistry and OOD generalization (foundation
model's strength = transferable representation). Yield prediction with current
v2 corpus still loses to RF — yield depends heavily on weather and management
not present in the microbiome signal alone, and v2 still under-covers
yield-paired domains. v5 (post-EMP continual pretrain) is the next yield rerun
target.

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

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- **Code**: Bug fixes, new features, pipeline improvements
- **Data**: Standardized soil microbiome datasets
- **Science**: New benchmark tasks, ecological validation, domain expertise

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

---

*This project is under active development. Star this repo to stay updated!*
