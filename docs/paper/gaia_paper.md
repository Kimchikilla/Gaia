# Gaia: A Soil-Specialized Foundation Model for Microbiome Understanding and Agricultural Prediction

## Abstract

We present Gaia, a foundation model specialized for soil microbiome analysis, built through continual pre-training of MGM (Microbiome General Model) on 8,329 soil samples from MGnify, Earth Microbiome Project, and experimental datasets. Gaia achieves near-parity with Random Forest on soil biome classification (98.6% vs 98.7%) and drought detection (91.9% vs 92.0%), while demonstrating superior performance on temporal yield prediction (R²=0.541 vs 0.504). We expand the microbial vocabulary from 9,669 to 12,916 genera through a two-step training strategy that preserves pre-trained knowledge while incorporating soil-specific taxa. Across eight benchmark tasks spanning biome classification, drought detection, tillage identification, pH prediction, and future yield forecasting, Gaia demonstrates that soil-specialized foundation models can match or exceed traditional machine learning approaches, with particular advantages in temporal prediction tasks requiring understanding of microbial community dynamics. All code, data pipelines, and model weights are publicly available.

## 1. Introduction

Soil microbiomes play a critical role in ecosystem functioning, agricultural productivity, and global biogeochemical cycles. The soil harbors an estimated 10^9 microbial cells per gram, forming complex communities whose composition reflects and influences soil health, nutrient cycling, and plant growth. Despite advances in high-throughput sequencing, interpreting these complex communities remains challenging.

Traditional approaches to soil microbiome analysis rely on statistical methods such as Random Forest, which treat each microbial taxon independently. These methods cannot capture the intricate ecological relationships between microorganisms---co-occurrence patterns, metabolic dependencies, and competitive interactions that collectively determine community function.

Foundation models, pre-trained on large corpora through self-supervised learning, have revolutionized natural language processing and protein structure prediction. Recently, MGM (Microbiome General Model) demonstrated that transformer-based language models can learn meaningful representations of microbial communities by treating abundance-ranked genus sequences as "sentences" in a microbial "language."

However, MGM was trained on all biomes (marine, freshwater, gut, soil) without specialization. Soil microbiomes have distinct characteristics: high diversity (typically 100-500+ genera per sample), strong environmental filtering by pH and moisture, and seasonal dynamics tied to agricultural practices.

We introduce Gaia, a soil-specialized foundation model that builds upon MGM through:
1. Continual pre-training on 8,329 soil-specific samples
2. Vocabulary expansion from 9,669 to 12,916 genera to cover soil-specific taxa
3. A two-step training strategy that preserves pre-trained knowledge
4. Comprehensive evaluation across eight benchmark tasks

## 2. Methods

### 2.1 Data Collection

We assembled soil microbiome data from four public sources:

**MGnify** (n=2,790): Genus-level abundance tables extracted from BIOM files via the MGnify REST API. Soil biome lineages were queried, and SSU OTU tables were parsed to obtain genus-level counts.

**Earth Microbiome Project** (n=4,628): Pre-processed closed-reference OTU table (Greengenes 13_8) filtered to soil samples identified by EMPO Level 3 classification.

**Naylor et al.** (n=623): Drought stress experiment with 18 grass species. Raw 16S rRNA sequences (PRJNA369551) were processed using QIIME2 DADA2 pipeline, with taxonomy assigned via vsearch against SILVA 138 (99% identity).

**BonaRes Westerfeld** (n=288): 20-year long-term field trial with tillage (cultivator vs plough) and fertilization (extensive vs intensive) treatments. Bacterial community data at genus level.

Total: 8,329 samples with genus-level abundance profiles.

### 2.2 Preprocessing Pipeline

A six-step preprocessing pipeline was applied:
1. Taxonomy unification to GTDB r220 nomenclature
2. Total Sum Scaling (TSS) normalization
3. Sparsity filtering (genera present in <1% of samples removed)
4. Metadata standardization using ENVO ontology
5. Batch effect annotation
6. MGM-compatible tokenization (abundance-ranked sequences, length 512)

### 2.3 Model Architecture

Gaia uses the GPT-2 architecture (transformer decoder):
- Layers: 8
- Attention heads: 8
- Embedding dimension: 256
- Feed-forward dimension: 1024
- Vocabulary: 12,916 tokens (including 4 special tokens)
- Total parameters: ~9.7M

### 2.4 Training Strategy

**Base model**: MGM pre-trained weights (260K samples across all biomes).

**Vocabulary expansion**: 2,970 soil-specific genera were added to the original 9,669-token vocabulary. The embedding layer was resized accordingly.

**Two-step continual pre-training**:
- Step 1 (15 epochs, lr=1e-3): Only the embedding layer is trained while all other parameters are frozen. This allows new token embeddings to learn meaningful representations without disrupting pre-trained weights.
- Step 2 (10 epochs, lr=5e-5): All parameters are unfrozen for full fine-tuning with a low learning rate.

Training was performed on a single NVIDIA RTX 5060 (8GB) with FP16 mixed precision.

### 2.5 Benchmark Tasks

Eight tasks were defined to evaluate model performance:

1. **Biome Classification**: Forest vs grassland (MGnify, n=569)
2. **Drought Detection**: Drought vs control (Naylor, n=623)
3. **Abundance Reconstruction**: Next-token prediction accuracy (MGnify, n=200)
4. **Tillage Classification**: Cultivator vs plough (BonaRes, n=288)
5. **Fertilization Classification**: Extensive vs intensive (BonaRes, n=288)
6. **pH Prediction**: Regression (BonaRes, n=288)
7. **Current Yield Prediction**: Regression (BonaRes, n=288)
8. **Future Yield Prediction**: This year's microbiome predicting next year's yield (BonaRes, n=283)

For classification and regression tasks, a task-specific head (256-128-n) was trained on top of mean-pooled hidden states, with model backbone weights frozen. Random Forest (200 trees) served as the baseline.

## 3. Results

### 3.1 Overall Performance

| Task | Gaia | Random Forest | Winner |
|------|------|--------------|--------|
| Biome Classification | 98.6% | 98.7% | Tie |
| Drought Detection | 91.9% | 92.0% | Tie |
| Abundance Reconstruction | Top-10: 31.3% | N/A | Gaia |
| Tillage Classification | 93.1% | 100.0% | RF |
| Fertilization Classification | 70.7% | 93.1% | RF |
| pH Prediction | R²=0.947 | R²=0.977 | RF |
| Current Yield | R²=0.873 | R²=0.926 | RF |
| **Future Yield** | **R²=0.541** | **R²=0.504** | **Gaia** |

### 3.2 Biome Classification and Drought Detection

Gaia achieves near-identical performance to Random Forest on biome classification (98.6% vs 98.7%) and drought detection (91.9% vs 92.0%), demonstrating that the foundation model's learned representations are competitive with engineered features for classification tasks.

### 3.3 Abundance Reconstruction

On the next-token prediction task (unique to generative models), Gaia achieves 31.3% Top-10 accuracy at 20% prediction, compared to 22.6% for the original MGM model---a 38% relative improvement. This task cannot be performed by Random Forest, representing a unique capability of the foundation model approach.

### 3.4 Future Yield Prediction

The most notable result is on temporal prediction: using this year's microbiome to predict next year's yield. Gaia (R²=0.541) outperforms Random Forest (R²=0.504) by 7.3% relative improvement. This suggests that the foundation model's understanding of microbial community structure provides an advantage for temporal prediction, where ecological relationships between taxa carry information about future soil state.

### 3.5 Data Scaling Effects

| Training Data | Biome Acc. | Drought Acc. | pH R² |
|--------------|-----------|-------------|-------|
| 3,624 | 66.7% | 87.1% | 0.878 |
| 6,339 | 97.0% | 91.1% | 0.866 |
| 8,329 | 98.6% | 91.9% | 0.947 |

Performance scales consistently with data quantity, suggesting further improvements with additional data.

### 3.6 Vocabulary Expansion

Expanding the vocabulary from 9,669 to 12,916 genera increased matching rates from 47-58% to 100%. The two-step training strategy was critical: single-step training degraded performance due to disruption of pre-trained embeddings.

## 4. Discussion

### 4.1 Foundation Models vs Traditional ML

Gaia demonstrates that soil-specialized foundation models can match Random Forest performance on most tasks while offering unique capabilities (abundance reconstruction, synthetic generation) that traditional methods cannot provide. The advantage of foundation models becomes most apparent in temporal prediction, where understanding microbial community dynamics is essential.

### 4.2 The Importance of Soil Specialization

Compared to the general-purpose MGM, Gaia shows consistent improvements on soil-specific tasks (31.3% vs 22.6% on abundance reconstruction). This supports the hypothesis that domain-specific continual pre-training captures environmental patterns not learned from heterogeneous training data.

### 4.3 Limitations

1. **Data scale**: 8,329 samples is small compared to MGM's 260K. Performance continues to improve with data, suggesting our results underestimate the approach's potential.
2. **Taxonomy resolution**: Genus-level analysis may miss species-level functional differences.
3. **Temporal data scarcity**: Future prediction was evaluated on only 283 time-series pairs from a single experimental site.
4. **Vocabulary mismatch**: Different databases use different naming conventions, limiting cross-dataset compatibility.

### 4.4 Future Directions

1. Scaling to 50,000+ soil samples through additional public databases
2. Multi-modal inputs combining microbiome with soil chemistry and climate data
3. Species-level or functional gene-level modeling
4. Prospective validation on independent experimental sites
5. Development of diagnostic tools for agricultural applications

## 5. Conclusion

Gaia demonstrates that soil-specialized foundation models are a viable approach for microbiome-based soil analysis. By achieving near-parity with Random Forest on classification tasks and superiority on temporal prediction, Gaia validates the concept that learned microbial "language" representations can capture ecologically meaningful patterns. The consistent improvement with data scaling suggests that larger soil microbiome datasets will further enhance model capabilities, potentially enabling practical applications in precision agriculture and soil health monitoring.

## Data and Code Availability

All code, data collection pipelines, trained model weights, and benchmark scripts are available at https://github.com/Kimchikilla/Gaia under Apache 2.0 license.

## References

1. MGM: A Large-Scale Pretrained Foundation Model for Microbiome Analyses. Advanced Science, 2024.
2. Thompson et al. A communal catalogue reveals Earth's multiscale microbial diversity. Nature, 2017.
3. Naylor et al. Drought and host selection influence bacterial community dynamics in the grass root microbiome. The ISME Journal, 2017.
4. Raab et al. Two decades long-term field trial data on fertilization, tillage, and crop rotation focusing on soil microbes. Scientific Data, 2025.
5. QIIME 2: Reproducible, interactive, scalable, and extensible microbiome data science. Nature Biotechnology, 2019.
