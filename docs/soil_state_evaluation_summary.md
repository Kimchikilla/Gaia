# Soil-State Microbiome Evaluation Summary

Date: 2026-05-29

## Bottom Line

The current public-data benchmark is not sufficient to claim that Gaia can diagnose and prescribe soil-state interventions from soil microbiomes.

It can support a limited diagnostic prototype for pH-like soil-state signals, but the result still fails the honest diagnostic gate because the microbiome matrix strongly encodes site identity. It also fails the prescription gate because there is no intervention, control-plot, follow-up outcome dataset.

## What Was Fixed

- Added explicit diagnostic and prescription acceptance gates.
- Added shortcut-resistant evaluation helpers: mean/majority baselines, leave-group-out validation, and shortcut probes.
- Added soil microbiome cleaning utilities: taxon canonicalization, invalid taxon filtering, prevalence filtering, relative abundance, CLR transform, and cross-dataset feature alignment.
- Added reproducible scripts to build cleaned soil-state datasets and run the honest benchmark.
- Updated CLI reporting so old checkpoint R2 values are described as source-validation scores, not reliable deployment confidence.

## Data Used

| Dataset | Samples | Groups | Target | Notes |
|---|---:|---:|---|---|
| NEON | 2,482 | 20 sites | pH | Site-level mean pH joined to microbiome samples; useful for stress testing, not per-sample calibrated chemistry. |
| Westerfeld | 192 | 3 years | pH, total carbon, total nitrogen | Paired long-term field-trial microbiome and soil chemistry. |
| Bernburg | 94 | 3 years | pH, total carbon, total nitrogen, organic matter | Paired long-term field-trial microbiome and soil chemistry. |

Processed files are stored under `data/processed_real/soil_state_*.csv`.

## Cleaning Results

| Dataset | Original Features | Valid Taxa | Kept After Prevalence Filter | Dropped Invalid Taxa | Dropped Low-Prevalence |
|---|---:|---:|---:|---:|---:|
| NEON | 1,088 | 986 | 315 | 102 | 671 |
| Westerfeld | 1,864 | 1,813 | 1,107 | 51 | 706 |
| Bernburg | 780 | 744 | 607 | 36 | 137 |

## Benchmark Results

### NEON Leave-Site-Out pH

Best model: `RandomForest_CLR`

| Metric | Value |
|---|---:|
| OOF R2 | 0.659 |
| OOF RMSE | 0.805 |
| OOF MAE | 0.612 |
| Mean fold R2 | 0.295 |
| RMSE improvement over train-mean baseline | 0.678 |

This beats a naive mean baseline, but it does not pass the diagnostic gate because the shortcut probe shows that site identity is highly recoverable from the microbiome features.

### Shortcut Probe

| Metric | Value |
|---|---:|
| Site prediction accuracy | 0.730 |
| Majority baseline accuracy | 0.094 |
| Accuracy over majority baseline | 0.636 |
| Allowed threshold | 0.250 |

Interpretation: the model is learning substantial site/protocol/ecology fingerprint information. This makes the apparent diagnostic performance unsafe to treat as generalizable soil-state diagnosis.

### Cross-Dataset Transfer: Westerfeld to Bernburg pH

| Model | R2 | RMSE | Mean Baseline RMSE | RMSE Improvement |
|---|---:|---:|---:|---:|
| Ridge_CLR | -7.491 | 0.296 | 0.218 | -0.078 |
| RandomForest_CLR | -1.733 | 0.168 | 0.218 | 0.050 |

Interpretation: external transfer is still weak. Random forest improves RMSE slightly over the mean baseline, but R2 remains negative, so this is not deployment-grade generalization.

## Gate Status

| Gate | Status | Reason |
|---|---|---|
| Diagnostic | Fail | Sample count, group count, group R2, and baseline improvement pass; shortcut probe fails. |
| Prescription | Fail | No intervention records, intervention types, sites, follow-up months, or control plots. |

## Data Required For The Real Product Goal

To diagnose and prescribe soil state from soil microbiomes, the next dataset must be plot-level and longitudinal:

- Microbiome before treatment.
- Soil chemistry before treatment: pH, organic matter, total carbon, total nitrogen, phosphorus, potassium, CEC, moisture, texture.
- Management/intervention records: lime, compost, fertilizer, cover crop, tillage, irrigation, pesticide, inoculant, application amount, timing, and method.
- Soil chemistry after treatment at 3+ months.
- Plant or agronomic outcome after treatment: yield, biomass, disease pressure, crop quality, or recovery score.
- Control plots or untreated matched plots.
- Site, crop, climate, sampling depth, sequencing protocol, and lab metadata.

Without this intervention/outcome layer, Gaia can estimate correlations between taxa and soil properties, but it cannot recommend reliable prescriptions.

## Reproduction Commands

```powershell
python scripts\build_soil_state_datasets.py
python scripts\run_honest_soil_state_benchmark.py
python -m pytest -q
```
