# Gaia Data Catalog

## Public Data Sources

| Source | Type | Estimated Samples | Priority |
|--------|------|-------------------|----------|
| MGnify | Taxonomic abundance tables | 5,000-15,000 | 1 (Primary) |
| NEON | Paired microbiome + environmental | ~2,000 | 1 (Highest value) |
| Earth Microbiome Project | Standardized global samples | ~5,000 | 2 |
| SMAG Catalog | 40,039 soil MAGs | Reference DB | 3 |
| Naylor et al. | Drought stress dataset | 623 | 2 (Benchmark) |

## Directory Structure

- Data collection scripts live in `../scripts/data_collection/`; dataset build scripts live in `../scripts/data/`.
- `configs/` - Per-source configuration files

## Data Standards

See [docs/standards/data_standard.md](../docs/standards/data_standard.md) for the full data standardization protocol.
