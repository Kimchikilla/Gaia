"""
Process Bernburg long-term trial data into Gaia-compatible format.
- Aggregate ASVs to genus level
- Build sample x genus abundance matrix
- Join with metadata + soil chemistry
- Save to data/processed_real/bernburg_*
"""

import pandas as pd
from pathlib import Path

BASE = Path("data/raw/bernburg/Synthesis_Three_Years_Bernburg-main")
OUT = Path("data/processed_real")
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load ASV table
print("Loading ASV table...")
asv = pd.read_csv(BASE / "InputData/16S_ASV_Table_LTE_three_years.csv", low_memory=False)
print(f"  {asv.shape[0]} ASVs x {asv.shape[1]} cols")

# Identify sample columns
meta_cols = ["Unnamed: 0", "Sequence", "ASV"] + [c for c in asv.columns if c.startswith(("tax.", "boot."))]
sample_cols = [c for c in asv.columns if c not in meta_cols]
print(f"  {len(sample_cols)} sample columns")

# 2. Aggregate to genus level
print("Aggregating to genus...")
asv_g = asv.dropna(subset=["tax.Genus"]).copy()
print(f"  {len(asv_g)} ASVs with genus assignment ({len(asv) - len(asv_g)} dropped)")
genus_table = asv_g.groupby("tax.Genus")[sample_cols].sum()
print(f"  {genus_table.shape[0]} unique genera")

# 3. Transpose to sample x genus (Gaia format)
abundance = genus_table.T  # rows = samples, cols = genera
abundance.index.name = "sample_id"
abundance.columns.name = None
print(f"  Sample x Genus matrix: {abundance.shape}")

# 4. Load metadata
print("Loading metadata...")
meta = pd.read_csv(BASE / "InputData/metadata1.csv")
print(f"  {len(meta)} samples in metadata")

# 5. Load soil chemistry
soil = pd.read_csv(BASE / "InputData/soil_lab.csv")
print(f"  {len(soil)} soil chemistry rows")
print(f"  Soil columns: {list(soil.columns)}")

# 6. Build unified metadata table joining sample -> treatment + soil chem
# Match metadata Sample -> abundance index (sample_id)
sample_set = set(abundance.index)
meta_filt = meta[meta["Sample"].isin(sample_set)].copy()
print(f"  Metadata matches: {len(meta_filt)} / {len(abundance)}")

# Normalize join keys (meta uses CT/MP, Ext/Int; soil uses Cultivator/Plough, extensive/intensive)
tillage_map = {"CT": "Cultivator", "MP": "Plough"}
fert_map = {"Ext": "extensive", "Int": "intensive"}
meta_filt["Tillage_norm"] = meta_filt["Tillage"].map(tillage_map)
meta_filt["Fertilization_norm"] = meta_filt["Int"].map(fert_map)
meta_filt = meta_filt.rename(columns={"Year": "Experimental_Year"})

# Aggregate soil chemistry per (Year, Tillage, Fertilization) — averaging over blocks
chem_cols = [c for c in soil.columns if c not in ["Experimental_Year", "Date", "Block", "Tillage", "Fertilization"]]
soil_agg = soil.groupby(["Experimental_Year", "Tillage", "Fertilization"])[chem_cols].mean().reset_index()
soil_agg = soil_agg.rename(columns={"Tillage": "Tillage_norm", "Fertilization": "Fertilization_norm"})
print(f"  Soil aggregated to {len(soil_agg)} treatment-year cells")

merged = meta_filt.merge(
    soil_agg,
    on=["Experimental_Year", "Tillage_norm", "Fertilization_norm"],
    how="left",
)
matched = merged[chem_cols[0]].notna().sum()
print(f"  Samples with soil chemistry: {matched} / {len(merged)}")

print(f"  Merged metadata: {merged.shape}")
print(f"  Columns: {list(merged.columns)}")

# 7. Save outputs (NEW files, don't touch existing gaia-* files)
abundance.to_csv(OUT / "bernburg_abundance.csv")
merged.to_csv(OUT / "bernburg_metadata.csv", index=False)
print(f"\nSaved:")
print(f"  {OUT}/bernburg_abundance.csv  ({abundance.shape[0]} samples x {abundance.shape[1]} genera)")
print(f"  {OUT}/bernburg_metadata.csv  ({len(merged)} rows)")
