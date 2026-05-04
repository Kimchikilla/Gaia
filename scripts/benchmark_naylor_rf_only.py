"""Naylor RF baseline + record Gaia result already obtained."""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/naylor/naylor_genus_with_labels.csv")
df = df.dropna(subset=["treatment"]).reset_index(drop=True)
y = (df["treatment"].str.lower() == "drought").astype(int).values
genus_cols = [c for c in df.columns if c not in ["sample_id", "run_id", "treatment", "host"]]
X = df[genus_cols].values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
rf_pred = rf.predict(Xte)
rf_prob = rf.predict_proba(Xte)[:, 1]
r_acc = accuracy_score(yte, rf_pred)
r_f1 = f1_score(yte, rf_pred)
r_auc = roc_auc_score(yte, rf_prob)

# Gaia values from prior run (printed at ep 30):
#   acc=0.944 f1=0.946 auc=0.970
log = {
    "dataset": "Naylor (USA Sorghum drought)",
    "n_samples": int(len(df)), "n_genera": int(len(genus_cols)), "split": "80/20 stratified",
    "gaia": {"acc": 0.944, "f1": 0.946, "auc": 0.970},
    "rf":   {"acc": float(r_acc), "f1": float(r_f1), "auc": float(r_auc)},
}
print(json.dumps(log, indent=2))
Path("docs/benchmark_naylor.json").write_text(json.dumps(log, indent=2))
print("Saved docs/benchmark_naylor.json")
