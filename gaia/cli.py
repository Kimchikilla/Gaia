"""Gaia 명령행 도구.

사용 예:
    gaia diagnose abundance.csv --output report.json
    gaia diagnose abundance.csv --markdown report.md

abundance.csv 형식:
    sample_id, <genus_1>, <genus_2>, ...
    s001, 0.0123, 0.0456, ...

genus 컬럼 이름은 학명 (예: 'Bacillus', 'Pseudomonas') — 'g__' 접두어는 자동으로 붙는다.
"""

import torch  # torch first (Windows c10.dll workaround)
import torch.nn as nn
import argparse
import json
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel


DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "gaia_v4"


@dataclass
class Prediction:
    label: str
    value: float
    unit: str
    confidence_r2: float


@dataclass
class SampleReport:
    sample_id: str
    n_genera: int
    predictions: list[Prediction]
    keystone_genera: list[tuple[str, float]]
    health_score: float | None = None
    notes: list[str] = None

    def to_text(self) -> str:
        lines = [f"=== Sample: {self.sample_id} ==="]
        lines.append(f"Genera detected: {self.n_genera}")
        lines.append("")
        lines.append("Predicted soil chemistry:")
        for p in self.predictions:
            lines.append(f"  {p.label:>14}: {p.value:7.3f} {p.unit:6} (training R^2={p.confidence_r2:.2f})")
        lines.append("")
        lines.append("Top keystone genera (by abundance):")
        for genus, weight in self.keystone_genera[:5]:
            lines.append(f"  - {genus:30s} {weight:.4f}")
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        out = [f"## Sample `{self.sample_id}`", ""]
        out.append(f"- Genera detected: **{self.n_genera}**")
        if self.health_score is not None:
            out.append(f"- Health score: **{self.health_score:.2f}** / 1.00")
        out.append("")
        out.append("### Predicted soil chemistry")
        out.append("| Property | Value | Unit | Training R² |")
        out.append("|---|---|---|---|")
        for p in self.predictions:
            out.append(f"| {p.label} | {p.value:.3f} | {p.unit} | {p.confidence_r2:.2f} |")
        out.append("")
        out.append("### Top keystone genera (by abundance)")
        for genus, weight in self.keystone_genera[:5]:
            out.append(f"- *{genus}* — {weight:.4f}")
        if self.notes:
            out.append("")
            out.append("### Notes")
            for n in self.notes:
                out.append(f"- {n}")
        return "\n".join(out)


class DiagnosisHead(nn.Module):
    """진단 헤드 (gpt 동결 + MLP). 학습 스크립트와 구조 일치."""

    def __init__(self, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )


HEAD_LABELS = {
    "ph":           ("pH",            ""),
    "total_carbon": ("Total Carbon",  "%"),
    "total_n":      ("Total Nitrogen", "%"),
    "organic_matter": ("Organic Matter", "%"),
}


def encode_row(row, genus_cols, tokenizer, max_len=512):
    bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    nz = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
    tokens = [bos]
    matched = []
    for genus in nz.index:
        # try with and without g__ prefix
        for cand in (genus, f"g__{genus}", genus.replace("g__", "")):
            if cand in tokenizer.vocab:
                tokens.append(tokenizer.vocab[cand])
                matched.append((genus, float(nz[genus])))
                break
        if len(tokens) >= max_len - 1:
            break
    tokens.append(eos)
    while len(tokens) < max_len:
        tokens.append(pad)
    return torch.tensor(tokens[:max_len], dtype=torch.long), matched


@torch.no_grad()
def get_embedding(gpt, x, device):
    h = gpt(x, output_hidden_states=True).hidden_states[-1]
    mask = (x != 0).unsqueeze(-1).float()
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled


def load_heads(heads_dir: Path, device):
    out = {}
    if not heads_dir.exists():
        return out
    for pt_file in sorted(heads_dir.glob("*.pt")):
        name = pt_file.stem
        ckpt = torch.load(pt_file, map_location=device, weights_only=False)
        head = DiagnosisHead(hidden=ckpt.get("hidden_size", 256)).to(device)
        head.mlp.load_state_dict(ckpt["state_dict"])
        head.eval()
        out[name] = {
            "head": head,
            "y_mean": ckpt["y_mean"],
            "y_std":  ckpt["y_std"],
            "r2":     ckpt.get("best_r2", 0.0),
            "label_col": ckpt.get("label_col", name),
        }
    return out


def diagnose_file(abundance_path, ckpt_dir=DEFAULT_CKPT, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gaia] device: {device}", file=sys.stderr)

    print("[gaia] loading backbone...", file=sys.stderr)
    gpt = GPT2LMHeadModel.from_pretrained(str(ckpt_dir / "best")).to(device)
    gpt.eval()
    with open(ckpt_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    heads = load_heads(ckpt_dir / "heads", device)
    print(f"[gaia] loaded {len(heads)} prediction head(s): {list(heads)}", file=sys.stderr)

    df = pd.read_csv(abundance_path)
    if "sample_id" not in df.columns:
        df.insert(0, "sample_id", [f"s{i:04d}" for i in range(len(df))])
    genus_cols = [c for c in df.columns if c != "sample_id"]
    print(f"[gaia] {len(df)} samples × {len(genus_cols)} genus columns", file=sys.stderr)

    reports = []
    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        x_cpu, matched = encode_row(row, genus_cols, tokenizer)
        x = x_cpu.unsqueeze(0).to(device)
        emb = get_embedding(gpt, x, device)

        preds = []
        notes = []
        if not matched:
            notes.append("0 genera matched the model vocabulary — predictions skipped.")
        else:
            for name, h in heads.items():
                with torch.no_grad():
                    z = h["head"].mlp(emb).squeeze(-1).item()
                value = z * h["y_std"] + h["y_mean"]
                label, unit = HEAD_LABELS.get(name, (name, ""))
                preds.append(Prediction(label=label, value=value, unit=unit, confidence_r2=h["r2"]))

        # keystone = top abundance genera that hit vocab
        matched_sorted = sorted(matched, key=lambda kv: -kv[1])

        if matched and len(matched) < 20:
            notes.append(f"Only {len(matched)} genera matched vocab — low coverage may reduce reliability.")

        reports.append(SampleReport(
            sample_id=sid,
            n_genera=len(matched),
            predictions=preds,
            keystone_genera=matched_sorted,
            notes=notes or None,
        ))
    return reports


def cmd_design(args):
    from gaia.inference.inverse_design import (
        DesignTarget, design_consortium, build_reference_index,
    )
    target = DesignTarget(
        ph=args.ph,
        total_carbon=args.carbon,
        total_nitrogen=args.nitrogen,
    )
    if not target.as_dict():
        print("Specify at least one of --ph / --carbon / --nitrogen", file=sys.stderr)
        sys.exit(2)
    print(f"[gaia] target: {target.as_dict()}", file=sys.stderr)
    index = build_reference_index(ckpt_dir=args.checkpoint)
    rec = design_consortium(target, index=index, k=args.k)

    if args.output:
        Path(args.output).write_text(
            json.dumps({
                "target": rec.target,
                "n_neighbors": rec.n_neighbors,
                "achieved_chemistry": rec.achieved_chemistry,
                "reference_samples": rec.reference_samples,
                "consortium": rec.genera[:args.top_n],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[gaia] wrote: {args.output}", file=sys.stderr)
    else:
        print(rec.to_text(top_n=args.top_n))


def cmd_diagnose(args):
    reports = diagnose_file(args.abundance, ckpt_dir=Path(args.checkpoint))

    if args.output:
        payload = [
            {
                "sample_id": r.sample_id,
                "n_genera": r.n_genera,
                "predictions": [asdict(p) for p in r.predictions],
                "keystone_genera": r.keystone_genera[:20],
                "notes": r.notes,
            }
            for r in reports
        ]
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[gaia] wrote JSON: {args.output}", file=sys.stderr)

    if args.markdown:
        md = "\n\n".join(r.to_markdown() for r in reports)
        Path(args.markdown).write_text(
            "# Gaia Soil Health Report\n\n" + md, encoding="utf-8"
        )
        print(f"[gaia] wrote Markdown: {args.markdown}", file=sys.stderr)

    if not args.output and not args.markdown:
        for r in reports:
            print(r.to_text())
            print()


def main():
    parser = argparse.ArgumentParser(prog="gaia", description="Gaia soil microbiome CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_d = sub.add_parser("diagnose", help="Predict soil chemistry from abundance CSV")
    p_d.add_argument("abundance", help="Path to abundance CSV")
    p_d.add_argument("--checkpoint", default=str(DEFAULT_CKPT), help="Checkpoint dir")
    p_d.add_argument("--output", help="JSON output path")
    p_d.add_argument("--markdown", help="Markdown report output path")
    p_d.set_defaults(func=cmd_diagnose)

    p_g = sub.add_parser("design", help="Recommend microbial consortium for target soil chemistry")
    p_g.add_argument("--ph", type=float, default=None)
    p_g.add_argument("--carbon", type=float, default=None, help="Total Carbon %")
    p_g.add_argument("--nitrogen", type=float, default=None, help="Total Nitrogen %")
    p_g.add_argument("-k", type=int, default=8, help="k-nearest reference samples")
    p_g.add_argument("--top-n", type=int, default=15)
    p_g.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p_g.add_argument("--output", help="JSON output path")
    p_g.set_defaults(func=cmd_design)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
