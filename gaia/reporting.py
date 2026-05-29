"""Reporting structures shared by CLI and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SHORTCUT_PRONE_CHECKPOINTS = {"gaia_v4", "gaia_v5", "gaia_v6", "gaia_v7"}


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
    notes: list[str] | None = None

    def to_text(self) -> str:
        lines = [f"=== Sample: {self.sample_id} ==="]
        lines.append(f"Genera detected: {self.n_genera}")
        lines.append("")
        lines.append("Predicted soil chemistry:")
        for prediction in self.predictions:
            lines.append(
                f"  {prediction.label:>14}: {prediction.value:7.3f} "
                f"{prediction.unit:6} "
                f"(source validation R^2={prediction.confidence_r2:.2f})"
            )
        lines.append("")
        lines.append("Top keystone genera (by abundance):")
        for genus, weight in self.keystone_genera[:5]:
            lines.append(f"  - {genus:30s} {weight:.4f}")
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        out = [f"## Sample `{self.sample_id}`", ""]
        out.append(f"- Genera detected: **{self.n_genera}**")
        if self.health_score is not None:
            out.append(f"- Health score: **{self.health_score:.2f}** / 1.00")
        out.append("")
        out.append("### Predicted soil chemistry")
        out.append("| Property | Value | Unit | Source validation R^2 |")
        out.append("|---|---|---|---|")
        for prediction in self.predictions:
            out.append(
                f"| {prediction.label} | {prediction.value:.3f} | "
                f"{prediction.unit} | {prediction.confidence_r2:.2f} |"
            )
        out.append("")
        out.append("### Top keystone genera (by abundance)")
        for genus, weight in self.keystone_genera[:5]:
            out.append(f"- *{genus}* - {weight:.4f}")
        if self.notes:
            out.append("")
            out.append("### Notes")
            for note in self.notes:
                out.append(f"- {note}")
        return "\n".join(out)


def checkpoint_reliability_notes(ckpt_dir: Path, heads: dict) -> list[str]:
    """Explain what the reported head R^2 does and does not mean."""
    notes = []
    ckpt_name = Path(ckpt_dir).name

    if ckpt_name in SHORTCUT_PRONE_CHECKPOINTS:
        notes.append(
            "Reliability: this checkpoint family is known to carry lab/country "
            "batch-shortcut signal. Treat predictions for new labs or new "
            "geographies as screening only, not calibrated soil chemistry."
        )

    if heads:
        notes.append(
            "The R^2 column is the source validation score for each prediction "
            "head; it is not per-sample confidence and does not estimate OOD "
            "performance."
        )

    return notes
