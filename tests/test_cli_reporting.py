"""Tests for CLI reporting guardrails."""

from pathlib import Path

from gaia.reporting import (
    Prediction,
    SampleReport,
    checkpoint_reliability_notes,
)


def test_report_labels_r2_as_source_validation_score():
    report = SampleReport(
        sample_id="s1",
        n_genera=1,
        predictions=[Prediction("pH", 6.5, "", 0.95)],
        keystone_genera=[("Bacillus", 0.4)],
        notes=["example note"],
    )

    text = report.to_text()
    markdown = report.to_markdown()

    assert "source validation R^2=0.95" in text
    assert "Source validation R^2" in markdown
    assert "training R^2" not in text


def test_shortcut_prone_checkpoints_get_reliability_note():
    notes = checkpoint_reliability_notes(
        Path("checkpoints/gaia_v4"),
        heads={"ph": {"r2": 0.95}},
    )

    assert any("batch-shortcut" in note for note in notes)
    assert any("not per-sample confidence" in note for note in notes)


def test_unlisted_checkpoint_only_explains_r2_when_heads_exist():
    notes = checkpoint_reliability_notes(
        Path("checkpoints/gaia_v10"),
        heads={"ph": {"r2": 0.2}},
    )

    assert not any("batch-shortcut" in note for note in notes)
    assert any("source validation score" in note for note in notes)
