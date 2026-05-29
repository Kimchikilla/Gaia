"""Acceptance criteria for Gaia soil-state models."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class DiagnosticCriteria:
    min_samples: int = 300
    min_groups: int = 5
    min_group_r2: float = 0.10
    min_rmse_improvement_over_mean: float = 0.0
    max_shortcut_accuracy_over_majority: float = 0.25


@dataclass(frozen=True)
class PrescriptionCriteria:
    min_intervention_records: int = 1000
    min_intervention_types: int = 5
    min_sites: int = 10
    min_followup_months: int = 3
    requires_control_plots: bool = True


def evaluate_diagnostic_gate(
    report: dict[str, Any],
    criteria: DiagnosticCriteria = DiagnosticCriteria(),
) -> dict[str, Any]:
    """Return pass/fail checks for a diagnostic model report."""
    checks = {
        "sample_count": int(report.get("n_samples", 0)) >= criteria.min_samples,
        "group_count": int(report.get("n_groups", 0)) >= criteria.min_groups,
        "group_r2": float(report.get("group_r2_mean", float("-inf")))
        >= criteria.min_group_r2,
        "beats_mean_baseline": float(
            report.get("rmse_improvement_over_mean", float("-inf"))
        )
        > criteria.min_rmse_improvement_over_mean,
        "shortcut_probe": float(
            report.get("shortcut_accuracy_over_majority", float("inf"))
        )
        <= criteria.max_shortcut_accuracy_over_majority,
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "criteria": asdict(criteria),
    }


def evaluate_prescription_gate(
    report: dict[str, Any],
    criteria: PrescriptionCriteria = PrescriptionCriteria(),
) -> dict[str, Any]:
    """Return pass/fail checks for a prescription/intervention dataset."""
    checks = {
        "intervention_records": int(report.get("n_intervention_records", 0))
        >= criteria.min_intervention_records,
        "intervention_types": int(report.get("n_intervention_types", 0))
        >= criteria.min_intervention_types,
        "sites": int(report.get("n_sites", 0)) >= criteria.min_sites,
        "followup_months": int(report.get("max_followup_months", 0))
        >= criteria.min_followup_months,
        "control_plots": (not criteria.requires_control_plots)
        or bool(report.get("has_control_plots", False)),
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "criteria": asdict(criteria),
    }
