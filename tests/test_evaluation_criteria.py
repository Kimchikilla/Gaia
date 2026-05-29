"""Tests for model acceptance criteria."""

from gaia.evaluation.criteria import (
    evaluate_diagnostic_gate,
    evaluate_prescription_gate,
)


def test_diagnostic_gate_requires_honest_group_performance_and_low_shortcut():
    passed = evaluate_diagnostic_gate(
        {
            "n_samples": 500,
            "n_groups": 8,
            "group_r2_mean": 0.2,
            "rmse_improvement_over_mean": 0.1,
            "shortcut_accuracy_over_majority": 0.1,
        }
    )
    failed = evaluate_diagnostic_gate(
        {
            "n_samples": 500,
            "n_groups": 8,
            "group_r2_mean": 0.2,
            "rmse_improvement_over_mean": 0.1,
            "shortcut_accuracy_over_majority": 0.8,
        }
    )

    assert passed["passed"]
    assert not failed["passed"]
    assert not failed["checks"]["shortcut_probe"]


def test_prescription_gate_fails_without_intervention_data():
    result = evaluate_prescription_gate(
        {
            "n_intervention_records": 0,
            "n_intervention_types": 0,
            "n_sites": 0,
            "max_followup_months": 0,
            "has_control_plots": False,
        }
    )

    assert not result["passed"]
    assert not result["checks"]["intervention_records"]
