"""Evaluation modules and metrics."""

from gaia.evaluation.honest import (
    leave_one_group_out_classification,
    leave_one_group_out_regression,
    majority_classification_baseline,
    mean_regression_baseline,
    regression_split_report,
    shortcut_probe,
)
from gaia.evaluation.criteria import (
    DiagnosticCriteria,
    PrescriptionCriteria,
    evaluate_diagnostic_gate,
    evaluate_prescription_gate,
)

__all__ = [
    "DiagnosticCriteria",
    "PrescriptionCriteria",
    "evaluate_diagnostic_gate",
    "evaluate_prescription_gate",
    "leave_one_group_out_classification",
    "leave_one_group_out_regression",
    "majority_classification_baseline",
    "mean_regression_baseline",
    "regression_split_report",
    "shortcut_probe",
]
