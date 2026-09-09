from evofact.core.models import EvaluationResult, SampleEvaluation, UsageRecord
from evofact.evaluation.calibration import brier_score, expected_calibration_error
from evofact.evaluation.metrics import compute_metrics, grouped_metrics


def evaluate(rows: list[SampleEvaluation], confidence_intervals=None) -> EvaluationResult:
    metrics = compute_metrics(rows)
    metrics["ece"] = expected_calibration_error(rows)
    metrics["brier"] = brier_score(rows)
    return EvaluationResult(
        tuple(rows),
        metrics,
        grouped_metrics(rows, "domain"),
        grouped_metrics(rows, "temporal_window"),
        confidence_intervals or {},
        UsageRecord(estimated_cost=sum(r.cost for r in rows)),
    )
