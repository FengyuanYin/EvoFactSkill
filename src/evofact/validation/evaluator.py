from evofact.core.models import EvaluationResult, SampleEvaluation, UsageRecord
from evofact.evaluation.calibration import brier_score, expected_calibration_error
from evofact.evaluation.metrics import compute_metrics, grouped_metrics


def evaluate(rows: list[SampleEvaluation], confidence_intervals=None) -> EvaluationResult:
    """函数作用：把逐样本评估汇总为总体、领域、时间和校准指标。
    输入要求：`rows`（list[SampleEvaluation]）需符合函数签名约定；`confidence_intervals`（未显式标注，默认 `None`）需符合函数签名约定。
    输出：返回 `EvaluationResult` 类型结果；校验或下游调用失败时异常向上传递。"""
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
