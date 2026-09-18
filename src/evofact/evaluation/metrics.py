from collections import defaultdict

from evofact.core.models import SampleEvaluation


def _validate_schema(rows: list[SampleEvaluation]) -> tuple[str, ...]:
    signatures = {row.allowed_labels for row in rows}
    if len(signatures) != 1:
        raise ValueError("rows in one label schema must use the same allowed_labels")
    labels = next(iter(signatures), ())
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("allowed_labels must be non-empty and unique")
    if any(row.gold not in labels for row in rows):
        raise ValueError("gold label is outside allowed_labels")
    return labels


def _single_schema_metrics(rows: list[SampleEvaluation]) -> dict[str, float]:
    labels = _validate_schema(rows)
    n = len(rows)
    covered = [row for row in rows if row.decision_origin != "runtime" and row.predicted in labels]
    correct = sum(row.gold == row.predicted for row in rows)
    f1s = []
    for label in labels:
        tp = sum(row.gold == label and row.predicted == label for row in rows)
        fp = sum(row.gold != label and row.predicted == label for row in rows)
        fn = sum(row.gold == label and row.predicted != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    covered_correct = sum(row.gold == row.predicted for row in covered)
    metrics = {
        "n": float(n),
        "accuracy_all": correct / n if n else 0.0,
        "macro_f1_all": sum(f1s) / len(labels),
        "coverage": len(covered) / n if n else 0.0,
        "covered_accuracy": covered_correct / len(covered) if covered else 0.0,
        "selective_risk": 1 - covered_correct / len(covered) if covered else 1.0,
        "mean_cost": sum(row.cost for row in rows) / n if n else 0.0,
        "cost_available": float(
            bool(rows) and all(row.cost_status != "unavailable" for row in rows)
        ),
        "evidence_coverage": sum(row.evidence_available for row in rows) / n if n else 0.0,
    }
    for label in labels:
        support = sum(row.gold == label for row in rows)
        correct_for_label = sum(row.gold == label and row.predicted == label for row in rows)
        metrics[f"label.{label}.support"] = float(support)
        metrics[f"label.{label}.recall"] = correct_for_label / support if support else 0.0
        metrics[f"label.{label}.error_rate"] = 1 - metrics[f"label.{label}.recall"]
        for predicted in (*labels, "__RUNTIME_ABSTAIN__", "__INVALID__"):
            count = sum(
                row.gold == label
                and (
                    (predicted == "__RUNTIME_ABSTAIN__" and row.decision_origin == "runtime")
                    or (
                        predicted == "__INVALID__"
                        and row.decision_origin != "runtime"
                        and row.predicted not in labels
                    )
                    or (
                        predicted in labels
                        and row.decision_origin != "runtime"
                        and row.predicted == predicted
                    )
                )
                for row in rows
            )
            metrics[f"confusion.{label}->{predicted}"] = float(count)
    return metrics


def compute_metrics(rows: list[SampleEvaluation]) -> dict[str, float]:
    """函数作用：计算包含弃权在分母中的分类、覆盖率、风险和成本指标。
    输入要求：`rows`（list[SampleEvaluation]）需符合函数签名约定。
    输出：返回 `dict[str, float]` 类型结果；校验或下游调用失败时异常向上传递。"""
    if not rows:
        return {
            "n": 0.0,
            "accuracy_all": 0.0,
            "macro_f1_all": 0.0,
            "coverage": 0.0,
            "covered_accuracy": 0.0,
            "selective_risk": 1.0,
            "mean_cost": 0.0,
            "cost_available": 0.0,
            "evidence_coverage": 0.0,
        }
    by_schema: dict[str, list[SampleEvaluation]] = defaultdict(list)
    for row in rows:
        by_schema[row.label_schema_id].append(row)
    metrics = [_single_schema_metrics(group) for group in by_schema.values()]
    n = len(rows)
    covered_n = sum(item["coverage"] * item["n"] for item in metrics)
    covered_correct = sum(
        item["covered_accuracy"] * item["coverage"] * item["n"] for item in metrics
    )
    return {
        "n": float(n),
        "accuracy_all": sum(item["accuracy_all"] * item["n"] for item in metrics) / n,
        "macro_f1_all": sum(item["macro_f1_all"] * item["n"] for item in metrics) / n,
        "coverage": covered_n / n,
        "covered_accuracy": covered_correct / covered_n if covered_n else 0.0,
        "selective_risk": 1 - covered_correct / covered_n if covered_n else 1.0,
        "mean_cost": sum(row.cost for row in rows) / n,
        "cost_available": float(all(row.cost_status != "unavailable" for row in rows)),
        "evidence_coverage": sum(row.evidence_available for row in rows) / n,
    }


def schema_metrics(rows: list[SampleEvaluation]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[SampleEvaluation]] = defaultdict(list)
    for row in rows:
        groups[row.label_schema_id].append(row)
    return {key: _single_schema_metrics(value) for key, value in sorted(groups.items())}


def grouped_metrics(rows: list[SampleEvaluation], field: str) -> dict[str, dict[str, float]]:
    """函数作用：负责当前模块中的 `grouped_metrics` 处理，封装调用方需要复用的业务步骤。
    输入要求：`rows`（list[SampleEvaluation]）需符合函数签名约定；`field`（str）需符合函数签名约定。
    输出：返回 `dict[str, dict[str, float]]` 类型结果；校验或下游调用失败时异常向上传递。"""
    groups = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, field) or "unknown")].append(row)
    return {k: compute_metrics(v) for k, v in sorted(groups.items())}
