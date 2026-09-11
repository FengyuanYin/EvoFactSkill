from evofact.core.models import Evidence, SpecialistReport


def aggregate_evidence(reports: tuple[SpecialistReport, ...]) -> tuple[Evidence, ...]:
    """函数作用：在保留专家报告顺序的同时，以确定性规则去除重复证据。
    输入要求：`reports`（tuple[SpecialistReport, ...]）需符合函数签名约定。
    输出：返回 `tuple[Evidence, ...]` 类型结果；校验或下游调用失败时异常向上传递。"""
    seen = set()
    result = []
    for report in reports:
        for evidence in report.evidence:
            key = (
                " ".join(evidence.text.casefold().split()),
                evidence.source,
                evidence.published_at,
            )
            if key not in seen:
                seen.add(key)
                result.append(evidence)
    return tuple(result)
