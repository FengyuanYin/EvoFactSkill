from collections import Counter

from evofact.core.models import AttributionReport, InferenceTrace


def distill(traces: list[InferenceTrace], reports: list[AttributionReport]) -> dict:
    """函数作用：将轨迹和归因压缩为不含原始标签与完整轨迹的经验摘要。
    输入要求：`traces`（list[InferenceTrace]）需符合函数签名约定；`reports`（list[AttributionReport]）需符合函数签名约定。
    输出：返回 `dict` 类型结果；校验或下游调用失败时异常向上传递。"""
    allowed_ids = {r.trace_id for r in reports}
    selected = [t for t in traces if t.trace_id in allowed_ids]
    errors = Counter(e.value for r in reports for e in r.error_types)
    skills = Counter(s for t in selected for s in t.routing.selected_skill_ids)
    return {
        "trace_ids": [t.trace_id for t in selected],
        "error_counts": dict(errors),
        "skill_counts": dict(skills),
        "lessons": [
            f"Address recurring {name} failures ({count})" for name, count in errors.most_common()
        ],
        "raw_traces_included": False,
        "labels_included": False,
    }
