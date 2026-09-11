from collections.abc import Callable

from evofact.core.models import InferenceTrace


def leave_one_skill_out(
    trace: InferenceTrace, scorer: Callable[[InferenceTrace, tuple[str, ...]], float]
) -> dict[str, float]:
    """函数作用：负责当前模块中的 `leave_one_skill_out` 处理，封装调用方需要复用的业务步骤。
    输入要求：`trace`（InferenceTrace）需符合函数签名约定；`scorer`（Callable[[InferenceTrace, tuple[str, ...]], float]）需符合函数签名约定。
    输出：返回 `dict[str, float]` 类型结果；校验或下游调用失败时异常向上传递。"""
    baseline = scorer(trace, trace.routing.selected_skill_ids)
    result = {}
    for sid in trace.routing.selected_skill_ids:
        remaining = tuple(x for x in trace.routing.selected_skill_ids if x != sid)
        result[sid] = baseline - scorer(trace, remaining)
    return result
