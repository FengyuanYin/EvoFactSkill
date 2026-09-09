from collections.abc import Callable

from evofact.core.models import InferenceTrace


def leave_one_skill_out(
    trace: InferenceTrace, scorer: Callable[[InferenceTrace, tuple[str, ...]], float]
) -> dict[str, float]:
    baseline = scorer(trace, trace.routing.selected_skill_ids)
    result = {}
    for sid in trace.routing.selected_skill_ids:
        remaining = tuple(x for x in trace.routing.selected_skill_ids if x != sid)
        result[sid] = baseline - scorer(trace, remaining)
    return result
