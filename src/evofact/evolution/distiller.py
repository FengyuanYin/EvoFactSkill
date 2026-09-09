from collections import Counter

from evofact.core.models import AttributionReport, InferenceTrace


def distill(traces: list[InferenceTrace], reports: list[AttributionReport]) -> dict:
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
