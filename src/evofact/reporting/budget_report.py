from __future__ import annotations


def budget_report(snapshot, usage=()) -> dict:
    statuses = sorted({getattr(item, "cost_status", "unavailable") for item in usage}, key=str)
    return {
        "snapshot": snapshot.model_dump() if hasattr(snapshot, "model_dump") else snapshot,
        "cost_statuses": [item.value if hasattr(item, "value") else str(item) for item in statuses],
        "usage": [item.model_dump() if hasattr(item, "model_dump") else item for item in usage],
    }
