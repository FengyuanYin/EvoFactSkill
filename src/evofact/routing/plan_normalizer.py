from __future__ import annotations

import hashlib
import json
from typing import Any

from evofact.core.dag_models import ExecutionPlan, PlanNode
from evofact.core.models import RunBudget


def normalize_plan(
    nodes: list[PlanNode | dict[str, Any]] | tuple[PlanNode | dict[str, Any], ...],
    *,
    reasons: dict[str, str] | None = None,
    confidence: float = 1.0,
    budget: RunBudget = RunBudget(),
    fallback_used: bool = False,
) -> ExecutionPlan:
    prepared: list[PlanNode] = []
    for raw in nodes:
        if isinstance(raw, PlanNode):
            node = raw
        else:
            allowed = {
                "node_id",
                "skill_id",
                "depends_on",
                "required",
                "upstream_outputs",
                "priority",
                "timeout_ms",
            }
            unknown = set(raw) - allowed
            if unknown:
                raise ValueError(f"unknown plan node fields: {sorted(unknown)}")
            data = dict(raw)
            data["depends_on"] = tuple(data.get("depends_on", ()))
            data["upstream_outputs"] = tuple(data.get("upstream_outputs", ()))
            if not data.get("node_id"):
                identity = json.dumps(
                    {
                        "skill_id": data.get("skill_id"),
                        "depends_on": sorted(data["depends_on"]),
                        "priority": data.get("priority", 0),
                    },
                    sort_keys=True,
                )
                data["node_id"] = "node-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
            node = PlanNode(**data)
        prepared.append(node)
    prepared.sort(key=lambda item: (item.priority, item.node_id))
    normalized = tuple(
        PlanNode(
            node_id=item.node_id,
            skill_id=item.skill_id,
            depends_on=tuple(sorted(item.depends_on)),
            required=item.required,
            upstream_outputs=tuple(sorted(item.upstream_outputs)),
            priority=item.priority,
            timeout_ms=item.timeout_ms,
        )
        for item in prepared
    )
    normalized_reasons = {key: str(value) for key, value in sorted((reasons or {}).items())}
    payload = {
        "nodes": [item.model_dump() for item in normalized],
        "reasons": normalized_reasons,
        "confidence": confidence,
    }
    plan_id = (
        "plan-"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    )
    return ExecutionPlan(plan_id, normalized, normalized_reasons, confidence, budget, fallback_used)
