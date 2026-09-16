from __future__ import annotations

from evofact.core.dag_models import ExecutionPlan, PlanNode
from evofact.core.models import RoutingDecision, RunBudget

from .plan_normalizer import normalize_plan


def selection_to_plan(
    selected_skill_ids: tuple[str, ...] | list[str],
    *,
    reasons: dict[str, str] | None = None,
    confidence: float = 1.0,
    budget: RunBudget = RunBudget(),
    strict_serial: bool = False,
    fallback_used: bool = False,
) -> ExecutionPlan:
    nodes = []
    previous: str | None = None
    for index, skill_id in enumerate(selected_skill_ids):
        node_id = f"legacy-{index:03d}-{skill_id}"
        dependencies = (previous,) if strict_serial and previous else ()
        nodes.append(PlanNode(node_id, skill_id, dependencies))
        previous = node_id
    return normalize_plan(
        nodes,
        reasons=reasons,
        confidence=confidence,
        budget=budget,
        fallback_used=fallback_used,
    )


def routing_decision_to_plan(
    decision: RoutingDecision, *, strict_serial: bool = False
) -> ExecutionPlan:
    return selection_to_plan(
        decision.selected_skill_ids,
        reasons=decision.reasons,
        confidence=decision.confidence,
        budget=decision.budget,
        strict_serial=strict_serial,
        fallback_used=decision.fallback_used,
    )
