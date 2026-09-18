from __future__ import annotations

import asyncio

import pytest

from evofact.core.budget_models import BudgetLimits
from evofact.core.dag_models import NodeStatus, PlanNode
from evofact.core.models import RunBudget, SkillKind, SkillScope, SkillSpec, SkillStatus
from evofact.core.package_models import SkillContract
from evofact.governance.dag_policy import validate_plan
from evofact.routing.plan_normalizer import normalize_plan
from evofact.routing.router import SkillRouter
from evofact.routing.rule_planner import RulePlanner
from evofact.runtime.dag_executor import DAGExecutor
from evofact.runtime.mock_backend import MockBackend


def _skill(name: str, consumes=(), produces=()):
    return SkillSpec(
        name,
        name,
        SkillKind.SPECIALIST,
        "0.1.0",
        SkillStatus.ACTIVE,
        name,
        scope=SkillScope(),
        contract=SkillContract(
            tuple(consumes),
            tuple(produces),
            allow_root=not consumes or consumes == ("sample_public",),
        ),
    )


def _plan():
    return normalize_plan(
        [
            PlanNode("a", "a"),
            PlanNode("b", "b", ("a",)),
            PlanNode("c", "c", ("a",)),
            PlanNode("d", "d", ("b", "c")),
        ],
        budget=RunBudget(max_skills=4, max_calls=6),
    )


def _skills():
    return [
        _skill("a", ("sample_public",), ("claims",)),
        _skill("b", ("claims",), ("b_report",)),
        _skill("c", ("claims",), ("c_report",)),
        _skill("d", ("b_report", "c_report"), ("final_report",)),
    ]


def test_dag_validation_computes_stable_levels() -> None:
    report = validate_plan(_plan(), _skills())
    assert report.valid
    assert report.levels == (("a",), ("b", "c"), ("d",))


def test_dag_executes_ready_nodes_concurrently_and_passes_upstream() -> None:
    backend = MockBackend(delays={"b": 0.03, "c": 0.03})
    result = asyncio.run(DAGExecutor(backend, _skills()).execute(_plan(), {"text": "claim"}))

    assert backend.max_active_calls == 2
    assert [item.status for item in result.nodes] == [NodeStatus.SUCCEEDED] * 4
    assert len(result.nodes[-1].report.claims) >= 3
    assert result.summary.required_complete


def test_failure_skips_only_dependent_branch() -> None:
    backend = MockBackend(failures={"c"})
    result = asyncio.run(DAGExecutor(backend, _skills()).execute(_plan(), {"text": "claim"}))
    states = {item.node_id: item.status for item in result.nodes}

    assert states["a"] == NodeStatus.SUCCEEDED
    assert states["b"] == NodeStatus.SUCCEEDED
    assert states["c"] == NodeStatus.FAILED
    assert states["d"] == NodeStatus.SKIPPED_DEPENDENCY
    assert not result.summary.required_complete


def test_cycle_is_rejected_before_execution() -> None:
    plan = normalize_plan(
        [PlanNode("a", "a", ("b",)), PlanNode("b", "b", ("a",))],
        budget=RunBudget(max_skills=2),
    )
    report = validate_plan(plan, _skills())
    assert not report.valid
    assert "cycle" in {item.code for item in report.findings}


def test_rule_planner_uses_configured_limits_and_timeout() -> None:
    limits = BudgetLimits(max_nodes=2, call_timeout_ms=1234)
    planner = RulePlanner(SkillRouter("all-experts"), limits=limits)
    with pytest.raises(ValueError, match="max_nodes"):
        asyncio.run(planner.plan({"text": "claim"}, _skills(), {}, RunBudget(max_skills=4)))

    allowed = RulePlanner(
        SkillRouter("all-experts"),
        limits=BudgetLimits(max_nodes=4, call_timeout_ms=1234),
    )
    result = asyncio.run(allowed.plan({"text": "claim"}, _skills(), {}, RunBudget(max_skills=4)))
    assert all(node.timeout_ms == 1234 for node in result.value.nodes)
