from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from evofact.config import AppConfig, BudgetConfig, ExecutionConfig
from evofact.core.budget_models import BudgetLimits, CostStatus, UsageDetails
from evofact.core.models import RunBudget, UsageRecord
from evofact.experiments.runner import ExperimentRunner, fixture_samples, load_seed_skills
from evofact.routing.llm_planner import LLMPlanner
from evofact.routing.router import LLMSkillRouter, SkillRouter
from evofact.routing.rule_planner import RulePlanner
from evofact.runtime.backend import BackendResult
from evofact.runtime.budget import BudgetManager
from evofact.runtime.inference import InferenceRuntime
from evofact.runtime.mock_backend import MockBackend
from evofact.runtime.node_runner import merge_usage_details

ROOT = Path(__file__).resolve().parents[1]


def _skills():
    return load_seed_skills(ROOT / "skills" / "seeds")


def _public_sample(sample_id: str = "budget-accounting") -> dict:
    return {
        "sample_id": sample_id,
        "dataset": "fixture",
        "domain": "social",
        "text": "官方通报：道路已经恢复通行。",
        "metadata": {},
    }


def test_inference_runtime_degrades_to_rule_routing_when_run_budget_is_exhausted() -> None:
    """run 级预算耗尽时，LLM 路由必须降级而不是让整条样本抛异常。"""
    sample = fixture_samples()[0]
    runtime = InferenceRuntime(
        MockBackend(),
        LLMSkillRouter(MockBackend(), SkillRouter(strategy="utility-aware", seed=42)),
        _skills(),
        budget_manager=BudgetManager(BudgetLimits(max_tokens_per_run=1)),
    )

    trace = asyncio.run(runtime.infer(sample, RunBudget(3)))

    assert trace.decision.label == "ABSTAIN"
    assert trace.specialist_reports == ()
    assert any("router budget denied" in error for error in trace.errors)


def test_runner_survives_run_budget_exhaustion_for_every_sample() -> None:
    """单次 run() 预算耗尽不能再终止整轮实验（原先会丢掉全部结果）。"""
    config = replace(
        AppConfig(),
        routing_strategy="llm",
        execution=ExecutionConfig(batch_size=4, max_concurrent_samples=2),
        budget=BudgetConfig(max_tokens_per_run=1),
    )
    runner = ExperimentRunner(config, ROOT)
    runner._backend = lambda: MockBackend()
    samples = fixture_samples()

    traces, result = asyncio.run(runner.run(samples))

    assert [trace.sample_id for trace in traces] == [sample.sample_id for sample in samples]
    assert [row.sample_id for row in result.per_sample] == [sample.sample_id for sample in samples]
    assert all(trace.decision.label == "ABSTAIN" for trace in traces)
    assert all(any("router budget denied" in error for error in trace.errors) for trace in traces)


def test_repeated_validation_inferences_get_separate_sample_call_budgets() -> None:
    sample = fixture_samples()[0]
    config = replace(AppConfig(), budget=BudgetConfig(max_calls_per_sample=5))
    runner = ExperimentRunner(config, ROOT)

    async def infer_twice():
        first, _ = await runner.run(
            [sample], task_name="validation", phase="baseline r1", isolate_sample_budget=True
        )
        second, _ = await runner.run(
            [sample], task_name="validation", phase="candidate r1", isolate_sample_budget=True
        )
        return first[0], second[0]

    first, second = asyncio.run(infer_twice())
    assert first.sample_id == second.sample_id == sample.sample_id
    assert first.decision.label != "ABSTAIN"
    assert second.decision.label != "ABSTAIN"
    assert first.usage.calls == second.usage.calls == 4
    assert runner._budget_manager().snapshot().calls_used == 8
    assert len(runner._budget_manager().export_state()["samples"]) == 2


def test_runner_keeps_later_batches_after_mid_run_exhaustion() -> None:
    """预算在一轮中途耗尽时，后续批次仍必须跑完，而不是整轮被取消。"""
    config = replace(
        AppConfig(),
        routing_strategy="llm",
        execution=ExecutionConfig(batch_size=4, max_concurrent_samples=2),
        budget=BudgetConfig(max_calls_per_run=5),
    )
    runner = ExperimentRunner(config, ROOT)
    runner._backend = lambda: MockBackend()
    samples = [
        replace(sample, sample_id=f"s-{index:02d}")
        for index, sample in enumerate(fixture_samples() * 3)
    ]

    traces, result = asyncio.run(runner.run(samples))

    assert [trace.sample_id for trace in traces] == [sample.sample_id for sample in samples]
    assert len(result.per_sample) == len(samples)
    # 第一条样本在预算耗尽前正常完成，说明耗尽确实发生在中途。
    assert not any("budget" in error for error in traces[0].errors)
    assert any(any("budget" in error for error in trace.errors) for trace in traces)


class _RejectedPlanBackend:
    """模型调用成功但返回非法计划，用于验证这次调用的 usage 不会丢失。"""

    def __init__(self) -> None:
        self.calls = 0

    async def route(self, sample, candidates, router_skill, utilities, budget) -> BackendResult:
        del sample, candidates, router_skill, utilities, budget
        self.calls += 1
        return BackendResult(
            {"nodes": [], "unexpected_field": 1},
            UsageRecord(calls=1, prompt_tokens=1234, completion_tokens=56, latency_ms=10.0),
        )


def test_llm_planner_keeps_usage_of_a_rejected_plan_response() -> None:
    """回退到规则规划时，已经发生的模型调用必须计入 usage 台账。"""
    backend = _RejectedPlanBackend()
    planner = LLMPlanner(backend, RulePlanner(SkillRouter(strategy="utility-aware", seed=42)))

    result = asyncio.run(
        planner.plan(
            _public_sample(),
            _skills(),
            {},
            RunBudget(max_skills=3, max_calls=6),
        )
    )

    assert backend.calls == 1
    assert result.value.fallback_used is True
    assert result.usage.calls == 1
    assert result.usage.prompt_tokens == 1234
    assert result.usage.completion_tokens == 56


_PRICED_DETAILS = UsageDetails(
    calls=1,
    input_tokens=100,
    output_tokens=50,
    latency_ms=12.0,
    cost=Decimal("0.0123"),
    cost_status=CostStatus.ESTIMATED,
    provider="deepseek",
    model="deepseek-flash",
    pricing_version="pricing_v1",
)


class _RoutableBackend:
    """Returns one valid plan so the LLM planning path succeeds instead of falling back."""

    def __init__(self, nodes: list[dict]) -> None:
        self.nodes = nodes

    async def route(self, sample, candidates, router_skill, utilities, budget) -> BackendResult:
        del sample, candidates, router_skill, utilities, budget
        return BackendResult(
            {"nodes": self.nodes, "reasons": {}, "confidence": 0.9},
            UsageRecord(calls=1, prompt_tokens=100, completion_tokens=50, estimated_cost=0.0123),
            _PRICED_DETAILS,
        )


def test_llm_planner_preserves_rich_usage_details() -> None:
    """router 必须把权威 usage_details 带出来：它保留了 provider/model/pricing_version。

    丢失它会让下游退回有损的 UsageRecord 转换，把"成本为零"和"成本未知"混为一谈。
    """
    skills = _skills()
    baseline = asyncio.run(
        RulePlanner(SkillRouter(strategy="utility-aware", seed=42)).plan(
            _public_sample(), skills, {}, RunBudget(max_skills=3, max_calls=6)
        )
    )
    nodes = [node.model_dump() for node in baseline.value.nodes]
    planner = LLMPlanner(
        _RoutableBackend(nodes), RulePlanner(SkillRouter(strategy="utility-aware", seed=42))
    )

    result = asyncio.run(
        planner.plan(_public_sample(), skills, {}, RunBudget(max_skills=3, max_calls=6))
    )

    assert result.value.fallback_used is False
    assert result.usage_details is _PRICED_DETAILS
    assert result.usage_details.pricing_version == "pricing_v1"
    assert result.usage_details.cost == Decimal("0.0123")


def test_merge_usage_details_ignores_nodes_that_made_no_call() -> None:
    """跳过/超时/失败节点贡献的空 usage 不能把整条轨迹标成成本未知。"""
    merged = merge_usage_details([_PRICED_DETAILS, UsageDetails()])

    assert merged.cost == Decimal("0.0123")
    assert merged.cost_status == CostStatus.ESTIMATED
    assert merged.calls == 1


def test_merge_usage_details_keeps_unavailable_cost_for_a_real_call() -> None:
    """真正发生过、但确实无法定价的调用仍必须让整条轨迹的成本不可用。"""
    unpriced = UsageDetails(calls=1, input_tokens=80, cost_status=CostStatus.UNAVAILABLE)

    merged = merge_usage_details([_PRICED_DETAILS, unpriced])

    assert merged.calls == 2
    assert merged.cost is None
    assert merged.cost_status == CostStatus.UNAVAILABLE
