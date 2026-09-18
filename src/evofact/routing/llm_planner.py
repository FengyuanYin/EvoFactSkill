from __future__ import annotations

from evofact.core.budget_models import BudgetLimits
from evofact.core.models import (
    RoutingDecision,
    RunBudget,
    SkillKind,
    SkillSpec,
    SkillStatus,
    SkillUtility,
    UsageRecord,
)
from evofact.governance.dag_policy import require_valid_plan
from evofact.runtime.backend import BackendResult

from .plan_normalizer import normalize_plan
from .rule_planner import contract_plan_from_decision


def _merge_usage(left: UsageRecord | None, right: UsageRecord) -> UsageRecord:
    """合并两次调用的资源记录，用于保留失败请求已实际产生的 usage。
    输入要求：`left`（UsageRecord | None）为已发生的模型调用记录，可为空；`right`（UsageRecord）为回退路径记录。
    输出：返回 `UsageRecord` 类型结果（逐字段求和）。"""
    if left is None:
        return right
    return UsageRecord(
        calls=left.calls + right.calls,
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        latency_ms=left.latency_ms + right.latency_ms,
        estimated_cost=left.estimated_cost + right.estimated_cost,
    )


class LLMPlanner:
    def __init__(self, backend, fallback, *, limits: BudgetLimits = BudgetLimits()):
        self.backend = backend
        self.fallback = fallback
        self.limits = limits

    async def plan(
        self,
        sample: dict,
        skills: list[SkillSpec],
        utilities: dict[str, SkillUtility],
        budget: RunBudget,
    ) -> BackendResult:
        candidates = tuple(
            item
            for item in skills
            if item.kind == SkillKind.SPECIALIST
            and item.status in {SkillStatus.ACTIVE, SkillStatus.FROZEN}
        )
        routers = [
            item
            for item in skills
            if item.kind == SkillKind.ROUTER
            and item.status in {SkillStatus.ACTIVE, SkillStatus.FROZEN}
        ]
        if not candidates or not routers:
            return await self._fallback(
                sample, skills, utilities, budget, "missing candidates or router"
            )
        # 只要模型调用已经发生，它的 usage 就必须被记账：否则预算台账会低估真实消耗。
        usage = UsageRecord()
        try:
            result = await self.backend.route(
                sample,
                candidates,
                sorted(routers, key=lambda item: item.name)[0],
                utilities,
                budget,
            )
            usage = result.usage
            raw = result.value
            if isinstance(raw, RoutingDecision):
                plan = contract_plan_from_decision(
                    raw, skills, timeout_ms=self.limits.call_timeout_ms
                )
            elif isinstance(raw, dict) and "nodes" in raw:
                allowed = {"nodes", "reasons", "confidence"}
                if set(raw) - allowed:
                    raise ValueError("unknown plan response fields")
                plan = normalize_plan(
                    raw["nodes"],
                    reasons=raw.get("reasons", {}),
                    confidence=float(raw.get("confidence", 0)),
                    budget=budget,
                    default_timeout_ms=self.limits.call_timeout_ms,
                    max_timeout_ms=self.limits.call_timeout_ms,
                )
            else:
                from evofact.routing.router import LLMSkillRouter

                decision = LLMSkillRouter._parse_decision(raw, candidates, budget)
                plan = contract_plan_from_decision(
                    decision, skills, timeout_ms=self.limits.call_timeout_ms
                )
            require_valid_plan(plan, skills, limits=self.limits)
            return BackendResult(plan, result.usage)
        except Exception as exc:
            return await self._fallback(
                sample,
                skills,
                utilities,
                budget,
                f"LLM plan failed: {type(exc).__name__}",
                usage,
            )

    async def _fallback(self, sample, skills, utilities, budget, cause, usage=None):
        result = await self.fallback.plan(sample, skills, utilities, budget)
        plan = result.value
        plan = normalize_plan(
            plan.nodes,
            reasons={**plan.reasons, "fallback": cause},
            confidence=plan.confidence,
            budget=budget,
            fallback_used=True,
            default_timeout_ms=self.limits.call_timeout_ms,
            max_timeout_ms=self.limits.call_timeout_ms,
        )
        require_valid_plan(plan, skills, limits=self.limits)
        return BackendResult(plan, _merge_usage(usage, result.usage))
