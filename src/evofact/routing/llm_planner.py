from __future__ import annotations

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


class LLMPlanner:
    def __init__(self, backend, fallback):
        self.backend = backend
        self.fallback = fallback

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
        try:
            result = await self.backend.route(
                sample,
                candidates,
                sorted(routers, key=lambda item: item.name)[0],
                utilities,
                budget,
            )
            raw = result.value
            if isinstance(raw, RoutingDecision):
                plan = contract_plan_from_decision(raw, skills)
            elif isinstance(raw, dict) and "nodes" in raw:
                allowed = {"nodes", "reasons", "confidence"}
                if set(raw) - allowed:
                    raise ValueError("unknown plan response fields")
                plan = normalize_plan(
                    raw["nodes"],
                    reasons=raw.get("reasons", {}),
                    confidence=float(raw.get("confidence", 0)),
                    budget=budget,
                )
            else:
                from evofact.routing.router import LLMSkillRouter

                decision = LLMSkillRouter._parse_decision(raw, candidates, budget)
                plan = contract_plan_from_decision(decision, skills)
            require_valid_plan(plan, skills)
            return BackendResult(plan, result.usage)
        except Exception as exc:
            return await self._fallback(
                sample, skills, utilities, budget, f"LLM plan failed: {type(exc).__name__}"
            )

    async def _fallback(self, sample, skills, utilities, budget, cause):
        result = await self.fallback.plan(sample, skills, utilities, budget)
        plan = result.value
        plan = normalize_plan(
            plan.nodes,
            reasons={**plan.reasons, "fallback": cause},
            confidence=plan.confidence,
            budget=budget,
            fallback_used=True,
        )
        require_valid_plan(plan, skills)
        return BackendResult(plan, UsageRecord(calls=result.usage.calls))
