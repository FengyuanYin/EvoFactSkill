from __future__ import annotations

from evofact.core.budget_models import BudgetLimits
from evofact.core.dag_models import PlanNode
from evofact.core.models import RunBudget, SkillSpec, SkillUtility
from evofact.governance.dag_policy import require_valid_plan

from .plan_normalizer import normalize_plan


def contract_plan_from_decision(decision, skills, *, timeout_ms: int | None = None):
    by_id = {item.skill_id: item for item in skills}
    selected = [skill_id for skill_id in decision.selected_skill_ids if skill_id in by_id]
    available = [item for item in skills if item.skill_id not in selected]
    required_inputs = set()
    for skill_id in selected:
        contract = getattr(by_id[skill_id], "contract", None)
        if contract:
            required_inputs.update(value for value in contract.consumes if value != "sample_public")
    produced = {
        value
        for skill_id in selected
        for value in getattr(getattr(by_id[skill_id], "contract", None), "produces", ())
    }
    for missing in sorted(required_inputs - produced):
        provider = next(
            (
                item
                for item in sorted(available, key=lambda value: value.name)
                if missing in getattr(getattr(item, "contract", None), "produces", ())
            ),
            None,
        )
        if provider and len(selected) < decision.budget.max_skills:
            selected.insert(0, provider.skill_id)
            available.remove(provider)
            produced.update(provider.contract.produces)
    nodes = []
    node_ids = {skill_id: f"node-{index:03d}-{skill_id}" for index, skill_id in enumerate(selected)}
    for skill_id in selected:
        contract = getattr(by_id[skill_id], "contract", None)
        dependencies = []
        consumes = set(getattr(contract, "consumes", ())) - {"sample_public"}
        for upstream_id in selected:
            if upstream_id == skill_id:
                continue
            upstream_contract = getattr(by_id[upstream_id], "contract", None)
            if consumes & set(getattr(upstream_contract, "produces", ())):
                dependencies.append(node_ids[upstream_id])
        if contract is not None and not contract.allow_root and not dependencies:
            continue
        nodes.append(
            PlanNode(
                node_ids[skill_id],
                skill_id,
                tuple(sorted(dependencies)),
                timeout_ms=timeout_ms,
            )
        )
    return normalize_plan(
        nodes,
        reasons=decision.reasons,
        confidence=decision.confidence,
        budget=decision.budget,
        fallback_used=decision.fallback_used,
    )


class RulePlanner:
    def __init__(
        self,
        router,
        *,
        strict_serial: bool = False,
        limits: BudgetLimits = BudgetLimits(),
    ):
        self.router = router
        self.strict_serial = strict_serial
        self.limits = limits

    async def plan(
        self,
        sample: dict,
        skills: list[SkillSpec],
        utilities: dict[str, SkillUtility],
        budget: RunBudget,
    ):
        result = await self.router.route(sample, skills, utilities, budget)
        if self.strict_serial:
            from .legacy_adapter import routing_decision_to_plan

            plan = routing_decision_to_plan(
                result.value,
                strict_serial=True,
                timeout_ms=self.limits.call_timeout_ms,
            )
        else:
            plan = contract_plan_from_decision(
                result.value, skills, timeout_ms=self.limits.call_timeout_ms
            )
        require_valid_plan(plan, skills, limits=self.limits)
        return type(result)(plan, result.usage)
