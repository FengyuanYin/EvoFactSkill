from __future__ import annotations

from collections import defaultdict

from evofact.core.budget_models import BudgetLimits
from evofact.core.dag_models import ExecutionPlan, PlanFinding, PlanValidationReport
from evofact.core.models import SkillSpec


def validate_plan(
    plan: ExecutionPlan,
    skills: list[SkillSpec] | tuple[SkillSpec, ...] | dict[str, SkillSpec],
    *,
    limits: BudgetLimits = BudgetLimits(),
    available_capabilities: tuple[str, ...] = ("llm", "references", "template"),
) -> PlanValidationReport:
    by_skill = skills if isinstance(skills, dict) else {item.skill_id: item for item in skills}
    findings: list[PlanFinding] = []
    warnings: list[PlanFinding] = []
    nodes = {item.node_id: item for item in plan.nodes}
    if len(nodes) != len(plan.nodes):
        findings.append(PlanFinding("duplicate_node", "node IDs must be unique"))
    if len(plan.nodes) > limits.max_nodes:
        findings.append(PlanFinding("max_nodes", "plan exceeds maximum node count"))
    for node in plan.nodes:
        if node.skill_id not in by_skill:
            findings.append(
                PlanFinding("unknown_skill", "node references an unknown skill", node.node_id)
            )
        for dependency in node.depends_on:
            if dependency not in nodes:
                findings.append(
                    PlanFinding("dangling_dependency", "dependency does not exist", node.node_id)
                )

    indegree = {node_id: 0 for node_id in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for node in plan.nodes:
        for dependency in node.depends_on:
            if dependency in nodes:
                indegree[node.node_id] += 1
                children[dependency].append(node.node_id)
    ready = sorted(node_id for node_id, count in indegree.items() if count == 0)
    order: list[str] = []
    levels: list[tuple[str, ...]] = []
    while ready:
        current = tuple(ready)
        levels.append(current)
        next_ready: list[str] = []
        for node_id in current:
            order.append(node_id)
            for child in sorted(children[node_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_ready.append(child)
        ready = sorted(next_ready)
    if len(order) != len(nodes):
        findings.append(PlanFinding("cycle", "execution plan contains a dependency cycle"))
    if len(levels) > limits.max_depth:
        findings.append(PlanFinding("max_depth", "plan exceeds maximum depth"))

    for level in levels:
        if len(level) > 1:
            for node_id in level:
                skill = by_skill.get(nodes[node_id].skill_id)
                contract = getattr(skill, "contract", None) if skill else None
                if contract is not None and not contract.allow_parallel:
                    findings.append(
                        PlanFinding(
                            "parallel_forbidden", "skill forbids parallel execution", node_id
                        )
                    )
    for node in plan.nodes:
        skill = by_skill.get(node.skill_id)
        contract = getattr(skill, "contract", None) if skill else None
        if contract is None:
            continue
        missing_capabilities = set(contract.requires_capabilities) - set(available_capabilities)
        if missing_capabilities:
            findings.append(
                PlanFinding(
                    "capability_mismatch",
                    f"missing runtime capabilities: {sorted(missing_capabilities)}",
                    node.node_id,
                )
            )
        missing_optional = set(contract.optional_capabilities) - set(available_capabilities)
        if missing_optional:
            warnings.append(
                PlanFinding(
                    "optional_capability_unavailable",
                    f"optional runtime capabilities unavailable: {sorted(missing_optional)}",
                    node.node_id,
                )
            )
        if not node.depends_on and not contract.allow_root:
            findings.append(
                PlanFinding("root_forbidden", "skill cannot be a root node", node.node_id)
            )
        if contract.consumes:
            produced = set()
            for dependency in node.depends_on:
                upstream = by_skill.get(nodes[dependency].skill_id) if dependency in nodes else None
                upstream_contract = getattr(upstream, "contract", None) if upstream else None
                if upstream_contract:
                    produced.update(upstream_contract.produces)
            missing = set(contract.consumes) - produced - {"sample_public"}
            if missing:
                findings.append(
                    PlanFinding(
                        "contract_mismatch",
                        f"missing upstream outputs: {sorted(missing)}",
                        node.node_id,
                    )
                )
    required = sum(1 for node in plan.nodes if node.required)
    if required + limits.judge_reserved_calls > plan.budget.max_calls:
        findings.append(
            PlanFinding("budget_infeasible", "required nodes leave no reserved Judge call")
        )
    return PlanValidationReport(
        valid=not findings,
        plan=plan if not findings else None,
        topological_order=tuple(order),
        levels=tuple(levels),
        depth=len(levels),
        findings=tuple(findings),
        warnings=tuple(warnings),
    )


def require_valid_plan(
    plan: ExecutionPlan,
    skills,
    *,
    limits: BudgetLimits = BudgetLimits(),
    available_capabilities: tuple[str, ...] = ("llm", "references", "template"),
) -> PlanValidationReport:
    report = validate_plan(
        plan,
        skills,
        limits=limits,
        available_capabilities=available_capabilities,
    )
    if not report.valid:
        detail = "; ".join(f"{item.code}: {item.message}" for item in report.findings)
        raise ValueError(f"invalid execution plan: {detail}")
    return report
