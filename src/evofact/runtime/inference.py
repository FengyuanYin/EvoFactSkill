import asyncio
import hashlib
from dataclasses import replace

from evofact.core.budget_models import BudgetLimits, BudgetRequest
from evofact.core.dag_models import ExecutionPlan, NodeStatus
from evofact.core.label_models import RUNTIME_ABSTAIN_LABEL, DecisionOrigin
from evofact.core.models import (
    InferenceTrace,
    Prediction,
    RoutingDecision,
    RunBudget,
    Sample,
    SkillKind,
    SkillSpec,
    UsageRecord,
)
from evofact.data.label_registry import LabelContractRegistry, fixture_binary_contract
from evofact.routing.legacy_adapter import routing_decision_to_plan
from evofact.routing.router import LLMSkillRouter, SkillRouter, matches_scope

from .aggregator import aggregate_evidence
from .backend import BackendResult, JudgeContractError
from .budget import BudgetExceeded
from .dag_executor import DAGExecutor
from .node_runner import _usage_details, merge_usage_details
from .resource_runtime import ResourceRuntime


class InferenceRuntime:
    def __init__(
        self,
        backend,
        router: SkillRouter | LLMSkillRouter,
        skills: list[SkillSpec],
        utilities=None,
        *,
        budget_manager=None,
        label_contract_registry: LabelContractRegistry | None = None,
    ):
        """函数作用：创建并初始化 `InferenceRuntime` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `InferenceRuntime` 实例；`backend`（未显式标注）需符合函数签名约定；`router`（SkillRouter）需符合函数签名约定；`skills`（list[SkillSpec]）需符合函数签名约定；`utilities`（未显式标注，默认 `None`）需符合函数签名约定。
        输出：返回 `None`；初始化 `InferenceRuntime` 的实例状态，构造参数非法时可能抛出异常。"""
        self.backend = backend
        self.router = router
        self.skills = skills
        self.utilities = utilities or {}
        self.budget_manager = budget_manager
        self.label_contract_registry = label_contract_registry or LabelContractRegistry(
            (fixture_binary_contract(),)
        )

    async def infer(self, sample: Sample, budget: RunBudget = RunBudget()) -> InferenceTrace:
        """函数作用：对单条样本执行技能路由、专家分析、证据聚合和最终判定，形成完整轨迹。
        输入要求：`self` 应为已初始化的 `InferenceRuntime` 实例；`sample`（Sample）需符合函数签名约定；`budget`（RunBudget，默认 `RunBudget()`）需符合函数签名约定。
        输出：异步返回 `InferenceTrace` 类型结果；校验或下游调用失败时异常向上传递。"""
        contract = self.label_contract_registry.resolve_sample(sample)
        if sample.label is not None:
            contract.normalize(sample.label)
        public = sample.public_view()
        scope_view = {
            "dataset": sample.dataset,
            "domain": sample.domain,
            "metadata": {"temporal_window": sample.metadata.get("temporal_window")},
        }
        eligible_skills = [skill for skill in self.skills if matches_scope(skill, scope_view)]
        evidence_available = bool(sample.evidence)

        route_reservation = None
        budget_denied: str | None = None
        try:
            if self.budget_manager is not None and isinstance(self.router, LLMSkillRouter):
                try:
                    route_reservation = await self.budget_manager.reserve(
                        sample.sample_id,
                        BudgetRequest(calls=1, tokens=1000, purpose="router"),
                    )
                except BudgetExceeded as exc:
                    # 预算耗尽不应让整条样本失败：退化为零模型调用的规则路由，
                    # 与 specialist 的 SKIPPED_BUDGET、judge 的 ABSTAIN 保持一致的降级语义。
                    budget_denied = str(exc)

            if budget_denied is not None:
                routing_result = await self._rule_routing_result(public, eligible_skills, budget)
            else:

                async def invoke_router():
                    if hasattr(self.router, "plan"):
                        return await self.router.plan(
                            public, eligible_skills, self.utilities, budget
                        )
                    return await self.router.route(public, eligible_skills, self.utilities, budget)

                if self.budget_manager is not None and isinstance(self.router, LLMSkillRouter):
                    async with self.budget_manager.concurrency(sample.sample_id):
                        routing_result = await asyncio.wait_for(
                            invoke_router(),
                            self.budget_manager.limits.call_timeout_ms / 1000,
                        )
                else:
                    routing_result = await invoke_router()
            if route_reservation is not None:
                await self.budget_manager.reconcile(
                    route_reservation,
                    _usage_details(routing_result),
                )
                route_reservation = None
        finally:
            if route_reservation is not None:
                await self.budget_manager.release(route_reservation)

        if isinstance(routing_result.value, RoutingDecision):
            decision = routing_result.value
            plan = routing_decision_to_plan(
                decision,
                timeout_ms=(
                    self.budget_manager.limits.call_timeout_ms
                    if self.budget_manager is not None
                    else None
                ),
            )
        elif isinstance(routing_result.value, ExecutionPlan):
            plan = routing_result.value
            selected = tuple(node.skill_id for node in plan.nodes)
            decision = RoutingDecision(
                selected,
                tuple(
                    skill.skill_id
                    for skill in eligible_skills
                    if skill.kind == SkillKind.SPECIALIST and skill.skill_id not in selected
                ),
                plan.reasons,
                plan.confidence,
                plan.budget,
                plan.fallback_used,
            )
        else:
            raise TypeError("router must return a RoutingDecision or ExecutionPlan")

        if not evidence_available:
            skills_by_id = {skill.skill_id: skill for skill in eligible_skills}
            plan = replace(
                plan,
                nodes=tuple(
                    replace(node, required=False)
                    if getattr(
                        getattr(skills_by_id.get(node.skill_id), "report_contract", None),
                        "report_type",
                        None,
                    )
                    == "evidence_assessment"
                    else node
                    for node in plan.nodes
                ),
            )

        detailed_usage = [_usage_details(routing_result)]

        executor = DAGExecutor(
            self.backend,
            eligible_skills,
            resource_runtime=ResourceRuntime(),
            budget_manager=self.budget_manager,
            plan_limits=(
                self.budget_manager.limits if self.budget_manager is not None else BudgetLimits()
            ),
            sample_timeout_ms=(
                self.budget_manager.limits.sample_timeout_ms
                if self.budget_manager is not None
                else None
            ),
        )
        execution = await executor.execute(plan, public, sample_id=sample.sample_id)
        reports = [
            item.report
            for item in execution.nodes
            if item.status == NodeStatus.SUCCEEDED and item.report is not None
        ]
        errors = list(execution.summary.errors)
        forbidden_finding_types = {label.casefold() for label in contract.allowed_labels}
        reports_valid = not any(
            finding.finding_type.casefold() in forbidden_finding_types
            for report in reports
            for finding in report.findings
        )
        if not reports_valid:
            errors.append("specialist report used a reserved business-label finding type")
        if budget_denied is not None:
            errors.insert(0, f"router budget denied: {budget_denied}")
        detailed_usage.append(execution.usage_details)
        judges = [
            s
            for s in eligible_skills
            if s.kind == SkillKind.JUDGE and s.status.value in {"active", "frozen"}
        ]
        if not judges:
            raise RuntimeError("no active judge skill")
        if execution.summary.required_complete and reports_valid:
            judge_reservation = None
            try:

                async def invoke_judge():
                    return await self.backend.judge(
                        public,
                        tuple(reports),
                        judges[0],
                        label_contract=contract,
                        execution_summary=execution.summary,
                    )

                if self.budget_manager is not None:
                    async with self.budget_manager.concurrency(sample.sample_id):
                        judge_reservation = await self.budget_manager.reserve(
                            sample.sample_id,
                            BudgetRequest(calls=1, tokens=1000, purpose="judge"),
                            judge=True,
                        )
                        result = await asyncio.wait_for(
                            invoke_judge(),
                            self.budget_manager.limits.call_timeout_ms / 1000,
                        )
                else:
                    result = await invoke_judge()
                if self.budget_manager is not None:
                    await self.budget_manager.reconcile(
                        judge_reservation,
                        _usage_details(result),
                    )
                    judge_reservation = None
                prediction = result.value
                detailed_usage.append(_usage_details(result))
                try:
                    contract.require_label(prediction.label)
                except ValueError as exc:
                    raise JudgeContractError(str(exc)) from exc
            except BudgetExceeded:
                prediction = Prediction(
                    RUNTIME_ABSTAIN_LABEL,
                    0.0,
                    "Judge budget unavailable",
                    DecisionOrigin.RUNTIME,
                )
            except asyncio.TimeoutError:
                errors.append("judge timeout")
                prediction = Prediction(
                    RUNTIME_ABSTAIN_LABEL,
                    0.0,
                    "Judge timed out",
                    DecisionOrigin.RUNTIME,
                )
            except JudgeContractError as exc:
                errors.append(f"judge contract violation: {exc}")
                prediction = Prediction(
                    RUNTIME_ABSTAIN_LABEL,
                    0.0,
                    "Judge returned an invalid dataset label",
                    DecisionOrigin.RUNTIME,
                )
            finally:
                if judge_reservation is not None and self.budget_manager is not None:
                    await self.budget_manager.release(judge_reservation)
        else:
            prediction = Prediction(
                RUNTIME_ABSTAIN_LABEL,
                0.0,
                "required specialist reports are incomplete",
                DecisionOrigin.RUNTIME,
            )
        trace_id = hashlib.sha256(
            f"{sample.sample_id}:{','.join(decision.selected_skill_ids)}:{contract.digest}".encode()
        ).hexdigest()[:20]
        return InferenceTrace(
            trace_id=trace_id,
            sample_id=sample.sample_id,
            sample_public=public,
            routing=decision,
            specialist_reports=tuple(reports),
            decision=prediction,
            skill_versions={s.skill_id: s.package_digest or s.version for s in self.skills},
            aggregated_evidence=aggregate_evidence(tuple(reports)),
            usage=merge_usage_details(detailed_usage),
            errors=tuple(errors),
            execution_plan=plan,
            node_executions=execution.nodes,
            execution_summary=execution.summary,
            label_schema_id=contract.schema_id,
            label_contract_digest=contract.digest,
        )

    async def _rule_routing_result(
        self,
        public: dict,
        eligible_skills: list[SkillSpec],
        budget: RunBudget,
    ) -> BackendResult:
        """预算不可用时退化为规则路由，保证不再产生任何未记账的模型调用。
        输入要求：`self` 应为已初始化的 `InferenceRuntime` 实例；`public`（dict）需符合函数签名约定；`eligible_skills`（list[SkillSpec]）需符合函数签名约定；`budget`（RunBudget）需符合函数签名约定。
        输出：返回 `BackendResult`；规则回退本身也不可用时返回空计划的 `BackendResult`。"""
        fallback = getattr(self.router, "fallback", None)
        if fallback is not None:
            try:
                return await fallback.plan(public, eligible_skills, self.utilities, budget)
            except Exception:
                # 规则回退失败时继续走空计划：后续节点全部跳过、judge 弃权，而不是抛异常终止整轮。
                pass
        return BackendResult(
            RoutingDecision((), (), {"router": "budget unavailable"}, 0.0, budget, True),
            UsageRecord(),
        )
