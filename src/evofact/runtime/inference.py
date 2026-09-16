import asyncio
import hashlib

from evofact.core.budget_models import BudgetRequest
from evofact.core.dag_models import ExecutionPlan, NodeStatus
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
from evofact.routing.legacy_adapter import routing_decision_to_plan
from evofact.routing.router import LLMSkillRouter, SkillRouter, matches_scope

from .aggregator import aggregate_evidence
from .budget import BudgetExceeded
from .dag_executor import DAGExecutor
from .node_runner import _usage_details
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
    ):
        """函数作用：创建并初始化 `InferenceRuntime` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `InferenceRuntime` 实例；`backend`（未显式标注）需符合函数签名约定；`router`（SkillRouter）需符合函数签名约定；`skills`（list[SkillSpec]）需符合函数签名约定；`utilities`（未显式标注，默认 `None`）需符合函数签名约定。
        输出：返回 `None`；初始化 `InferenceRuntime` 的实例状态，构造参数非法时可能抛出异常。"""
        self.backend = backend
        self.router = router
        self.skills = skills
        self.utilities = utilities or {}
        self.budget_manager = budget_manager

    async def infer(self, sample: Sample, budget: RunBudget = RunBudget()) -> InferenceTrace:
        """函数作用：对单条样本执行技能路由、专家分析、证据聚合和最终判定，形成完整轨迹。
        输入要求：`self` 应为已初始化的 `InferenceRuntime` 实例；`sample`（Sample）需符合函数签名约定；`budget`（RunBudget，默认 `RunBudget()`）需符合函数签名约定。
        输出：异步返回 `InferenceTrace` 类型结果；校验或下游调用失败时异常向上传递。"""
        public = sample.public_view()
        scope_view = {
            "dataset": sample.dataset,
            "domain": sample.domain,
            "metadata": {"temporal_window": sample.metadata.get("temporal_window")},
        }
        eligible_skills = [skill for skill in self.skills if matches_scope(skill, scope_view)]

        route_reservation = None
        try:
            if self.budget_manager is not None and isinstance(self.router, LLMSkillRouter):
                route_reservation = await self.budget_manager.reserve(
                    sample.sample_id,
                    BudgetRequest(calls=1, tokens=1000, purpose="router"),
                )

            async def invoke_router():
                if hasattr(self.router, "plan"):
                    return await self.router.plan(public, eligible_skills, self.utilities, budget)
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
            plan = routing_decision_to_plan(decision)
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

        usage = _merge(
            UsageRecord(),
            routing_result.usage,
        )

        executor = DAGExecutor(
            self.backend,
            eligible_skills,
            resource_runtime=ResourceRuntime(),
            budget_manager=self.budget_manager,
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
        usage = _merge(usage, execution.usage)
        judges = [
            s
            for s in eligible_skills
            if s.kind == SkillKind.JUDGE and s.status.value in {"active", "frozen"}
        ]
        if not judges:
            raise RuntimeError("no active judge skill")
        if execution.summary.required_complete:
            judge_reservation = None
            try:
                if self.budget_manager is not None:
                    judge_reservation = await self.budget_manager.reserve(
                        sample.sample_id,
                        BudgetRequest(calls=1, tokens=1000, purpose="judge"),
                        judge=True,
                    )

                async def invoke_judge():
                    try:
                        return await self.backend.judge(
                            public,
                            tuple(reports),
                            judges[0],
                            execution_summary=execution.summary,
                        )
                    except TypeError:
                        return await self.backend.judge(public, tuple(reports), judges[0])

                if self.budget_manager is not None:
                    async with self.budget_manager.concurrency(sample.sample_id):
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
                usage = _merge(usage, result.usage)
            except BudgetExceeded:
                prediction = Prediction("ABSTAIN", 0.0, "Judge budget unavailable")
            except asyncio.TimeoutError:
                errors.append("judge timeout")
                prediction = Prediction("ABSTAIN", 0.0, "Judge timed out")
            finally:
                if judge_reservation is not None and self.budget_manager is not None:
                    await self.budget_manager.release(judge_reservation)
        else:
            prediction = Prediction("ABSTAIN", 0.0, "required specialist reports are incomplete")
        trace_id = hashlib.sha256(
            f"{sample.sample_id}:{','.join(decision.selected_skill_ids)}".encode()
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
            usage=usage,
            errors=tuple(errors),
            execution_plan=plan,
            node_executions=execution.nodes,
            execution_summary=execution.summary,
        )


def _merge(a: UsageRecord, b: UsageRecord) -> UsageRecord:
    """函数作用：负责当前模块中的 `_merge` 处理，封装调用方需要复用的业务步骤。
    输入要求：`a`（UsageRecord）需符合函数签名约定；`b`（UsageRecord）需符合函数签名约定。
    输出：返回 `UsageRecord` 类型结果；校验或下游调用失败时异常向上传递。"""
    return UsageRecord(
        a.calls + b.calls,
        a.prompt_tokens + b.prompt_tokens,
        a.completion_tokens + b.completion_tokens,
        a.latency_ms + b.latency_ms,
        a.estimated_cost + b.estimated_cost,
    )
