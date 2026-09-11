import hashlib

from evofact.core.models import InferenceTrace, RunBudget, Sample, SkillKind, SkillSpec, UsageRecord
from evofact.routing.router import SkillRouter, matches_scope

from .aggregator import aggregate_evidence


class InferenceRuntime:
    def __init__(self, backend, router: SkillRouter, skills: list[SkillSpec], utilities=None):
        """函数作用：创建并初始化 `InferenceRuntime` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `InferenceRuntime` 实例；`backend`（未显式标注）需符合函数签名约定；`router`（SkillRouter）需符合函数签名约定；`skills`（list[SkillSpec]）需符合函数签名约定；`utilities`（未显式标注，默认 `None`）需符合函数签名约定。
        输出：返回 `None`；初始化 `InferenceRuntime` 的实例状态，构造参数非法时可能抛出异常。"""
        self.backend = backend
        self.router = router
        self.skills = skills
        self.utilities = utilities or {}

    async def infer(self, sample: Sample, budget: RunBudget = RunBudget()) -> InferenceTrace:
        """函数作用：对单条样本执行技能路由、专家分析、证据聚合和最终判定，形成完整轨迹。
        输入要求：`self` 应为已初始化的 `InferenceRuntime` 实例；`sample`（Sample）需符合函数签名约定；`budget`（RunBudget，默认 `RunBudget()`）需符合函数签名约定。
        输出：异步返回 `InferenceTrace` 类型结果；校验或下游调用失败时异常向上传递。"""
        public = sample.public_view()
        decision = self.router.select(public, self.skills, self.utilities, budget)
        by_id = {s.skill_id: s for s in self.skills}
        reports = []
        usage = UsageRecord()
        errors = []
        for sid in (
            decision.selected_skill_ids
        ):  # 把skill 加载到上下文 执行分析得到result append到reports中
            try:
                result = await self.backend.analyze(public, by_id[sid])
                reports.append(result.value)
                usage = _merge(usage, result.usage)
            except Exception as exc:
                errors.append(f"specialist {sid}: {type(exc).__name__}: {exc}")
        judges = [
            s
            for s in self.skills
            if s.kind == SkillKind.JUDGE
            and s.status.value in {"active", "frozen"}
            and matches_scope(s, public)
        ]
        if not judges:
            raise RuntimeError("no active judge skill")
        result = await self.backend.judge(public, tuple(reports), judges[0])
        usage = _merge(usage, result.usage)
        trace_id = hashlib.sha256(
            f"{sample.sample_id}:{','.join(decision.selected_skill_ids)}".encode()
        ).hexdigest()[:20]
        return InferenceTrace(
            trace_id,
            public,
            decision,
            tuple(reports),
            result.value,
            {s.skill_id: s.version for s in self.skills},
            aggregated_evidence=aggregate_evidence(tuple(reports)),
            usage=usage,
            errors=tuple(errors),
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
