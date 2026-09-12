from dataclasses import dataclass
from typing import Any, Protocol

from evofact.core.models import (
    RunBudget,
    SkillSpec,
    SkillUtility,
    SpecialistReport,
    UsageRecord,
)


@dataclass(frozen=True)
class BackendResult:
    value: Any
    usage: UsageRecord = UsageRecord()


class ModelBackend(Protocol):
    async def route(
        self,
        sample: dict[str, Any],
        candidates: tuple[SkillSpec, ...],
        router_skill: SkillSpec,
        utilities: dict[str, SkillUtility],
        budget: RunBudget,
    ) -> BackendResult:
        """让模型从合法候选技能中选择本次需要执行的专家技能。"""
        ...

    async def analyze(self, sample: dict[str, Any], skill: SkillSpec) -> BackendResult:
        """函数作用：负责`ModelBackend` 中的 `analyze` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `ModelBackend` 实例；`sample`（dict[str, Any]）需符合函数签名约定；`skill`（SkillSpec）需符合函数签名约定。
        输出：异步返回 `BackendResult` 类型结果；校验或下游调用失败时异常向上传递。"""
        ...

    async def judge(
        self, sample: dict[str, Any], reports: tuple[SpecialistReport, ...], skill: SkillSpec
    ) -> BackendResult:
        """函数作用：负责`ModelBackend` 中的 `judge` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `ModelBackend` 实例；`sample`（dict[str, Any]）需符合函数签名约定；`reports`（tuple[SpecialistReport, ...]）需符合函数签名约定；`skill`（SkillSpec）需符合函数签名约定。
        输出：异步返回 `BackendResult` 类型结果；校验或下游调用失败时异常向上传递。"""
        ...
