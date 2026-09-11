from dataclasses import replace

from evofact.core.models import SkillUtility


class UtilityTracker:
    def __init__(self, retire_after: int = 3):
        """函数作用：创建并初始化 `UtilityTracker` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `UtilityTracker` 实例；`retire_after`（int，默认 `3`）需符合函数签名约定。
        输出：返回 `None`；初始化 `UtilityTracker` 的实例状态，构造参数非法时可能抛出异常。"""
        self.values: dict[str, SkillUtility] = {}
        self.retire_after = retire_after

    def update(
        self,
        skill_id: str,
        *,
        success: bool,
        delta: float,
        cost: float,
        domain: str | None = None,
        window: str | None = None,
    ) -> SkillUtility:
        """函数作用：负责`UtilityTracker` 中的 `update` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `UtilityTracker` 实例；`skill_id`（str）需符合函数签名约定；`success`（bool）需以关键字传入并符合签名约定；`delta`（float）需以关键字传入并符合签名约定；`cost`（float）需以关键字传入并符合签名约定；`domain`（str | None，默认 `None`）需以关键字传入并符合签名约定；`window`（str | None，默认 `None`）需以关键字传入并符合签名约定。
        输出：返回 `SkillUtility` 类型结果；校验或下游调用失败时异常向上传递。"""
        old = self.values.get(skill_id, SkillUtility(skill_id))
        uses = old.uses + 1
        du = dict(old.domain_utility)
        tu = dict(old.temporal_utility)
        if domain:
            du[domain] = (du.get(domain, 0) * (uses - 1) + delta) / uses
        if window:
            tu[window] = (tu.get(window, 0) * (uses - 1) + delta) / uses
        value = replace(
            old,
            uses=uses,
            successes=old.successes + int(success),
            marginal_utility=(old.marginal_utility * (uses - 1) + delta) / uses,
            mean_cost=(old.mean_cost * (uses - 1) + cost) / uses,
            domain_utility=du,
            temporal_utility=tu,
            negative_transfer_count=old.negative_transfer_count + int(delta < 0),
        )
        self.values[skill_id] = value
        return value

    def recommendation(self, skill_id: str) -> str:
        """函数作用：负责`UtilityTracker` 中的 `recommendation` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `UtilityTracker` 实例；`skill_id`（str）需符合函数签名约定。
        输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
        value = self.values[skill_id]
        return (
            "retire"
            if value.negative_transfer_count >= self.retire_after and value.marginal_utility < 0
            else ("downweight" if value.marginal_utility < 0 else "keep")
        )
