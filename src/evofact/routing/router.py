import random

from evofact.core.models import (
    RoutingDecision,
    RunBudget,
    SkillKind,
    SkillSpec,
    SkillStatus,
    SkillUtility,
)


# 规则匹配的方式选择agent路由
class SkillRouter:
    """根据样本特征、技能历史效用和运行预算选择要调用的专家技能。"""

    def __init__(self, strategy: str = "utility-aware", seed: int = 42):
        """函数作用：创建并初始化 `SkillRouter` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `SkillRouter` 实例；`strategy`（str，默认 `'utility-aware'`）需符合函数签名约定；`seed`（int，默认 `42`）需符合函数签名约定。
        输出：返回 `None`；初始化 `SkillRouter` 的实例状态，构造参数非法时可能抛出异常。"""
        # strategy 决定选择规则；seed 仅供 random 策略生成可复现的结果。
        self.strategy = strategy
        self.seed = seed

    def select(
        self,
        sample: dict,
        skills: list[SkillSpec],
        utilities: dict[str, SkillUtility],
        budget: RunBudget,
    ) -> RoutingDecision:
        """函数作用：为单个样本选择技能，并返回选中项、拒绝项及路由元数据。
        输入要求：`self` 应为已初始化的 `SkillRouter` 实例；`sample`（dict）需符合函数签名约定；`skills`（list[SkillSpec]）需符合函数签名约定；`utilities`（dict[str, SkillUtility]）需符合函数签名约定；`budget`（RunBudget）需符合函数签名约定。
        输出：返回 `RoutingDecision` 类型结果；校验或下游调用失败时异常向上传递。"""

        # 先建立候选池：只允许处于可用状态、且适用范围匹配的专家技能。
        candidates = [
            s
            for s in skills
            if s.kind == SkillKind.SPECIALIST
            and s.status in {SkillStatus.ACTIVE, SkillStatus.FROZEN}
            and matches_scope(s, sample)
        ]

        if self.strategy == "all-experts":
            # 按输入顺序选取尽可能多的专家，但不超过预算上限。
            chosen = candidates[: budget.max_skills]
        elif self.strategy == "random":
            # 将全局 seed 与 sample_id 组合，使同一样本的随机结果可以复现。
            chosen = random.Random(f"{self.seed}:{sample.get('sample_id')}").sample(
                candidates, min(len(candidates), budget.max_skills)
            )
        elif self.strategy == "static":
            # 固定选择三个通用技能；候选池中不存在的技能会被自动忽略。
            chosen = [
                s
                for s in candidates
                if s.name in {"claim_decomposition", "evidence_assessment", "temporal_reasoning"}
            ][: budget.max_skills]
        else:
            # 默认使用效用感知策略：结合范围、文本触发词、历史收益和成本打分。
            text = str(sample.get("text", "")).casefold()
            domain = str(sample.get("domain") or "")

            def score(skill):
                """函数作用：负责当前模块中的 `score` 处理，封装调用方需要复用的业务步骤。
                输入要求：`skill`（未显式标注）需符合函数签名约定。
                输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
                # matches_scope 已过滤不匹配项；这里保留范围分，强调领域匹配程度。
                scope = 1 if not skill.scope.domains or domain in skill.scope.domains else -10
                # 文本每命中一个 trigger.pattern，就累加该触发器的权重。
                trigger = sum(t.weight for t in skill.triggers if t.pattern.casefold() in text)
                # 没有历史统计时，以全零的默认效用记录参与计算。
                u = utilities.get(skill.skill_id, SkillUtility(skill.skill_id))
                return (
                    scope
                    + trigger
                    + u.marginal_utility
                    - 0.1 * u.mean_cost
                    - 0.2 * u.negative_transfer_count
                )

            # 分数越高越优先；同分时按名称排序，保证结果稳定。
            chosen = sorted(candidates, key=lambda s: (-score(s), s.name))[: budget.max_skills]

        # 主策略未选出技能时，尝试使用声明拆解技能作为兜底。
        fallback = False
        if not chosen:
            chosen = [s for s in candidates if s.name == "claim_decomposition"][:1]
            fallback = True

        # 返回 ID 而不是完整 SkillSpec，并记录候选池中未被选择的技能。
        selected = tuple(s.skill_id for s in chosen)
        rejected = tuple(s.skill_id for s in candidates if s.skill_id not in selected)
        return RoutingDecision(
            selected,
            rejected,
            {s.skill_id: self.strategy for s in chosen},
            1.0 if chosen else 0.0,
            budget,
            fallback,
        )


def matches_scope(skill: SkillSpec, sample: dict) -> bool:
    """函数作用：检查样本的领域、数据集和时间窗口是否都落在技能的适用范围内。 某项范围为空表示该项不设限制；非空时则要求样本值包含在允许列表中。
    输入要求：`skill`（SkillSpec）需符合函数签名约定；`sample`（dict）需符合函数签名约定。
    输出：返回 `bool` 类型结果；校验或下游调用失败时异常向上传递。"""
    scope = skill.scope
    return (
        (not scope.domains or sample.get("domain") in scope.domains)
        and (not scope.datasets or sample.get("dataset") in scope.datasets)
        and (
            not scope.temporal_windows
            or str(sample.get("metadata", {}).get("temporal_window")) in scope.temporal_windows
        )
    )
