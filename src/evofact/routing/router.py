import random
from dataclasses import replace

from evofact.core.models import (
    RoutingDecision,
    RunBudget,
    SkillKind,
    SkillSpec,
    SkillStatus,
    SkillUtility,
    UsageRecord,
)
from evofact.runtime.backend import BackendResult, ModelBackend


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

    async def route(
        self,
        sample: dict,
        skills: list[SkillSpec],
        utilities: dict[str, SkillUtility],
        budget: RunBudget,
    ) -> BackendResult:

        decision = self.select(
            sample,
            skills,
            utilities,
            budget
        )

        return BackendResult(
            decision,
            UsageRecord(),
        )


class LLMSkillRouter:
    def __init__(
        self,
        backend: ModelBackend,
        fallback: SkillRouter | None = None,
    ):
        self.backend = backend
        self.fallback = (
            fallback
            if fallback is not None
            else SkillRouter("utility-aware")
        )

    async def route(
        self,
        sample: dict,
        skills: list[SkillSpec],
        utilities: dict[str, SkillUtility],
        budget: RunBudget,
    ) -> BackendResult:
        candidates = tuple(
            skill
            for skill in skills
            if skill.kind == SkillKind.SPECIALIST
            and skill.status
            in {
                SkillStatus.ACTIVE,
                SkillStatus.FROZEN,
            }
            and matches_scope(skill, sample)
        )

        if not candidates:
            decision = self._fallback_decision(
                sample,
                skills,
                utilities,
                budget,
                "no legal specialist candidates",
            )
            return BackendResult(
                decision,
                UsageRecord(),
            )

        router_skills = sorted(
        (
            skill
            for skill in skills
            if skill.kind == SkillKind.ROUTER
            and skill.status
            in {
                SkillStatus.ACTIVE,
                SkillStatus.FROZEN,
            }
            and matches_scope(skill, sample)
        ),
            key = lambda skill:skill.name
        )

        if not router_skills:
            decision = self._fallback_decision(
                sample,
                skills,
                utilities,
                budget,
                "no active router skill",
            )
            return BackendResult(
                decision,
                UsageRecord(),
            )

        router_skill = router_skills[0]

        try:
            result = await self.backend.route(
                sample,
                candidates,
                router_skill,
                utilities,
                budget,
            )

            decision = self._parse_decision(
                result.value,
                candidates,
                budget,
            )

            return BackendResult(
                decision,
                result.usage,
            )
        except Exception as exc:
            decision = self._fallback_decision(
                sample,
                skills,
                utilities,
                budget,
                f"LLM router failed: {type(exc).__name__}",
            )

            # 这里只能粗略记录一次失败调用。
            return BackendResult(
                decision,
                UsageRecord(calls=1),
            )

    @staticmethod
    def _parse_decision(
        data: object,
        candidates: tuple[SkillSpec, ...],
        budget: RunBudget,
    ) -> RoutingDecision:
        if not isinstance(data, dict):
            raise ValueError(
                "router response must be a JSON object"
            )
        raw_ids = data.get("selected_skill_ids")

        if not isinstance(raw_ids, list):
            raise ValueError(
                "selected_skill_ids must be a list"
            )

        if not all(
            isinstance(skill_id, str)
            for skill_id in raw_ids
        ):
            raise ValueError(
                "selected_skill_ids must contain strings"
            )

        selected = tuple(dict.fromkeys(raw_ids))
        if not selected:
            raise ValueError(
                "LLM router selected no skills"
            )


        if len(selected) > budget.max_skills:
            raise ValueError(
                "LLM router exceeded max_skills"
            )

        allowed = {
            skill.skill_id: skill
            for skill in candidates
        }
        unknown = [
            skill_id
            for skill_id in selected
            if skill_id not in allowed
        ]
        if unknown:
            raise ValueError(
                f"LLM router returned unknown skills: "
                f"{unknown}"
            )

        raw_confidence = data.get("confidence", 0.0)
        if isinstance(raw_confidence, bool):
            raise ValueError(
                "router confidence must be numeric"
            )

        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "router confidence must be numeric"
            ) from exc

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "router confidence must be in [0, 1]"
            )

        raw_reasons = data.get("reasons", {})
        if not isinstance(raw_reasons, dict):
            raw_reasons = {}

        reasons = {
            skill_id: str(
                raw_reasons.get(
                    skill_id,
                    "selected by LLM router",
                )
            )
            for skill_id in selected
        }

        rejected = tuple(
            skill.skill_id
            for skill in candidates
            if skill.skill_id not in selected
        )

        return RoutingDecision(
            selected_skill_ids=selected,
            rejected_skill_ids=rejected,
            reasons=reasons,
            confidence=confidence,
            budget=budget,
            fallback_used=False,
        )

    def _fallback_decision(
            self,
            sample: dict,
            skills: list[SkillSpec],
            utilities: dict[str, SkillUtility],
            budget: RunBudget,
            cause: str,
    ) -> RoutingDecision:

        decision = self.fallback.select(
            sample,
            skills,
            utilities,
            budget,
        )

        reasons = dict(decision.reasons)

        for skill_id in decision.selected_skill_ids:
            previous = reasons.get(
                skill_id,
                self.fallback.strategy,
            )
            reasons[skill_id] = (
                f"{previous}; fallback reason: {cause}"
            )

        return replace(
            decision,
            reasons = reasons,
            fallback_used = True,
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

async def _router_demo(
        *,
        live: bool = False,
        config_path: str = "configs/adversarial_llm.yaml",
) -> None:
    import json
    from pathlib import Path

    from evofact.config import load_config
    from evofact.runtime.mock_backend import MockBackend
    from evofact.runtime.openai_backend import OpenAICompatibleBackend
    from evofact.skills.loader import load_skill_package

    project_root = Path(__file__).resolve().parents[3]
    seed_root = project_root / "skills" / "seeds"

    skills = [
        load_skill_package(path)
        for path in sorted(seed_root.iterdir())
        if (path / "SKILL.md").is_file()
    ]

    if live:
        path = Path(config_path)

        if not path.is_absolute():
            path = project_root / path

        config = load_config(path)

        backend = OpenAICompatibleBackend(
            config.base_url,
            config.resolved_api_key(),
            config.model,
        )

        mode = "live LLM"

    else:
        backend = MockBackend()
        mode = "mock"


    router = LLMSkillRouter(
        backend=backend,
        fallback=SkillRouter(
            strategy="utility-aware",
            seed = 42,
        ),
    )


    budget = RunBudget(
        max_skills=3,
        max_calls=6,
        max_tokens=4000,
    )

    samples = [
        {
            "sample_id": "router-test-1",
            "dataset": "manual",
            "domain": "finance",
            "text": (
                "官方报告称，2025年该公司的收入"
                "同比增长了30%。"
            ),
            "metadata": {
                "temporal_window": "2025",
            },
        },
        {
            "sample_id": "router-test-2",
            "dataset": "manual",
            "domain": "health",
            "text": (
                "网传某种饮料可以治愈所有疾病，"
                "但没有提供临床研究证据。"
            ),
            "metadata": {},
        },
        {
            "sample_id": "router-test-3",
            "dataset": "manual",
            "domain": "science",
            "text": (
                "两家研究机构对同一实验给出了"
                "互相矛盾的结论。"
            ),
            "metadata": {},
        },
    ]

    skill_names = {
        skill.skill_id: skill.name
        for skill in skills
    }

    print(f"\nRouter mode: {mode}")
    print(f"Loaded skills: {len(skills)}")
    print(
        "Router skills:",
        [
            skill.name
            for skill in skills
            if skill.kind == SkillKind.ROUTER
        ],
    )

    for sample in samples:
        result = await router.route(
            sample,
            skills,
            {},
            budget,
        )

        decision = result.value

        output = {
            "sample_id": sample["sample_id"],
            "text": sample["text"],
            "selected": [
                {
                    "skill_id": skill_id,
                    "name": skill_names.get(
                        skill_id,
                        "<unknown>",
                    ),
                    "reason": decision.reasons.get(
                        skill_id,
                        "",
                    ),
                }
                for skill_id
                in decision.selected_skill_ids
            ],
            "rejected_names": [
                skill_names.get(
                    skill_id,
                    "<unknown>",
                )
                for skill_id
                in decision.rejected_skill_ids
            ],
            "confidence": decision.confidence,
            "fallback_used": decision.fallback_used,
            "router_usage": {
                "calls": result.usage.calls,
                "prompt_tokens":
                    result.usage.prompt_tokens,
                "completion_tokens":
                    result.usage.completion_tokens,
                "latency_ms":
                    result.usage.latency_ms,
                "estimated_cost":
                    result.usage.estimated_cost,
            },
        }

        print(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            )
        )

    if live:
        fallback_count = 0

        for sample in samples:
            result = await router.route(
                sample,
                skills,
                {},
                budget,
            )

            if result.value.fallback_used:
                fallback_count += 1

        if fallback_count:
            print(
                "\nLive connection result: "
                f"{fallback_count}/{len(samples)} "
                "requests used fallback."
            )
            print(
                "The API request, response format, "
                "or validation may have failed."
            )
        else:
            print(
                "\nLive connection result: "
                "all LLM router requests succeeded."
            )

async def _invalid_output_demo() -> None:
    """验证 LLM 返回非法 Skill ID 时是否触发规则回退。"""

    import json
    from pathlib import Path

    from evofact.runtime.mock_backend import MockBackend
    from evofact.skills.loader import load_skill_package

    class InvalidRouterBackend(MockBackend):
        async def route(
            self,
            sample,
            candidates,
            router_skill,
            utilities,
            budget,
        ) -> BackendResult:
            del (
                sample,
                candidates,
                router_skill,
                utilities,
                budget,
            )

            return BackendResult(
                {
                    "selected_skill_ids": [
                        "invented-skill-id"
                    ],
                    "reasons": {
                        "invented-skill-id":
                            "invalid test output"
                    },
                    "confidence": 0.99,
                },
                UsageRecord(calls=1),
            )

    project_root = Path(__file__).resolve().parents[3]
    seed_root = project_root / "skills" / "seeds"

    skills = [
        load_skill_package(path)
        for path in sorted(seed_root.iterdir())
        if (path / "SKILL.md").is_file()
    ]

    router = LLMSkillRouter(
        backend=InvalidRouterBackend(),
        fallback=SkillRouter(
            "utility-aware",
            seed=42,
        ),
    )

    result = await router.route(
        {
            "sample_id": "invalid-output-test",
            "dataset": "manual",
            "domain": "health",
            "text": "一项未经证实的健康声明。",
            "metadata": {},
        },
        skills,
        {},
        RunBudget(max_skills=2),
    )

    decision = result.value

    print("\nInvalid-output fallback test:")
    print(
        json.dumps(
            {
                "selected_skill_ids":
                    decision.selected_skill_ids,
                "reasons": decision.reasons,
                "fallback_used":
                    decision.fallback_used,
                "expected_fallback": True,
                "test_passed":
                    decision.fallback_used is True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

def _main() -> None:
    """解析 Router 演示参数并运行异步测试。"""

    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description=(
            "Test the rule-constrained LLM skill router"
        )
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="connect to the configured real LLM API",
    )

    parser.add_argument(
        "--config",
        default="configs/adversarial_llm.yaml",
        help="configuration used by --live",
    )

    parser.add_argument(
        "--skip-invalid-test",
        action="store_true",
        help="skip the invalid-output fallback test",
    )

    args = parser.parse_args()

    asyncio.run(
        _router_demo(
            live=args.live,
            config_path=args.config,
        )
    )

    if not args.skip_invalid_test:
        asyncio.run(_invalid_output_demo())


if __name__ == "__main__":
    _main()
