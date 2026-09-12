from evofact.core.models import (
    Prediction,
    RunBudget,
    SkillSpec,
    SkillUtility,
    SpecialistReport,
    UsageRecord,
)

from .backend import BackendResult

FAKE_CUES = ("谣言", "不实", "假消息", "fake", "hoax", "fabricated")


class MockBackend:
    async def route(
        self,
        sample: dict,
        candidates: tuple[SkillSpec, ...],
        router_skill: SkillSpec,
        utilities: dict[str, SkillUtility],
        budget: RunBudget,
    ) -> BackendResult:
        del router_skill, utilities

        text = str(sample.get("text", "")).casefold()
        by_name = {
            skill.name: skill
            for skill in candidates
        }

        selected_names: list[str] = []
        def select(name: str) -> None:
            """技能存在且尚未选择时，将它加入结果。"""
            if (
                name in by_name
                and name not in selected_names
            ):
                selected_names.append(name)

        select("claim_decomposition")

        if any(character.isdigit() for character in text):
            select("numerical_consistency")

        temporal_cues = ("过去", "未来", "现在", "曾经", "将来", "前几天", "后天", "昨天", "明天")
        if any(cue in text for cue in temporal_cues):
            select("temporal_reasoning")

        source_cues = ("来源", "出处", "来源于", "根据", "据说", "据报道")
        if any(cue in text for cue in source_cues):
            select("source_credibility")


        evaluation_cues = ("评估", "评价", "分析", "判断", "结论", "观点")
        if any(cue in text for cue in evaluation_cues):
            select("evidence_assessment")

        if not selected_names and candidates:
            selected_names.append(candidates[0].name)

        selected_names = selected_names[: budget.max_skills]

        selected_ids = [
            by_name[name].skill_id
            for name in selected_names
        ]

        return BackendResult(
            {
                "selected_skill_ids": selected_ids,
                "reasons": {
                    by_name[name].skill_id:
                     f"mock router selected {name}"

                     for name in selected_names
                },
                "confidence": 0.8 if selected_ids else 0.0,
            },

            UsageRecord(
                calls=1,
                prompt_tokens=len(text.split()),
                completion_tokens=10,
                latency_ms=1,
            ),
        )



    async def analyze(self, sample: dict, skill: SkillSpec) -> BackendResult:
        """函数作用：负责`MockBackend` 中的 `analyze` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `MockBackend` 实例；`sample`（dict）需符合函数签名约定；`skill`（SkillSpec）需符合函数签名约定。
        输出：异步返回 `BackendResult` 类型结果；校验或下游调用失败时异常向上传递。"""
        text = str(sample.get("text", "")).casefold()
        fake = any(x in text for x in FAKE_CUES)
        if "recurring reasoning_error" in skill.instructions and "未经证实" in text:
            fake = True
        assessment = "fake" if fake else "real"
        report = SpecialistReport(
            skill.skill_id,
            (str(sample.get("text", "")),),
            (),
            assessment,
            0.75,
            () if fake else ("offline heuristic",),
        )
        return BackendResult(
            report,
            UsageRecord(
                calls=1, prompt_tokens=len(text.split()), completion_tokens=20, latency_ms=1
            ),
        )

    async def judge(
        self, sample: dict, reports: tuple[SpecialistReport, ...], skill: SkillSpec
    ) -> BackendResult:
        """函数作用：负责`MockBackend` 中的 `judge` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `MockBackend` 实例；`sample`（dict）需符合函数签名约定；`reports`（tuple[SpecialistReport, ...]）需符合函数签名约定；`skill`（SkillSpec）需符合函数签名约定。
        输出：异步返回 `BackendResult` 类型结果；校验或下游调用失败时异常向上传递。"""
        votes = [r.assessment for r in reports]
        if not votes:
            label = "ABSTAIN"
        elif votes.count("fake") == votes.count("real"):
            label = votes[0].upper()
        else:
            label = "FAKE" if votes.count("fake") > votes.count("real") else "REAL"
        return BackendResult(
            Prediction(
                label, 0.5 if label == "ABSTAIN" else 0.75, "deterministic offline consensus"
            ),
            UsageRecord(calls=1, prompt_tokens=10, completion_tokens=8, latency_ms=1),
        )
