import asyncio
import hashlib

from evofact.core.label_models import DatasetLabelContract, DecisionOrigin
from evofact.core.models import (
    Prediction,
    RunBudget,
    SkillSpec,
    SkillUtility,
    SpecialistFinding,
    SpecialistReport,
    UsageRecord,
)
from evofact.data.label_registry import fixture_binary_contract

from .backend import BackendResult

FAKE_CUES = ("谣言", "不实", "假消息", "fake", "hoax", "fabricated")

_MOCK_FINDING_TYPES = {
    "claim_decomposition": "atomic_claim",
    "linguistic_manipulation": "neutral_style",
    "numerical_consistency": "consistent_comparison",
    "source_credibility": "source_missing",
    "temporal_reasoning": "missing_time",
    "evidence_assessment": "evidence_missing",
    "cross_source_contradiction": "insufficient_sources",
}


class MockBackend:
    def __init__(self, *, delays: dict[str, float] | None = None, failures: set[str] | None = None):
        self.delays = delays or {}
        self.failures = failures or set()
        self.active_calls = 0
        self.max_active_calls = 0

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
        by_name = {skill.name: skill for skill in candidates}

        selected_names: list[str] = []

        def select(name: str) -> None:
            """技能存在且尚未选择时，将它加入结果。"""
            if name in by_name and name not in selected_names:
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

        selected_ids = [by_name[name].skill_id for name in selected_names]

        return BackendResult(
            {
                "selected_skill_ids": selected_ids,
                "reasons": {
                    by_name[name].skill_id: f"mock router selected {name}"
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

    async def optimize(
        self,
        context: dict,
        optimizer_skill: SkillSpec,
    ) -> BackendResult:
        """离线优化器实现，固定返回 NO_CHANGE。"""
        del optimizer_skill

        report_count = len(context.get("reports", ()))
        skill_count = len(context.get("skills", ()))

        return BackendResult(
            {
                "action": "no_change",
                "rationale": (
                    "mock optimizer reviewed "
                    f"{report_count} reports and "
                    f"{skill_count} skills; "
                    "no deterministic change was proposed"
                ),
                "confidence": 0.5,
            },
            UsageRecord(
                calls=1,
                prompt_tokens=len(str(context).split()),
                completion_tokens=12,
                latency_ms=1,
            ),
        )

    async def optimize_package(self, context, optimizer_skill) -> BackendResult:
        del optimizer_skill
        return BackendResult(
            {
                "action": "no_change",
                "rationale": "mock package optimizer proposed no deterministic change",
                "file_operations": [],
            },
            UsageRecord(
                calls=1,
                prompt_tokens=len(str(context).split()),
                completion_tokens=8,
                latency_ms=1,
            ),
        )

    async def analyze(
        self,
        sample: dict,
        skill: SkillSpec,
        *,
        upstream: tuple[SpecialistReport, ...] = (),
        resources=None,
    ) -> BackendResult:
        """函数作用：负责`MockBackend` 中的 `analyze` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `MockBackend` 实例；`sample`（dict）需符合函数签名约定；`skill`（SkillSpec）需符合函数签名约定。
        输出：异步返回 `BackendResult` 类型结果；校验或下游调用失败时异常向上传递。"""
        del resources
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if skill.skill_id in self.failures or skill.name in self.failures:
                raise RuntimeError(f"configured mock failure: {skill.name}")
            delay = self.delays.get(skill.skill_id, self.delays.get(skill.name, 0))
            if delay:
                await asyncio.sleep(delay)
            text = str(sample.get("text", "")).casefold()
        finally:
            self.active_calls -= 1
        fake = any(x in text for x in FAKE_CUES)
        if "recurring reasoning_error" in skill.instructions and "未经证实" in text:
            fake = True
        assessment = "fake" if fake else "real"
        input_claim_ids = tuple(
            dict.fromkeys(
                finding.claim_id
                for report in upstream
                for finding in report.findings
                if finding.claim_id
            )
        )
        claim_id = (
            "claim-1"
            if skill.name == "claim_decomposition"
            else (input_claim_ids[0] if input_claim_ids else None)
        )
        report = SpecialistReport(
            skill.skill_id,
            (
                str(sample.get("text", "")),
                *(claim for report in upstream for claim in report.claims),
            ),
            (),
            assessment,
            0.75,
            () if fake else ("offline heuristic",),
            report_type=skill.name,
            input_claim_ids=input_claim_ids,
            findings=(
                SpecialistFinding(
                    finding_type=_MOCK_FINDING_TYPES.get(skill.name, "domain_observation"),
                    conclusion="mock heuristic observation",
                    explanation="deterministic offline backend finding",
                    confidence=0.75,
                    claim_id=claim_id,
                    text_span=str(sample.get("text", "")) or None,
                    details={"mock": True, "supports_label_index": int(fake)},
                ),
            ),
        )
        return BackendResult(
            report,
            UsageRecord(
                calls=1, prompt_tokens=len(text.split()), completion_tokens=20, latency_ms=1
            ),
        )

    async def judge(
        self,
        sample: dict,
        reports: tuple[SpecialistReport, ...],
        skill: SkillSpec,
        *,
        label_contract: DatasetLabelContract | None = None,
        execution_summary=None,
    ) -> BackendResult:
        """函数作用：负责`MockBackend` 中的 `judge` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `MockBackend` 实例；`sample`（dict）需符合函数签名约定；`reports`（tuple[SpecialistReport, ...]）需符合函数签名约定；`skill`（SkillSpec）需符合函数签名约定。
        输出：异步返回 `BackendResult` 类型结果；校验或下游调用失败时异常向上传递。"""
        del execution_summary, skill
        label_contract = label_contract or fixture_binary_contract()
        text = str(sample.get("text", ""))
        if label_contract.dataset_id == "fixture" and len(label_contract.allowed_labels) == 2:
            indices = [
                int(finding.details.get("supports_label_index", 0))
                for report in reports
                for finding in report.findings
                if finding.details.get("mock") is True
            ]
            index = max(indices, default=int(any(cue in text.casefold() for cue in FAKE_CUES)))
        else:
            index = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % len(
                label_contract.allowed_labels
            )
        label = label_contract.allowed_labels[index]
        return BackendResult(
            Prediction(
                label,
                0.75,
                "deterministic contract-aware offline decision",
                DecisionOrigin.JUDGE,
            ),
            UsageRecord(calls=1, prompt_tokens=10, completion_tokens=8, latency_ms=1),
        )
