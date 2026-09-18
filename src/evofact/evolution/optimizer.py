from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import replace
from typing import Any

from evofact.config import EvolutionConfig
from evofact.core.budget_models import BudgetRequest
from evofact.core.models import (
    AttributionReport,
    EvolutionOperation,
    EvolutionProposal,
    InferenceTrace,
    OptimizerAction,
    SkillKind,
    SkillOptimizerDecision,
    SkillSpec,
    SkillStatus,
)
from evofact.core.package_models import (
    SkillContract,
    SkillEntrypoints,
    SpecialistReportContract,
)
from evofact.runtime.backend import ModelBackend
from evofact.runtime.node_runner import _usage_details

from .proposer import propose_from_cluster

_ALLOWED_KEYS = frozenset(
    {
        "action",
        "rationale",
        "confidence",
        "target_skill_id",
        "skill_name",
        "skill_kind",
        "instructions",
    }
)


def _clip(value: object, limit: int) -> str:
    """截断发送给 LLM 的文本，并保证长度不超过 limit。"""
    text = str(value)
    if len(text) <= limit:
        return text

    return text[: limit - 3] + "..."


def _context_chars(context: dict[str, Any]) -> int:
    """计算上下文序列化后的字符数。"""
    return len(
        json.dumps(
            context,
            ensure_ascii=False,
            default=str,
        )
    )


def build_optimizer_context(
    cluster_id: str,
    reports: list[AttributionReport],
    traces: list[InferenceTrace],
    bank: list[SkillSpec],
    config: EvolutionConfig,
) -> dict[str, Any]:
    """构造经过裁剪、不包含真实标签的优化器输入。"""
    if not cluster_id.strip():
        raise ValueError("cluster_id must not be empty")

    if not reports:
        raise ValueError("optimizer requires at least one attribution report")

    selected_reports = reports[: config.max_reports]

    reports_view = [
        {
            "trace_id": report.trace_id,
            "error_types": [error.value for error in report.error_types],
            "responsible_skill_ids": list(report.responsible_skill_ids),
            "confidence": report.confidence,
            "evidence": [_clip(item, config.max_text_chars) for item in report.evidence[:3]],
            "counterfactual_deltas": dict(report.counterfactual_deltas),
        }
        for report in selected_reports
    ]

    trace_by_id = {trace.trace_id: trace for trace in traces}

    trace_views = []

    for report in selected_reports:
        trace = trace_by_id.get(report.trace_id)

        if trace is None:
            continue

        trace_views.append(
            {
                "trace_id": trace.trace_id,
                # 只提供公开新闻内容，不提供 gold label。
                "sample": {
                    "text": _clip(
                        trace.sample_public.get("text", ""),
                        config.max_text_chars,
                    ),
                    "domain": trace.sample_public.get("domain"),
                },
                "routing": {
                    "selected_skill_ids": list(trace.routing.selected_skill_ids),
                    "reasons": {
                        skill_id: _clip(
                            reason,
                            config.max_text_chars,
                        )
                        for skill_id, reason in trace.routing.reasons.items()
                    },
                    "confidence": trace.routing.confidence,
                    "fallback_used": trace.routing.fallback_used,
                },
                "decision": {
                    "label": trace.decision.label,
                    "confidence": trace.decision.confidence,
                    "rationale": _clip(
                        trace.decision.rationale,
                        config.max_text_chars,
                    ),
                },
                "runtime_errors": [
                    _clip(error, config.max_text_chars) for error in trace.errors[:5]
                ],
            }
        )

    responsible_ids = {
        skill_id for report in selected_reports for skill_id in report.responsible_skill_ids
    }

    editable = [
        skill
        for skill in bank
        if skill.kind
        in {
            SkillKind.SPECIALIST,
            SkillKind.ROUTER,
            SkillKind.JUDGE,
        }
    ]

    def skill_priority(skill: SkillSpec) -> tuple:
        if skill.skill_id in responsible_ids:
            priority = 0
        elif skill.kind in {
            SkillKind.ROUTER,
            SkillKind.JUDGE,
        }:
            priority = 1
        else:
            priority = 2

        return priority, skill.name, skill.skill_id

    selected_skills = sorted(
        editable,
        key=skill_priority,
    )[: config.max_skills]

    skill_views = [
        {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "kind": skill.kind.value,
            "version": skill.version,
            "status": skill.status.value,
            "instructions": _clip(
                skill.instructions,
                config.max_instructions_chars,
            ),
            "scope": {
                "domains": list(skill.scope.domains),
                "datasets": list(skill.scope.datasets),
                "temporal_windows": list(skill.scope.temporal_windows),
                "tags": list(skill.scope.tags),
            },
        }
        for skill in selected_skills
    ]

    context = {
        "schema_version": "optimizer_context_v1",
        "cluster_id": cluster_id,
        "reports": reports_view,
        "traces": trace_views,
        "skills": skill_views,
    }

    # 优先移除低优先级 Skill，再减少 trace/report 数量。
    while _context_chars(context) > config.max_total_chars:
        if len(context["skills"]) > 1:
            context["skills"].pop()
        elif len(context["traces"]) > 1:
            context["traces"].pop()
        elif len(context["reports"]) > 1:
            context["reports"].pop()
        else:
            raise ValueError("optimizer context cannot fit max_total_chars")

    return context


_SKILL_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{1,63}")


def _optional_text(
    data: dict[str, Any],
    key: str,
) -> str | None:
    value = data.get(key)

    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(f"optimizer field {key} must be a string or null")

    value = value.strip()

    if not value:
        raise ValueError(f"optimizer field {key} must not be empty")

    return value


def parse_optimizer_decision(data: object, bank: list[SkillSpec]) -> SkillOptimizerDecision:

    if not isinstance(data, dict):
        raise TypeError("optimizer response must be a JSON object")

    unknown = set(data) - _ALLOWED_KEYS

    if unknown:
        raise ValueError("unexpected optimizer fields: " + ", ".join(sorted(unknown)))

    action_text = _optional_text(data, "action")
    if action_text is None:
        raise ValueError("optimizer response must contain action")

    try:
        action = OptimizerAction(action_text)

    except ValueError as exc:
        raise ValueError(f"invalid optimizer action: {action_text}") from exc

    rationale = _optional_text(data, "rationale")
    if rationale is None:
        raise ValueError("optimizer response must contain rationale")

    confidence_raw = data.get("confidence", 0.0)
    if isinstance(confidence_raw, bool) or not isinstance(confidence_raw, (int, float)):
        raise TypeError("optimizer confidence must be a number")

    kind_text = _optional_text(data, "skill_kind")

    try:
        skill_kind = SkillKind(kind_text) if kind_text is not None else None
    except ValueError as exc:
        raise ValueError(f"invalid optimizer skill_kind: {kind_text}") from exc

    decision = SkillOptimizerDecision(
        action=action,
        rationale=rationale,
        confidence=float(confidence_raw),
        target_skill_id=_optional_text(
            data,
            "target_skill_id",
        ),
        skill_name=_optional_text(
            data,
            "skill_name",
        ),
        skill_kind=skill_kind,
        instructions=_optional_text(
            data,
            "instructions",
        ),
    )
    if decision.action == OptimizerAction.ADD:
        if not _SKILL_NAME_PATTERN.fullmatch(decision.skill_name or ""):
            raise ValueError("new skill_name must match [a-z][a-z0-9_]{1,63}")

        if any(skill.name == decision.skill_name for skill in bank):
            raise ValueError("new skill_name already exists")
    if decision.action == OptimizerAction.EDIT:
        matches = [skill for skill in bank if skill.skill_id == decision.target_skill_id]

        if len(matches) != 1:
            raise ValueError("edit target must match exactly one skill_id")

        target = matches[0]

        if target.kind not in {
            SkillKind.SPECIALIST,
            SkillKind.ROUTER,
            SkillKind.JUDGE,
        }:
            raise ValueError("optimizer cannot edit this skill kind")

        if target.status in {
            SkillStatus.FROZEN,
            SkillStatus.RETIRED,
        }:
            raise ValueError("optimizer cannot edit frozen or retired skills")

        if decision.skill_kind != target.kind:
            raise ValueError("skill_kind does not match edit target")

    return decision


def _stable_hash(
    *parts: str,
    length: int = 20,
) -> str:
    payload = json.dumps(
        parts,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _bump_patch_version(version: str) -> str:
    parts = version.split(".")

    try:
        parts[-1] = str(int(parts[-1]) + 1)
    except (ValueError, IndexError):
        return version + ".1"

    return ".".join(parts)


def decision_to_proposal(
    decision: SkillOptimizerDecision,
    bank: list[SkillSpec],
    *,
    cluster_id: str,
    source_trace_ids: tuple[str, ...],
) -> EvolutionProposal | None:
    """将 Agent 决策转换为受项目约束的正式进化提案。"""
    if not cluster_id.strip():
        raise ValueError("cluster_id must not be empty")

    if decision.action == OptimizerAction.NO_CHANGE:
        return None

    if decision.action == OptimizerAction.ADD:
        if decision.skill_name is None or decision.instructions is None:
            raise ValueError("add decision is incomplete")

        report_contract = SpecialistReportContract(
            report_type=decision.skill_name,
            finding_types=("domain_observation", "insufficient_analysis"),
            guidance=(
                "Produce role-local, auditable findings for this specialist. "
                "Do not issue any final dataset business label or Runtime abstention outcome."
            ),
        )
        schema_path = "schemas/specialist_report.json"
        metadata = {
            "name": decision.skill_name,
            "kind": "specialist",
            "version": "0.1.0",
            "safety_level": "text_only",
            "contract": {
                "consumes": ["atomic_claims"],
                "produces": ["specialist_report"],
                "requires_capabilities": ["llm"],
                "optional_capabilities": [],
                "allow_root": False,
                "allow_parallel": True,
                "output_schema": "specialist_report_v2",
            },
            "entrypoints": {
                "instructions": "SKILL.md",
                "template": None,
                "script": None,
                "output_schema": schema_path,
                "tests": [],
            },
        }
        candidate = SkillSpec(
            skill_id=_stable_hash(
                "skill",
                "add",
                cluster_id,
                decision.skill_name,
                decision.instructions,
            ),
            name=decision.skill_name,
            kind=SkillKind.SPECIALIST,
            version="0.1.0",
            status=SkillStatus.CANDIDATE,
            instructions=decision.instructions,
            resources={
                "metadata.json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                schema_path: json.dumps(
                    report_contract.model_dump(), ensure_ascii=False, sort_keys=True
                ),
            },
            contract=SkillContract(
                consumes=("atomic_claims",),
                produces=("specialist_report",),
                requires_capabilities=("llm",),
                allow_root=False,
                output_schema="specialist_report_v2",
            ),
            entrypoints=SkillEntrypoints(output_schema=schema_path),
            report_contract=report_contract,
        )

        operation = EvolutionOperation.ADD
        targets: tuple[str, ...] = ()

    elif decision.action == OptimizerAction.EDIT:
        matches = [skill for skill in bank if skill.skill_id == decision.target_skill_id]

        if len(matches) != 1:
            raise ValueError("edit target must resolve to exactly one skill")

        old = matches[0]
        if old.kind not in {
            SkillKind.SPECIALIST,
            SkillKind.ROUTER,
            SkillKind.JUDGE,
        }:
            raise ValueError("optimizer cannot edit this skill kind")

        if old.status in {
            SkillStatus.FROZEN,
            SkillStatus.RETIRED,
        }:
            raise ValueError("optimizer cannot edit frozen or retired skills")

        if old.kind != decision.skill_kind:
            raise ValueError("edit target kind changed after parsing")

        if decision.instructions is None:
            raise ValueError("edit decision is missing instructions")

        candidate = replace(
            old,
            skill_id=_stable_hash(
                "skill",
                "edit",
                old.skill_id,
                cluster_id,
                decision.instructions,
            ),
            version=_bump_patch_version(old.version),
            status=SkillStatus.CANDIDATE,
            instructions=decision.instructions,
            parent_ids=(old.skill_id,),
        )

        operation = EvolutionOperation.EDIT

        targets = (old.skill_id,)

    else:
        raise ValueError(f"unsupported optimizer action: {decision.action}")

    trace_ids = tuple(dict.fromkeys(source_trace_ids))

    proposal_id = "proposal-" + _stable_hash(
        "proposal",
        cluster_id,
        operation.value,
        candidate.skill_id,
        length=10,
    )

    return EvolutionProposal(
        proposal_id=proposal_id,
        operation=operation,
        rationale=decision.rationale,
        target_skill_ids=targets,
        candidate_skills=(candidate,),
        source_trace_ids=trace_ids,
        error_cluster_id=cluster_id,
    )


class SkillOptimizerAgent:
    """调用模型生成一个受约束的 Skill 进化提案。"""

    def __init__(
        self,
        backend: ModelBackend,
        config: EvolutionConfig,
        budget_manager=None,
    ):
        self.backend = backend
        self.config = config
        self.budget_manager = budget_manager

    def _find_optimizer_skill(
        self,
        bank: list[SkillSpec],
    ) -> SkillSpec:
        matches = [
            skill
            for skill in bank
            if skill.name == self.config.optimizer_skill
            and skill.kind == SkillKind.META
            and skill.status
            in {
                SkillStatus.ACTIVE,
                SkillStatus.PARETO,
                SkillStatus.FROZEN,
            }
        ]

        if len(matches) != 1:
            raise ValueError(
                "optimizer Skill must resolve to exactly one "
                f"available meta Skill: {self.config.optimizer_skill}"
            )

        return matches[0]

    async def propose(
        self,
        cluster_id: str,
        reports: list[AttributionReport],
        traces: list[InferenceTrace],
        bank: list[SkillSpec],
    ) -> EvolutionProposal | None:
        """生成一个提案, NO_CHANGE 返回 None, 失败时可回退规则。"""
        try:
            optimizer_skill = self._find_optimizer_skill(bank)

            context = build_optimizer_context(
                cluster_id,
                reports,
                traces,
                bank,
                self.config,
            )

            reservation = None
            try:
                if self.budget_manager is not None:
                    async with self.budget_manager.concurrency(f"optimizer:{cluster_id}"):
                        reservation = await self.budget_manager.reserve(
                            f"optimizer:{cluster_id}",
                            BudgetRequest(calls=1, tokens=4000, purpose="optimizer"),
                        )
                        backend_result = await asyncio.wait_for(
                            self.backend.optimize(context, optimizer_skill),
                            self.budget_manager.limits.call_timeout_ms / 1000,
                        )
                else:
                    backend_result = await self.backend.optimize(context, optimizer_skill)
                if reservation is not None:
                    await self.budget_manager.reconcile(reservation, _usage_details(backend_result))
                    reservation = None
            finally:
                if reservation is not None:
                    await self.budget_manager.release(reservation)

            decision = parse_optimizer_decision(
                backend_result.value,
                bank,
            )

            if decision.action == OptimizerAction.EDIT:
                visible_skill_ids = {item["skill_id"] for item in context["skills"]}

                if decision.target_skill_id not in visible_skill_ids:
                    raise ValueError("optimizer selected a Skill outside the visible context")

            source_trace_ids = tuple(item["trace_id"] for item in context["reports"])

            return decision_to_proposal(
                decision,
                bank,
                cluster_id=cluster_id,
                source_trace_ids=source_trace_ids,
            )
        except Exception as exc:
            if not self.config.fallback_to_rule:
                raise

            fallback = propose_from_cluster(
                cluster_id,
                reports,
                bank,
            )

            # 记录发生过 LLM 回退，但不把异常内容写进提案。
            return replace(
                fallback,
                risk_flags=(
                    *fallback.risk_flags,
                    "llm_optimizer_fallback:" + type(exc).__name__,
                ),
            )
