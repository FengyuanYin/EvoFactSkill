import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evofact.config import EvolutionConfig
from evofact.core.models import (
    AttributionReport,
    ErrorType,
    EvolutionOperation,
    OptimizerAction,
    SkillKind,
    SkillSpec,
    SkillStatus,
)
from evofact.evolution.optimizer import (
    SkillOptimizerAgent,
    decision_to_proposal,
    parse_optimizer_decision,
)
from evofact.runtime.mock_backend import MockBackend


def _skill(
    skill_id: str,
    name: str,
    kind: SkillKind,
    *,
    version: str = "1.2.3",
) -> SkillSpec:
    return SkillSpec(
        skill_id=skill_id,
        name=name,
        kind=kind,
        version=version,
        status=SkillStatus.ACTIVE,
        instructions=f"Instructions for {name}.",
    )


def _bank() -> list[SkillSpec]:
    return [
        _skill("specialist-1", "evidence_check", SkillKind.SPECIALIST),
        _skill("optimizer-1", "skill_optimizer", SkillKind.META),
    ]


def _report() -> AttributionReport:
    return AttributionReport(
        trace_id="trace-1",
        error_types=(ErrorType.REASONING_ERROR,),
        responsible_skill_ids=("specialist-1",),
        confidence=0.9,
        evidence=("The same reasoning failure appeared repeatedly.",),
    )


def test_no_change_decision_produces_no_proposal():
    bank = _bank()
    decision = parse_optimizer_decision(
        {
            "action": "no_change",
            "rationale": "Existing skills already cover the failure pattern.",
            "confidence": 0.8,
        },
        bank,
    )

    proposal = decision_to_proposal(
        decision,
        bank,
        cluster_id="cluster-1",
        source_trace_ids=("trace-1",),
    )

    assert decision.action is OptimizerAction.NO_CHANGE
    assert proposal is None


def test_add_decision_creates_one_specialist_candidate():
    bank = _bank()
    decision = parse_optimizer_decision(
        {
            "action": "add",
            "rationale": "The current bank lacks this bounded capability.",
            "confidence": 0.9,
            "target_skill_id": None,
            "skill_name": "new_evidence_specialist",
            "skill_kind": "specialist",
            "instructions": "Analyze the missing evidence condition.",
        },
        bank,
    )

    proposal = decision_to_proposal(
        decision,
        bank,
        cluster_id="cluster-add",
        source_trace_ids=("trace-1", "trace-1", "trace-2"),
    )

    assert proposal is not None
    assert proposal.operation is EvolutionOperation.ADD
    assert proposal.target_skill_ids == ()
    assert proposal.source_trace_ids == ("trace-1", "trace-2")
    assert len(proposal.candidate_skills) == 1

    candidate = proposal.candidate_skills[0]
    assert candidate.name == "new_evidence_specialist"
    assert candidate.kind is SkillKind.SPECIALIST
    assert candidate.version == "0.1.0"
    assert candidate.status is SkillStatus.CANDIDATE


def test_edit_decision_preserves_identity_and_bumps_version():
    target = _skill("router-1", "coordinator_routing", SkillKind.ROUTER)
    bank = [target, _skill("optimizer-1", "skill_optimizer", SkillKind.META)]
    replacement_instructions = "Select the smallest sufficient specialist set."

    decision = parse_optimizer_decision(
        {
            "action": "edit",
            "rationale": "Repeated routing misses require a narrower policy.",
            "confidence": 0.85,
            "target_skill_id": target.skill_id,
            "skill_name": None,
            "skill_kind": "router",
            "instructions": replacement_instructions,
        },
        bank,
    )

    proposal = decision_to_proposal(
        decision,
        bank,
        cluster_id="cluster-edit",
        source_trace_ids=("trace-1",),
    )

    assert proposal is not None
    assert proposal.operation is EvolutionOperation.EDIT
    assert proposal.target_skill_ids == (target.skill_id,)
    assert len(proposal.candidate_skills) == 1

    candidate = proposal.candidate_skills[0]
    assert candidate.skill_id != target.skill_id
    assert candidate.name == target.name
    assert candidate.kind is target.kind
    assert candidate.version == "1.2.4"
    assert candidate.status is SkillStatus.CANDIDATE
    assert candidate.instructions == replacement_instructions
    assert candidate.parent_ids == (target.skill_id,)


def test_parser_rejects_fields_controlled_by_the_application():
    with pytest.raises(ValueError, match="unexpected optimizer fields: version"):
        parse_optimizer_decision(
            {
                "action": "no_change",
                "rationale": "No change is needed.",
                "confidence": 0.8,
                "version": "9.9.9",
            },
            _bank(),
        )


def test_parser_rejects_adding_non_specialist_skill():
    with pytest.raises(ValueError, match="only add specialist"):
        parse_optimizer_decision(
            {
                "action": "add",
                "rationale": "Attempt to create another router.",
                "confidence": 0.8,
                "skill_name": "another_router",
                "skill_kind": "router",
                "instructions": "Route all claims.",
            },
            _bank(),
        )


async def test_agent_returns_none_for_mock_no_change():
    agent = SkillOptimizerAgent(
        backend=MockBackend(),
        config=EvolutionConfig(proposer="llm"),
    )

    proposal = await agent.propose(
        cluster_id="cluster-no-change",
        reports=[_report()],
        traces=[],
        bank=_bank(),
    )

    assert proposal is None


class FailingOptimizerBackend:
    async def optimize(self, context: dict, optimizer_skill: SkillSpec):
        del context, optimizer_skill
        raise RuntimeError("simulated optimizer failure")


async def test_agent_falls_back_to_rule_proposer_on_backend_failure():
    agent = SkillOptimizerAgent(
        backend=FailingOptimizerBackend(),
        config=EvolutionConfig(proposer="llm", fallback_to_rule=True),
    )

    proposal = await agent.propose(
        cluster_id="cluster-fallback",
        reports=[_report()],
        traces=[],
        bank=_bank(),
    )

    assert proposal is not None
    assert proposal.operation is EvolutionOperation.EDIT
    assert "llm_optimizer_fallback:RuntimeError" in proposal.risk_flags
