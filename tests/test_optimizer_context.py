"""Optimizer prompt payload: what the model may see, and what must never reach it."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from evofact.config import EvolutionConfig
from evofact.core.models import AttributionReport, ErrorType, EvolutionOperation, EvolutionProposal
from evofact.core.package_models import SkillPackage
from evofact.evolution.optimizer_context import build_optimizer_context
from evofact.evolution.package_optimizer import PackageOptimizerAgent
from evofact.experiments.runner import (
    ExperimentRunner,
    counterfactual_harm,
    rank_optimizer_targets,
)
from evofact.runtime.backend import BackendResult
from evofact.runtime.mock_backend import MockBackend
from evofact.skills.package_loader import load_package

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _trace(**overrides):
    base = dict(
        trace_id="t1",
        sample_public={"text": "claim text", "published_at": None, "evidence": []},
        routing=SimpleNamespace(
            selected_skill_ids=("s1", "s2"),
            reasons={"s1": "root"},
            confidence=0.8,
            fallback_used=False,
        ),
        decision=SimpleNamespace(label="REAL", origin="judge", confidence=0.7, rationale="why"),
        specialist_reports=(),
        errors=("node timeout",),
        usage={"cost": "0.01", "calls": 5, "input_tokens": 100},
        execution_plan={"plan_id": "p"},
        node_executions=({"node_id": "n1"},),
        execution_summary={"succeeded": ["n1"]},
        label_schema_id="weibo21-binary-v1",
        label_contract_digest="deadbeef",
        schema_version="inference_trace_v1",
        skill_versions={"s1": "1.0.0"},
        sample_id="sample-1",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _report(**overrides):
    base = dict(
        trace_id="t1",
        error_types=(ErrorType.REASONING_ERROR,),
        responsible_skill_ids=("s1", "s2"),
        confidence=0.9,
        evidence=("decision abstained on a labeled item",),
        counterfactual_deltas={"s1": 0.0, "s2": -1.0},
    )
    base.update(overrides)
    return AttributionReport(**base)


def test_context_shows_the_prediction_but_not_ground_truth() -> None:
    """The model's own answer is the signal; gold and label governance stay hidden."""
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    context = build_optimizer_context(target=package, reports=[_report()], traces=[_trace()])

    decision = context["traces"][0]["decision"]
    assert decision["label"] == "REAL"
    assert decision["origin"] == "judge"
    blob = json.dumps(context)
    assert "weibo21-binary-v1" not in blob
    assert "deadbeef" not in blob


def test_context_drops_runtime_bookkeeping() -> None:
    """Cost, tokens, plans and node bookkeeping go nowhere near the prompt."""
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    context = build_optimizer_context(target=package, reports=[_report()], traces=[_trace()])

    view = context["traces"][0]
    assert sorted(view) == [
        "decision",
        "errors",
        "routing",
        "sample",
        "specialist_reports",
        "trace_id",
    ]
    blob = json.dumps(context)
    for noise in ("node_executions", "execution_plan", "execution_summary", "skill_versions"):
        assert noise not in blob
    assert "input_tokens" not in blob


def test_context_carries_distillation_and_prior_proposals() -> None:
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    context = build_optimizer_context(
        target=package,
        reports=[_report()],
        traces=[_trace()],
        distillation={"lessons": ["Address recurring reasoning_error failures (92)"]},
        prior_proposals=[{"target": "claim_decomposition", "disposition": "rejected"}],
    )
    assert context["distillation"]["lessons"]
    assert context["prior_proposals"][0]["disposition"] == "rejected"
    assert "generation_audits" not in context


def test_context_trims_evidence_to_fit_the_total_budget() -> None:
    package = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    traces = [_trace(trace_id=f"t{index}") for index in range(12)]
    full = build_optimizer_context(target=package, traces=traces)
    trimmed = build_optimizer_context(target=package, traces=traces, max_total_chars=5000)

    assert len(full["traces"]) == 12
    assert len(trimmed["traces"]) < 12
    assert len(json.dumps(trimmed, ensure_ascii=False)) <= 5000


def test_target_ranking_prefers_the_measured_culprit() -> None:
    """A negative counterfactual delta means removing the Skill repaired the failure."""
    packages = [
        load_package(PROJECT_ROOT / "skills" / "seeds" / name)
        for name in ("claim_decomposition", "source_credibility", "judge_decision")
    ]
    first, second, third = (package.skill_id for package in packages)
    reports = [
        _report(responsible_skill_ids=(first, second, third), counterfactual_deltas={third: -1.0}),
        _report(responsible_skill_ids=(first, second, third), counterfactual_deltas={third: -1.0}),
    ]

    harm = counterfactual_harm(reports)
    assert harm == {third: -2.0}
    ranked = rank_optimizer_targets(packages, {first, second, third}, harm)
    assert [package.skill_id for package in ranked] == [third, first, second]


class _CapturingBackend(MockBackend):
    def __init__(self):
        self.contexts = []

    async def optimize_package(self, context, optimizer_skill):
        del optimizer_skill
        self.contexts.append(context)
        return BackendResult({"action": "no_change"})


def test_agent_applies_config_limits_and_prompt_evidence() -> None:
    target = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    optimizer = load_package(PROJECT_ROOT / "skills" / "seeds" / "skill_optimizer")
    backend = _CapturingBackend()
    agent = PackageOptimizerAgent(
        backend, optimizer, config=replace(EvolutionConfig(), max_reports=1)
    )

    outcome = asyncio.run(
        agent.propose(
            target,
            [target, optimizer],
            reports=[_report(), _report(trace_id="t2")],
            traces=[_trace()],
            distillation={"lessons": ["x"]},
            prior_proposals=[{"target": "claim_decomposition"}],
        )
    )

    assert outcome.status == "no_change"
    context = backend.contexts[0]
    assert len(context["reports"]) == 1
    assert context["distillation"] == {"lessons": ["x"]}
    assert context["prior_proposals"] == [{"target": "claim_decomposition"}]


def test_agent_falls_back_to_the_rule_proposer_when_configured() -> None:
    target = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    optimizer = load_package(PROJECT_ROOT / "skills" / "seeds" / "skill_optimizer")

    class InvalidBackend(MockBackend):
        async def optimize_package(self, context, optimizer_skill):
            del context, optimizer_skill
            return BackendResult({"action": "edit", "unknown_field": 1})

    rule_proposal = EvolutionProposal(
        "rule-1", EvolutionOperation.EDIT, "rule fallback", (target.skill_id,), ()
    )
    calls = []

    def rule_fallback(cluster_id, group):
        calls.append((cluster_id, len(group)))
        return rule_proposal

    agent = PackageOptimizerAgent(
        InvalidBackend(),
        optimizer,
        config=replace(EvolutionConfig(), fallback_to_rule=True),
        rule_fallback=rule_fallback,
    )
    outcome = asyncio.run(
        agent.propose(target, [target, optimizer], cluster_id="c1", reports=[_report()])
    )

    assert outcome.status == "fallback"
    assert outcome.proposal is rule_proposal
    assert calls == [("c1", 1)]
    assert agent.skips["fallback:invalid:ValueError"] == 1

    # With the switch off the same failure is simply skipped and recorded instead.
    strict = PackageOptimizerAgent(
        InvalidBackend(),
        optimizer,
        config=replace(EvolutionConfig(), fallback_to_rule=False),
        rule_fallback=rule_fallback,
    )
    skipped = asyncio.run(strict.propose(target, [target, optimizer], reports=[_report()]))
    assert skipped.status == "skipped"
    assert strict.skips["invalid:ValueError"] == 1
    assert len(calls) == 1


def test_optimizer_backend_receives_the_output_cap(monkeypatch) -> None:
    monkeypatch.setenv("EVOFACT_OPTIMIZER_API_KEY", "k")
    monkeypatch.setenv("EVOFACT_OPTIMIZER_MODEL", "m")
    monkeypatch.setenv("EVOFACT_OPTIMIZER_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv(
        "EVOFACT_OPTIMIZER_PRICING_TABLE",
        str(PROJECT_ROOT / "pricing/deepseek-2026-09-18-peak.json"),
    )
    from evofact.config import OptimizerBackendConfig, load_config

    config = load_config(PROJECT_ROOT / "configs" / "weibo21_cross_domain.yaml")
    assert config.optimizer_backend == replace(
        config.optimizer_backend,
        temperature=0.2,
        max_output_tokens=8192,
    )
    backend = ExperimentRunner(config, PROJECT_ROOT)._optimizer_backend()
    assert backend.temperature == 0.2
    assert backend.max_output_tokens == 8192

    with pytest.raises(ValueError, match="max_output_tokens"):
        OptimizerBackendConfig(backend="mock", max_output_tokens=0)


def test_forward_inference_backend_is_left_uncapped(monkeypatch) -> None:
    """The optimizer's output cap and temperature must never reach the inference path."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    from evofact.config import load_config

    config = load_config(PROJECT_ROOT / "configs" / "weibo21_cross_domain.yaml")
    forward = ExperimentRunner(config, PROJECT_ROOT)._backend()

    assert forward.max_output_tokens is None
    assert forward.temperature == 0
    assert config.optimizer_backend.max_output_tokens == 8192
    assert config.optimizer_backend.temperature == 0.2


def test_prompt_never_contains_ground_truth_keys() -> None:
    """Guard the exact-key exception for ``label``: only the prediction is allowed."""
    package: SkillPackage = load_package(PROJECT_ROOT / "skills" / "seeds" / "source_credibility")
    context = build_optimizer_context(
        target=package,
        traces=[_trace(sample_public={"text": "x", "gold": "FAKE", "label": "FAKE"})],
    )
    sample = context["traces"][0]["sample"]
    assert "gold" not in sample
    assert "label" not in sample
