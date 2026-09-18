from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from evofact.core.label_models import DatasetLabelContract, DecisionOrigin, LabelDefinition
from evofact.core.models import Prediction, Sample, SampleEvaluation, UsageRecord
from evofact.data.label_registry import LabelContractRegistry
from evofact.data.registry import DataRegistry
from evofact.evaluation.metrics import compute_metrics, schema_metrics
from evofact.experiments.runner import load_seed_skills
from evofact.routing.router import SkillRouter
from evofact.runtime.backend import BackendResult
from evofact.runtime.inference import InferenceRuntime
from evofact.runtime.mock_backend import MockBackend
from evofact.runtime.openai_backend import JudgeContractError, OpenAICompatibleBackend

ROOT = Path(__file__).resolve().parents[1]


def tri_contract() -> DatasetLabelContract:
    return DatasetLabelContract(
        "tri-fixture",
        "claim-status-v1",
        "1",
        (
            LabelDefinition("SUPPORTED", "Evidence supports the claim.", "Preserve support."),
            LabelDefinition("REFUTED", "Evidence refutes the claim.", "Preserve refutation."),
            LabelDefinition(
                "NOT_ENOUGH_INFORMATION",
                "Evidence cannot resolve the claim.",
                "Keep the claim genuinely unresolved by available evidence.",
            ),
        ),
        (
            ("supported", "SUPPORTED"),
            ("refuted", "REFUTED"),
            ("nei", "NOT_ENOUGH_INFORMATION"),
        ),
    )


class JudgeBackend(OpenAICompatibleBackend):
    def __init__(self, label: str):
        super().__init__("https://example.invalid/v1", "unused", "test-model")
        self.label = label
        self.system = ""

    async def _call(self, system: str, payload: dict) -> dict:
        self.system = system
        return {"label": self.label, "confidence": 0.4, "rationale": "best supported"}


class InvalidJudgeBackend(MockBackend):
    async def judge(self, *args, **kwargs) -> BackendResult:
        del args, kwargs
        return BackendResult(
            Prediction("ABSTAIN", 0.5, "invalid model abstention", DecisionOrigin.JUDGE),
            UsageRecord(calls=1),
        )


def _judge_skill():
    return next(
        skill
        for skill in load_seed_skills(ROOT / "skills" / "seeds")
        if skill.name == "judge_decision"
    )


def test_contract_normalization_digest_and_registry_are_fail_closed() -> None:
    contract = tri_contract()
    registry = LabelContractRegistry((contract,))
    sample = Sample(
        "tri-1",
        "tri-fixture",
        "claim",
        "nei",
        label_schema_id="claim-status-v1",
    )
    assert registry.resolve_sample(sample) is contract
    assert contract.normalize(sample.label) == "NOT_ENOUGH_INFORMATION"
    assert contract.require_label("REFUTED") == "REFUTED"
    assert contract.digest == tri_contract().digest
    with pytest.raises(ValueError, match="unknown native label"):
        contract.normalize("invented")
    with pytest.raises(ValueError, match="outside"):
        contract.require_label("ABSTAIN")
    with pytest.raises(KeyError, match="unknown label contract"):
        registry.resolve("tri-fixture", "missing")


def test_judge_prompt_and_parser_use_the_active_multiclass_contract() -> None:
    contract = tri_contract()
    backend = JudgeBackend("NOT_ENOUGH_INFORMATION")
    result = asyncio.run(
        backend.judge(
            {"text": "claim", "evidence": []},
            (),
            _judge_skill(),
            label_contract=contract,
        )
    )
    assert result.value.label == "NOT_ENOUGH_INFORMATION"
    assert result.value.origin == DecisionOrigin.JUDGE
    assert all(label in backend.system for label in contract.allowed_labels)
    assert "ABSTAIN is a reserved runtime outcome" in backend.system

    invalid = JudgeBackend("ABSTAIN")
    with pytest.raises(JudgeContractError):
        asyncio.run(
            invalid.judge(
                {"text": "claim", "evidence": []},
                (),
                _judge_skill(),
                label_contract=contract,
            )
        )


def test_multiclass_metrics_follow_contract_order_and_separate_runtime_abstain() -> None:
    labels = tri_contract().allowed_labels
    rows = [
        SampleEvaluation(
            "1",
            "SUPPORTED",
            "SUPPORTED",
            0.9,
            label_schema_id="tri",
            allowed_labels=labels,
            positive_label=None,
            decision_origin="judge",
        ),
        SampleEvaluation(
            "2",
            "REFUTED",
            "SUPPORTED",
            0.6,
            label_schema_id="tri",
            allowed_labels=labels,
            positive_label=None,
            decision_origin="judge",
        ),
        SampleEvaluation(
            "3",
            "NOT_ENOUGH_INFORMATION",
            "ABSTAIN",
            0.0,
            label_schema_id="tri",
            allowed_labels=labels,
            positive_label=None,
            decision_origin="runtime",
        ),
    ]
    metrics = compute_metrics(rows)
    detailed = schema_metrics(rows)["tri"]
    assert metrics["accuracy_all"] == pytest.approx(1 / 3)
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert detailed["confusion.NOT_ENOUGH_INFORMATION->__RUNTIME_ABSTAIN__"] == 1.0
    assert detailed["label.REFUTED.error_rate"] == 1.0


def test_runtime_converts_invalid_judge_label_to_runtime_abstain() -> None:
    contract = tri_contract()
    runtime = InferenceRuntime(
        InvalidJudgeBackend(),
        SkillRouter(strategy="static"),
        load_seed_skills(ROOT / "skills" / "seeds"),
        label_contract_registry=LabelContractRegistry((contract,)),
    )
    trace = asyncio.run(
        runtime.infer(
            Sample(
                "tri-invalid",
                "tri-fixture",
                "A claim with evidence.",
                "supported",
                label_schema_id="claim-status-v1",
            )
        )
    )
    assert trace.decision.label == "ABSTAIN"
    assert trace.decision.origin == DecisionOrigin.RUNTIME
    assert any("judge contract violation" in error for error in trace.errors)


def test_builtin_registry_declares_livefact_as_three_class_and_weibo_mapping() -> None:
    registry = DataRegistry().label_contracts
    assert registry.resolve("weibo21", "weibo21-binary-v1").normalize(1) == "FAKE"
    for schema in ("livefact-cls-v1", "livefact-inf-v1"):
        assert registry.resolve("livefact", schema).allowed_labels == (
            "real",
            "fake",
            "ambiguous",
        )
