from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evofact.config import AppConfig
from evofact.core.models import SpecialistFinding, SpecialistReport
from evofact.experiments.runner import ExperimentRunner, fixture_samples, load_seed_skills
from evofact.runtime.openai_backend import OpenAICompatibleBackend, _specialist_json_contract


def _skill(name: str):
    return next(
        skill for skill in load_seed_skills(ROOT / "skills" / "seeds") if skill.name == name
    )


class DataBackend(OpenAICompatibleBackend):
    def __init__(self, data: dict):
        super().__init__("https://example.invalid/v1", "unused", "test-model")
        self.data = data
        self.system = ""
        self.payload = {}

    async def _call(self, system: str, payload: dict) -> dict:
        self.system = system
        self.payload = payload
        return self.data


def _finding(**overrides):
    value = {
        "finding_type": "evidence_supports",
        "claim_id": "claim-1",
        "conclusion": "supported by supplied evidence",
        "explanation": "the supplied record directly states the claim",
        "text_span": None,
        "evidence_indices": [0],
        "confidence": 0.8,
        "details": {"stance": "support"},
    }
    value.update(overrides)
    return value


def test_specialist_report_v2_preserves_legacy_constructor() -> None:
    report = SpecialistReport("skill", ("claim",), (), "unknown", 0.5, ())

    assert report.report_type == "generic"
    assert report.findings == ()
    assert report.schema_version == "specialist_report_v2"
    assert report.model_dump()["claims"] == ("claim",)


def test_specialist_finding_validates_required_fields_and_confidence() -> None:
    with pytest.raises(ValueError, match="finding_type"):
        SpecialistFinding("", "conclusion", "explanation", 0.5)
    with pytest.raises(ValueError, match="confidence"):
        SpecialistFinding("type", "conclusion", "explanation", 1.1)
    with pytest.raises(ValueError, match="outside"):
        SpecialistReport(
            "skill",
            evidence=(),
            findings=(SpecialistFinding("type", "result", "why", 0.5, evidence_indices=(0,)),),
        )


@pytest.mark.parametrize(
    ("skill_name", "expected"),
    [
        ("claim_decomposition", "atomic_claim"),
        ("linguistic_manipulation", "sensational_framing"),
        ("numerical_consistency", "unit_mismatch"),
        ("source_credibility", "source_identified"),
        ("temporal_reasoning", "timeline_conflict"),
        ("evidence_assessment", "evidence_supports"),
        ("cross_source_contradiction", "genuine_contradiction"),
        ("future_evolved_specialist", "domain_observation"),
    ],
)
def test_each_specialist_receives_role_specific_contract(skill_name, expected) -> None:
    contract = _specialist_json_contract(skill_name)

    assert expected in contract
    assert f"report_type is exactly '{skill_name}'" in contract
    assert '"assessment": "unknown"' in contract
    assert "judge owns the final label" in contract


def test_backend_keeps_only_referenced_trusted_evidence_and_remaps_indices() -> None:
    backend = DataBackend(
        {
            "claims": ["claim"],
            "input_claim_ids": ["claim-1", "invented-claim"],
            "findings": [_finding(evidence_indices=[2, 99])],
            "evidence": [{"text": "invented model evidence", "source": "model"}],
            "assessment": "fake",
            "confidence": 0.8,
            "limitations": [],
        }
    )
    sample = {
        "text": "claim",
        "evidence": [
            {"text": "unused", "source": "source-0"},
            {"text": "also unused", "source": "source-1"},
            {
                "text": "trusted cited evidence",
                "source": "source-2",
                "published_at": "2026-01-01T00:00:00Z",
                "stance": "support",
            },
        ],
    }
    upstream = SpecialistReport(
        "decomposition",
        claims=("claim",),
        report_type="claim_decomposition",
        findings=(SpecialistFinding("atomic_claim", "extracted", "from input", 0.9, "claim-1"),),
    )

    report = asyncio.run(
        backend.analyze(sample, _skill("evidence_assessment"), upstream=(upstream,))
    ).value

    assert report.assessment == "unknown"
    assert report.report_type == "evidence_assessment"
    assert report.input_claim_ids == ("claim-1",)
    assert len(report.evidence) == 1
    assert report.evidence[0].text == "trusted cited evidence"
    assert report.evidence[0].published_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert report.findings[0].evidence_indices == (0,)
    assert all(item.text != "invented model evidence" for item in report.evidence)
    assert len(backend.payload["available_evidence"]) == 3


def test_backend_rejects_response_without_a_valid_typed_finding() -> None:
    backend = DataBackend(
        {
            "claims": ["claim"],
            "assessment": "unknown",
            "confidence": 0.5,
            "limitations": [],
            "findings": [],
        }
    )

    with pytest.raises(ValueError, match="typed finding"):
        asyncio.run(backend.analyze({"text": "claim"}, _skill("claim_decomposition")))


def test_backend_rejects_wrong_role_finding_and_does_not_split_string_arrays() -> None:
    backend = DataBackend(
        {
            "claims": "must not become characters",
            "assessment": "unknown",
            "confidence": 0.5,
            "limitations": "must be an array",
            "findings": [
                _finding(
                    finding_type="sensational_framing",
                    evidence_indices="not-an-array",
                )
            ],
        }
    )

    with pytest.raises(ValueError, match="typed finding"):
        asyncio.run(backend.analyze({"text": "claim"}, _skill("claim_decomposition")))


def test_mock_end_to_end_emits_serializable_v2_reports() -> None:
    runner = ExperimentRunner(AppConfig(), ROOT)
    traces, _ = asyncio.run(runner.run(fixture_samples()[:1]))

    reports = traces[0].specialist_reports
    assert reports
    assert all(report.schema_version == "specialist_report_v2" for report in reports)
    assert all(report.findings for report in reports)
    json.dumps(traces[0].model_dump(), ensure_ascii=False, default=str)


def test_seed_specialists_declare_v2_output_schema() -> None:
    for skill in load_seed_skills(ROOT / "skills" / "seeds"):
        if skill.kind.value == "specialist":
            assert skill.contract.output_schema == "specialist_report_v2"
            assert skill.version == "0.2.0"
