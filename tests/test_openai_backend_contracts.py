from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evofact.core.models import Prediction, SpecialistReport
from evofact.experiments.runner import load_seed_skills
from evofact.runtime.openai_backend import BackendHTTPError, OpenAICompatibleBackend


class ContractCapturingBackend(OpenAICompatibleBackend):
    def __init__(self):
        super().__init__("https://example.invalid/v1", "unused", "test-model")
        self.systems: list[str] = []
        self.payloads: list[dict] = []

    async def _call(self, system: str, payload: dict) -> dict:
        self.systems.append(system)
        self.payloads.append(payload)
        if '"label"' in system:
            return {"label": "REAL", "confidence": 0.8, "rationale": "supported"}
        return {
            "claims": ["claim"],
            "assessment": "real",
            "confidence": 0.75,
            "limitations": [],
            "findings": [
                {
                    "finding_type": "atomic_claim",
                    "claim_id": "claim-1",
                    "conclusion": "extracted",
                    "explanation": "claim copied from the input",
                    "text_span": "example",
                    "evidence_indices": [],
                    "confidence": 0.75,
                    "details": {"modality": "asserted"},
                }
            ],
        }


def _skill(name: str):
    return next(
        skill for skill in load_seed_skills(ROOT / "skills" / "seeds") if skill.name == name
    )


def test_specialist_and_judge_prompts_include_explicit_json_contracts() -> None:
    backend = ContractCapturingBackend()
    specialist = _skill("claim_decomposition")
    judge = _skill("judge_decision")

    report = asyncio.run(backend.analyze({"text": "example"}, specialist)).value
    decision = asyncio.run(backend.judge({"text": "example"}, (report,), judge)).value

    assert isinstance(report, SpecialistReport)
    assert isinstance(decision, Prediction)
    assert '"claims"' in backend.systems[0]
    assert '"findings"' in backend.systems[0]
    assert "atomic_claim" in backend.systems[0]
    assert '"assessment"' in backend.systems[0]
    assert "valid json object" in backend.systems[0].lower()
    assert report.report_type == "claim_decomposition"
    assert report.assessment == "unknown"
    assert report.findings[0].finding_type == "atomic_claim"
    assert backend.payloads[0]["available_evidence"] == []
    assert '"label"' in backend.systems[1]
    assert '"rationale"' in backend.systems[1]
    assert "typed specialist findings" in backend.systems[1].lower()
    assert "provides no external evidence" in backend.systems[1].lower()
    assert backend.payloads[1]["evidence_mode"] == "unavailable"
    assert "valid json object" in backend.systems[1].lower()


def test_call_adds_json_instruction_when_caller_prompt_omits_it() -> None:
    backend = OpenAICompatibleBackend("https://example.invalid/v1", "test-key", "test-model")
    response = {
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    with patch("evofact.runtime.openai_backend.urllib.request.urlopen") as request:
        request.return_value.__enter__.return_value.read.return_value = json.dumps(
            response
        ).encode()
        result = asyncio.run(backend._call("perform the task", {"value": 1}))

    body = json.loads(request.call_args.args[0].data)
    assert "valid json object" in body["messages"][0]["content"].lower()
    assert result == {"ok": True}


def test_http_error_exposes_bounded_provider_detail_and_redacts_keys() -> None:
    backend = OpenAICompatibleBackend("https://example.invalid/v1", "test-key", "test-model")
    error = urllib.error.HTTPError(
        "https://example.invalid/v1/chat/completions",
        400,
        "Bad Request",
        {},
        BytesIO(
            json.dumps(
                {
                    "error": {
                        "message": "prompt must contain json; token sk-secret123",
                        "type": "invalid_request_error",
                    }
                }
            ).encode()
        ),
    )

    with patch("evofact.runtime.openai_backend.urllib.request.urlopen", side_effect=error):
        with pytest.raises(BackendHTTPError) as captured:
            asyncio.run(backend._call("json response", {"value": 1}))

    message = str(captured.value)
    assert "HTTP 400 Bad Request" in message
    assert "prompt must contain json" in message
    assert "invalid_request_error" in message
    assert "sk-secret123" not in message
    assert "[REDACTED]" in message
