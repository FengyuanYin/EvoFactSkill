"""Generator memory is run-local, versioned, blind-review-isolated and resumable."""

import asyncio
import copy
import json
from dataclasses import asdict, replace

import pytest
from test_adversarial_batch_runner import invoke, setup

from evofact.config import GeneratorMemoryConfig, config_identity
from evofact.core.models import Sample
from evofact.generation.memory import (
    GeneratorContextFailure,
    GeneratorMemoryBackend,
    bind_memory,
    build_memory_view,
    record_round,
    reduce_generator_context,
)


def enable(runner, *, window=1000000):
    runner.config = replace(
        runner.config, generator_memory=GeneratorMemoryConfig(enabled=True, context_window=window)
    )


def test_feedback_and_cross_batch_memory_are_sent_to_generator_only(tmp_path, monkeypatch):
    runner, backend, contexts, _, samples = setup(tmp_path, monkeypatch)
    enable(runner)
    outcome = invoke(runner, samples)
    assert outcome["status"] == "completed"
    assert len(backend.requests) == 3
    assert not backend.requests[0]["generation_memory"]["coverage"]
    assert backend.requests[1]["generation_memory"]["coverage"]
    assert backend.requests[2]["generation_memory"]["capability_boundaries"]
    for payload in backend.verifications:
        assert set(payload) == {"samples"}
        assert "generation_memory" not in json.dumps(payload)
        assert all("label" not in row and "strategy" not in row for row in payload["samples"])
    assert all("generation_memory" not in c for c in contexts)
    state = json.loads(runner.config.meta_learning.checkpoint_path.read_text(encoding="utf-8"))
    assert len(state["generator_memory"]["rounds"]) == 3
    assert len(state["generator_requests"]) == 3
    view = backend.requests[2]["generation_memory"]
    assert any(not c["current_bank"] for c in view["capability_boundaries"])
    assert state["generator_capabilities"]["identity"]["model"] == type(backend).__name__


def test_rejected_labels_are_remembered_without_becoming_hard_cases(tmp_path, monkeypatch):
    runner, backend, _, _, samples = setup(tmp_path, monkeypatch, reject=True)
    enable(runner)
    invoke(runner, samples)
    memory = backend.requests[1]["generation_memory"]
    assert memory["label_feedback"]
    assert not memory["capability_boundaries"]
    assert not memory["retrieved_examples"]


def test_disabled_policy_preserves_identity_and_legacy_requests(tmp_path, monkeypatch):
    runner, backend, _, _, samples = setup(tmp_path, monkeypatch)
    expected = asdict(runner.config)
    for key in ("generator_memory", "optimizer_memory", "independent_validation"):
        expected.pop(key)
    assert config_identity(runner.config) == expected
    invoke(runner, samples)
    assert all("generation_memory" not in r for r in backend.requests)


def test_missing_window_stops_before_training(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    enable(runner, window=None)
    result = invoke(runner, samples)
    assert result["status"] == "generator_context_failure"
    assert not calls and not contexts and not backend.requests
    assert "generator_memory.context_window" in result["generator_memory"]["error"]


def test_cross_run_memory_is_rejected():
    state = {"run_id": "fold1"}
    bind_memory(state, "gen-version")
    state["run_id"] = "fold2"
    with pytest.raises(GeneratorContextFailure, match="different run"):
        bind_memory(state, "gen-version")


def round_record(*, outcome="FAKE", errors=(), target="REAL"):
    sample = {
        "sample_id": "generated-1",
        "text": "A sufficiently long generated news story.",
        "dataset": "fixture",
        "label_schema_id": "fixture",
        "domain": "train",
        "label": target,
    }
    return {
        "round_id": "r1",
        "batch": 1,
        "request": {
            "examples": [
                {
                    "source_sample": {"dataset": "fixture", "label_schema_id": "fixture"},
                    "label_contract": {"allowed_labels": ["REAL", "FAKE"]},
                }
            ]
        },
        "response": {
            "samples": [sample],
            "decisions": [
                {
                    "sample_id": sample["sample_id"],
                    "strategy": "style_decoupling",
                    "reason": "Official style interference",
                    "source_label": target,
                }
            ],
        },
        "verification": {
            "accepted": [sample],
            "rejected": [],
            "reviews": [
                {
                    "sample_id": sample["sample_id"],
                    "status": "classified",
                    "predicted_label": target,
                    "accepted": True,
                }
            ],
        },
        "traces": [
            {
                "sample_id": sample["sample_id"],
                "decision": {"label": outcome, "origin": "judge"},
                "errors": list(errors),
            }
        ],
    }


def test_runtime_failures_are_not_attacks_and_observations_are_idempotent():
    state = {"run_id": "run"}
    current = round_record(errors=["node timeout"])
    for _ in range(2):
        record_round(state, current, bank_digest="bank1", generator_digest="gen")
    assert len(state["generator_memory"]["rounds"]) == 1
    rows = [Sample("s", "fixture", "training", "REAL", "train")]
    view = build_memory_view(state, rows, bank_digest="bank1")
    assert view["capability_boundaries"][0]["runtime_failure"] == 1
    assert view["capability_boundaries"][0]["wrong"] == 0
    assert not view["retrieved_examples"]
    current["traces"][0]["errors"] = []
    record_round(state, current, bank_digest="bank1", generator_digest="gen")
    view = build_memory_view(state, rows, bank_digest="bank2")
    assert view["capability_boundaries"][0]["wrong"] == 1
    assert not view["capability_boundaries"][0]["current_bank"]


def test_custom_label_contract_is_used_for_difficulty_scoring():
    state = {"run_id": "run"}
    current = round_record(target="SUPPORT", outcome="REFUTE")
    current["request"]["examples"][0]["label_contract"]["allowed_labels"] = ["SUPPORT", "REFUTE"]
    record_round(state, current, bank_digest="bank", generator_digest="gen")
    obs = state["generator_memory"]["rounds"][0]["observations"][0]
    assert obs["detector_outcome"] == "wrong"


def test_heldout_overlap_is_removed_from_memory_views():
    state = {"run_id": "run"}
    current = round_record()
    record_round(state, current, bank_digest="bank", generator_digest="gen")
    heldout = Sample(
        "heldout", "fixture", current["response"]["samples"][0]["text"], "REAL", "final"
    )
    view = build_memory_view(state, [], bank_digest="bank", forbidden=[heldout])
    assert not view["coverage"] and not view["capability_boundaries"]
    assert not view["retrieved_examples"]


def test_cross_batch_generated_text_duplicates_are_rejected(tmp_path, monkeypatch):
    runner, backend, _, _, samples = setup(tmp_path, monkeypatch)
    enable(runner)
    invoke(runner, samples)
    state = json.loads(runner.config.meta_learning.checkpoint_path.read_text(encoding="utf-8"))
    old = state["generator_memory"]["rounds"][0]["observations"][0]["sample"]
    # A candidate with a fresh ID but the same text must still be rejected.
    current = copy.deepcopy(state["generation_audit"][1])
    current["response"]["samples"][0]["text"] = old["text"]
    from evofact.experiments.adversarial_batch_runner import sample_from_dict

    _, rejected = runner._validate_response(current, [], [sample_from_dict(old)])
    assert any("duplicate generated" in r["reason"] for r in rejected)


def test_token_pruning_preserves_source_schema_and_records_actual_payload(tmp_path, monkeypatch):
    runner, backend, _, _, samples = setup(tmp_path, monkeypatch)
    enable(runner)
    state = {"run_id": "run", "train_traces": []}
    request = {
        "batch_size": 2,
        "id_prefix": "id-",
        "label_contracts": [{"allowed_labels": ["REAL", "FAKE"]}],
        "examples": [{"source_sample": {"text": "a" * 100, "label": "REAL"}}],
        "generation_memory": {"retrieved_examples": [{"text": "x" * 20000}]},
    }

    class Backend:
        async def _call(self, system, payload):
            return {"samples": [], "decisions": []}

    manager = GeneratorMemoryBackend(
        Backend(),
        GeneratorMemoryConfig(enabled=True, context_window=12000),
        state,
        lambda: None,
        round_id="r1",
    )
    response, _, effective = asyncio.run(manager.execute("json generation rules", request))
    assert response == {"samples": [], "decisions": []}
    assert effective["examples"] == request["examples"]
    assert effective["label_contracts"] == request["label_contracts"]
    assert not effective["generation_memory"]["retrieved_examples"]
    assert state["generator_requests"]["r1"]["generation"]["omitted"]
    assert not state.get("optimizer_capabilities")
    assert not state.get("optimizer_requests")


def test_completed_generation_response_is_reused_after_resume(tmp_path, monkeypatch):
    runner, backend, _, _, _ = setup(tmp_path, monkeypatch)
    enable(runner)
    state = {"run_id": "run", "train_traces": []}
    request = {"batch_size": 2, "id_prefix": "id-", "label_contracts": [], "examples": []}
    calls = []

    class Backend:
        async def _call(self, system, payload):
            calls.append(copy.deepcopy(payload))
            return {"samples": [], "decisions": []}

    for _ in range(2):
        manager = GeneratorMemoryBackend(
            Backend(), runner.config.generator_memory, state, lambda: None, round_id="r1"
        )
        asyncio.run(manager.execute("json generation", request))
        state = json.loads(json.dumps(state))
    assert len(calls) == 1


def test_memory_reduction_removes_old_examples_before_current_sources():
    context = {
        "generation_memory": {"retrieved_examples": [{"id": "new"}, {"id": "old"}]},
        "examples": [{"id": "source"}],
    }
    reduce_generator_context(context)
    assert context["generation_memory"]["retrieved_examples"] == [{"id": "new"}]
    assert context["examples"] == [{"id": "source"}]


def test_json_failure_is_recorded_and_resume_preserves_original_request(tmp_path, monkeypatch):
    from evofact.generation.prompts import KNOWLEDGE_VERIFIER_SYSTEM, VERIFIER_SYSTEM

    runner, backend, _, _, samples = setup(tmp_path, monkeypatch)
    enable(runner)
    original = backend._call
    failed_payloads = []

    async def fail_once(system, payload):
        if system not in {KNOWLEDGE_VERIFIER_SYSTEM, VERIFIER_SYSTEM} and not failed_payloads:
            failed_payloads.append(copy.deepcopy(payload))
            raise json.JSONDecodeError("invalid generation JSON", "broken", 0)
        return await original(system, payload)

    monkeypatch.setattr(backend, "_call", fail_once)
    with pytest.raises(json.JSONDecodeError):
        invoke(runner, samples)
    outcome = invoke(runner, samples, resume=True)
    assert outcome["status"] == "completed"
    assert backend.requests[0] == failed_payloads[0]
    assert any(
        r["reason"] == "invalid generator JSON"
        for r in backend.requests[1]["generation_memory"]["protocol_feedback"]
    )


def test_manual_runtime_resume_does_not_consume_context_overflow_retry_allowance():
    failures = [TimeoutError(), TimeoutError(), TimeoutError()]
    calls = []

    class Backend:
        async def _call(self, system, payload):
            calls.append(payload)
            if failures:
                raise failures.pop()
            return {"samples": [], "decisions": []}

    state = {"run_id": "run"}
    policy = GeneratorMemoryConfig(enabled=True, context_window=30000, max_overflow_retries=1)
    for i in range(4):
        manager = GeneratorMemoryBackend(Backend(), policy, state, lambda: None, round_id="r1")
        if i < 3:
            with pytest.raises(TimeoutError):
                asyncio.run(manager.execute("json generation", {}))
        else:
            asyncio.run(manager.execute("json generation", {}))
    assert len(calls) == 4
    assert all(c["retry"] == 0 for c in state["generator_requests"]["r1"]["generation"]["calls"])


def test_protocol_fifo_uses_last_observation_not_first_insertion():
    state = {
        "generator_memory": {
            "rounds": [
                {
                    "round_id": f"r{i}",
                    "observations": [],
                    "protocol_rejections": [{"reason": reason}],
                }
                for i, reason in enumerate(("A", "B", "B", "A"))
            ]
        }
    }
    view = build_memory_view(state, [], bank_digest="bank")
    assert [r["reason"] for r in view["protocol_feedback"]] == ["A", "B"]
    assert all(r["count"] == 2 for r in view["protocol_feedback"])
    reduce_generator_context({"generation_memory": view})
    assert [r["reason"] for r in view["protocol_feedback"]] == ["A"]


@pytest.mark.parametrize("kind", ["label", "capability"])
def test_equal_priority_generator_groups_keep_the_most_recent_observation(kind):
    state = {"run_id": "run"}
    a, b = "style_decoupling", "quantity_time_change"
    for i, strategy in enumerate((a, b, b, a)):
        current = round_record()
        current["round_id"] = f"r{i}"
        current["batch"] = i + 1
        current["response"]["decisions"][0]["strategy"] = strategy
        if kind == "label":
            current["verification"]["accepted"] = []
            current["verification"]["reviews"][0]["accepted"] = False
            current["verification"]["reviews"][0]["predicted_label"] = "FAKE"
            current["traces"] = []
        record_round(state, current, bank_digest="bank", generator_digest="gen")
    batch = [Sample("s", "fixture", "training", "REAL", "train")]
    view = build_memory_view(state, batch, bank_digest="bank")
    key = "label_feedback" if kind == "label" else "capability_boundaries"
    assert [r["strategy"] for r in view[key]] == [a, b]
    assert [r["strategy"] for r in view["coverage"]] == [a, b]
    if kind == "capability":
        assert view["retrieved_examples"][0]["source_round"] == "r3"
    view["retrieved_examples"] = []
    reduce_generator_context({"generation_memory": view})
    assert [r["strategy"] for r in view["coverage"]] == [a]
