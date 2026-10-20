"""Supervised feedback is optimizer-only; identical no-change retries are bounded."""

import asyncio
import json
from copy import deepcopy

import pytest
from test_meta_batch_runner import invoke, setup_runner

from evofact.core.models import Sample
from evofact.data.label_registry import LabelContractRegistry, fixture_binary_contract
from evofact.evolution.training_feedback import (
    FEEDBACK_POLICY_VERSION,
    build_training_supervision,
    information_fingerprint,
    record_no_change,
)
from evofact.runtime.backend import BackendResult
from evofact.skills.package_adapter import package_bank_digest


def test_supervision_distinguishes_wrong_correct_and_runtime_without_mutating_public_input():
    sources = [
        (Sample("real", "fixture", "Training text", "REAL", "train"), "real_training"),
        (Sample("fake", "fixture", "Other training text", "FAKE", "train"), "real_training"),
        (Sample("generated", "fixture", "Verified text", "FAKE", "train"), "verified_challenge"),
    ]
    traces = [
        {
            "trace_id": "t1",
            "sample_id": "real",
            "sample_public": {"text": "Training text"},
            "decision": {"label": "FAKE", "origin": "model"},
        },
        {"trace_id": "t2", "sample_id": "fake", "decision": {"label": "FAKE", "origin": "model"}},
        {
            "trace_id": "t3",
            "sample_id": "generated",
            "decision": {"label": "ABSTAIN", "origin": "runtime"},
        },
    ]
    before = deepcopy(traces)
    registry = LabelContractRegistry((fixture_binary_contract(),))
    result = build_training_supervision(traces, sources, registry)
    assert [r["outcome"] for r in result["records"]] == ["wrong", "correct", "runtime_failure"]
    assert result["records"][0]["gold_label"] == "REAL"
    assert result["summary"]["real_training"] == {"wrong": 1, "correct": 1}
    assert result["summary"]["verified_challenge"] == {"runtime_failure": 1}
    assert traces == before
    with pytest.raises(ValueError, match="authorized training source"):
        build_training_supervision([dict(traces[0], sample_id="heldout")], sources, registry)
    with pytest.raises(ValueError, match="not training"):
        build_training_supervision([], [(sources[0][0], "independent_validation")], registry)


@pytest.mark.parametrize("change", ["bank", "traces", "feedback", "supervision"])
def test_new_information_resets_no_change_counter(change):
    values = ["bank-v1", [{"trace_id": "t1"}], [], {"records": []}]
    first = information_fingerprint(*values)
    state = {}
    assert not record_no_change(state, first, no_change=True)
    assert not record_no_change(state, first, no_change=True)
    index = ["bank", "traces", "feedback", "supervision"].index(change)
    values[index] = "changed" if index == 0 else [{"new": True}]
    second = information_fingerprint(*values)
    assert first != second
    assert not record_no_change(state, second, no_change=True)
    assert state["optimizer_no_change"]["count"] == 1
    assert not record_no_change(state, second, no_change=False)
    assert "optimizer_no_change" not in state


def install_no_change(runner, contexts, monkeypatch):
    class NoChange:
        async def optimize_package(self, context, skill):
            contexts.append(deepcopy(context))
            return BackendResult(
                {"action": "no_change", "rationale": "No supported new hypothesis."}
            )

    monkeypatch.setattr(runner.base, "_optimizer_backend", NoChange)


def test_identical_no_change_stops_after_three_without_consuming_validation(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    install_no_change(runner, contexts, monkeypatch)
    before = package_bank_digest(runner.base.packages)
    result = invoke(runner, samples, max_attempts_per_batch=10)
    assert result["status"] == "completed"
    assert len(contexts) == 6  # Two batches, three requests each.
    assert len(result["skipped_batches"]) == 2
    assert all(
        r["reason"] == "no_new_information" and r["attempts"] == 3
        for r in result["skipped_batches"]
    )
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert state["validation_cursor"] == 0
    assert "optimizer_no_change" not in state
    assert package_bank_digest(runner.base.packages) == before
    assert state["optimizer_feedback_policy"] == FEEDBACK_POLICY_VERSION


def test_no_change_counter_survives_resume(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    install_no_change(runner, contexts, monkeypatch)
    original = runner._optimizer_traces

    async def pause(*args, **kwargs):
        if len(contexts) == 2:
            raise RuntimeError("pause after two no_change results")
        return await original(*args, **kwargs)

    monkeypatch.setattr(runner, "_optimizer_traces", pause)
    with pytest.raises(RuntimeError, match="pause after two"):
        invoke(runner, samples, train_limit=2, max_attempts_per_batch=10)
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved["optimizer_no_change"]["count"] == 2
    monkeypatch.setattr(runner, "_optimizer_traces", original)
    result = invoke(runner, samples, train_limit=2, max_attempts_per_batch=10, resume=True)
    assert len(contexts) == 3
    assert result["skipped_batches"][0]["attempts"] == 3


def test_old_checkpoint_keeps_old_supervision_and_retry_policy(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    install_no_change(runner, contexts, monkeypatch)
    original = runner._optimizer_traces

    async def pause(*args, **kwargs):
        raise RuntimeError("pause before first proposal")

    monkeypatch.setattr(runner, "_optimizer_traces", pause)
    with pytest.raises(RuntimeError, match="pause before first"):
        invoke(runner, samples, train_limit=2, max_attempts_per_batch=5)
    checkpoint = tmp_path / "checkpoint.json"
    state = json.loads(checkpoint.read_text())
    state.pop("optimizer_feedback_policy")
    checkpoint.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(runner, "_optimizer_traces", original)
    result = invoke(runner, samples, train_limit=2, max_attempts_per_batch=5, resume=True)
    assert len(contexts) == 5
    assert all("training_supervision" not in context for context in contexts)
    assert result["skipped_batches"][0]["reason"] == "attempt_limit"


def test_adversarial_supervision_uses_only_current_training_and_verified_challenges(
    tmp_path, monkeypatch
):
    from test_adversarial_batch_runner import invoke as adversarial_invoke
    from test_adversarial_batch_runner import setup as adversarial_setup

    runner, backend, contexts, _, samples = adversarial_setup(tmp_path, monkeypatch)
    install_no_change(runner, contexts, monkeypatch)
    result = adversarial_invoke(runner, samples, max_attempts_per_batch=10)
    assert result["status"] == "completed"
    assert len(contexts) == 14  # Each batch exhausts its two no_change probes, then stops.
    by_id = {s.sample_id: s.label for s in samples if s.domain == "train"}
    for round_ in result["generation_rounds"]:
        by_id.update({s["sample_id"]: s["label"] for s in round_["verification"]["accepted"]})
    for context in contexts:
        supervision = context["training_supervision"]
        assert {r["provenance"] for r in supervision["records"]} == {
            "real_training",
            "verified_challenge",
        }
        assert {r["sample_id"] for r in supervision["records"]} == {
            t["sample_id"] for t in context["traces"]
        }
        assert all(r["gold_label"] == by_id[r["sample_id"]] for r in supervision["records"])
    assert len(backend.requests) == 6  # Initial plus two bounded probes per batch.
    assert all("training_supervision" not in r for r in backend.requests + backend.verifications)


def test_supervision_survives_two_stage_context_selection():
    from test_optimizer_memory import context, setup

    manager, backend, _, skill = setup(window=30000)
    payload = context(size=6000, skills=5)
    supervision = {
        "records": [
            {
                "sample_id": "train",
                "gold_label": "FAKE",
                "predicted_label": "REAL",
                "outcome": "wrong",
            }
        ]
    }
    payload["training_supervision"] = deepcopy(supervision)
    asyncio.run(manager.optimize_package(payload, skill))
    assert [name for name, _ in backend.calls] == ["selection", "proposal"]
    assert all(call["training_supervision"] == supervision for _, call in backend.calls)
