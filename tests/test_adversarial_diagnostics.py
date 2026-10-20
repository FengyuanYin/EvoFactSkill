"""Error-driven generation without weakening labels, provenance or retry limits."""

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict

import pytest
from test_adversarial_batch_runner import setup
from test_training_feedback import install_no_change

from evofact.experiments.meta_batch_runner import _json
from evofact.generation.diagnostics import (
    challenge_statistics,
    enrich_request,
    policy_marker,
)
from evofact.generation.memory import reduce_generator_context
from evofact.runtime.backend import BackendResult


def example(sid, gold, predicted, *, errors=()):
    return {
        "source_sample": {"sample_id": sid, "label": gold, "text": "Claim " + sid},
        "label_contract": {"allowed_labels": ["REAL", "FAKE"]},
        "trace": {
            "sample_id": sid,
            "trace_id": "trace-" + sid,
            "decision": {"label": predicted, "origin": "judge"},
            "errors": list(errors),
        },
    }


def request(examples, size=4):
    return {"examples": examples, "batch_size": size}


def test_explicit_diagnosis_and_three_to_one_allocation_without_label_quota():
    payload = enrich_request(
        request(
            [
                example("wrong", "REAL", "FAKE"),
                example("correct", "FAKE", "FAKE"),
                example("runtime", "FAKE", "REAL", errors=["timeout"]),
            ]
        ),
        [],
    )
    assert [e["diagnosis"]["outcome"] for e in payload["examples"]] == [
        "wrong",
        "correct",
        "runtime_failure",
    ]
    plan = payload["generation_plan"]
    assert plan["error_extension_slots"] == 3 and plan["boundary_exploration_slots"] == 1
    assert [s["source_sample_id"] for s in plan["slots"]] == ["wrong"] * 3 + ["correct"]
    assert all("target_label" not in slot for slot in plan["slots"])


@pytest.mark.parametrize("size,focus", [(1, 1), (2, 1), (3, 2), (4, 3)])
def test_small_budgets_keep_boundary_slot_when_possible(size, focus):
    plan = enrich_request(request([example("a", "REAL", "FAKE")], size), [])["generation_plan"]
    assert plan["error_extension_slots"] == focus


def test_without_scored_error_all_slots_explore():
    payload = enrich_request(
        request(
            [
                example("a", "REAL", "REAL"),
                example("b", "FAKE", "REAL", errors=["network"]),
            ]
        ),
        [],
    )
    assert payload["generation_plan"]["error_extension_slots"] == 0
    assert all(s["source_sample_id"] == "a" for s in payload["generation_plan"]["slots"])


def verified_round():
    accepted = [{"sample_id": f"g{i}", "label": "FAKE"} for i in range(3)]
    return {
        "request": {"examples": [example("a", "REAL", "REAL")]},
        "response": {
            "samples": [{"sample_id": f"g{i}"} for i in range(5)],
            "decisions": [
                {"sample_id": f"g{i}", "source_sample_id": "a", "strategy": "minimal_contrast"}
                for i in range(5)
            ],
        },
        "structural_candidates": 4,
        "verification": {"accepted": accepted},
        "traces": [
            {"sample_id": "g0", "trace_id": "t0", "decision": {"label": "FAKE"}},
            {"sample_id": "g1", "trace_id": "t1", "decision": {"label": "REAL"}},
            {
                "sample_id": "g2",
                "trace_id": "t2",
                "decision": {"label": "REAL"},
                "errors": ["timeout"],
            },
        ],
        "detector_bank_digest": "bank-a",
        "complete": True,
    }


def test_verified_challenge_errors_are_diagnostics_not_new_source_ids():
    round_ = verified_round()
    payload = enrich_request(request([example("a", "REAL", "REAL")]), [round_])
    plan = payload["generation_plan"]
    assert plan["error_extension_slots"] == 3
    assert all(s["source_sample_id"] == "a" for s in plan["slots"])
    assert plan["slots"][0]["diagnosis"]["trace_id"] == "t1"
    assert plan["slots"][0]["provenance"] == "verified_challenge"


def test_attack_success_excludes_runtime_and_has_distinct_verification_denominator():
    stats = challenge_statistics([verified_round()])
    assert stats["proposed"] == 5 and stats["structurally_valid"] == 4
    assert stats["accepted"] == 3 and stats["scored"] == 2
    assert stats["runtime_failure"] == 1
    assert stats["attack_success_rate"] == 0.5
    assert stats["verification_pass_rate"] == 0.75
    assert stats["by_detector_bank"]["bank-a"]["wrong"] == 1
    assert stats["by_strategy"]["minimal_contrast"]["attack_success_rate"] == 0.5
    assert challenge_statistics([])["attack_success_rate"] is None


def test_window_reduction_preserves_every_assigned_source_and_failure_trace():
    payload = enrich_request(
        request(
            [
                example("a", "REAL", "FAKE"),
                example("b", "REAL", "REAL"),
                example("c", "REAL", "REAL"),
            ]
        ),
        [],
    )
    before = deepcopy(payload["generation_plan"])
    while reduce_generator_context(payload) is not None:
        pass
    assert payload["generation_plan"] == before
    remaining = {e["source_sample"]["sample_id"] for e in payload["examples"]}
    assert {s["source_sample_id"] for s in before["slots"]} <= remaining


def test_wrong_assigned_source_is_rejected_without_relaxing_verifier(tmp_path, monkeypatch):
    runner, generator, _, _, samples = setup(tmp_path, monkeypatch)
    batch = [s for s in samples if s.domain == "train"][:2]
    traces, _ = asyncio.run(runner.base.run(batch, phase="train inference"))
    state = {
        "train_traces": _json([asdict(t) for t in traces]),
        "feedback": [],
        "adversarial_generation_policy": policy_marker(),
    }
    payload = runner._request(state, batch, "round-a")
    response = asyncio.run(generator._call("generator", payload))
    first = response["samples"][0]
    other = next(
        e["source_sample"]
        for e in payload["examples"]
        if e["source_sample"]["sample_id"]
        != payload["generation_plan"]["slots"][0]["source_sample_id"]
    )
    response["decisions"][0]["source_sample_id"] = other["sample_id"]
    response["decisions"][0]["source_trace_ids"] = [
        next(
            e["trace"]["trace_id"]
            for e in payload["examples"]
            if e["source_sample"]["sample_id"] == other["sample_id"]
        )
    ]
    response["samples"][0] = {
        **other,
        "sample_id": first["sample_id"],
        "text": first["text"],
        "label": first["label"],
    }
    accepted, rejected = runner._validate_response(
        {"request": payload, "response": response}, [], batch
    )
    assert first["sample_id"] not in {s.sample_id for s in accepted}
    assert any(r["reason"] == "source violates fixed generation slot" for r in rejected)
    # A malformed individual row is rejected without crashing otherwise valid rows.
    response = asyncio.run(generator._call("generator", payload))
    response["samples"][0] = None
    accepted, rejected = runner._validate_response(
        {"request": payload, "response": response}, [], batch
    )
    assert len(accepted) == 1 and rejected


def invoke(runner, samples, **kwargs):
    return asyncio.run(
        runner.run(
            samples,
            final_test_domains=("final",),
            validation_domains=("valid",),
            train_limit=2,
            max_attempts_per_batch=10,
            **kwargs,
        )
    )


def test_two_no_change_requests_a_probe_with_shared_finite_budget(tmp_path, monkeypatch):
    runner, generator, contexts, _, samples = setup(tmp_path, monkeypatch, max_supplement_rounds=2)
    install_no_change(runner, contexts, monkeypatch)
    result = invoke(runner, samples)
    assert result["status"] == "completed"
    assert len(generator.requests) == 3 and len(contexts) == 7
    assert [r["trigger"] for r in result["generation_rounds"]] == [
        "initial",
        "optimizer_no_change",
        "optimizer_no_change",
    ]
    assert result["challenge_statistics"]["scored"] == 6
    assert result["skipped_batches"][0]["reason"] == "no_new_information"


def test_rejected_probes_do_not_create_infinite_no_change_retries(tmp_path, monkeypatch):
    runner, generator, contexts, _, samples = setup(tmp_path, monkeypatch, reject=True)
    install_no_change(runner, contexts, monkeypatch)
    result = invoke(runner, samples)
    assert len(generator.requests) == 2 and len(contexts) == 3
    assert result["challenge_statistics"]["accepted"] == 0
    assert result["skipped_batches"][0]["reason"] == "no_new_information"


def test_validation_rejections_and_no_change_share_supplement_budget(tmp_path, monkeypatch):
    runner, generator, contexts, _, samples = setup(tmp_path, monkeypatch, max_supplement_rounds=2)
    original = runner.base._optimizer_backend

    class Mixed:
        async def optimize_package(self, context, skill):
            if not contexts:
                return await original().optimize_package(context, skill)
            contexts.append(deepcopy(context))
            return BackendResult({"action": "no_change", "rationale": "Need diagnostic evidence."})

    monkeypatch.setattr(runner.base, "_optimizer_backend", Mixed)
    result = invoke(runner, samples)
    assert len(generator.requests) == 3
    assert [r["trigger"] for r in result["generation_rounds"]] == [
        "initial",
        "validation_rejection",
        "optimizer_no_change",
    ]


def test_optimizer_timeout_does_not_request_new_generation(tmp_path, monkeypatch):
    runner, generator, _, _, samples = setup(tmp_path, monkeypatch)

    class Timeout:
        async def optimize_package(self, context, skill):
            raise TimeoutError("optimizer transport failure")

    monkeypatch.setattr(runner.base, "_optimizer_backend", Timeout)
    result = asyncio.run(
        runner.run(
            samples,
            final_test_domains=("final",),
            validation_domains=("valid",),
            train_limit=2,
            max_attempts_per_batch=2,
        )
    )
    assert len(generator.requests) == 1
    assert result["skipped_batches"][0]["reason"] == "attempt_limit"


def test_no_change_probe_resumes_without_repeating_completed_generation(tmp_path, monkeypatch):
    runner, generator, contexts, _, samples = setup(tmp_path, monkeypatch)
    install_no_change(runner, contexts, monkeypatch)
    original = runner.workflow.run
    calls = 0

    async def pause(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupt probe")
        return await original(*args, **kwargs)

    monkeypatch.setattr(runner.workflow, "run", pause)
    with pytest.raises(RuntimeError, match="interrupt probe"):
        invoke(runner, samples)
    state = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert state["adversarial_no_change"]["requested"] == 1
    assert len(contexts) == 2 and len(generator.requests) == 1
    monkeypatch.setattr(runner.workflow, "run", original)
    result = invoke(runner, samples, resume=True)
    assert result["status"] == "completed"
    assert len(generator.requests) == 3 and len(contexts) == 7


def test_legacy_checkpoint_retains_original_prompt_and_no_change_behavior(tmp_path, monkeypatch):
    runner, generator, contexts, _, samples = setup(tmp_path, monkeypatch)
    install_no_change(runner, contexts, monkeypatch)
    original = runner._optimizer_traces

    async def pause(*args, **kwargs):
        raise RuntimeError("before legacy generation")

    monkeypatch.setattr(runner, "_optimizer_traces", pause)
    with pytest.raises(RuntimeError, match="before legacy"):
        invoke(runner, samples)
    path = tmp_path / "checkpoint.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state.pop("adversarial_generation_policy") == policy_marker()
    path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(runner, "_optimizer_traces", original)
    result = invoke(runner, samples, resume=True)
    assert len(generator.requests) == 1 and len(contexts) == 3
    assert "generation_plan" not in generator.requests[0]
    assert "challenge_statistics" not in result
