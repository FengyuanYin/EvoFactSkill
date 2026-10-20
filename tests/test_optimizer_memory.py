"""Window management is exercised without calling any model service."""

import asyncio
import copy
import json
from dataclasses import asdict, replace
from decimal import Decimal
from types import SimpleNamespace

import pytest
from test_meta_batch_runner import invoke, setup_runner

from evofact.config import GeneratorMemoryConfig, OptimizerMemoryConfig, config_identity
from evofact.core.budget_models import CostStatus, UsageDetails
from evofact.core.models import Sample
from evofact.evolution.optimizer_memory import (
    ManagedOptimizerBackend,
    OptimizerContextFailure,
    OptimizerProtocolFailure,
    memory_view,
    reduce_context,
    remember,
)
from evofact.generation.memory import GeneratorMemoryBackend
from evofact.runtime.backend import BackendResult, ModelOutputTruncated
from evofact.runtime.budget import BudgetManager
from evofact.runtime.openai_backend import BackendHTTPError


class Backend:
    model = "test-model"
    url = "https://example.invalid/v1/chat/completions"
    max_output_tokens = None

    def __init__(self, failures=()):
        self.calls = []
        self.failures = list(failures)
        self.metadata_calls = []

    async def optimizer_model_capabilities(self):
        self.metadata_calls.append(1)
        return {"context_window": 30000}

    def optimizer_system(self, context, skill):
        return skill.instructions + " proposal instructions"

    def optimizer_selection_system(self, skill):
        return skill.instructions + " selection instructions"

    async def select_optimizer_target(self, context, skill):
        self.calls.append(("selection", copy.deepcopy(context)))
        return BackendResult(
            {"action": "edit", "target_skill_id": "s0"},
            usage_details=UsageDetails(calls=1, input_tokens=100),
        )

    async def optimize_package(self, context, skill):
        self.calls.append(("proposal", copy.deepcopy(context)))
        if self.failures:
            raise self.failures.pop(0)
        return BackendResult(
            {"action": "no_change"}, usage_details=UsageDetails(calls=1, input_tokens=100)
        )


def context(size=100, skills=1):
    return {
        "skills": [
            {
                "skill_id": f"s{i}",
                "manifest": {"name": f"skill{i}", "kind": "specialist", "version": "1"},
                "files": [{"path": "SKILL.md", "content": "x" * size, "digest": str(i)}],
            }
            for i in range(skills)
        ],
        "traces": [
            {"trace_id": "t1", "decision": {"label": "REAL"}},
            {"trace_id": "t2", "decision": {"label": "FAKE"}},
        ],
    }


def setup(backend=None, *, window=30000, save=None, state=None, budget=None):
    backend = backend or Backend()
    state = state if state is not None else {"completed_batches": 0, "attempt": 1}
    policy = OptimizerMemoryConfig(enabled=True, context_window=window)
    manager = ManagedOptimizerBackend(backend, policy, state, save or (lambda: None), budget=budget)
    return manager, backend, state, SimpleNamespace(instructions="optimizer instructions")


def test_actual_window_is_cached_and_output_reserved():
    manager, backend, state, skill = setup(window=None)
    asyncio.run(manager.optimize_package(context(), skill))
    asyncio.run(manager.optimize_package(context(), skill))
    assert len(backend.metadata_calls) == 1
    assert len(backend.calls) == 1  # Completed request response reused.
    call = state["optimizer_request"]["proposal"]["calls"][0]
    assert call["input_budget"] == 27000 - 8192
    assert call["estimated_input_tokens"] <= call["input_budget"]
    assert backend.max_output_tokens is None  # Only the managed clone is modified.


def test_unknown_window_fails_before_llm_call():
    backend = Backend()
    backend.optimizer_model_capabilities = None
    manager, _, _, skill = setup(backend, window=None)
    with pytest.raises(OptimizerContextFailure, match="window unknown"):
        asyncio.run(manager.optimize_package(context(), skill))
    assert not backend.calls


def test_large_bank_uses_two_calls_and_full_target_and_individual_budgets():
    budget = BudgetManager()
    manager, backend, state, skill = setup(budget=budget)
    payload = context(size=6000, skills=5)
    asyncio.run(manager.optimize_package(payload, skill))
    assert [c[0] for c in backend.calls] == ["selection", "proposal"]
    assert "skills" not in backend.calls[0][1]
    assert backend.calls[1][1]["skills"] == [payload["skills"][0]]
    assert budget.snapshot().calls_used == 2
    assert budget.snapshot().calls_reserved == 0
    assert state["optimizer_request"]["selection"]["response"]["target_skill_id"] == "s0"


def test_resume_reuses_completed_selection_and_saved_effective_payload():
    state = {"completed_batches": 0, "attempt": 1}
    stopped = False

    def save():
        nonlocal stopped
        if not stopped and state.get("optimizer_request", {}).get("selection", {}).get("response"):
            stopped = True
            raise KeyboardInterrupt()

    manager, backend, _, skill = setup(save=save, state=state)
    payload = context(size=6000, skills=5)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(manager.optimize_package(payload, skill))
    restored = json.loads(json.dumps(state))
    resumed, _, _, _ = setup(backend, state=restored)
    asyncio.run(resumed.optimize_package(payload, skill))
    assert [c[0] for c in backend.calls] == ["selection", "proposal"]
    changed = copy.deepcopy(payload)
    changed["skills"][0]["files"][0]["content"] += "changed"
    with pytest.raises(OptimizerContextFailure, match="no longer matches"):
        asyncio.run(resumed.optimize_package(changed, skill))


def test_overflow_only_retries_with_smaller_payload_and_accounts_for_failed_call():
    backend = Backend([BackendHTTPError("HTTP 400: context_length_exceeded")])
    budget = BudgetManager()
    manager, _, state, skill = setup(backend, budget=budget)
    asyncio.run(manager.optimize_package(context(), skill))
    assert len(backend.calls) == 2
    assert len(backend.calls[1][1]["traces"]) < len(backend.calls[0][1]["traces"])
    assert state["optimizer_request"]["proposal"]["calls"][0]["status"] == "context_overflow"
    assert budget.snapshot().calls_used == 2


@pytest.mark.parametrize(
    "error,status",
    [
        (TimeoutError(), "runtime_failure"),
        (BackendHTTPError("HTTP 429: too many tokens per minute"), "runtime_failure"),
    ],
)
def test_other_failures_do_not_trigger_context_pruning(error, status):
    backend = Backend([error])
    manager, _, state, skill = setup(backend)
    with pytest.raises(type(error)):
        asyncio.run(manager.optimize_package(context(), skill))
    assert len(backend.calls) == 1
    assert state["optimizer_request"]["proposal"]["calls"][0]["status"] == status


def test_protected_input_is_not_truncated():
    manager, backend, _, skill = setup(window=10000)
    with pytest.raises(OptimizerContextFailure, match="does not fit"):
        asyncio.run(manager.optimize_package(context(size=10000), skill))
    assert backend.calls[0][0] == "selection"
    assert all(c[0] != "proposal" for c in backend.calls)


def test_relevance_precedes_fifo_and_repeated_observations_are_grouped():
    records = []
    for i, target in enumerate(("critical", "unrelated", "critical")):
        state = {"optimizer_memories": records}
        remember(state, {"batch": i + 1, "attempt": 1, "status": "rejected", "reason": target})
        records[-1]["summary"].update(target=target)
    view = memory_view(records, {"traces": [{"skill_id": "critical"}]}, ())
    assert view["observations"][0]["target"] == "critical"
    assert view["observations"][0]["observation_count"] == 2
    payload = {"training_memory": view, "traces": [{"trace_id": "required"}]}
    reduce_context(payload)
    assert payload["training_memory"]["observations"][0]["target"] == "critical"
    assert len(records) == 3


def test_memory_filters_protected_text_and_never_uses_independent_rows():
    heldout = Sample("secret-id", "fixture", "private heldout news content", "FAKE", "x")
    state = {}
    remember(
        state,
        {
            "batch": 1,
            "attempt": 1,
            "reason": heldout.text,
            "status": "rejected",
            "independent_validation": {"sample_ids": [heldout.sample_id]},
        },
    )
    assert not memory_view(state["optimizer_memories"], {}, [heldout])["observations"]
    assert "independent_validation" not in json.dumps(state["optimizer_memories"])


def test_disabled_policy_keeps_checkpoint_identity_unchanged(tmp_path, monkeypatch):
    runner, _, _, _ = setup_runner(tmp_path, monkeypatch)
    expected = asdict(runner.config)
    expected.pop("optimizer_memory")
    expected.pop("generator_memory")
    expected.pop("independent_validation")
    assert config_identity(runner.config) == expected


def test_batch_memory_crosses_batch_boundary_and_keeps_full_audit(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config, optimizer_memory=OptimizerMemoryConfig(enabled=True, context_window=1000000)
    )
    result = invoke(runner, samples)
    assert result["status"] == "completed"
    assert not contexts[0]["training_memory"]["observations"]
    assert contexts[2]["training_memory"]["observations"]
    state = json.loads(runner.config.meta_learning.checkpoint_path.read_text(encoding="utf-8"))
    assert len(state["optimizer_memories"]) == 3
    assert len(state["optimizer_requests"]) == 3
    assert state["optimizer_memories"][0]["baseline_traces"]
    assert not state.get("optimizer_request")


@pytest.mark.parametrize("route", ["feedback", "adversarial"])
def test_other_feedback_routes_use_cross_batch_memory(tmp_path, monkeypatch, route):
    if route == "feedback":
        from test_feedback_evolve import invoke as run
        from test_feedback_evolve import setup as prepare

        runner, train, valid, contexts, _ = prepare(tmp_path, monkeypatch)
        arguments = (train, valid)
    else:
        from test_adversarial_batch_runner import invoke as run
        from test_adversarial_batch_runner import setup as prepare

        runner, _, contexts, _, samples = prepare(tmp_path, monkeypatch)
        arguments = (samples,)
    runner.config = replace(
        runner.config, optimizer_memory=OptimizerMemoryConfig(enabled=True, context_window=1000000)
    )
    assert run(runner, *arguments)["status"] == "completed"
    assert contexts[2]["training_memory"]["observations"]


def test_ordinary_evolve_never_enters_managed_optimizer_even_with_flag(monkeypatch):
    from pathlib import Path

    from optimizer_stub import EditingOptimizerBackend

    from evofact.config import load_config
    from evofact.experiments.runner import ExperimentRunner, fixture_samples

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/dry_run.yaml")
    cfg = replace(cfg, optimizer_memory=OptimizerMemoryConfig(enabled=True, context_window=1))
    runner = ExperimentRunner(cfg, root)
    contexts = []

    class Capturing(EditingOptimizerBackend):
        async def optimize_package(self, context, skill):
            contexts.append(context)
            return await super().optimize_package(context, skill)

    monkeypatch.setattr(runner, "_optimizer_backend", Capturing)
    asyncio.run(runner.closed_loop(fixture_samples()))
    assert contexts and all("training_memory" not in c for c in contexts)


def test_output_truncation_detection_is_only_enabled_on_managed_backend(monkeypatch):
    from evofact.runtime.openai_backend import OpenAICompatibleBackend

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"action":"no_change"}'},
                        }
                    ]
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    backend = OpenAICompatibleBackend("https://example.invalid/v1", "test", "test")
    assert asyncio.run(backend._call("json", {})) == {"action": "no_change"}
    backend._managed_optimizer = True
    with pytest.raises(ModelOutputTruncated):
        asyncio.run(backend._call("json", {}))


def test_invalid_window_stops_batch_before_training_inference(tmp_path, monkeypatch):
    runner, contexts, samples, calls = setup_runner(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config,
        optimizer_memory=OptimizerMemoryConfig(enabled=True, context_window=1000),
    )
    outcome = invoke(runner, samples)
    assert outcome["status"] == "optimizer_context_failure"
    assert not calls and not contexts
    assert "no optimizer input budget" in outcome["optimizer_memory"]["error"]


@pytest.mark.parametrize("component", ["optimizer", "generator"])
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_failed_response_usage_survives_child_task_and_is_charged(
    monkeypatch, component, finish_reason
):
    from evofact.runtime.openai_backend import OpenAICompatibleBackend

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "model": "test",
                    "choices": [
                        {"finish_reason": finish_reason, "message": {"content": '{"action":'}}
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response())
    monkeypatch.setattr(
        "evofact.runtime.openai_backend.price_usage",
        lambda usage, table: replace(usage, cost=Decimal("0.25"), cost_status=CostStatus.ACTUAL),
    )
    backend = OpenAICompatibleBackend(
        "https://example.invalid/v1", "test", "test", pricing_table=object()
    )
    # Keep this accounting test to one paid call; adaptive retry is tested separately.
    backend.max_output_tokens = 8192
    budget = BudgetManager()
    if component == "optimizer":
        manager, _, state, skill = setup(backend, budget=budget)
        request = manager.optimize_package(context(), skill)
    else:
        state = {"run_id": "run"}
        manager = GeneratorMemoryBackend(
            backend,
            GeneratorMemoryConfig(enabled=True, context_window=30000),
            state,
            lambda: None,
            round_id="r1",
            budget=budget,
        )
        request = manager.execute("json generation", {})
    error = ModelOutputTruncated if finish_reason == "length" else json.JSONDecodeError
    with pytest.raises(error):
        asyncio.run(request)
    assert budget.snapshot().tokens_used == 300
    assert budget.snapshot().calls_used == 1
    assert budget.snapshot().cost_used == Decimal("0.25")
    assert budget.snapshot().calls_reserved == 0
    if component == "optimizer":
        call = state["optimizer_request"]["proposal"]["calls"][0]
    else:
        call = state["generator_requests"]["r1"]["generation"]["calls"][0]
    assert call["usage"]["input_tokens"] == 100
    assert call["usage"]["output_tokens"] == 200
    assert call["usage"]["cost"] == "0.25"


@pytest.mark.parametrize(
    "choice",
    [
        {"action": "edit", "target_skill_id": "nonexistent"},
        {"action": "edit", "target_skill_id": []},
        {"action": "unknown"},
        {"action": "no_change", "target_skill_id": "s0"},
    ],
)
def test_bad_selection_is_a_protocol_error_not_a_context_failure(choice):
    class BadSelection(Backend):
        async def select_optimizer_target(self, context, skill):
            return BackendResult(choice)

    manager, _, _, skill = setup(BadSelection())
    with pytest.raises(OptimizerProtocolFailure) as caught:
        asyncio.run(manager.optimize_package(context(size=6000, skills=5), skill))
    assert not isinstance(caught.value, OptimizerContextFailure)


def test_batch_can_continue_after_a_selection_protocol_failure(tmp_path, monkeypatch):
    runner, _, samples, _ = setup_runner(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config, optimizer_memory=OptimizerMemoryConfig(enabled=True, context_window=1000000)
    )
    original = ManagedOptimizerBackend._optimize_package
    seen = []

    async def once(self, context, skill):
        if not seen:
            seen.append(1)
            raise OptimizerProtocolFailure("invalid selection")
        return await original(self, context, skill)

    monkeypatch.setattr(ManagedOptimizerBackend, "_optimize_package", once)
    result = invoke(runner, samples)
    assert result["status"] == "completed"
    assert result["attempts"][0]["reason"] == "call:OptimizerProtocolFailure"
    assert any(a.get("decision", {}).get("accepted") for a in result["attempts"])


def test_large_rejected_package_is_compacted_while_current_target_stays_complete():
    manager, backend, state, skill = setup()
    payload = context()
    payload["validation_feedback"] = [
        {
            "rejected_package": {
                "files": [{"path": "SKILL.md", "digest": "old", "content": "x" * 30000}]
            },
            "baseline_traces": [{"trace_id": "b"}],
            "candidate_traces": [{"trace_id": "c"}],
            "gate": {"reason": "macro-F1 did not improve"},
        }
    ]
    asyncio.run(manager.optimize_package(payload, skill))
    assert [c[0] for c in backend.calls] == ["proposal"]
    effective = backend.calls[0][1]
    assert effective["skills"] == payload["skills"]
    feedback = effective["validation_feedback"][0]
    assert feedback["rejected_package"]["files"] == [{"path": "SKILL.md", "digest": "old"}]
    assert feedback["gate"] == payload["validation_feedback"][0]["gate"]
    assert feedback["baseline_traces"] and feedback["candidate_traces"]
    assert (
        len(
            state["optimizer_request"]["payload"]["validation_feedback"][0]["rejected_package"][
                "files"
            ][0]["content"]
        )
        == 30000
    )


@pytest.mark.parametrize("component", ["optimizer", "generator"])
def test_failed_metadata_cache_is_retried_after_resume(component):
    class Recovering(Backend):
        async def optimizer_model_capabilities(self):
            self.metadata_calls.append(1)
            if len(self.metadata_calls) == 1:
                raise TimeoutError("temporary outage")
            return {"context_window": 30000}

    backend = Recovering()
    state = {"run_id": "run"}
    for attempt in range(2):
        if component == "optimizer":
            manager, _, _, _ = setup(backend, window=None, state=state)
        else:
            manager = GeneratorMemoryBackend(
                backend, GeneratorMemoryConfig(enabled=True), state, lambda: None, round_id="r1"
            )
        if attempt == 0:
            with pytest.raises(OptimizerContextFailure):
                asyncio.run(manager.capabilities())
            state[f"{component}_context_error"] = "model window unknown"
            state = json.loads(json.dumps(state))
        else:
            assert asyncio.run(manager.capabilities())[0] == 30000
    assert len(backend.metadata_calls) == 2
    assert f"{component}_context_error" not in state


@pytest.mark.parametrize("component", ["optimizer", "generator"])
def test_calibration_multiplies_the_existing_factor(component):
    class ActualUsage(Backend):
        def result(self, system, payload, value):
            raw = len((system + json.dumps(payload, ensure_ascii=False, default=str)).encode()) + 64
            return BackendResult(value, usage_details=UsageDetails(calls=1, input_tokens=raw * 3))

        async def optimize_package(self, context, skill):
            return self.result(
                self.optimizer_system(context, skill), context, {"action": "no_change"}
            )

        async def _call_with_usage(self, system, payload):
            result = self.result(system, payload, {"samples": [], "decisions": []})
            return result.value, result.usage_details

    state = {
        "completed_batches": 0,
        "attempt": 1,
        "run_id": "run",
        f"{component}_token_factor": 2.0,
    }
    if component == "optimizer":
        manager, _, _, skill = setup(ActualUsage(), state=state, window=1000000)
        asyncio.run(manager.optimize_package(context(), skill))
    else:
        manager = GeneratorMemoryBackend(
            ActualUsage(),
            GeneratorMemoryConfig(enabled=True, context_window=1000000),
            state,
            lambda: None,
            round_id="r1",
        )
        asyncio.run(manager.execute("json generation", {}))
    assert state[f"{component}_token_factor"] == pytest.approx(3.3)
