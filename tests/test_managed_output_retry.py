"""Adaptive output retry is bounded, charged, resumable and batch-route only."""

import asyncio
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest
from test_adversarial_batch_runner import invoke, setup

from evofact.config import GeneratorMemoryConfig, OptimizerMemoryConfig
from evofact.core.budget_models import CostStatus, UsageDetails
from evofact.evolution.optimizer_memory import ManagedOptimizerBackend, digest
from evofact.generation.memory import GeneratorMemoryBackend
from evofact.runtime.backend import BackendResult, ModelOutputTruncated
from evofact.runtime.budget import BudgetManager


class Backend:
    max_output_tokens = None

    def __init__(self, failures):
        self.failures = list(failures)
        self.caps = []
        self.payloads = []

    async def reply(self, payload):
        self.caps.append(self.max_output_tokens)
        self.payloads.append(deepcopy(payload))
        truncate = self.failures.pop(0) if self.failures else False
        output = self.max_output_tokens if truncate else 10
        usage = UsageDetails(
            calls=1,
            input_tokens=100,
            output_tokens=output,
            reasoning_tokens=output if truncate else 0,
            cost=Decimal("0.1"),
            cost_status=CostStatus.ACTUAL,
        )
        if truncate:
            exc = ModelOutputTruncated("provider length stop")
            exc.usage_details = usage
            raise exc
        return {"action": "no_change"}, usage

    async def optimize_package(self, context, skill):
        response, usage = await self.reply(context)
        return BackendResult(response, usage_details=usage)

    async def _call_with_usage(self, system, payload):
        return await self.reply(payload)


def manager_for(component, backend, state, budget):
    if component == "generator":
        manager = GeneratorMemoryBackend(
            backend,
            GeneratorMemoryConfig(enabled=True, context_window=1000000),
            state,
            lambda: None,
            round_id="r1",
            budget=budget,
        )
        return manager, lambda: manager.execute("json rules", {})
    manager = ManagedOptimizerBackend(
        backend,
        OptimizerMemoryConfig(enabled=True, context_window=1000000),
        state,
        lambda: None,
        budget=budget,
    )
    return manager, lambda: manager.optimize_package(
        {"skills": [], "traces": []}, SimpleNamespace(instructions="json rules")
    )


@pytest.mark.parametrize("component", ["optimizer", "generator"])
def test_output_cap_grows_and_every_paid_attempt_is_charged(component):
    backend = Backend([True, True, False])
    state = {"completed_batches": 0, "attempt": 1}
    budget = BudgetManager()
    _, call = manager_for(component, backend, state, budget)
    asyncio.run(call())
    assert backend.caps == [8192, 16384, 32768]
    assert backend.max_output_tokens is None  # The original inference backend is untouched.
    assert all(p == backend.payloads[0] for p in backend.payloads)
    assert budget.snapshot().calls_used == 3
    assert budget.snapshot().cost_used == Decimal("0.3")
    assert budget.snapshot().tokens_used > 0
    assert budget.snapshot().active_reservations == 0
    before = len(backend.caps)
    _, resumed = manager_for(component, backend, state, budget)
    asyncio.run(resumed())
    assert len(backend.caps) == before  # A complete response still comes from cache.


@pytest.mark.parametrize("component", ["optimizer", "generator"])
def test_output_retry_budget_survives_resume(component):
    backend = Backend([True] * 10)
    state = {"completed_batches": 0, "attempt": 1}
    budget = BudgetManager()
    for _ in range(2):
        _, call = manager_for(component, backend, state, budget)
        with pytest.raises(ModelOutputTruncated):
            asyncio.run(call())
    assert backend.caps == [8192, 16384, 32768]
    assert budget.snapshot().calls_used == 3


@pytest.mark.parametrize("component", ["optimizer", "generator"])
def test_explicit_output_ceiling_is_not_overridden(component):
    backend = Backend([True] * 10)
    backend.max_output_tokens = 8192
    state = {"completed_batches": 0, "attempt": 1}
    _, call = manager_for(component, backend, state, BudgetManager())
    with pytest.raises(ModelOutputTruncated):
        asyncio.run(call())
    assert backend.caps == [8192]


def test_old_generator_checkpoint_with_one_truncated_call_resumes_at_larger_cap():
    backend = Backend([])
    state = {
        "generator_requests": {
            "r1": {
                "id": "r1",
                "window": 1000000,
                "source_digest": digest({"system": "json rules", "payload": {}}),
                "generation": {
                    "payload": {},
                    "calls": [
                        {
                            "status": "output_truncated",
                            "output_reserve": 8192,
                            "usage": {"calls": 1, "input_tokens": 100, "output_tokens": 8192},
                        }
                    ],
                },
            }
        }
    }
    _, call = manager_for("generator", backend, state, BudgetManager())
    _, usage, _ = asyncio.run(call())
    assert backend.caps == [16384]
    assert usage.calls == 2


def test_exhausted_supplement_is_audited_without_discarding_initial_challenges(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    runner, backend, contexts, _, samples = setup(tmp_path, monkeypatch, max_supplement_rounds=1)
    runner.config = replace(
        runner.config, generator_memory=GeneratorMemoryConfig(enabled=True, context_window=1000000)
    )
    original = backend._call
    failures = []

    async def fail_supplement(system, payload):
        if "examples" in payload and "-r1-" in payload["id_prefix"]:
            failures.append(1)
            raise ModelOutputTruncated("supplement output exhausted")
        return await original(system, payload)

    monkeypatch.setattr(backend, "_call", fail_supplement)
    result = invoke(runner, samples)
    assert result["status"] == "completed"
    assert len(failures) == 3
    failed = [r for r in result["generation_rounds"] if r.get("generation_error")]
    assert len(failed) == 1
    assert failed[0]["complete"]
    assert failed[0]["generator_usage"]["calls"] == 3
    assert not failed[0]["verification"]["accepted"]
    assert "output truncated" in failed[0]["verification"]["rejected"][0]["reason"]
    assert result["accepted_challenge_samples"] > 0
    assert any(t["sample_id"].startswith("gen-") for c in contexts for t in c["traces"])
