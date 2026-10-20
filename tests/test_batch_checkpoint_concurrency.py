"""Checkpoint completed samples while sibling inference still owns reservations."""

import asyncio
import json
from collections import Counter
from dataclasses import replace

import pytest
from test_meta_batch_runner import invoke, setup_runner

from evofact.config import IndependentValidationConfig
from evofact.core.budget_models import BudgetRequest
from evofact.runtime.budget import BudgetManager
from evofact.runtime.inference import InferenceRuntime


@pytest.mark.parametrize("interrupt", [False, True])
def test_sample_checkpoint_with_inflight_sibling_and_resume(tmp_path, monkeypatch, interrupt):
    runner, _, samples, _ = setup_runner(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config,
        execution=replace(runner.config.execution, max_concurrent_samples=2),
        independent_validation=IndependentValidationConfig(
            enabled=True, samples_per_domain=2, evaluation_samples_per_domain=2
        ),
    )
    _, _, train, _, _, _, _ = runner._prepare_plan(
        samples,
        final_test_domains=("final",),
        train_domains=None,
        validation_domains=("valid",),
        train_limit=2,
        match_evolve_training=False,
    )
    fast_id, slow_id = [s.sample_id for s in train]
    reserved = asyncio.Event()
    saved = asyncio.Event()
    counts = Counter()
    concurrent_exports = []
    original_infer = InferenceRuntime.infer
    original_export = BudgetManager.export_reconciled_state
    first_run = True

    def export(manager):
        snapshot = original_export(manager)
        if manager.snapshot().active_reservations:
            concurrent_exports.append(snapshot)
            saved.set()
        return snapshot

    async def infer(runtime, sample, *args, **kwargs):
        counts[sample.sample_id] += 1
        if not first_run or sample.sample_id not in {fast_id, slow_id}:
            return await original_infer(runtime, sample, *args, **kwargs)
        if sample.sample_id == fast_id:
            await reserved.wait()
            return await original_infer(runtime, sample, *args, **kwargs)
        trace = await original_infer(runtime, sample, *args, **kwargs)
        manager = runner.base._budget_manager()
        reservation = await manager.reserve("inflight-sibling", BudgetRequest(calls=1))
        reserved.set()
        try:
            await saved.wait()
            if interrupt:
                raise RuntimeError("interrupted after sibling checkpoint")
            return trace
        finally:
            await manager.release(reservation)

    monkeypatch.setattr(BudgetManager, "export_reconciled_state", export)
    monkeypatch.setattr(InferenceRuntime, "infer", infer)
    if interrupt:
        with pytest.raises(RuntimeError, match="interrupted after sibling checkpoint"):
            invoke(runner, samples, train_limit=2, max_attempts_per_batch=1)
        checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
        assert fast_id in checkpoint["training_inference"]["training_progress"]["completed"]
        restored = BudgetManager(runner.base._budget_manager().limits)
        restored.restore(checkpoint["budget"])
        assert restored.snapshot().calls_used > 0
        assert restored.snapshot().active_reservations == 0
        first_run = False
        result = invoke(runner, samples, train_limit=2, max_attempts_per_batch=1, resume=True)
    else:
        result = invoke(runner, samples, train_limit=2, max_attempts_per_batch=1)
    assert result["status"] == "completed"
    assert concurrent_exports
    assert counts[fast_id] == 1
    assert counts[slow_id] == (2 if interrupt else 1)
    for snapshot in concurrent_exports:
        for ledger in [snapshot["run"], *snapshot["samples"].values()]:
            assert all(
                ledger[k] == 0
                for k in (
                    "calls_reserved",
                    "tokens_reserved",
                    "cost_reserved",
                    "active_reservations",
                )
            )
    assert runner.base._budget_manager().snapshot().active_reservations == 0
