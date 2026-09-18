from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from evofact.config import AppConfig, ExecutionConfig
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.runtime.batching import ordered_batched_map
from evofact.runtime.mock_backend import MockBackend

ROOT = Path(__file__).resolve().parents[1]


def test_ordered_batched_map_limits_concurrency_and_preserves_order() -> None:
    async def scenario():
        active = 0
        maximum = 0
        batches = []

        async def worker(value):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep((6 - value) * 0.002)
                return value * 2
            finally:
                active -= 1

        result = await ordered_batched_map(
            list(range(6)),
            worker,
            batch_size=3,
            concurrency=2,
            on_batch_start=lambda number, count, start, stop: batches.append(
                ("start", number, count, start, stop)
            ),
            on_batch_done=lambda number, count, start, stop: batches.append(
                ("done", number, count, start, stop)
            ),
        )
        return result, maximum, batches

    result, maximum, batches = asyncio.run(scenario())
    assert result == [0, 2, 4, 6, 8, 10]
    assert maximum == 2
    assert batches == [
        ("start", 1, 2, 0, 3),
        ("done", 1, 2, 0, 3),
        ("start", 2, 2, 3, 6),
        ("done", 2, 2, 3, 6),
    ]


def test_ordered_batched_map_cancels_batch_and_skips_later_batches() -> None:
    started = []
    cancelled = []

    async def worker(value):
        started.append(value)
        if value == 1:
            raise RuntimeError("boom")
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.append(value)
            raise
        return value

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(ordered_batched_map(list(range(6)), worker, batch_size=3, concurrency=3))
    assert set(started) == {0, 1, 2}
    assert set(cancelled) <= {0, 2}


def test_experiment_runner_processes_samples_concurrently_in_input_order() -> None:
    backend = MockBackend(delays={"claim_decomposition": 0.03})
    config = replace(
        AppConfig(),
        execution=ExecutionConfig(batch_size=4, max_concurrent_samples=2),
    )
    runner = ExperimentRunner(config, ROOT)
    runner._backend = lambda: backend
    samples = fixture_samples()

    traces, result = asyncio.run(runner.run(samples))

    assert backend.max_active_calls == 2
    assert [trace.sample_id for trace in traces] == [sample.sample_id for sample in samples]
    assert [row.sample_id for row in result.per_sample] == [sample.sample_id for sample in samples]


def test_execution_config_validation() -> None:
    with pytest.raises(ValueError):
        ExecutionConfig(batch_size=0)
    with pytest.raises(ValueError):
        ExecutionConfig(max_concurrent_samples=0)
    with pytest.raises(ValueError):
        ExecutionConfig(progress="sometimes")
