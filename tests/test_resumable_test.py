import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from evofact.config import AppConfig, ExecutionConfig
from evofact.core.budget_models import BudgetRequest, UsageDetails
from evofact.experiments.runner import ExperimentRunner, _budget_abstention_trace, fixture_samples
from evofact.experiments.test_runner import resumable_test
from evofact.runtime.budget import BudgetManager
from evofact.runtime.inference import InferenceRuntime

ROOT = Path(__file__).resolve().parents[1]


def runner(concurrency=1):
    return ExperimentRunner(
        AppConfig(execution=ExecutionConfig(batch_size=4, max_concurrent_samples=concurrency)), ROOT
    )


def read_events(output):
    return [
        json.loads(line)
        for line in output.with_suffix(".samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_timeout_recorded_continue_and_resume_only_failed(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    rows = fixture_samples()
    original = InferenceRuntime.infer
    attempted = []

    async def infer(self, sample, *args, **kwargs):
        attempted.append(sample.sample_id)
        if sample.sample_id == rows[1].sample_id:
            raise TimeoutError("fixture timeout")
        return await original(self, sample, *args, **kwargs)

    monkeypatch.setattr(InferenceRuntime, "infer", infer)
    result = asyncio.run(resumable_test(runner(), rows, output=output, sample_retries=1))
    assert result["status"] == "incomplete"
    assert result["completed_samples"] == len(rows) - 1
    assert result["failed_sample_ids"] == [rows[1].sample_id]
    assert attempted.count(rows[1].sample_id) == 2
    assert "traces" not in result
    events = read_events(output)
    failure = next(e for e in events if e.get("status") == "error")
    assert failure["error_type"] == "TimeoutError" and failure["attempts"] == 2
    assert (
        json.loads(output.read_text(encoding="utf-8"))["metrics_scope"] == "completed samples only"
    )
    attempted.clear()

    async def succeed(self, sample, *args, **kwargs):
        attempted.append(sample.sample_id)
        return await original(self, sample, *args, **kwargs)

    monkeypatch.setattr(InferenceRuntime, "infer", succeed)
    result = asyncio.run(resumable_test(runner(), rows, output=output, resume=True))
    assert attempted == [rows[1].sample_id]
    assert result["status"] == "complete" and result["completed_samples"] == len(rows)
    assert result["resumed_samples"] == len(rows) - 1
    assert result["failed_samples"] == 0 and result["metrics_scope"] == "all samples"
    attempts = len(attempted)
    asyncio.run(resumable_test(runner(), rows, output=output, resume=True))
    assert len(attempted) == attempts


def test_interrupted_run_has_durable_samples_and_budget(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    rows = fixture_samples()
    original = InferenceRuntime.infer

    async def interrupt(self, sample, *args, **kwargs):
        if sample.sample_id == rows[2].sample_id:
            # Results must be readable before the whole run returns.
            events = read_events(output)
            assert len([e for e in events if e.get("status") == "complete"]) == 2
            assert json.loads(output.read_text(encoding="utf-8"))["completed_samples"] == 2
            raise asyncio.CancelledError()
        return await original(self, sample, *args, **kwargs)

    monkeypatch.setattr(InferenceRuntime, "infer", interrupt)
    first_runner = runner()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(resumable_test(first_runner, rows, output=output))
    partial = json.loads(output.read_text(encoding="utf-8"))
    assert partial["status"] == "interrupted" and partial["completed_samples"] == 2
    first_calls = int(partial["budget"]["calls_used"])
    assert first_calls > 0
    seen = []

    async def finish(self, sample, *args, **kwargs):
        seen.append(sample.sample_id)
        return await original(self, sample, *args, **kwargs)

    monkeypatch.setattr(InferenceRuntime, "infer", finish)
    result = asyncio.run(resumable_test(runner(), rows, output=output, resume=True))
    assert seen == [s.sample_id for s in rows[2:]]
    assert int(result["budget"]["calls_used"]) > first_calls
    assert result["completed_samples"] == len(rows)


def test_out_of_order_completions_and_partial_last_line(tmp_path, monkeypatch):
    rows = fixture_samples()
    output = tmp_path / "summary.json"
    original = InferenceRuntime.infer

    async def delayed(self, sample, *args, **kwargs):
        if sample.sample_id == rows[0].sample_id:
            await asyncio.sleep(0.03)
        return await original(self, sample, *args, **kwargs)

    monkeypatch.setattr(InferenceRuntime, "infer", delayed)
    first = asyncio.run(resumable_test(runner(4), rows, output=output))
    complete = [e for e in read_events(output) if e.get("status") == "complete"]
    assert complete[0]["sample_id"] != rows[0].sample_id
    assert sorted(e["sample_index"] for e in complete) == list(range(len(rows)))
    path = output.with_suffix(".samples.jsonl")
    with path.open("ab") as stream:
        stream.write(b'{"kind":"sample","broken":"\xe4')

    def forbidden(*args, **kwargs):
        raise AssertionError("completed samples must not be inferred again")

    fresh = runner(4)
    monkeypatch.setattr(fresh, "_backend", forbidden)
    # Concurrency/batch size can change without changing the test's identity.
    second = asyncio.run(resumable_test(fresh, rows, output=output, resume=True))
    assert second["metrics"] == first["metrics"]
    assert second["completed_samples"] == len(rows)
    assert all(isinstance(e, dict) for e in read_events(output))


@pytest.mark.parametrize("change", ["model", "skills", "order"])
def test_resume_rejects_changed_identity_without_writing(tmp_path, change):
    rows = fixture_samples()
    output = tmp_path / "summary.json"
    asyncio.run(resumable_test(runner(), rows, output=output))
    before = output.with_suffix(".samples.jsonl").read_bytes()
    fresh = runner()
    if change == "model":
        fresh.config = replace(fresh.config, model="different-model")
    elif change == "skills":
        fresh.skills[0] = replace(fresh.skills[0], instructions="different instructions")
    else:
        rows = list(reversed(rows))
    with pytest.raises(ValueError, match="identity differs"):
        asyncio.run(resumable_test(fresh, rows, output=output, resume=True))
    assert output.with_suffix(".samples.jsonl").read_bytes() == before


def test_fresh_run_does_not_overwrite_and_complete_corrupt_line_fails(tmp_path):
    output = tmp_path / "summary.json"
    rows = fixture_samples()
    asyncio.run(resumable_test(runner(), rows, output=output))
    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(resumable_test(runner(), rows, output=output))
    with output.with_suffix(".samples.jsonl").open("ab") as stream:
        stream.write(b'{"broken":}\n')
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(resumable_test(runner(), rows, output=output, resume=True))


def test_reconciled_budget_snapshot_does_not_restore_live_reservations():
    async def run():
        budget = BudgetManager()
        first = await budget.reserve("first", BudgetRequest(tokens=10))
        await budget.reserve("in-flight", BudgetRequest(tokens=20))
        await budget.reconcile(first, UsageDetails(calls=1, input_tokens=5))
        with pytest.raises(ValueError, match="active reservations"):
            budget.export_state()
        state = budget.export_reconciled_state()
        restored = BudgetManager()
        restored.restore(state)
        assert restored.snapshot().calls_used == 1
        assert restored.snapshot().calls_reserved == 0
        assert restored.snapshot().active_reservations == 0

    asyncio.run(run())


def test_initial_progress_and_heartbeat_before_first_completion(tmp_path, monkeypatch):
    original_infer = InferenceRuntime.infer
    original_sleep = asyncio.sleep
    events = []

    class Progress:
        def update(self, event):
            events.append(event)

    async def quick_heartbeat(seconds):
        assert seconds == 5
        await original_sleep(0.001)

    async def slow_sample(self, sample, *args, **kwargs):
        assert events and events[0].completed == 0
        await original_sleep(0.02)
        assert len([e for e in events if e.completed == 0]) >= 2
        return await original_infer(self, sample, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", quick_heartbeat)
    monkeypatch.setattr(InferenceRuntime, "infer", slow_sample)
    result = asyncio.run(
        resumable_test(
            runner(), fixture_samples()[:1], output=tmp_path / "summary.json", progress=Progress()
        )
    )
    assert result["status"] == "complete"
    assert events[-1].completed == 1
    assert "saved=1" in events[-1].detail


def test_runtime_timeout_abstention_is_retried(tmp_path, monkeypatch):
    instance = runner()
    rows = fixture_samples()[:1]
    original = InferenceRuntime.infer
    calls = []

    async def infer(self, sample, *args, **kwargs):
        calls.append(sample.sample_id)
        if len(calls) == 1:
            return replace(
                _budget_abstention_trace(sample, instance.skills, "fixture"),
                errors=("node timeout",),
            )
        return await original(self, sample, *args, **kwargs)

    monkeypatch.setattr(InferenceRuntime, "infer", infer)
    result = asyncio.run(
        resumable_test(instance, rows, output=tmp_path / "summary.json", sample_retries=1)
    )
    assert result["status"] == "complete"
    assert calls == [rows[0].sample_id] * 2


def test_resume_retries_old_timeout_marked_complete(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    rows = fixture_samples()[:1]
    instance = runner()
    asyncio.run(resumable_test(instance, rows, output=output))
    events = read_events(output)
    for event in events:
        if event.get("kind") == "sample":
            trace = replace(
                _budget_abstention_trace(rows[0], instance.skills, "fixture"),
                errors=("node timeout",),
            )
            event["trace"] = asdict(trace)
    output.with_suffix(".samples.jsonl").write_text(
        "".join(json.dumps(e, default=str) + "\n" for e in events), encoding="utf-8"
    )
    original = InferenceRuntime.infer
    called = []

    async def infer(self, sample, *args, **kwargs):
        called.append(sample.sample_id)
        return await original(self, sample, *args, **kwargs)

    monkeypatch.setattr(InferenceRuntime, "infer", infer)
    result = asyncio.run(resumable_test(runner(), rows, output=output, resume=True))
    assert result["status"] == "complete"
    assert result["resumed_samples"] == 0
    assert called == [rows[0].sample_id]
