import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from optimizer_stub import EditingOptimizerBackend

from evofact.config import load_config
from evofact.core.models import Sample
from evofact.experiments.meta_batch_runner import MetaBatchRunner

ROOT = Path(__file__).resolve().parents[1]


def setup_runner(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/demse_dry_run.yaml")
    config = replace(
        config,
        execution=replace(config.execution, batch_size=2),
        data=replace(config.data, evolution_validation_samples=2),
        meta_learning=replace(config.meta_learning, checkpoint_path=tmp_path / "checkpoint.json"),
    )
    runner = MetaBatchRunner(config, ROOT)
    contexts = []

    class Optimizer(EditingOptimizerBackend):
        async def optimize_package(self, context, skill):
            contexts.append(context)
            result = await super().optimize_package(context, skill)
            result.value["file_operations"][0]["content"] += f"Revision {len(contexts)}.\n"
            return result

    monkeypatch.setattr(runner.base, "_optimizer_backend", Optimizer)
    samples = [
        Sample(
            f"{domain}-{i}",
            "fixture",
            f"Unique claim about {domain} number {i}.",
            "REAL" if i % 2 else "FAKE",
            domain,
        )
        for domain, count in (("train", 4), ("valid", 8), ("final", 2))
        for i in range(count)
    ]
    calls = []
    original = runner.base.run

    async def run(rows, **kwargs):
        phase = kwargs["phase"]
        calls.append(
            (phase, [s.sample_id for s in rows], tuple(s.instructions for s in runner.base.skills))
        )
        traces, result = await original(rows, **kwargs)
        if phase.startswith("validation"):
            value = 0.5
            if phase == "validation candidate" and len(contexts) >= 2:
                value = 0.75
            result = replace(
                result, aggregate_metrics={**result.aggregate_metrics, "macro_f1_all": value}
            )
        return traces, result

    monkeypatch.setattr(runner.base, "run", run)
    return runner, contexts, samples, calls


def invoke(runner, samples, **kwargs):
    return asyncio.run(
        runner.run(samples, final_test_domains=("final",), validation_domains=("valid",), **kwargs)
    )


def test_rejection_feedback_fresh_validation_and_immediate_promotion(tmp_path, monkeypatch):
    runner, contexts, samples, calls = setup_runner(tmp_path, monkeypatch)
    result = invoke(runner, samples)
    assert result["status"] == "completed"
    assert result["completed_batches"] == 2
    assert len(contexts) == 3
    assert all(context["feedback_enabled"] is True for context in contexts)
    assert "validation_feedback" not in contexts[0]
    feedback = contexts[1]["validation_feedback"][0]
    assert feedback["baseline_traces"] and feedback["candidate_traces"]
    assert feedback["gate"]["accepted"] is False
    assert "validation_feedback" not in contexts[2]
    train_calls = [c for c in calls if c[0] == "train inference"]
    assert len(train_calls) == 2
    assert train_calls[0][2] != train_calls[1][2]
    validations = [c[1] for c in calls if c[0] == "validation baseline"]
    assert len(set(sum(validations, []))) == 6
    assert not any(s.startswith("final-") for _, ids, _ in calls for s in ids)
    assert all(
        s.startswith("train-") for c in contexts for t in c["traces"] for s in [t["sample_id"]]
    )


def test_resume_failed_batch_retains_train_and_feedback(tmp_path, monkeypatch):
    runner, contexts, samples, calls = setup_runner(tmp_path, monkeypatch)
    original = runner._optimizer_traces
    interrupted = False

    async def pause(*args, **kwargs):
        nonlocal interrupted
        if len(contexts) == 1 and not interrupted:
            interrupted = True
            raise RuntimeError("pause after rejection")
        return await original(*args, **kwargs)

    monkeypatch.setattr(runner, "_optimizer_traces", pause)
    with pytest.raises(RuntimeError, match="pause after rejection"):
        invoke(runner, samples)
    result = invoke(runner, samples, resume=True, max_attempts_per_batch=3)
    assert result["status"] == "completed"
    assert len([c for c in calls if c[0] == "train inference"]) == 2
    assert contexts[1]["validation_feedback"]


def test_attempt_limit_advances_without_committing_rejected_candidate(tmp_path, monkeypatch):
    runner, contexts, samples, calls = setup_runner(tmp_path, monkeypatch)
    result = invoke(runner, samples, max_attempts_per_batch=1)
    assert result["status"] == "completed" and result["completed_batches"] == 2
    assert [b["batch"] for b in result["skipped_batches"]] == [1]
    assert not contexts[1].get("validation_feedback")
    assert contexts[0]["skills"] == contexts[1]["skills"]


def test_resume_mid_validation_does_not_repeat_completed_baseline(tmp_path, monkeypatch):
    runner, contexts, samples, calls = setup_runner(tmp_path, monkeypatch)
    run = runner.base.run
    interrupted = False

    async def fail(rows, **kwargs):
        nonlocal interrupted
        if kwargs["phase"] == "validation candidate" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted")
        return await run(rows, **kwargs)

    monkeypatch.setattr(runner.base, "run", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        invoke(runner, samples)
    result = invoke(runner, samples, resume=True)
    assert result["status"] == "completed"
    assert len(contexts) == 3
    assert len([c for c in calls if c[0] == "validation baseline"]) == 3


def test_pool_exhaustion_stops_current_batch(tmp_path, monkeypatch):
    runner, contexts, samples, calls = setup_runner(tmp_path, monkeypatch)
    samples = [
        s for s in samples if not s.sample_id.startswith("valid-") or s.sample_id == "valid-0"
    ]
    result = invoke(runner, samples)
    assert result["status"] == "validation_pool_exhausted"
    assert result["completed_batches"] == 0
    assert not contexts


def test_resume_recovers_commit_after_repository_write(tmp_path, monkeypatch):
    from evofact.skills.package_adapter import package_bank_digest
    from evofact.skills.repository import SkillRepository

    runner, contexts, samples, calls = setup_runner(tmp_path, monkeypatch)
    repo = SkillRepository(tmp_path / "skills")
    repo.commit_package_bank(
        runner.base.packages, run_id="initialize", expected_active={}, audit={}
    )
    runner.repository = repo
    commit = repo.commit_package_bank

    def interrupted_commit(*args, **kwargs):
        commit(*args, **kwargs)
        raise RuntimeError("crash after commit")

    monkeypatch.setattr(repo, "commit_package_bank", interrupted_commit)
    with pytest.raises(RuntimeError, match="crash after commit"):
        invoke(runner, samples, train_limit=2)
    expected = package_bank_digest(list(repo.active_packages().values()))
    restored = MetaBatchRunner(runner.config, ROOT, repo)
    result = invoke(restored, samples, train_limit=2, resume=True)
    assert result["status"] == "completed" and result["completed_batches"] == 1
    assert package_bank_digest(restored.base.packages) == expected


def test_resume_rejects_changed_domain_split(tmp_path, monkeypatch):
    runner, contexts, samples, calls = setup_runner(tmp_path, monkeypatch)
    invoke(runner, samples, max_attempts_per_batch=1)
    with pytest.raises(ValueError, match="incompatible"):
        asyncio.run(
            runner.run(
                samples, final_test_domains=("final",), validation_domains=("train",), resume=True
            )
        )


def test_final_event_cannot_leak_into_training(tmp_path, monkeypatch):
    runner, contexts, samples, calls = setup_runner(tmp_path, monkeypatch)
    samples = [
        replace(s, event_id="shared-event") if s.sample_id in ("train-0", "final-0") else s
        for s in samples
    ]
    with pytest.raises(ValueError, match="event crosses"):
        invoke(runner, samples)
    assert not calls
