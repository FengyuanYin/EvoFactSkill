from __future__ import annotations

import asyncio
import tempfile
from dataclasses import replace
from pathlib import Path

from evofact.config import AppConfig, ExecutionConfig
from evofact.core.models import (
    EvaluationResult,
    EvolutionOperation,
    EvolutionProposal,
    GateDecision,
    SampleEvaluation,
    StatisticalTestResult,
)
from evofact.evaluation.metrics import compute_metrics
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.skills.repository import SkillRepository

ROOT = Path(__file__).resolve().parents[1]


def _evaluation(samples) -> EvaluationResult:
    rows = tuple(
        SampleEvaluation(sample.sample_id, str(sample.label), str(sample.label), 1.0)
        for sample in samples
    )
    return EvaluationResult(rows, compute_metrics(rows))


class AcceptedBatchRunner(ExperimentRunner):
    def __init__(self, config, root):
        super().__init__(config, root)
        self.seen_versions = []

    async def closed_loop(
        self,
        samples=None,
        validation_samples=None,
        *,
        progress=None,
        task_name="evolve",
    ):
        del validation_samples, progress, task_name
        self.seen_versions.append(tuple(skill.version for skill in self.skills))
        target = self.skills[0]
        major, minor, patch = (int(part) for part in target.version.split("."))
        candidate = replace(target, version=f"{major}.{minor}.{patch + 1}")
        proposal = EvolutionProposal(
            f"proposal-{len(self.seen_versions)}",
            EvolutionOperation.EDIT,
            "test batch update",
            (target.name,),
            (candidate,),
        )
        evaluation = _evaluation(samples)
        decision = GateDecision(
            True,
            "active",
            evaluation,
            evaluation,
            StatisticalTestResult("fixture", 1.0, 0.01, True),
        )
        return {
            "traces": [],
            "evaluation": evaluation,
            "attributions": [],
            "utilities": {},
            "distillation": {},
            "proposals": [proposal],
            "gate_decisions": [decision],
        }


def test_evolve_updates_only_between_batches() -> None:
    config = replace(AppConfig(), execution=ExecutionConfig(batch_size=2, max_concurrent_samples=2))
    runner = AcceptedBatchRunner(config, ROOT)
    initial_version = runner.skills[0].version
    with tempfile.TemporaryDirectory() as temp:
        repository = SkillRepository(Path(temp) / "skills")
        result = asyncio.run(
            runner.closed_loop_batched(
                fixture_samples(),
                fixture_samples()[::-1],
                repository=repository,
            )
        )

    assert len(runner.seen_versions) == 2
    assert runner.seen_versions[0][0] == initial_version
    assert runner.seen_versions[1][0] != initial_version
    assert len(result["batches"]) == 2
    assert all(
        len(batch["committed_snapshots"]) == len(runner.skills) for batch in result["batches"]
    )
    assert (
        result["batches"][0]["result_fingerprint"] == result["batches"][1]["baseline_fingerprint"]
    )


class FailingRepository(SkillRepository):
    def commit_package_bank(self, *args, **kwargs):
        raise RuntimeError("injected commit failure")


def test_failed_batch_commit_does_not_publish_working_bank() -> None:
    config = replace(AppConfig(), execution=ExecutionConfig(batch_size=2, max_concurrent_samples=2))
    runner = AcceptedBatchRunner(config, ROOT)
    before = tuple((skill.name, skill.version) for skill in runner.skills)
    with tempfile.TemporaryDirectory() as temp:
        repository = FailingRepository(Path(temp) / "skills")
        try:
            asyncio.run(
                runner.closed_loop_batched(
                    fixture_samples()[:2],
                    fixture_samples()[2:],
                    repository=repository,
                )
            )
        except RuntimeError as error:
            assert "injected" in str(error)
        else:
            raise AssertionError("commit failure was not propagated")
    assert tuple((skill.name, skill.version) for skill in runner.skills) == before


def test_evolve_resume_skips_atomically_completed_batches() -> None:
    config = replace(AppConfig(), execution=ExecutionConfig(batch_size=2, max_concurrent_samples=2))
    rows = fixture_samples()
    captured = {}

    def interrupt_after_first_batch(state):
        captured.update(state)
        raise RuntimeError("simulated interruption")

    with tempfile.TemporaryDirectory() as temp:
        repository = SkillRepository(Path(temp) / "skills")
        first = AcceptedBatchRunner(config, ROOT)
        try:
            asyncio.run(
                first.closed_loop_batched(
                    rows,
                    rows[::-1],
                    repository=repository,
                    checkpoint_callback=interrupt_after_first_batch,
                    training_run_id="resume-test",
                    provenance={"identity_digest": "identity"},
                )
            )
        except RuntimeError as error:
            assert "simulated interruption" in str(error)
        else:
            raise AssertionError("interruption was not propagated")

        assert captured["completed_batches"] == 1
        resumed = AcceptedBatchRunner(config, ROOT)
        result = asyncio.run(
            resumed.closed_loop_batched(
                rows,
                rows[::-1],
                repository=repository,
                start_batch=1,
                prior_batch_audit=captured["batch_audit"],
                prior_state=captured["accumulated"],
                training_run_id="resume-test",
                provenance={"identity_digest": "identity"},
            )
        )

    assert len(resumed.seen_versions) == 1
    assert result["resumed_from_batch"] == 1
    assert [row["batch_index"] for row in result["batches"]] == [1, 2]
    assert len(result["evaluation"].per_sample) == len(rows)
