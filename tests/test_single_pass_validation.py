import asyncio
from collections import Counter
from dataclasses import replace
from pathlib import Path

from optimizer_stub import EditingOptimizerBackend

from evofact.config import load_config
from evofact.experiments.meta_runner import MetaEvolutionRunner, fixture_meta_samples
from evofact.experiments.runner import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]


def test_evolve_validates_each_candidate_once_with_legacy_repeat_config(monkeypatch):
    config = load_config(ROOT / "configs/dry_run.yaml")
    config = replace(config, gate=replace(config.gate, repeats=3))
    runner = ExperimentRunner(config, ROOT)
    monkeypatch.setattr(runner, "_optimizer_backend", EditingOptimizerBackend)
    original_run = runner.run
    validation_calls = []

    async def counted_run(samples=None, **kwargs):
        if kwargs.get("phase", "").startswith("validation"):
            validation_calls.append(kwargs["phase"])
        return await original_run(samples, **kwargs)

    monkeypatch.setattr(runner, "run", counted_run)
    outcome = asyncio.run(runner.closed_loop())
    assert outcome["proposals"]
    assert validation_calls == ["validation baseline", "validation candidate"] * len(
        outcome["proposals"]
    )
    for decision in outcome["gate_decisions"]:
        assert all(":r" not in row.sample_id for row in decision.candidate_result.per_sample)


def test_meta_validates_each_episode_candidate_once_with_legacy_repeat_config(
    monkeypatch, tmp_path
):
    config = load_config(ROOT / "configs/demse_dry_run.yaml")
    config = replace(
        config,
        gate=replace(config.gate, repeats=3),
        meta_learning=replace(
            config.meta_learning,
            evaluation_repeats=3,
            isolate_candidate_budget=True,
            checkpoint_path=tmp_path / "checkpoint.json",
        ),
    )
    runner = MetaEvolutionRunner(config, ROOT)
    monkeypatch.setattr(runner.base, "_optimizer_backend", EditingOptimizerBackend)
    original_run = runner.base.run
    validation_calls = []

    async def counted_run(samples=None, **kwargs):
        if kwargs.get("phase", "").startswith("meta-test"):
            validation_calls.append((kwargs["task_name"], kwargs["phase"]))
        return await original_run(samples, **kwargs)

    monkeypatch.setattr(runner.base, "run", counted_run)
    outcome = asyncio.run(runner.run(fixture_meta_samples(), final_test_domains=("outer_holdout",)))
    assert validation_calls
    assert sum(phase == "meta-test baseline" for _, phase in validation_calls) == len(
        outcome.episode_results
    )
    assert sum(phase == "meta-test candidate" for _, phase in validation_calls) == len(
        outcome.episode_results
    )
    assert Counter(task for task, _ in validation_calls) == Counter(
        {
            f"meta-evolve {episode}": count * 2
            for episode, count in Counter(row.episode_id for row in outcome.episode_results).items()
        }
    )
