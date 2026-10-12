import asyncio
from dataclasses import replace
from pathlib import Path

from evofact.config import load_config
from evofact.experiments.meta_runner import MetaEvolutionRunner, fixture_meta_samples
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.runtime.backend import BackendResult
from evofact.runtime.mock_backend import MockBackend

ROOT = Path(__file__).resolve().parents[1]


class CapturingOptimizer(MockBackend):
    def __init__(self):
        super().__init__()
        self.contexts = []

    async def optimize_package(self, context, optimizer_skill):
        self.contexts.append(context)
        return BackendResult({"action": "no_change"})


def test_training_traces_go_directly_to_optimizer_without_replays(monkeypatch):
    runner = ExperimentRunner(load_config(ROOT / "configs/dry_run.yaml"), ROOT)
    backend = CapturingOptimizer()
    monkeypatch.setattr(runner, "_optimizer_backend", lambda: backend)
    original_run = runner.run
    phases = []

    async def counted_run(samples=None, **kwargs):
        phases.append(kwargs["phase"])
        return await original_run(samples, **kwargs)

    monkeypatch.setattr(runner, "run", counted_run)
    outcome = asyncio.run(runner.evolve_once(fixture_samples()))
    assert phases == ["train inference"]
    assert len(backend.contexts) == 1
    context = backend.contexts[0]
    assert set(context) == {"skills", "traces"}
    assert len(context["traces"]) == len(fixture_samples())
    assert {package["skill_id"] for package in context["skills"]} == {
        package.skill_id for package in runner.packages
    }
    assert all(
        isinstance(item["content"], str)
        for package in context["skills"]
        for item in package["files"]
        if item["path"] == "SKILL.md"
    )
    assert outcome["attributions"] == []
    assert outcome["utilities"] == {}
    assert outcome["distillation"] == {}
    for trace in context["traces"]:
        assert "decision" in trace
        assert "routing" in trace
        assert "gold" not in trace["sample_public"]


def test_meta_optimizer_receives_only_episode_train_traces(monkeypatch, tmp_path):
    config = load_config(ROOT / "configs/demse_dry_run.yaml")
    config = replace(
        config,
        meta_learning=replace(config.meta_learning, checkpoint_path=tmp_path / "checkpoint.json"),
    )
    runner = MetaEvolutionRunner(config, ROOT)
    backend = CapturingOptimizer()
    monkeypatch.setattr(runner.base, "_optimizer_backend", lambda: backend)
    outcome = asyncio.run(
        runner.run(
            fixture_meta_samples(), final_test_domains=("outer_holdout",), evaluation_only=True
        )
    )
    assert len(backend.contexts) == len(outcome.episodes)
    for episode, context in zip(outcome.episodes, backend.contexts):
        ids = {trace["sample_id"] for trace in context["traces"]}
        assert ids == set(episode.meta_train_sample_ids)
        assert not ids.intersection(episode.meta_test_sample_ids)
        assert set(context) == {"skills", "traces"}


def test_optimizer_failure_does_not_restore_rule_proposals(monkeypatch):
    config = load_config(ROOT / "configs/dry_run.yaml")
    config = replace(config, evolution=replace(config.evolution, fallback_to_rule=True))
    runner = ExperimentRunner(config, ROOT)

    class InvalidOptimizer(MockBackend):
        async def optimize_package(self, context, optimizer_skill):
            return BackendResult({"action": "edit", "unknown_field": True})

    monkeypatch.setattr(runner, "_optimizer_backend", InvalidOptimizer)
    outcome = asyncio.run(runner.evolve_once(fixture_samples()))
    assert outcome["proposals"] == []
    assert outcome["optimizer_skips"]["invalid:ValueError"] == 1
