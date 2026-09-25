import asyncio
import json
from dataclasses import replace
from pathlib import Path

from evofact.config import load_config
from evofact.core.models import (
    EvolutionOperation,
    EvolutionProposal,
    Sample,
    SampleEvaluation,
)
from evofact.experiments.meta_runner import MetaEvolutionRunner, fixture_meta_samples
from evofact.validation.evaluator import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_meta_episode_uses_small_inner_updates_and_bounds_proposals(monkeypatch):
    config = load_config(ROOT / "configs/demse_dry_run.yaml")
    config = replace(
        config,
        evolution=replace(
            config.evolution,
            update_interval_samples=2,
            max_proposals_per_update=1,
        ),
    )
    runner = MetaEvolutionRunner(config, ROOT)
    rows = [Sample(f"s-{index}", "fixture", f"text {index}", "REAL", "a") for index in range(5)]
    batch_sizes = []

    async def fake_evolve_once(batch, **kwargs):
        del kwargs
        batch_sizes.append(len(batch))
        update = len(batch_sizes)
        proposals = [
            EvolutionProposal(f"p-{update}-{offset}", EvolutionOperation.EDIT, "test")
            for offset in range(2)
        ]
        evaluations = [
            SampleEvaluation(sample.sample_id, "REAL", "REAL", 1.0, sample.domain)
            for sample in batch
        ]
        return {
            "traces": [],
            "evaluation": evaluate(evaluations),
            "attributions": [],
            "utilities": {},
            "distillation": {},
            "proposals": proposals,
            "package_candidates": {},
        }

    monkeypatch.setattr(runner.base, "evolve_once", fake_evolve_once)
    outcome = asyncio.run(
        runner._evolve_rows_in_updates(
            rows,
            object(),
            object(),
            task_name="test",
        )
    )

    assert batch_sizes == [2, 2, 1]
    assert len(outcome["evaluation"].per_sample) == 5
    assert [proposal.proposal_id for proposal in outcome["proposals"]] == [
        "p-1-0",
        "p-2-0",
        "p-3-0",
    ]


def test_meta_inner_updates_carry_episode_scoped_history_and_skip_counts(monkeypatch):
    config = load_config(ROOT / "configs/demse_dry_run.yaml")
    config = replace(
        config,
        evolution=replace(
            config.evolution,
            update_interval_samples=2,
            max_proposals_per_update=1,
        ),
    )
    runner = MetaEvolutionRunner(config, ROOT)
    rows = [Sample(f"s-{index}", "fixture", f"text {index}", "REAL", "a") for index in range(5)]
    seen_history = []

    async def fake_evolve_once(batch, **kwargs):
        update = len(seen_history) + 1
        seen_history.append(kwargs["prior_proposals"])
        evaluations = [
            SampleEvaluation(sample.sample_id, "REAL", "REAL", 1.0, sample.domain)
            for sample in batch
        ]
        return {
            "traces": [],
            "evaluation": evaluate(evaluations),
            "attributions": [],
            "utilities": {},
            "distillation": {},
            "proposals": [
                EvolutionProposal(
                    f"p-{update}",
                    EvolutionOperation.EDIT,
                    f"rationale {update}",
                    ("some_skill",),
                )
            ],
            "package_candidates": {},
            "optimizer_skips": {"no_change": 1, "invalid:ValueError": update},
            "skipped_proposals": (("cluster-x", "invalid:ValueError"),),
        }

    monkeypatch.setattr(runner.base, "evolve_once", fake_evolve_once)
    outcome = asyncio.run(
        runner._evolve_rows_in_updates(rows, object(), object(), task_name="test")
    )

    # The first update has no memory; later updates see earlier proposals of the episode.
    assert seen_history[0] == ()
    assert len(seen_history[1]) == 1
    assert seen_history[1][0]["rationale"] == "rationale 1"
    assert seen_history[1][0]["disposition"] == "pending"
    assert [entry["rationale"] for entry in seen_history[2]] == ["rationale 1", "rationale 2"]
    # Skip statistics are aggregated over the whole episode.
    assert outcome["optimizer_skips"] == {"no_change": 3, "invalid:ValueError": 6}
    assert len(outcome["skipped_proposals"]) == 3


def test_meta_checkpoint_records_optimizer_skips(monkeypatch, tmp_path):
    config = load_config(ROOT / "configs/demse_dry_run.yaml")
    config = replace(
        config,
        meta_learning=replace(config.meta_learning, checkpoint_path=tmp_path / "checkpoint.json"),
    )
    runner = MetaEvolutionRunner(config, ROOT)

    async def fake_evolve_episode(train_rows, episode, firewall, *, progress=None):
        del train_rows, episode, firewall, progress
        return {
            "traces": [],
            "evaluation": evaluate([]),
            "attributions": [],
            "utilities": {},
            "distillation": [],
            "proposals": [],
            "package_candidates": {},
            "optimizer_skips": {"no_change": 2, "invalid:ValueError": 1},
            "skipped_proposals": (("cluster-x", "invalid:ValueError"),),
        }

    monkeypatch.setattr(runner, "_evolve_episode", fake_evolve_episode)
    asyncio.run(runner.run(fixture_meta_samples(), final_test_domains=("outer_holdout",)))

    saved = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))["state"]
    episodes = len(saved["completed_episode_ids"])
    assert episodes > 0
    assert saved["optimizer_skips"] == {"no_change": 2 * episodes, "invalid:ValueError": episodes}
    assert len(saved["skipped_proposals"]) == episodes


def test_meta_test_proxy_is_deterministic_and_domain_balanced():
    config = load_config(ROOT / "configs/demse_dry_run.yaml")
    config = replace(
        config,
        meta_learning=replace(config.meta_learning, max_meta_test_samples_per_domain=2),
    )
    runner = MetaEvolutionRunner(config, ROOT)
    rows = [
        Sample(f"{domain}-{index}", "fixture", "text", "REAL", domain)
        for domain in ("a", "b")
        for index in range(4)
    ]

    first = runner._bounded_meta_test_rows(rows, "episode-1")
    second = runner._bounded_meta_test_rows(list(reversed(rows)), "episode-1")

    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert {domain: sum(sample.domain == domain for sample in first) for domain in ("a", "b")} == {
        "a": 2,
        "b": 2,
    }


def test_weibo_meta_cost_controls_are_loaded():
    config = load_config(ROOT / "configs/weibo21_cross_domain.yaml")

    assert config.evolution.update_interval_samples == 25
    assert config.evolution.max_proposals_per_update == 1
    assert config.evolution.max_attribution_samples_per_update == 8
    assert config.evolution.max_counterfactuals_per_sample == 1
    assert config.meta_learning.evaluation_repeats == 1
    assert config.meta_learning.max_meta_test_samples_per_domain == 50
    assert config.meta_learning.isolate_candidate_budget
