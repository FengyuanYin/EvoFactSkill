import asyncio
from dataclasses import replace
from pathlib import Path

from evofact.config import load_config
from evofact.core.models import (
    EvolutionOperation,
    EvolutionProposal,
    Sample,
    SampleEvaluation,
)
from evofact.experiments.meta_runner import MetaEvolutionRunner
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
    assert config.meta_learning.max_meta_test_samples_per_domain == 10
    assert config.meta_learning.isolate_candidate_budget
