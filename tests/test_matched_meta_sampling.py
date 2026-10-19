import asyncio
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from optimizer_stub import EditingOptimizerBackend
from test_adversarial_batch_runner import Generator

from evofact.config import load_config
from evofact.core.models import Evidence, Sample
from evofact.data.manifests import build_manifest
from evofact.data.sampling import balanced_domain_sample
from evofact.experiments.adversarial_batch_runner import AdversarialBatchRunner
from evofact.experiments.meta_batch_runner import MetaBatchRunner
from evofact.experiments.meta_sampling import matched_meta_plan

ROOT = Path(__file__).resolve().parents[1]


def samples(domains, count):
    return [
        Sample(
            f"sample-{domain}-{i:04d}",
            "fixture",
            f"Claim from {domain} number {i}.",
            "REAL" if i % 2 else "FAKE",
            domain,
            evidence=(Evidence(f"Evidence from {domain} item {i}.", f"fixture://{domain}/{i}"),),
        )
        for domain in (*domains, "final")
        for i in range(count)
    ]


@pytest.mark.parametrize("domain_count,quota,batch_count", [(4, 50, 13), (5, 40, 13)])
def test_exact_ordinary_training_ids_and_mixed_domain_batches(domain_count, quota, batch_count):
    domains = tuple(f"domain-{i}" for i in range(domain_count))
    rows = samples(domains, 100)
    manifest = build_manifest(rows, seed=42, train_domains=domains, final_test_domains=("final",))
    expected, _ = balanced_domain_sample(
        [s for s in rows if s.sample_id in set(manifest.train_ids)], domains, 200, seed=42
    )
    selected, batches, validation, audit = matched_meta_plan(
        rows,
        source_domains=domains,
        final_domains=("final",),
        seed=42,
        batch_size=16,
        train_limit=200,
        label_contract_digest="",
    )
    assert [s.sample_id for s in selected] == [s.sample_id for s in expected]
    assert Counter(s.domain for s in selected) == {d: quota for d in domains}
    assert len(batches) == batch_count
    assert [len(batch) for batch in batches] == [16] * 12 + [8]
    assert all(2 <= len({s.domain for s in batch}) < domain_count for batch in batches)
    assert audit["training_layout"] == "matched_mixed_domain_v2"
    _, repeated, _, _ = matched_meta_plan(
        rows,
        source_domains=domains,
        final_domains=("final",),
        seed=42,
        batch_size=16,
        train_limit=200,
        label_contract_digest="",
    )
    assert [[s.sample_id for s in b] for b in repeated] == audit["batch_sample_ids"]
    assert Counter(s.sample_id for b in batches for s in b) == Counter(
        s.sample_id for s in selected
    )
    assert set(s.sample_id for s in selected) <= set(s.sample_id for s in validation)
    assert all(s.domain != "final" for s in validation)
    assert audit["validation_training_overlap_allowed"] is True


@pytest.mark.parametrize("adversarial", [False, True])
@pytest.mark.parametrize("resume_after_failure", [False, True])
def test_matched_routes_use_cross_domain_validation_and_allow_training_overlap(
    tmp_path, monkeypatch, adversarial, resume_after_failure
):
    domains = ("a", "b", "c")
    rows = samples(domains, 20)
    config = load_config(ROOT / "configs/adversarial_llm.yaml")
    config = replace(
        config,
        backend="mock",
        model="mock-v1",
        execution=replace(config.execution, batch_size=4),
        data=replace(
            config.data,
            dataset="fixture",
            root=tmp_path,
            train_domains=domains,
            final_test_domains=("final",),
            evolution_validation_samples=2,
        ),
        generation=replace(config.generation, batch_size=2),
        meta_learning=replace(config.meta_learning, checkpoint_path=tmp_path / "checkpoint.json"),
    )
    if adversarial:
        runner = AdversarialBatchRunner(config, ROOT, generation_backend=Generator())
    else:
        runner = MetaBatchRunner(config, ROOT)
    count = 0

    class Optimizer(EditingOptimizerBackend):
        async def optimize_package(self, context, skill):
            nonlocal count
            count += 1
            result = await super().optimize_package(context, skill)
            result.value["file_operations"][0]["content"] += f"Revision {count}.\n"
            return result

    monkeypatch.setattr(runner.base, "_optimizer_backend", Optimizer)
    original = runner.base.run
    training, validations, trained_ids, reused_trained = {}, [], set(), []

    async def run(selected, **kwargs):
        task, phase = kwargs["task_name"], kwargs["phase"]
        if phase == "train inference":
            assert 2 <= len({s.domain for s in selected}) < len(domains)
            training[task] = list(selected)
            trained_ids.update(s.sample_id for s in selected)
        if phase == "validation baseline":
            assert {s.domain for s in selected}.isdisjoint({s.domain for s in training[task]})
            validations.extend(s.sample_id for s in selected)
            reused_trained.extend(s.sample_id for s in selected if s.sample_id in trained_ids)
        traces, result = await original(selected, **kwargs)
        if phase.startswith("validation"):
            score = (
                0.75
                if phase == "validation candidate" and not (resume_after_failure and count == 1)
                else 0.5
            )
            result = replace(
                result, aggregate_metrics={**result.aggregate_metrics, "macro_f1_all": score}
            )
        return traces, result

    monkeypatch.setattr(runner.base, "run", run)
    options = dict(
        final_test_domains=("final",),
        train_domains=domains,
        train_limit=18,
        match_evolve_training=True,
    )
    if resume_after_failure:
        original_save = runner._after_save
        interrupted = False

        def pause_after_rejection(checkpoint):
            nonlocal interrupted
            original_save(checkpoint)
            import json

            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if state.get("feedback") and not state.get("pending") and not interrupted:
                interrupted = True
                raise RuntimeError("pause after rejection")

        monkeypatch.setattr(runner, "_after_save", pause_after_rejection)
        with pytest.raises(RuntimeError, match="pause after rejection"):
            asyncio.run(runner.run(rows, max_attempts_per_batch=2, **options))
        result = asyncio.run(runner.run(rows, resume=True, max_attempts_per_batch=2, **options))
    else:
        result = asyncio.run(runner.run(rows, **options))
    assert result["status"] == "completed"
    assert result["total_batches"] == 5
    assert Counter(s.domain for batch in training.values() for s in batch) == {
        d: 6 for d in domains
    }
    assert len(set(validations)) == len(validations)
    assert reused_trained, (
        "previously trained samples must remain eligible for other-domain validation"
    )
    assert result["training_sampling"]["selected_by_domain"] == {d: 6 for d in domains}
    assert all(
        set(a["train_domains"]).isdisjoint(a["validation_domains"]) for a in result["attempts"]
    )
