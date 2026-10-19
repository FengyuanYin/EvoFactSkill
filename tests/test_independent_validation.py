import asyncio
import json
from collections import Counter
from dataclasses import asdict, replace

import pytest
from test_adversarial_batch_runner import ROOT, setup

from evofact.config import IndependentValidationConfig, config_identity, load_config
from evofact.core.models import Sample, SampleEvaluation
from evofact.data.label_registry import LabelContractRegistry, fixture_binary_contract
from evofact.experiments.meta_batch_runner import _json
from evofact.runtime.inference import InferenceRuntime
from evofact.runtime.openai_backend import BackendHTTPError
from evofact.validation.evaluator import evaluate
from evofact.validation.independent import guard_failures, independent_pool, independent_rows


def test_pool_is_disjoint_balanced_and_cross_domain():
    samples = [
        Sample(
            f"{d}-{i:04d}",
            "fixture",
            f"Distinct factual statement from domain {d}, entry {i}.",
            "REAL" if i % 2 else "FAKE",
            d,
        )
        for d in ("a", "b", "c")
        for i in range(30)
    ]
    training = [s for s in samples if int(s.sample_id.split("-")[-1]) < 4]
    registry = LabelContractRegistry((fixture_binary_contract(),))
    pool, discovery = independent_pool(
        samples, training, [], registry=registry, per_domain=8, seed=42
    )
    assert Counter((s.domain, s.label) for s in pool) == {
        (d, label): 4 for d in ("a", "b", "c") for label in ("REAL", "FAKE")
    }
    assert not {s.sample_id for s in pool} & {s.sample_id for s in training + discovery}
    again, _ = independent_pool(samples, training, [], registry=registry, per_domain=8, seed=42)
    assert [s.sample_id for s in again] == [s.sample_id for s in pool]
    heldout = independent_rows(pool, [training[0], training[4]], cross_domain=True)
    assert {s.domain for s in heldout} == {"c"}
    with pytest.raises(ValueError, match="cross-domain"):
        independent_rows(pool, [training[0], training[4], training[8]], cross_domain=True)


def test_positive_recall_can_veto_an_f1_gain():
    def rows(real_correct, fake_correct):
        result = []
        for label, count in (("REAL", real_correct), ("FAKE", fake_correct)):
            for i in range(4):
                result.append(
                    SampleEvaluation(
                        f"{label}-{i}",
                        label,
                        label if i < count else ("REAL" if label == "FAKE" else "FAKE"),
                        0.8,
                        "heldout",
                        label_schema_id="fixture",
                        allowed_labels=("REAL", "FAKE"),
                        positive_label="FAKE",
                    )
                )
        return evaluate(result)

    baseline, candidate = rows(1, 4), rows(4, 3)
    assert candidate.aggregate_metrics["macro_f1_all"] > baseline.aggregate_metrics["macro_f1_all"]
    assert any(
        "positive-label recall" in s
        for s in guard_failures(baseline, candidate, IndependentValidationConfig())
    )
    assert not guard_failures(baseline, baseline, IndependentValidationConfig())


def test_evaluation_subset_preserves_pool_and_balances_each_eligible_domain():
    registry = LabelContractRegistry((fixture_binary_contract(),))
    pool = [
        Sample(f"{d}-{i}", "fixture", f"Story {d} {i}", "REAL" if i % 2 else "FAKE", d)
        for d in ("a", "b", "c")
        for i in range(32)
    ]
    original_ids = [s.sample_id for s in pool]
    kwargs = dict(cross_domain=True, per_domain=4, registry=registry, seed=50)
    selected = independent_rows(pool, [pool[0]], **kwargs)
    assert Counter((s.domain, s.label) for s in selected) == {
        (d, label): 2 for d in ("b", "c") for label in ("REAL", "FAKE")
    }
    assert [s.sample_id for s in selected] == [
        s.sample_id for s in independent_rows(pool, [pool[0]], **kwargs)
    ]
    assert [s.sample_id for s in pool] == original_ids
    all_domains = independent_rows(
        pool, [pool[0]], cross_domain=False, per_domain=4, registry=registry
    )
    assert len(all_domains) == 12


@pytest.mark.parametrize("size", [0, 3, 34, True])
def test_invalid_independent_evaluation_size(size):
    with pytest.raises(ValueError, match="evaluation_samples_per_domain"):
        IndependentValidationConfig(evaluation_samples_per_domain=size)


def test_disabled_policy_preserves_legacy_config_identity():
    config = load_config(ROOT / "configs/dry_run.yaml")
    legacy = asdict(config)
    legacy.pop("independent_validation")
    legacy.pop("optimizer_memory")
    legacy.pop("generator_memory")
    assert config_identity(config) == legacy
    enabled = replace(config, independent_validation=IndependentValidationConfig(enabled=True))
    assert config_identity(enabled)["independent_validation"]["enabled"] is True


def test_validation_resumes_only_failed_samples_and_keeps_wrong_predictions(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config,
        independent_validation=IndependentValidationConfig(
            enabled=True, samples_per_domain=2, sample_retries=0
        ),
    )
    original = InferenceRuntime.infer
    counts = Counter()
    fail = True

    async def infer(self, sample, *args, **kwargs):
        counts[sample.sample_id] += 1
        if sample.sample_id == samples[0].sample_id and fail:
            raise TimeoutError("temporary service failure")
        trace = await original(self, sample, *args, **kwargs)
        # An ordinary wrong business decision is a complete evaluation, not a retry.
        return replace(
            trace,
            decision=replace(trace.decision, label="REAL" if sample.label == "FAKE" else "FAKE"),
        )

    monkeypatch.setattr(InferenceRuntime, "infer", infer)
    pending = {}

    def save():
        pass

    async def run():
        return await runner._clean_validation(
            pending, "baseline", samples[:2], runner.base.skills, save, progress=None, task="test"
        )

    assert asyncio.run(run()) is False
    assert "baseline" not in pending
    assert pending["baseline_progress"]["failures"]
    fail = False
    assert asyncio.run(run()) is True
    assert counts[samples[0].sample_id] == 2
    assert counts[samples[1].sample_id] == 1
    assert pending["baseline"]["result"]["aggregate_metrics"]["n"] == 2
    assert pending["baseline"]["result"]["aggregate_metrics"]["accuracy_all"] == 0


def test_guard_vetoes_promotion_and_never_exposes_pool_to_generator(tmp_path, monkeypatch):
    runner, backend, contexts, calls, _ = setup(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config,
        execution=replace(runner.config.execution, batch_size=4),
        independent_validation=IndependentValidationConfig(
            enabled=True, samples_per_domain=8, evaluation_samples_per_domain=4
        ),
    )
    samples = [
        Sample(
            f"{d}-{i:04d}",
            "fixture",
            f"Factual story with domain {d} and entry number {i}.",
            "REAL" if i % 2 else "FAKE",
            d,
        )
        for d in ("a", "b", "c", "final")
        for i in range(30)
    ]
    training_domains = {}
    original = runner.base.run
    _, _, train, valid, _, _, _ = runner._prepare_plan(
        samples,
        final_test_domains=("final",),
        train_domains=("a", "b", "c"),
        validation_domains=None,
        train_limit=12,
        match_evolve_training=True,
    )
    pool, _ = independent_pool(
        valid,
        train,
        [s for s in samples if s.domain == "final"],
        registry=runner.base.label_contract_registry,
        per_domain=8,
        seed=runner.config.seed,
    )
    manifest_path = tmp_path / "independent.json"
    manifest_path.write_text(
        json.dumps(
            {
                "training_sample_ids": [s.sample_id for s in train],
                "independent_samples": [
                    {"sample_id": s.sample_id, "domain": s.domain, "label": s.label} for s in pool
                ],
            }
        ),
        encoding="utf-8",
    )
    runner.config = replace(
        runner.config,
        independent_validation=replace(
            runner.config.independent_validation, manifest_path=manifest_path
        ),
    )

    async def clean(pending, role, rows, skills, save, *, progress, task, phase=None):
        if role == "training":
            training_domains[task] = {s.domain for s in rows}
        if role.startswith("independent"):
            assert not training_domains[task] & {s.domain for s in rows}
            assert all(n == 2 for n in Counter((s.domain, s.label) for s in rows).values())
            if role == "independent_candidate":
                assert [s.sample_id for s in rows] == pending["independent_sample_ids"]
        traces, _ = await original(rows, skills=skills, phase=phase or "validation " + role)
        evaluations = [
            SampleEvaluation(
                s.sample_id,
                s.label,
                s.label,
                0.8,
                s.domain,
                label_schema_id="fixture",
                allowed_labels=("REAL", "FAKE"),
                positive_label="FAKE",
            )
            for s in rows
        ]
        if role == "baseline":
            evaluations[0] = replace(
                evaluations[0], predicted="FAKE" if evaluations[0].gold == "REAL" else "REAL"
            )
        if role == "independent_candidate":
            i = next(i for i, r in enumerate(evaluations) if r.gold == "FAKE")
            evaluations[i] = replace(evaluations[i], predicted="REAL")
        pending[role] = {
            "traces": _json([asdict(t) for t in traces]),
            "result": _json(asdict(evaluate(evaluations))),
        }
        pending[role + "_progress"] = {"failures": []}
        return True

    monkeypatch.setattr(runner, "_clean_validation", clean)
    result = asyncio.run(
        runner.run(
            samples,
            final_test_domains=("final",),
            train_domains=("a", "b", "c"),
            train_limit=12,
            match_evolve_training=True,
            max_attempts_per_batch=1,
        )
    )
    assert result["status"] == "completed"
    judged = [a for a in result["attempts"] if "decision" in a]
    assert judged and all(not a["decision"]["accepted"] for a in judged)
    assert all("positive-label recall" in a["decision"]["reason"] for a in judged)
    pool_ids = set(result["training_sampling"]["independent_validation"]["pool_sample_ids"])
    for request in backend.requests:
        assert not pool_ids & {e["source_sample"]["sample_id"] for e in request["examples"]}
        assert not pool_ids & {e["trace"]["sample_id"] for e in request["examples"]}
        assert not request["validation_feedback"]


def test_changed_manifest_fails_before_any_inference(tmp_path, monkeypatch):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    manifest = tmp_path / "independent.json"
    manifest.write_text(
        json.dumps({"training_sample_ids": [], "independent_samples": []}), encoding="utf-8"
    )
    runner.config = replace(
        runner.config,
        independent_validation=IndependentValidationConfig(
            enabled=True, samples_per_domain=2, manifest_path=manifest
        ),
    )
    with pytest.raises(ValueError, match="manifest does not match"):
        asyncio.run(
            runner.run(samples, final_test_domains=("final",), validation_domains=("valid",))
        )
    assert not calls and not backend.requests


@pytest.mark.parametrize("error_type", [TimeoutError, ValueError, BackendHTTPError])
def test_transient_runtime_errors_are_retried_and_audited(tmp_path, monkeypatch, error_type):
    runner, backend, contexts, calls, samples = setup(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config,
        independent_validation=IndependentValidationConfig(
            enabled=True, samples_per_domain=2, sample_retries=1
        ),
    )
    original = InferenceRuntime.infer
    counts = Counter()

    async def transient(self, sample, *args, **kwargs):
        counts[sample.sample_id] += 1
        if sample.sample_id == samples[0].sample_id and counts[sample.sample_id] == 1:
            raise error_type("temporary execution failure")
        return await original(self, sample, *args, **kwargs)

    monkeypatch.setattr(InferenceRuntime, "infer", transient)
    pending = {}

    async def run():
        return await runner._clean_validation(
            pending,
            "baseline",
            samples[:2],
            runner.base.skills,
            lambda: None,
            progress=None,
            task="test",
        )

    assert asyncio.run(run()) is True
    assert counts[samples[0].sample_id] == 2 and counts[samples[1].sample_id] == 1
    assert len(pending["baseline_progress"]["failures"]) == 1
    assert pending["baseline_progress"]["failures"][0]["error_type"] == error_type.__name__
    assert pending["baseline"]["result"]["aggregate_metrics"]["n"] == 2
