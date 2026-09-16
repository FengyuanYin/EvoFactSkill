from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from evofact.core.generation_models import GenerationMetrics, PairedGeneratorEvaluation
from evofact.core.models import Evidence, Sample
from evofact.experiments.checkpoint import CheckpointIdentity, CheckpointV2Store
from evofact.generation.lineage import group_real_samples
from evofact.generation.split import split_real_only
from evofact.validation.generator_gate import GeneratorGate


def _samples():
    return [
        Sample("a", "d", "same claim", "REAL", "one", "e1", evidence=(Evidence("fact a"),)),
        Sample("a2", "d", "same claim", "REAL", "one", "e1", evidence=(Evidence("fact a"),)),
        Sample("b", "d", "claim b", "FAKE", "two", "e2", evidence=(Evidence("fact b"),)),
        Sample("c", "d", "claim c", "REAL", "three", "e3", evidence=(Evidence("fact c"),)),
        Sample("d", "d", "claim d", "FAKE", "four", "e4", evidence=(Evidence("fact d"),)),
    ]


def test_lineage_and_real_only_split_are_deterministic() -> None:
    groups = group_real_samples(_samples())
    same = next(group for group in groups if "a" in group.sample_ids)
    assert same.sample_ids == ("a", "a2")

    first = split_real_only(_samples(), seed="42")
    second = split_real_only(_samples(), seed="42")
    assert first == second
    assert not (set(first.meta_train_ids) & set(first.meta_test_ids))
    assert not (set(first.meta_test_ids) & set(first.final_test_ids))
    assert (
        {"a", "a2"} <= set(first.meta_train_ids)
        or {"a", "a2"} <= set(first.meta_test_ids)
        or {"a", "a2"} <= set(first.final_test_ids)
    )


def _metrics(validity: float, cost: str) -> GenerationMetrics:
    return GenerationMetrics(validity, 1, 1, 1, 0, 0.5, 0.2, 0, 0, Decimal(cost))


def test_generator_gate_requires_independent_episode() -> None:
    evaluation = PairedGeneratorEvaluation(
        "same", "same", "old", "new", _metrics(0.8, "1"), _metrics(0.9, "1")
    )
    decision = GeneratorGate().decide(evaluation)
    assert not decision.accepted
    assert any("episodes" in failure for failure in decision.failures)


def test_generator_script_candidate_requires_review() -> None:
    evaluation = PairedGeneratorEvaluation(
        "source", "evaluation", "old", "new", _metrics(0.8, "1"), _metrics(0.9, "1")
    )
    decision = GeneratorGate().decide(evaluation, safety_level="review_required")
    assert decision.disposition == "review_required"
    assert not decision.accepted


def test_checkpoint_v2_rejects_identity_drift(tmp_path) -> None:
    identity = CheckpointIdentity(*("x" for _ in range(11)))
    store = CheckpointV2Store(tmp_path / "checkpoint.json")
    store.save(identity, {"completed": ["episode-1"]})
    assert store.load(identity)["completed"] == ["episode-1"]

    changed = replace(identity, pricing_version="new")
    with pytest.raises(ValueError, match="pricing_version"):
        store.load(changed)
