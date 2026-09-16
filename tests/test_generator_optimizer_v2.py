from __future__ import annotations

from decimal import Decimal

from evofact.core.generation_models import GenerationMetrics, PairedGeneratorEvaluation
from evofact.validation.generator_gate import GeneratorGate


def _metrics(**changes):
    values = {
        "validity_rate": 0.9,
        "agreement_rate": 0.9,
        "evidence_rate": 0.9,
        "diversity": 0.8,
        "duplicate_rate": 0.1,
        "difficulty": 0.7,
        "teaching_gain": 0.1,
        "leakage_rate": 0.0,
        "safety_rejection_rate": 0.0,
        "cost": Decimal("1"),
    }
    values.update(changes)
    return GenerationMetrics(**values)


def test_generator_gate_compares_diversity_and_difficulty():
    evaluation = PairedGeneratorEvaluation(
        "source-episode",
        "evaluation-episode",
        "champion",
        "challenger",
        _metrics(),
        _metrics(diversity=0.7, difficulty=0.6),
    )
    decision = GeneratorGate().decide(evaluation)
    assert not decision.accepted
    assert "diversity delta below threshold" in decision.failures
    assert "difficulty delta below threshold" in decision.failures


def test_generator_gate_requires_independent_episode_and_known_cost():
    same_episode = PairedGeneratorEvaluation(
        "episode",
        "episode",
        "champion",
        "challenger",
        _metrics(),
        _metrics(),
    )
    assert not GeneratorGate().decide(same_episode).accepted

    unavailable_cost = PairedGeneratorEvaluation(
        "source",
        "evaluation",
        "champion",
        "challenger",
        _metrics(),
        _metrics(cost=None),
    )
    assert "cost unavailable" in GeneratorGate().decide(unavailable_cost).failures
