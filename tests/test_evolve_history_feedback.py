"""Exercise history delivery to the model, including checkpoint-shaped records."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from optimizer_stub import EditingOptimizerBackend

from evofact.config import load_config
from evofact.evolution.optimizer_context import build_skillbank_optimizer_context
from evofact.experiments.runner import ExperimentRunner, fixture_samples, proposal_history

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("accepted", [False, True])
def test_validation_history_reaches_next_optimizer_after_serialization(monkeypatch, accepted):
    runner = ExperimentRunner(load_config(ROOT / "configs/dry_run.yaml"), ROOT)
    contexts = []

    class Optimizer(EditingOptimizerBackend):
        async def optimize_package(self, context, skill):
            contexts.append(context)
            return await super().optimize_package(context, skill)

    monkeypatch.setattr(runner, "_optimizer_backend", Optimizer)
    first = asyncio.run(runner.closed_loop(fixture_samples()))
    assert len(first["proposals"]) == 1
    # Exercise both outcomes independently of incidental mock predictions.
    reason = "macro-F1 improved" if accepted else "macro-F1 did not improve"
    decision = replace(
        first["gate_decisions"][0],
        accepted=accepted,
        disposition="active" if accepted else "rejected",
        reason=reason,
        regression_failures=() if accepted else (reason,),
    )
    from evofact.cli import _stable_value

    restored = json.loads(json.dumps(_stable_value([first["proposals"][0], decision])))
    history = proposal_history(runner.packages, [restored[0]], [restored[1]])
    asyncio.run(runner.closed_loop(fixture_samples(), prior_proposals=history))
    assert "prior_proposals" not in contexts[0]
    received = contexts[1]["prior_proposals"][0]
    assert received["proposal_id"] == first["proposals"][0].proposal_id
    assert received["target"] == "source_credibility"
    assert received["disposition"] == decision.disposition
    assert received["reason"] == decision.reason
    assert received["failures"] == list(decision.regression_failures)
    assert (
        received["metrics"]["baseline"]["macro_f1_all"]
        == decision.baseline_result.aggregate_metrics["macro_f1_all"]
    )
    assert "gold" not in json.dumps(contexts[1])


def test_history_is_bounded_sanitized_and_keeps_latest_when_context_is_full():
    histories = [
        {"proposal_id": str(i), "reason": "x" * 200, "gold": "FAKE", "api_key": "secret"}
        for i in range(20)
    ]
    context = build_skillbank_optimizer_context(packages=(), traces=(), prior_proposals=histories)
    assert [h["proposal_id"] for h in context["prior_proposals"]] == [
        str(i) for i in range(19, 7, -1)
    ]
    assert all("gold" not in h and "api_key" not in h for h in context["prior_proposals"])
    reduced = build_skillbank_optimizer_context(
        packages=(),
        traces=(),
        prior_proposals=histories,
        max_total_chars=len(json.dumps(context, ensure_ascii=False)) - 300,
    )
    assert 0 < len(reduced["prior_proposals"]) < 12
    assert reduced["prior_proposals"][0]["proposal_id"] == "19"
