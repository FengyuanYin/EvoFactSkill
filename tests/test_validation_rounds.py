"""Predetermined paired rounds aggregate predictions, never cherry-pick passes."""

import json
from collections import Counter
from dataclasses import asdict, replace

import pytest
from test_meta_batch_runner import invoke, setup_runner

from evofact.config import IndependentValidationConfig
from evofact.core.models import Sample, SampleEvaluation
from evofact.experiments.meta_batch_runner import MetaBatchRunner, _json
from evofact.validation.evaluator import evaluate
from evofact.validation.rounds import AGGREGATION_POLICY, independent_rounds


def install_clean(runner, monkeypatch, calls, *, guard_drop=False, pause=None):
    async def clean(pending, role, rows, skills, save, *, progress, task, phase=None):
        if role == "training":
            return await MetaBatchRunner._clean_validation(
                runner,
                pending,
                role,
                rows,
                skills,
                save,
                progress=progress,
                task=task,
                phase=phase,
            )
        phase = phase or role
        if pause:
            pause(phase)
        calls.append((phase, [s.sample_id for s in rows]))
        # All phases are paired on the same frozen candidate; no early first-round gate.
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
        if "discovery" in phase and role == "baseline" and "round 1/" not in phase:
            evaluations[0] = replace(
                evaluations[0], predicted="REAL" if rows[0].label == "FAKE" else "FAKE"
            )
        if guard_drop and "independent round 3/" in phase and role == "candidate":
            i = next(i for i, r in enumerate(evaluations) if r.gold == "FAKE")
            evaluations[i] = replace(evaluations[i], predicted="REAL")
        pending[role] = {
            "traces": [
                {"trace_id": "trace-" + s.sample_id, "sample_id": s.sample_id} for s in rows
            ],
            "result": _json(asdict(evaluate(evaluations))),
        }
        save()
        return True

    monkeypatch.setattr(runner, "_clean_validation", clean)


def test_three_rounds_run_even_when_first_baseline_is_perfect(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    calls = []
    install_clean(runner, monkeypatch, calls)
    result = invoke(runner, samples, validation_rounds=3, train_limit=2)
    assert result["status"] == "completed"
    assert len(contexts) == 1  # No intermediate validation feedback or optimization.
    audit = result["attempts"][0]
    assert audit["decision"]["accepted"]
    rounds = audit["validation_aggregation"]["stages"]["discovery"]
    assert len(rounds) == 3 and rounds[0]["baseline"]["macro_f1_all"] == 1
    ids = [sid for r in rounds for sid in r["sample_ids"]]
    assert len(set(ids)) == 6
    for i in range(3):
        assert calls[2 * i][1] == calls[2 * i + 1][1] == rounds[i]["sample_ids"]
    baseline = audit["decision"]["baseline_result"]
    assert len(baseline["per_sample"]) == 6
    pooled = evaluate(
        [
            SampleEvaluation(**{**r, "allowed_labels": tuple(r["allowed_labels"])})
            for r in baseline["per_sample"]
        ]
    )
    assert baseline["aggregate_metrics"]["macro_f1_all"] == pooled.aggregate_metrics["macro_f1_all"]
    assert baseline["aggregate_metrics"]["macro_f1_all"] != pytest.approx(
        sum(r["baseline"]["macro_f1_all"] for r in rounds) / 3
    )
    assert result["validation_samples_used"] == 6


def test_resume_uses_same_all_round_plan_and_skips_completed_roles(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    calls = []

    def pause(phase):
        if "round 2/3 candidate" in phase:
            raise RuntimeError("interrupt fixed plan")

    install_clean(runner, monkeypatch, calls, pause=pause)
    with pytest.raises(RuntimeError, match="interrupt fixed plan"):
        invoke(runner, samples, validation_rounds=3, train_limit=2)
    checkpoint = tmp_path / "checkpoint.json"
    saved = json.loads(checkpoint.read_text())
    plan = saved["pending"]["discovery_round_ids"]
    assert len(plan) == 3 and len(calls) == 3
    install_clean(runner, monkeypatch, calls)
    result = invoke(runner, samples, validation_rounds=3, train_limit=2, resume=True)
    assert result["status"] == "completed"
    assert len(contexts) == 1 and len(calls) == 6
    assert result["attempts"][0]["validation_sample_ids"] == sum(plan, [])


def test_round_count_cannot_change_on_resume(tmp_path, monkeypatch):
    runner, _, samples, _ = setup_runner(tmp_path, monkeypatch)
    install_clean(runner, monkeypatch, [])
    invoke(runner, samples, validation_rounds=3, train_limit=2)
    with pytest.raises(ValueError, match="checkpoint-recorded validation rounds"):
        invoke(runner, samples, validation_rounds=2, train_limit=2, resume=True)


def test_legacy_checkpoint_retains_single_round_with_new_cli_default(tmp_path, monkeypatch):
    runner, _, samples, _ = setup_runner(tmp_path, monkeypatch)
    original = runner._optimizer_traces

    async def pause(*args, **kwargs):
        raise RuntimeError("pause legacy")

    monkeypatch.setattr(runner, "_optimizer_traces", pause)
    with pytest.raises(RuntimeError, match="pause legacy"):
        invoke(runner, samples, train_limit=2)
    path = tmp_path / "checkpoint.json"
    saved = json.loads(path.read_text())
    saved.pop("validation_aggregation")
    path.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(runner, "_optimizer_traces", original)
    result = invoke(runner, samples, train_limit=2, validation_rounds=3, resume=True)
    assert result["validation_aggregation"]["rounds"] == 1
    assert all(len(a["validation_sample_ids"]) == 2 for a in result["attempts"])


def test_independent_rounds_are_disjoint_balanced_and_cross_domain(tmp_path, monkeypatch):
    runner, _, _, _ = setup_runner(tmp_path, monkeypatch)
    pool = [
        Sample(f"{d}-{i}", "fixture", f"Claim {d} {i}", "REAL" if i % 2 else "FAKE", d)
        for d in ("a", "b", "c")
        for i in range(32)
    ]
    plans = independent_rounds(
        pool,
        [pool[0]],
        rounds=3,
        per_domain=4,
        registry=runner.base.label_contract_registry,
        seed=42,
        cross_domain=True,
    )
    by_id = {s.sample_id: s for s in pool}
    assert len(set(sum(plans, []))) == 24
    for ids in plans:
        assert Counter((by_id[s].domain, by_id[s].label) for s in ids) == {
            (d, label): 2 for d in ("b", "c") for label in ("REAL", "FAKE")
        }
    assert len(pool) == 96
    with pytest.raises(ValueError, match="lacks"):
        independent_rounds(
            pool[:8],
            [],
            rounds=3,
            per_domain=4,
            registry=runner.base.label_contract_registry,
            seed=42,
            cross_domain=False,
        )


def test_pooled_independent_recall_still_vetoes_candidate(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config,
        independent_validation=IndependentValidationConfig(
            enabled=True,
            samples_per_domain=12,
            evaluation_samples_per_domain=4,
        ),
    )
    samples = [s for s in samples if s.domain != "valid"] + [
        Sample(
            f"valid-{i}",
            "fixture",
            f"Distinct valid factual claim {i}.",
            "REAL" if i % 2 else "FAKE",
            "valid",
        )
        for i in range(40)
    ]
    calls = []
    install_clean(runner, monkeypatch, calls, guard_drop=True)
    result = invoke(runner, samples, validation_rounds=3, train_limit=2, max_attempts_per_batch=1)
    audit = result["attempts"][0]
    assert not audit["decision"]["accepted"]
    assert "positive-label recall" in audit["decision"]["reason"]
    stages = audit["validation_aggregation"]["stages"]
    assert len(stages["independent"]) == 3
    assert len(audit["independent_validation"]["sample_ids"]) == 12
    assert len(contexts) == 1
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved["validation_aggregation"]["policy"] == AGGREGATION_POLICY


def test_insufficient_discovery_never_reduces_rounds_or_calls_optimizer(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    samples = [s for s in samples if s.domain != "valid" or int(s.sample_id[-1]) < 4]
    result = invoke(runner, samples, validation_rounds=3)
    assert result["status"] == "validation_pool_exhausted"
    assert not contexts and result["validation_samples_used"] == 0


def test_rejected_candidate_gets_feedback_only_after_all_rounds_and_new_ids(tmp_path, monkeypatch):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    samples = [s for s in samples if s.domain != "valid"] + [
        Sample(
            f"valid-{i}",
            "fixture",
            f"Distinct validation claim {i}.",
            "REAL" if i % 2 else "FAKE",
            "valid",
        )
        for i in range(20)
    ]
    install_clean(runner, monkeypatch, [])
    clean = runner._clean_validation

    async def reject_first(pending, role, rows, skills, save, **kwargs):
        success = await clean(pending, role, rows, skills, save, **kwargs)
        if len(contexts) == 1 and role == "candidate":
            result = pending[role]["result"]
            evaluations = [
                SampleEvaluation(
                    **{
                        **r,
                        "allowed_labels": tuple(r["allowed_labels"]),
                        "predicted": "REAL" if r["gold"] == "FAKE" else "FAKE",
                    }
                )
                for r in result["per_sample"]
            ]
            pending[role]["result"] = _json(asdict(evaluate(evaluations)))
        return success

    monkeypatch.setattr(runner, "_clean_validation", reject_first)
    result = invoke(runner, samples, validation_rounds=3, train_limit=2)
    assert len(contexts) == 2
    assert len(contexts[1]["validation_feedback"][0]["baseline_traces"]) == 6
    first, second = result["attempts"]
    assert not first["decision"]["accepted"] and second["decision"]["accepted"]
    assert not set(first["validation_sample_ids"]) & set(second["validation_sample_ids"])
    assert len(first["validation_sample_ids"]) == len(second["validation_sample_ids"]) == 6


def test_cli_defaults_three_rounds_without_changing_ordinary_evolve():
    from evofact.cli import build_parser

    for route in ("feedback-evolve", "meta-evolve", "adversarial-evolve"):
        assert build_parser().parse_args([route]).validation_rounds == 3
    assert not hasattr(build_parser().parse_args(["evolve"]), "validation_rounds")


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_invalid_round_count_rejected(tmp_path, monkeypatch, rounds):
    runner, _, samples, _ = setup_runner(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="positive integer"):
        invoke(runner, samples, validation_rounds=rounds)
