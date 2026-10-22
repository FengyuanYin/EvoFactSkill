import json
from dataclasses import asdict, replace

import pytest
from test_meta_batch_runner import invoke, setup_runner

from evofact.config import IndependentValidationConfig, config_identity
from evofact.core.models import Sample, SampleEvaluation
from evofact.evolution.optimizer_memory import reduce_context
from evofact.evolution.source_feedback import balanced_source_rows, source_recall_failures
from evofact.validation.evaluator import evaluate


def test_source_feedback_runs_without_protection_and_keeps_labels_across_batches(
    tmp_path,
    monkeypatch,
):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config,
        independent_validation=IndependentValidationConfig(enabled=True, source_feedback=True),
    )
    samples += [
        Sample(
            f"valid-{i}",
            "fixture",
            f"Another validation claim {i}.",
            "REAL" if i % 2 else "FAKE",
            "valid",
        )
        for i in range(8, 40)
    ]
    calls = []
    original = runner._clean_validation

    async def clean(pending, role, rows, skills, save, *, progress, task, phase=None):
        if role == "training":
            return await original(
                pending, role, rows, skills, save, progress=progress, task=task, phase=phase
            )
        assert role in {"baseline", "candidate"}  # No second independent stage.
        calls.append((len(contexts), role, [s.sample_id for s in rows]))
        evaluated = [
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
        # First candidate ties and is refused; subsequent candidates improve REAL recall.
        if len(contexts) > 1 and role == "baseline":
            index = next(i for i, s in enumerate(rows) if s.label == "REAL")
            evaluated[index] = replace(evaluated[index], predicted="FAKE")
        pending[role] = {
            "traces": [
                {
                    "trace_id": f"{role}-{s.sample_id}",
                    "sample_id": s.sample_id,
                    "decision": {"label": r.predicted},
                    "errors": [],
                }
                for s, r in zip(rows, evaluated)
            ],
            "result": asdict(evaluate(evaluated)),
        }
        save()
        return True

    def no_protection(*args, **kwargs):
        raise AssertionError("source feedback must not reserve an independent pool")

    monkeypatch.setattr(runner, "_clean_validation", clean)
    monkeypatch.setattr("evofact.experiments.meta_batch_runner.independent_pool", no_protection)
    result = invoke(runner, samples, validation_rounds=3)
    assert result["status"] == "completed"
    assert len(contexts) == 3 and len(calls) == 18
    assert not contexts[0]["training_supervision"]["source_validation_feedback"]
    feedback = contexts[1]["training_supervision"]["source_validation_feedback"]
    assert len(feedback) == 1 and feedback[0]["accepted"] is False
    paired = feedback[0]["paired_comparison"]
    assert paired["summary"]["preserved"] == 6
    assert paired["summary"]["total"] == 6
    for role in ("baseline", "candidate"):
        assert len(feedback[0][role]["traces"]) == 6
        records = feedback[0][role]["supervision"]["records"]
        assert {r["gold_label"] for r in records} == {"REAL", "FAKE"}
        assert {r["outcome"] for r in records} == {"correct"}
    history = contexts[2]["training_supervision"]["source_validation_feedback"]
    assert len(history) == 2 and history[0]["accepted"] is True
    assert history[0]["paired_comparison"]["summary"]["repaired"] == 3
    assert not any(s.sample_id in json.dumps(contexts) for s in samples if s.domain == "final")
    assert not any("independent_validation" in a for a in result["attempts"])
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert len(state["source_validation_feedback"]) == 3
    assert "independent_validation" not in state["training_plan"]
    assert not (tmp_path / "independent-validation-manifest.json").exists()


def test_source_fake_recall_protects_even_when_f1_improves():
    def result(real_correct, fake_correct):
        return evaluate(
            [
                SampleEvaluation(
                    f"{label}-{i}",
                    label,
                    label if i < correct else ("FAKE" if label == "REAL" else "REAL"),
                    0.8,
                    "source",
                    allowed_labels=("REAL", "FAKE"),
                    positive_label="FAKE",
                )
                for label, correct in (("REAL", real_correct), ("FAKE", fake_correct))
                for i in range(3)
            ]
        )

    baseline, candidate = result(0, 3), result(3, 2)
    assert candidate.aggregate_metrics["macro_f1_all"] > baseline.aggregate_metrics["macro_f1_all"]
    assert source_recall_failures(baseline, candidate, max_drop=0)
    assert not source_recall_failures(baseline, baseline, max_drop=0)


@pytest.mark.parametrize("policy", [None, "unknown_version"])
def test_paired_diagnostics_legacy_resume_and_unknown_policy(tmp_path, monkeypatch, policy):
    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config,
        independent_validation=IndependentValidationConfig(enabled=True, source_feedback=True),
    )
    original = runner._optimizer_traces

    async def pause(*args, **kwargs):
        raise RuntimeError("pause before optimizer")

    monkeypatch.setattr(runner, "_optimizer_traces", pause)
    with pytest.raises(RuntimeError, match="pause before optimizer"):
        invoke(runner, samples, train_limit=2)
    path = tmp_path / "checkpoint.json"
    state = json.loads(path.read_text())
    assert state.pop("paired_source_diagnostics_policy") == "paired_source_diagnostics_v1"
    if policy is not None:
        state["paired_source_diagnostics_policy"] = policy
    path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(runner, "_optimizer_traces", original)
    if policy is not None:
        unchanged = path.read_bytes()
        with pytest.raises(ValueError, match="unsupported paired source diagnostics policy"):
            invoke(runner, samples, train_limit=2, resume=True)
        assert path.read_bytes() == unchanged and not contexts
    else:
        invoke(runner, samples, train_limit=2, max_attempts_per_batch=2, resume=True)
        saved = json.loads(path.read_text())
        assert saved["source_validation_feedback"]
        assert "paired_source_diagnostics_policy" not in saved
        assert all(
            "paired_comparison" not in entry for entry in saved["source_validation_feedback"]
        )
        assert all(
            "paired_comparison" not in entry
            for context in contexts
            for entry in context["training_supervision"]["source_validation_feedback"]
        )


def test_source_feedback_window_discards_oldest_keeps_checkpoint_records():
    old, latest = {"attempt": 1}, {"attempt": 2}
    persisted = [old, latest]
    context = {"training_supervision": {"source_validation_feedback": list(reversed(persisted))}}
    assert reduce_context(context)["kind"] == "older_source_validation_feedback"
    assert context["training_supervision"]["source_validation_feedback"] == [latest]
    assert persisted == [old, latest]


def test_source_feedback_opt_in_does_not_change_legacy_identity(tmp_path, monkeypatch):
    runner, _, _, _ = setup_runner(tmp_path, monkeypatch)
    enabled = replace(
        runner.config, independent_validation=IndependentValidationConfig(enabled=True)
    )
    legacy = asdict(enabled)["independent_validation"]
    legacy.pop("source_feedback")
    legacy.pop("evaluation_samples_per_domain")
    assert config_identity(enabled)["independent_validation"] == legacy
    with pytest.raises(ValueError, match="source_feedback"):
        IndependentValidationConfig(source_feedback=True)


def test_source_selection_balances_labels_without_shrinking_eligible_pool(tmp_path, monkeypatch):
    runner, _, samples, _ = setup_runner(tmp_path, monkeypatch)
    rows = [s for s in samples if s.domain == "valid"]
    original = list(rows)
    chosen = balanced_source_rows(
        rows, size=2, rounds=3, registry=runner.base.label_contract_registry
    )
    assert len(chosen) == 6 and len({s.sample_id for s in chosen}) == 6
    assert [s.label for s in chosen] == ["FAKE", "REAL"] * 3
    assert rows == original


def test_source_feedback_resume_keeps_frozen_rounds(tmp_path, monkeypatch):
    from test_validation_rounds import install_clean

    runner, contexts, samples, _ = setup_runner(tmp_path, monkeypatch)
    runner.config = replace(
        runner.config,
        independent_validation=IndependentValidationConfig(enabled=True, source_feedback=True),
    )
    calls = []

    def pause(phase):
        if "round 2/3 candidate" in phase:
            raise RuntimeError("pause source feedback")

    install_clean(runner, monkeypatch, calls, pause=pause)
    with pytest.raises(RuntimeError, match="pause source feedback"):
        invoke(runner, samples, train_limit=2, validation_rounds=3)
    path = tmp_path / "checkpoint.json"
    saved = json.loads(path.read_text())
    plan = saved["pending"]["discovery_round_ids"]
    assert not saved.get("source_validation_feedback")  # No partial-round release.
    install_clean(runner, monkeypatch, calls)
    result = invoke(runner, samples, train_limit=2, validation_rounds=3, resume=True)
    assert result["status"] == "completed" and len(contexts) == 1
    assert result["attempts"][0]["validation_sample_ids"] == sum(plan, [])
    saved = json.loads(path.read_text())
    assert len(saved["source_validation_feedback"]) == 1
    runner.config = replace(
        runner.config,
        independent_validation=replace(runner.config.independent_validation, source_feedback=False),
    )
    unchanged = path.read_bytes()
    with pytest.raises(ValueError):
        invoke(runner, samples, train_limit=2, validation_rounds=3, resume=True)
    assert path.read_bytes() == unchanged


def test_source_feedback_prompt_authorizes_labels_without_changing_legacy_prompt(
    tmp_path, monkeypatch
):
    from evofact.evolution.source_feedback import SOURCE_TRAINING_FEEDBACK_POLICY
    from evofact.evolution.training_feedback import TRAINING_FEEDBACK_POLICY
    from evofact.runtime.openai_backend import OpenAICompatibleBackend

    runner, _, _, _ = setup_runner(tmp_path, monkeypatch)
    skill = next(s for s in runner.base.skills if s.name == "skill_optimizer")
    backend = OpenAICompatibleBackend("https://example.invalid/v1", "unused", "test")
    legacy = {"skills": [], "training_supervision": {"records": []}}
    assert backend.optimizer_system(legacy, skill).endswith(TRAINING_FEEDBACK_POLICY)
    current = {**legacy, "source_validation_feedback_policy": True}
    assert backend.optimizer_system(current, skill).endswith(SOURCE_TRAINING_FEEDBACK_POLICY)


def test_source_manifest_accepts_shuffle_but_rejects_changed_membership(tmp_path, monkeypatch):
    from evofact.evolution.source_feedback import (
        SOURCE_FEEDBACK_POLICY,
        validate_source_feedback_manifest,
    )

    runner, _, samples, _ = setup_runner(tmp_path, monkeypatch)
    train = [s for s in samples if s.domain == "train"]
    valid = [s for s in samples if s.domain == "valid"]
    manifest = {
        "validation_feedback_policy": SOURCE_FEEDBACK_POLICY,
        "training_sample_ids": [s.sample_id for s in train],
        "source_validation_sample_ids": [s.sample_id for s in valid],
    }
    validate_source_feedback_manifest(manifest, train, list(reversed(valid)))
    with pytest.raises(ValueError, match="validation candidates"):
        validate_source_feedback_manifest(manifest, train, valid[:-1])
    with pytest.raises(ValueError, match="validation candidates"):
        validate_source_feedback_manifest(manifest, train, valid[:-1] + [valid[0]])
    with pytest.raises(ValueError, match="training plan"):
        validate_source_feedback_manifest(manifest, list(reversed(train)), valid)
    with pytest.raises(ValueError, match="policy"):
        validate_source_feedback_manifest(
            {**manifest, "validation_feedback_policy": "old"}, train, valid
        )
