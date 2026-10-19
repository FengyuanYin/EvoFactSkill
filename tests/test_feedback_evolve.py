import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from optimizer_stub import EditingOptimizerBackend

from evofact.cli import build_parser
from evofact.config import load_config
from evofact.core.models import Sample
from evofact.experiments.feedback_evolve_runner import FeedbackEvolveRunner

ROOT = Path(__file__).resolve().parents[1]


def setup(tmp_path, monkeypatch, *, valid_count=8):
    config = load_config(ROOT / "configs/dry_run.yaml")
    config = replace(
        config,
        output_dir=tmp_path / "output",
        execution=replace(config.execution, batch_size=2),
        data=replace(config.data, evolution_validation_samples=2),
    )
    runner = FeedbackEvolveRunner(config, ROOT, checkpoint_path=tmp_path / "checkpoint.json")
    contexts, calls = [], []

    class Optimizer(EditingOptimizerBackend):
        async def optimize_package(self, context, skill):
            contexts.append(context)
            response = await super().optimize_package(context, skill)
            response.value["file_operations"][0]["content"] += f"Revision {len(contexts)}."
            return response

    monkeypatch.setattr(runner.base, "_optimizer_backend", Optimizer)
    original = runner.base.run

    async def infer(rows, **kwargs):
        calls.append((kwargs["phase"], [s.sample_id for s in rows]))
        traces, result = await original(rows, **kwargs)
        if kwargs["phase"].startswith("validation"):
            value = (
                0.75 if kwargs["phase"] == "validation candidate" and len(contexts) >= 2 else 0.5
            )
            result = replace(
                result, aggregate_metrics={**result.aggregate_metrics, "macro_f1_all": value}
            )
        return traces, result

    monkeypatch.setattr(runner.base, "run", infer)
    # Ordinary evolution permits train and validation to share domains, not samples.
    train = [
        Sample(f"train-{i}", "fixture", f"Training assertion {i}", "REAL", "same-domain")
        for i in range(4)
    ]
    valid = [
        Sample(f"valid-{i}", "fixture", f"Validation statement {i}", "FAKE", "same-domain")
        for i in range(valid_count)
    ]
    return runner, train, valid, contexts, calls


def invoke(runner, train, valid, **kwargs):
    return asyncio.run(runner.run(train, valid, evaluation_only=True, **kwargs))


def test_feedback_retry_fresh_validation_and_cross_batch_history(tmp_path, monkeypatch):
    runner, train, valid, contexts, calls = setup(tmp_path, monkeypatch)
    outcome = invoke(runner, train, valid)
    assert outcome["status"] == "completed"
    assert outcome["completed_batches"] == 2
    assert len(contexts) == 3
    assert all(context["feedback_enabled"] is True for context in contexts)
    feedback = contexts[1]["validation_feedback"][0]
    assert feedback["gate"]["accepted"] is False
    assert feedback["baseline_traces"] and feedback["candidate_traces"]
    assert "gold" not in json.dumps(contexts[1])
    assert contexts[1]["prior_proposals"][0]["disposition"] == "rejected"
    assert not contexts[2].get("validation_feedback")
    assert contexts[2]["prior_proposals"][0]["disposition"] == "active"
    assert len([c for c in calls if c[0] == "train inference"]) == 2
    used = [sid for phase, ids in calls if phase == "validation baseline" for sid in ids]
    assert len(used) == len(set(used)) == 6
    assert all(t["sample_id"].startswith("train-") for c in contexts for t in c["traces"])
    assert len(outcome["proposals"]) == len(outcome["gate_decisions"]) == 3
    assert (tmp_path / "output/validation-curve.csv").read_text().count("package-") == 3


def test_attempt_limit_advances_without_promotion_and_resume_does_not_repeat(tmp_path, monkeypatch):
    runner, train, valid, contexts, calls = setup(tmp_path, monkeypatch)
    first = invoke(runner, train, valid, max_attempts_per_batch=1)
    assert first["status"] == "completed" and first["completed_batches"] == 2
    assert [b["batch"] for b in first["skipped_batches"]] == [1]
    assert first["skipped_batches"][0]["reason"] == "attempt_limit"
    second = invoke(runner, train, valid, max_attempts_per_batch=3, resume=True)
    assert second["status"] == "completed"
    assert not contexts[1].get("validation_feedback")
    assert len(contexts) == 2
    assert contexts[0]["skills"] == contexts[1]["skills"]
    assert len([c for c in calls if c[0] == "train inference"]) == 2


def test_no_change_rationale_is_saved_and_sent_to_next_attempt(tmp_path, monkeypatch):
    from evofact.runtime.backend import BackendResult

    runner, train, valid, contexts, _ = setup(tmp_path, monkeypatch)
    original_bank = [p.package_digest for p in runner.base.packages]

    class NoChange(EditingOptimizerBackend):
        async def optimize_package(self, context, skill):
            contexts.append(context)
            return BackendResult(
                {
                    "action": "no_change",
                    "rationale": "Routing and reports show no supported defect.",
                }
            )

    monkeypatch.setattr(runner.base, "_optimizer_backend", NoChange)
    result = invoke(runner, train, valid, max_attempts_per_batch=2)
    assert result["status"] == "completed"
    assert [b["batch"] for b in result["skipped_batches"]] == [1, 2]
    assert len(contexts) == 4
    assert [p.package_digest for p in runner.base.packages] == original_bank
    assert result["attempts"][0]["reason"] == "Routing and reports show no supported defect."
    prior = contexts[1]["prior_proposals"][0]
    assert prior["status"] == "no_change"
    assert prior["reason"] == result["attempts"][0]["reason"]


def test_resume_old_attempt_limit_checkpoint_skips_exhausted_batch(tmp_path, monkeypatch):
    runner, train, valid, contexts, calls = setup(tmp_path, monkeypatch)
    original = runner._after_save
    interrupted = False

    def save(checkpoint):
        nonlocal interrupted
        original(checkpoint)
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("feedback") and not state.get("pending") and not interrupted:
            interrupted = True
            raise RuntimeError("pause after rejection")

    monkeypatch.setattr(runner, "_after_save", save)
    with pytest.raises(RuntimeError, match="pause after rejection"):
        invoke(runner, train, valid, max_attempts_per_batch=1)
    checkpoint = tmp_path / "checkpoint.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    state["status"] = "attempt_limit"
    state.pop("skipped_batches", None)
    checkpoint.write_text(json.dumps(state), encoding="utf-8")
    outcome = invoke(runner, train, valid, resume=True, max_attempts_per_batch=1)
    assert outcome["status"] == "completed"
    assert [b["batch"] for b in outcome["skipped_batches"]] == [1]
    assert len(contexts) == 2
    assert len([c for c in calls if c[0] == "train inference"]) == 2


def test_resume_mid_validation_reuses_saved_baseline(tmp_path, monkeypatch):
    runner, train, valid, contexts, calls = setup(tmp_path, monkeypatch)
    original = runner.base.run
    interrupted = False

    async def infer(rows, **kwargs):
        nonlocal interrupted
        if kwargs["phase"] == "validation candidate" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted")
        return await original(rows, **kwargs)

    monkeypatch.setattr(runner.base, "run", infer)
    with pytest.raises(RuntimeError, match="interrupted"):
        invoke(runner, train, valid)
    result = invoke(runner, train, valid, resume=True)
    assert result["status"] == "completed"
    assert len(contexts) == 3
    assert len([c for c in calls if c[0] == "validation baseline"]) == 3


def test_validation_exhaustion_stops_without_reusing_samples(tmp_path, monkeypatch):
    runner, train, valid, contexts, calls = setup(tmp_path, monkeypatch, valid_count=2)
    result = invoke(runner, train, valid)
    assert result["status"] == "validation_pool_exhausted"
    assert result["completed_batches"] == 0 and len(contexts) == 1


def test_resume_rejects_modified_validation_content(tmp_path, monkeypatch):
    runner, train, valid, _, _ = setup(tmp_path, monkeypatch)
    invoke(runner, train, valid, max_attempts_per_batch=1)
    valid[0] = replace(valid[0], text="Changed validation content")
    with pytest.raises(ValueError, match="incompatible"):
        invoke(runner, train, valid, resume=True)


def test_rejects_sample_and_content_leakage(tmp_path, monkeypatch):
    runner, train, valid, _, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        invoke(runner, train, valid + [train[0]])
    valid[0] = replace(valid[0], text=train[0].text)
    with pytest.raises(ValueError, match="duplicate content"):
        invoke(runner, train, valid)


def test_ordinary_cli_has_same_attempt_limit_as_other_routes():
    parser = build_parser()
    for route in ("feedback-evolve", "meta-evolve", "adversarial-evolve"):
        assert parser.parse_args([route]).max_attempts_per_batch == 5
        assert (
            parser.parse_args([route, "--max-attempts-per-batch", "3"]).max_attempts_per_batch == 3
        )
    assert not hasattr(parser.parse_args(["evolve"]), "max_attempts_per_batch")


def test_feedback_route_paths_do_not_change_ordinary_evolve(tmp_path):
    from evofact.experiments.route_isolation import evolution_route_config, route_artifact_path

    config = replace(
        load_config(ROOT / "configs/dry_run.yaml"),
        output_dir=tmp_path / "output",
        skill_store=tmp_path / "store",
    )
    assert evolution_route_config(config, "evolve") is config
    feedback = evolution_route_config(config, "feedback-evolve")
    assert feedback.output_dir == config.output_dir / "feedback-evolve"
    assert feedback.skill_store == config.skill_store / "feedback-evolve"
    assert evolution_route_config(feedback, "feedback-evolve") == feedback
    output = route_artifact_path(tmp_path / "result.json", "feedback-evolve")
    assert output == tmp_path / "feedback-evolve/result.json"
    assert route_artifact_path(output, "feedback-evolve") == output


def test_cli_dispatches_ordinary_and_feedback_to_separate_loops(tmp_path, monkeypatch):
    from evofact.cli import _run
    from evofact.experiments.runner import ExperimentRunner, fixture_samples

    config = load_config(ROOT / "configs/dry_run.yaml")
    config = replace(
        config,
        output_dir=tmp_path / "output",
        skill_store=tmp_path / "store",
        execution=replace(config.execution, trace_log=tmp_path / "traces.jsonl"),
    )
    monkeypatch.setattr("evofact.cli.load_config", lambda _: config)
    observed = []

    async def ordinary(self, *args, **kwargs):
        observed.append(("ordinary", self.config.skill_store))
        _, evaluation = await self.run(fixture_samples())
        return {"batches": [], "evaluation": evaluation}

    async def feedback(self, *args, **kwargs):
        observed.append(("feedback", self.config.skill_store))
        return {"mode": "feedback-evolve", "status": "completed"}

    monkeypatch.setattr(ExperimentRunner, "closed_loop_batched", ordinary)
    monkeypatch.setattr(FeedbackEvolveRunner, "run", feedback)
    for command in ("evolve", "feedback-evolve"):
        args = build_parser().parse_args([command, "--evaluation-only"])
        asyncio.run(_run(args))
    assert observed == [
        ("ordinary", config.skill_store),
        ("feedback", config.skill_store / "feedback-evolve"),
    ]


def test_cli_writes_incomplete_outcome_then_exits_nonzero(tmp_path, monkeypatch):
    from evofact import cli

    async def stopped(args):
        return {"mode": "feedback-evolve", "status": "attempt_limit", "completed_batches": 0}

    monkeypatch.setattr(cli, "_run", stopped)
    output = tmp_path / "evolve.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(["feedback-evolve", "--output", str(output)])
    assert exc.value.code == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "attempt_limit"


def test_cli_supplies_full_validation_pool_to_retry_runner(tmp_path, monkeypatch):
    from evofact.experiments.feedback_evolve_runner import run_cli

    runner, train, valid, _, _ = setup(tmp_path, monkeypatch, valid_count=8)
    manifest = SimpleNamespace(
        train_ids=[s.sample_id for s in train],
        evolution_validation_ids=[s.sample_id for s in valid],
        manifest_id="test",
    )
    samples = {s.sample_id: s for s in train + valid}
    selected = []

    def select(ids, *, role="split"):
        selected.append((len(ids), role))
        return [samples[sid] for sid in ids]

    async def capture(self, training, validation, **kwargs):
        assert len(validation) == 8  # Each attempt is configured for just 2.
        assert kwargs["max_attempts_per_batch"] == 5
        return {"status": "completed"}

    monkeypatch.setattr(FeedbackEvolveRunner, "run", capture)
    args = build_parser().parse_args(
        ["--config", "configs/dry_run.yaml", "feedback-evolve", "--evaluation-only"]
    )
    result = asyncio.run(run_cli(args, runner.config, ROOT, manifest, select, None, {}))
    assert result["status"] == "completed"
    assert selected == [(4, "train"), (8, "split")]


def test_cli_rejects_old_checkpoint_without_overwriting_it(tmp_path, monkeypatch):
    from evofact.experiments.feedback_evolve_runner import run_cli

    runner, train, valid, _, _ = setup(tmp_path, monkeypatch)
    checkpoint = tmp_path / "old-checkpoint.json"
    original = '{"schema_version": "evolve_checkpoint_v1", "status": "complete"}'
    checkpoint.write_text(original)
    args = build_parser().parse_args(
        [
            "--config",
            "configs/dry_run.yaml",
            "feedback-evolve",
            "--resume",
            "--checkpoint",
            str(checkpoint),
        ]
    )
    with pytest.raises(ValueError, match="incompatible feedback-evolve checkpoint"):
        asyncio.run(run_cli(args, runner.config, ROOT, None, None, None, {}))
    assert checkpoint.read_text() == original
