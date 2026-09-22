from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evofact.cli import build_parser
from evofact.config import AppConfig, load_config
from evofact.experiments.runner import ExperimentRunner, fixture_samples
from evofact.runtime.trace_store import TraceStore

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_training_runner_writes_all_phases_to_one_jsonl(tmp_path: Path) -> None:
    runner = ExperimentRunner(AppConfig(), ROOT)
    path = tmp_path / "training-traces.jsonl"
    runner.enable_trace_logging(path, run_id="run-1")

    samples = fixture_samples()[:2]
    asyncio.run(runner.run(samples, task_name="evolve batch 1", phase="train inference"))
    asyncio.run(runner.run(samples[:1], task_name="evolve batch 1", phase="validation baseline"))

    rows = _rows(path)
    assert len(rows) == 3
    assert [row["training_context"]["sequence"] for row in rows] == [0, 1, 2]
    assert {row["training_context"]["run_id"] for row in rows} == {"run-1"}
    assert [row["training_context"]["phase"] for row in rows] == [
        "train inference",
        "train inference",
        "validation baseline",
    ]
    assert [row["sample_id"] for row in rows] == [
        samples[0].sample_id,
        samples[1].sample_id,
        samples[0].sample_id,
    ]
    assert len(TraceStore(path).read_raw()) == 3


def test_training_trace_resume_appends_and_fresh_run_resets(tmp_path: Path) -> None:
    path = tmp_path / "training-traces.jsonl"
    first = ExperimentRunner(AppConfig(), ROOT)
    first.enable_trace_logging(path, run_id="run-1")
    asyncio.run(first.run(fixture_samples()[:1], phase="first"))

    resumed = ExperimentRunner(AppConfig(), ROOT)
    resumed.enable_trace_logging(path, run_id="run-1", resume=True)
    asyncio.run(resumed.run(fixture_samples()[1:2], phase="resumed"))
    assert [row["training_context"]["sequence"] for row in _rows(path)] == [0, 1]

    fresh = ExperimentRunner(AppConfig(), ROOT)
    fresh.enable_trace_logging(path, run_id="run-2", resume=False)
    asyncio.run(fresh.run(fixture_samples()[2:3], phase="fresh"))
    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["training_context"]["run_id"] == "run-2"
    assert rows[0]["training_context"]["sequence"] == 0


def test_resume_requires_the_existing_training_trace_log(tmp_path: Path) -> None:
    runner = ExperimentRunner(AppConfig(), ROOT)
    with pytest.raises(ValueError, match="does not exist"):
        runner.enable_trace_logging(tmp_path / "missing.jsonl", run_id="run-1", resume=True)


def test_training_commands_accept_explicit_trace_log() -> None:
    parser = build_parser()
    for command in ("evolve", "meta-evolve", "adversarial-evolve"):
        args = parser.parse_args([command, "--trace-log", "outputs/all-traces.jsonl"])
        assert args.trace_log == "outputs/all-traces.jsonl"


def test_weibo_config_declares_training_trace_log() -> None:
    config = load_config(ROOT / "configs/weibo21_cross_domain.yaml")
    assert config.execution.trace_log == Path(
        "outputs/weibo21_cross_domain/training-traces.jsonl"
    )
