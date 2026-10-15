"""Durable per-sample evaluation journal with resumable, separate summaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

from evofact.core.models import SampleEvaluation
from evofact.data.domains import data_fingerprint, skillbank_fingerprint
from evofact.governance.pricing_policy import load_pricing_table
from evofact.runtime.progress import ProgressEvent
from evofact.validation.evaluator import evaluate

from .runner import retryable_runtime_failure


def _plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".evaluation-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


async def resumable_test(
    runner,
    samples,
    *,
    output,
    records=None,
    checkpoint=None,
    resume=False,
    sample_retries=2,
    progress=None,
    metadata=None,
):
    samples = list(samples)
    if not samples or len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("test requires non-empty samples with unique IDs")
    if sample_retries < 0:
        raise ValueError("sample retries must be non-negative")
    output = Path(output).resolve()
    records = Path(records).resolve() if records else output.with_suffix(".samples.jsonl")
    checkpoint = (
        Path(checkpoint).resolve() if checkpoint else output.with_suffix(".checkpoint.json")
    )
    if len({output, records, checkpoint}) != 3:
        raise ValueError("test summary, records, and checkpoint must use different files")
    config = runner.config
    pricing = config.pricing.table_path
    pricing_id = load_pricing_table(runner.root / pricing).identity if pricing else None
    identity_view = {
        "samples": data_fingerprint(samples),
        "sample_ids": [s.sample_id for s in samples],
        "skills": skillbank_fingerprint(runner.skills),
        "label_contract": runner.label_contract_registry.digest,
        "backend": config.backend,
        "provider": config.pricing.provider,
        "model": config.model,
        "base_url": config.base_url,
        "pricing": pricing_id,
        "routing": config.routing_strategy,
        "seed": config.seed,
        "max_skills": config.max_skills_per_item,
        "dag": asdict(config.dag),
        "budget": asdict(config.budget),
    }
    identity = hashlib.sha256(
        json.dumps(identity_view, sort_keys=True, default=str).encode()
    ).hexdigest()
    header = {
        "kind": "manifest",
        "schema_version": "test_samples_v1",
        "identity": identity,
        "sample_ids": identity_view["sample_ids"],
    }
    latest = {}
    budget_state = {"run": {}, "samples": {}}

    def restore_event(event):
        if event.get("kind") not in {"sample", "budget"}:
            raise ValueError("unknown test journal record")
        delta = event.get("budget", {})
        if "run" in delta:
            budget_state["run"] = delta["run"]
        budget_state["samples"].update(delta.get("samples", {}))
        if event["kind"] == "sample":
            sid = event["sample_id"]
            if sid not in sample_positions or event.get("sample_index") != sample_positions[sid]:
                raise ValueError("test journal sample does not match the ordered test set")
            if event.get("status") not in {"complete", "error"}:
                raise ValueError("invalid test journal sample status")
            if event["status"] == "complete" and (
                event["trace"]["sample_id"] != sid or event["evaluation"]["sample_id"] != sid
            ):
                raise ValueError("test journal trace/evaluation identity mismatch")
            if event["status"] == "complete" and retryable_runtime_failure(event["trace"]):
                event = {**event, "status": "error", "error_type": "RuntimeRequestFailure"}
            latest[sid] = event

    sample_positions = {s.sample_id: index for index, s in enumerate(samples)}
    if resume:
        if not records.is_file():
            raise ValueError(f"test records do not exist: {records}")
        # Only an unterminated final append is recoverable; complete malformed records fail closed.
        with records.open("r+b") as stream:
            first = stream.readline()
            if json.loads(first) != header:
                raise ValueError(
                    "test resume identity differs: samples/order, model, skills, or policy changed"
                )
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(position)
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                restore_event(json.loads(line))
    else:
        if records.exists() or output.exists() or checkpoint.exists():
            raise ValueError("test output already exists; use test --resume or a new --output path")
        records.parent.mkdir(parents=True, exist_ok=True)
        with records.open("xb") as stream:
            stream.write((json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    budget = runner._budget_manager()
    if budget_state["run"]:
        budget.restore(budget_state)
    last_budget = _plain(budget.export_reconciled_state())
    resumed_count = sum(e["status"] == "complete" for e in latest.values())
    summary = None

    def summarize(status):
        nonlocal summary
        rows = []
        for sample in samples:
            event = latest.get(sample.sample_id)
            if event and event["status"] == "complete":
                raw = dict(event["evaluation"])
                raw["allowed_labels"] = tuple(raw["allowed_labels"])
                rows.append(SampleEvaluation(**raw))
        failed = [
            s.sample_id for s in samples if latest.get(s.sample_id, {}).get("status") == "error"
        ]
        result = evaluate(rows)
        summary = {
            **(metadata or {}),
            "mode": "test",
            "status": status,
            "n_samples": len(samples),
            "n_traces": len(rows),
            "completed_samples": len(rows),
            "failed_samples": len(failed),
            "remaining_samples": len(samples) - len(rows),
            "failed_sample_ids": failed,
            "resumed_samples": resumed_count,
            "metrics_scope": "all samples"
            if len(rows) == len(samples)
            else "completed samples only",
            "metrics": result.aggregate_metrics,
            "domain_metrics": result.domain_metrics,
            "label_schema_metrics": result.label_schema_metrics,
            "records": str(records),
            "checkpoint": str(checkpoint),
            "identity_digest": identity,
            "budget": budget.export_reconciled_state()["run"],
        }
        _atomic(output, summary)
        _atomic(
            checkpoint,
            {
                "schema_version": "test_checkpoint_v1",
                "identity": identity,
                "records": str(records),
                "output": str(output),
                "status": status,
                "completed_samples": len(rows),
                "failed_samples": len(failed),
                "total_samples": len(samples),
            },
        )
        return summary

    def append(event):
        nonlocal last_budget
        current = _plain(budget.export_reconciled_state())
        # Persist changed cumulative ledgers instead of repeating the whole growing ledger on every line.
        event["budget"] = {
            "run": current["run"],
            "samples": {
                key: value
                for key, value in current["samples"].items()
                if last_budget["samples"].get(key) != value
            },
        }
        payload = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with records.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        last_budget = current
        if event["kind"] == "sample":
            latest[event["sample_id"]] = event
            summarize("running")

    def completed(sample, trace, evaluation):
        append(
            {
                "kind": "sample",
                "sample_id": sample.sample_id,
                "sample_index": sample_positions[sample.sample_id],
                "status": "complete",
                "trace": asdict(trace),
                "evaluation": asdict(evaluation),
            }
        )

    def failed(sample, exc):
        append(
            {
                "kind": "sample",
                "sample_id": sample.sample_id,
                "sample_index": sample_positions[sample.sample_id],
                "status": "error",
                "error_type": type(exc).__name__,
                "attempts": sample_retries + 1,
                **({"trace": asdict(exc.trace)} if hasattr(exc, "trace") else {}),
            }
        )

    last_progress = ProgressEvent("test", "inference", resumed_count, len(samples), "samples")

    def display(event):
        nonlocal last_progress
        last_progress = event
        if progress is not None:
            saved = sum(e["status"] == "complete" for e in latest.values())
            errors = sum(e["status"] == "error" for e in latest.values())
            progress.update(
                ProgressEvent(
                    event.task,
                    event.phase,
                    event.completed,
                    event.total,
                    event.unit,
                    f"saved={saved} failed={errors}",
                )
            )

    class Progress:
        def update(self, event):
            display(
                ProgressEvent(
                    event.task,
                    event.phase,
                    resumed_count + event.completed,
                    len(samples),
                    event.unit,
                    event.detail,
                )
            )

    async def heartbeat():
        while True:
            await asyncio.sleep(5)
            display(last_progress)

    pending = [s for s in samples if latest.get(s.sample_id, {}).get("status") != "complete"]
    summarize("running")
    display(last_progress)
    heartbeat_task = asyncio.create_task(heartbeat()) if progress is not None and pending else None
    try:
        if pending:
            await runner.run(
                pending,
                progress=Progress(),
                task_name="test",
                isolate_sample_budget=True,
                on_sample_complete=completed,
                on_sample_error=failed,
                sample_retries=sample_retries,
            )
    except BaseException:
        append({"kind": "budget"})
        summarize("interrupted")
        raise
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
    append({"kind": "budget"})
    status = (
        "complete"
        if all(e.get("status") == "complete" for e in latest.values())
        and len(latest) == len(samples)
        else "incomplete"
    )
    return summarize(status)
