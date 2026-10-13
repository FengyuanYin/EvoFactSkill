"""Re-cut every outputs/static result file to the 100-samples-per-domain final-test subset.

Background
----------
``configs/weibo21_cross_domain.yaml`` locks final-test membership to
``outputs/static/{dataset}-test-{domain}.json`` and keeps the first
``data.final_test_samples_per_domain`` (=100) traces of each file, in file order
(``evofact.data.sampling.select_final_test_samples`` -> ``static_result_order``).
Several static files hold far more than 100 traces (医药健康 955, 文体娱乐/灾难事故/
社会生活 400, 科技 225), so their stored aggregate metrics describe a larger
population than the run actually evaluates against.

This script re-selects the first 100 traces of each static file, recomputes the
aggregate/domain metrics with the project's own evaluator
(``evofact.validation.evaluator.evaluate``) and writes a new file next to the
source with a ``-100`` suffix. Source files are never modified.

For every domain that appears in ``outputs/weibo21_cross_domain_4/final-test.json``
the script also asserts that the selected 100 sample IDs are exactly the ones the
run evaluated, and records that verification in the output's ``subset`` block.

Usage:
    python scripts/recompute_static_100.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from evofact.core.models import SampleEvaluation  # noqa: E402
from evofact.data.domains import effective_domain  # noqa: E402
from evofact.data.registry import DataRegistry  # noqa: E402
from evofact.validation.evaluator import evaluate  # noqa: E402

DATASET = "weibo21"
DATA_ROOT = REPOSITORY_ROOT / "datasets" / "Weibo21"
EXCLUDED_DOMAINS = ("无法确定",)
STATIC_DIR = REPOSITORY_ROOT / "outputs" / "static"
STATIC_GLOB = f"{DATASET}-test-*.json"
REFERENCE_RESULT = REPOSITORY_ROOT / "outputs" / "weibo21_cross_domain_4" / "final-test.json"
SAMPLES_PER_DOMAIN = 100
SUFFIX = "-100"
BUDGET_KEYS = (
    "calls_used",
    "tokens_used",
    "cost_used",
    "calls_reserved",
    "tokens_reserved",
    "cost_reserved",
    "active_reservations",
    "unavailable_cost_calls",
)


def source_files() -> list[Path]:
    """Static result files, excluding this script's own ``-100`` outputs."""
    suffix = f"{SUFFIX}.json"
    return [path for path in sorted(STATIC_DIR.glob(STATIC_GLOB)) if not path.name.endswith(suffix)]


def domain_of(path: Path) -> str:
    return path.name[len(f"{DATASET}-test-") : -len(".json")]


def trace_cost(usage: dict | None) -> tuple[Decimal, str]:
    """Read per-trace cost and cost status across both historical usage schemas."""
    if not usage:
        return Decimal(0), "unavailable"
    if "cost" in usage or "input_tokens" in usage:  # current schema
        raw = usage.get("cost")
        cost = Decimal(str(raw)) if raw not in (None, "") else Decimal(0)
        status = usage.get("cost_status") or (
            "estimated" if raw not in (None, "") else "unavailable"
        )
        return cost, str(status)
    raw = usage.get("estimated_cost")  # legacy schema
    cost = Decimal(str(raw)) if raw not in (None, "") else Decimal(0)
    return cost, ("estimated" if cost else "unavailable")


def trace_tokens(usage: dict | None) -> int:
    """Total tokens per trace: input+output+reasoning, or prompt+completion."""
    if not usage:
        return 0
    if "input_tokens" in usage:
        return (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("output_tokens") or 0)
            + int(usage.get("reasoning_tokens") or 0)
        )
    return int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)


def usage_calls(usage: dict | None) -> int:
    return int((usage or {}).get("calls") or 0)


def sample_id_digest(sample_ids: list[str]) -> str:
    payload = json.dumps(sample_ids, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def to_evaluations(traces: list[dict], index: dict, contracts) -> list[SampleEvaluation]:
    """Rebuild the evaluator rows exactly the way ExperimentRunner.run does."""
    rows = []
    for trace in traces:
        sample = index[trace["sample_id"]]
        contract = contracts.resolve_sample(sample)
        decision = trace.get("decision") or {}
        cost, cost_status = trace_cost(trace.get("usage"))
        rows.append(
            SampleEvaluation(
                sample.sample_id,
                contract.normalize(sample.label),
                decision["label"],
                float(decision.get("confidence") or 0.0),
                effective_domain(sample),
                str(sample.metadata["temporal_window"])
                if sample.metadata.get("temporal_window") is not None
                else None,
                float(cost),
                contract.schema_id,
                contract.allowed_labels,
                contract.positive_label,
                str(decision.get("origin") or "legacy"),
                cost_status,
                bool(sample.evidence),
            )
        )
    return rows


def recompute_budget(traces: list[dict]) -> dict:
    cost = sum((trace_cost(t.get("usage"))[0] for t in traces), Decimal(0))
    unavailable = sum(1 for t in traces if trace_cost(t.get("usage"))[1] == "unavailable")
    return {
        "calls_used": sum(usage_calls(t.get("usage")) for t in traces),
        "tokens_used": sum(trace_tokens(t.get("usage")) for t in traces),
        "cost_used": format(cost, "f"),
        "calls_reserved": 0,
        "tokens_reserved": 0,
        "cost_reserved": "0",
        "active_reservations": 0,
        "unavailable_cost_calls": unavailable,
    }


def reference_ids(index: dict) -> dict[str, list[str]]:
    """Per-domain sample IDs, in order, from the cross-domain final-test result."""
    if not REFERENCE_RESULT.is_file():
        return {}
    payload = json.loads(REFERENCE_RESULT.read_text(encoding="utf-8"))
    grouped: dict[str, list[str]] = defaultdict(list)
    for trace in payload.get("traces", []):
        grouped[effective_domain(index[trace["sample_id"]])].append(trace["sample_id"])
    return dict(grouped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing -100 files against a fresh recomputation instead of writing",
    )
    args = parser.parse_args()

    registry = DataRegistry()
    samples = registry.load(DATASET, DATA_ROOT, excluded_domains=EXCLUDED_DOMAINS)
    index = {sample.sample_id: sample for sample in samples}
    contracts = registry.label_contracts
    reference = reference_ids(index)
    reference_label = f"{REFERENCE_RESULT.relative_to(REPOSITORY_ROOT).as_posix()}"

    sources = source_files()
    if not sources:
        raise ValueError(f"no static result files match {STATIC_DIR / STATIC_GLOB}")
    summary = []
    for source in sources:
        domain = domain_of(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        traces = payload["traces"]
        if len(traces) < SAMPLES_PER_DOMAIN:
            raise ValueError(
                f"{source} has only {len(traces)} traces; {SAMPLES_PER_DOMAIN} are required"
            )
        selected = traces[:SAMPLES_PER_DOMAIN]
        sample_ids = [trace["sample_id"] for trace in selected]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"duplicate sample_id in selection from {source}")
        wrong = [sid for sid in sample_ids if effective_domain(index[sid]) != domain]
        if wrong:
            raise ValueError(f"{source} selects samples outside domain {domain!r}: {wrong[:3]}")

        expected = reference.get(domain)
        verified = expected is not None and expected == sample_ids
        if expected is not None and not verified:
            raise ValueError(
                f"{source}: first {SAMPLES_PER_DOMAIN} traces do not match the "
                f"{reference_label} final-test selection for {domain!r}"
            )

        result = evaluate(to_evaluations(selected, index, contracts))
        metrics = dict(result.aggregate_metrics)

        document = {
            "mode": payload.get("mode", "test"),
            "manifest_id": payload.get("manifest_id"),
            "n_traces": len(selected),
            "metrics": metrics,
            "domain_metrics": result.domain_metrics,
            "traces": selected,
            "budget": recompute_budget(selected),
            "subset": {
                "schema_version": "static_final_test_subset_v1",
                "source_file": str(source.resolve()),
                "source_n_traces": len(traces),
                "domain": domain,
                "rule": "first_n_in_source_trace_order",
                "samples_per_domain": SAMPLES_PER_DOMAIN,
                "selected_total": len(selected),
                "selected_sample_ids_digest": sample_id_digest(sample_ids),
                "metrics_recomputed": "evofact.validation.evaluator.evaluate",
                "budget_derivation": "sum_of_selected_trace_usage",
                "reference": {
                    "path": reference_label,
                    "available": expected is not None,
                    "verified_identical_selection": verified,
                    "note": (
                        "selection matches the cross-domain final-test result for this domain"
                        if verified
                        else "domain is absent from the cross-domain final-test result; "
                        "the pipeline's identical first-N static_result_order rule was applied"
                    ),
                },
            },
        }

        target = source.with_name(f"{source.stem}{SUFFIX}{source.suffix}")
        if args.check:
            if not target.is_file():
                raise ValueError(f"missing recomputed file: {target}")
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != document:
                raise ValueError(f"{target} is stale relative to a fresh recomputation")
        else:
            target.write_text(
                json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )

        summary.append(
            {
                "domain": domain,
                "source_n": len(traces),
                "out": target.name,
                "verified": verified,
                "accuracy_all": metrics["accuracy_all"],
                "macro_f1_all": metrics["macro_f1_all"],
                "coverage": metrics["coverage"],
                "mean_cost": metrics["mean_cost"],
                "source_accuracy_all": payload.get("metrics", {}).get("accuracy_all"),
                "source_macro_f1_all": payload.get("metrics", {}).get("macro_f1_all"),
                "source_coverage": payload.get("metrics", {}).get("coverage"),
            }
        )

    print(
        f"{'domain':<10}{'srcN':>6}{'acc@100':>10}{'F1@100':>10}{'cov@100':>10}"
        f"{'acc@src':>10}{'F1@src':>10}{'verified':>10}"
    )
    for row in summary:
        print(
            f"{row['domain']:<10}{row['source_n']:>6}{row['accuracy_all']:>10.4f}"
            f"{row['macro_f1_all']:>10.4f}{row['coverage']:>10.4f}"
            f"{float(row['source_accuracy_all'] or 0):>10.4f}"
            f"{float(row['source_macro_f1_all'] or 0):>10.4f}"
            f"{str(row['verified']):>10}"
        )
    verified_total = sum(1 for row in summary if row["verified"])
    print(
        f"\n{len(summary)} files {'checked' if args.check else 'written'}; "
        f"{verified_total} selections verified against {reference_label}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
