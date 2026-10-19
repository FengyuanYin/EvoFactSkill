"""Derive first-50/first-25 per-domain results from completed 100/domain tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evofact.data.registry import DataRegistry  # noqa: E402
from evofact.validation.evaluator import evaluate  # noqa: E402
from scripts.recompute_static_100 import (  # noqa: E402
    recompute_budget,
    sample_id_digest,
    to_evaluations,
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_records(records):
    """Restore manifest order, never concurrent completion order or sample-ID sort."""
    manifests = [r for r in records if r.get("kind") == "manifest"]
    if len(manifests) != 1:
        raise ValueError("expected exactly one test manifest")
    ids = manifests[0]["sample_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs in manifest")
    latest = {r["sample_id"]: r for r in records if r.get("kind") == "sample"}
    if set(latest) != set(ids):
        raise ValueError("test records do not match manifest membership")
    if any(latest[sid].get("status") != "complete" for sid in ids):
        raise ValueError("test contains incomplete samples")
    return [latest[sid]["trace"] for sid in ids]


def prefix_by_domain(traces, index, size):
    grouped = defaultdict(list)
    ids = [t["sample_id"] for t in traces]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs in source test")
    for trace in traces:
        grouped[index[trace["sample_id"]].domain].append(trace)
    if not grouped or any(len(rows) != 100 for rows in grouped.values()):
        raise ValueError("source must contain exactly 100 samples per domain")
    selected = {t["sample_id"] for rows in grouped.values() for t in rows[:size]}
    return [t for t in traces if t["sample_id"] in selected]


def discover_sources():
    sources = []
    for folder in ("static", "static-100-before-pr-review"):
        for path in sorted((ROOT / "outputs" / folder).glob("weibo21-test-*-100.json")):
            sources.append(("no-evolve", folder, path))
    # Historical test initialized directly from seeds, without a training commit.
    historical = ROOT / "outputs/weibo21_cross_domain_4/final-test.json"
    if historical.is_file():
        bank = read(historical).get("skill_bank", {})
        if not bank.get("run_id", "").startswith("initialize-package-bank-"):
            raise ValueError("historical baseline is no longer a seed-only test")
        sources.append(("no-evolve", "weibo21_cross_domain_4", historical))
    for parent in (ROOT / "docs", ROOT / "outputs"):
        for directory in sorted(parent.glob("weibo21_evolve_fold_*")):
            path = directory / "final-test.json"
            if path.is_file():
                sources.append(("evolve", parent.name + "-" + directory.name, path))
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")
    out = args.output_root or ROOT / "outputs/final-test-subsets" / stamp
    if out.exists():
        raise ValueError(f"refusing to overwrite output directory: {out}")
    registry = DataRegistry()
    samples = registry.load("weibo21", ROOT / "datasets/Weibo21", excluded_domains=("无法确定",))
    index = {s.sample_id: s for s in samples}
    sources = discover_sources()
    if not sources:
        raise ValueError("no completed 100/domain tests found")
    protected = {}
    prepared, summary, report_rows = [], [], []
    canonical = {
        read(p)["subset"]["domain"]: [t["sample_id"] for t in read(p)["traces"]]
        for p in (ROOT / "outputs/static-100-before-pr-review").glob("*-100.json")
    }
    for method, experiment, source in sources:
        protected[source] = sha256(source)
        payload = read(source)
        if payload.get("status") not in (None, "complete") or payload.get("failed_samples", 0):
            raise ValueError(f"source test is incomplete: {source}")
        record_file = None
        if "traces" in payload:
            traces = payload["traces"]
        else:
            record_file = source.with_suffix(".samples.jsonl")
            protected[record_file] = sha256(record_file)
            with record_file.open(encoding="utf-8") as stream:
                traces = ordered_records([json.loads(line) for line in stream if line.strip()])
        if len(traces) != payload.get("completed_samples", payload.get("n_traces")):
            raise ValueError(f"source count mismatch: {source}")
        # Validate stored metrics before deriving subsets.
        original = evaluate(to_evaluations(traces, index, registry.label_contracts))
        for key in ("accuracy_all", "macro_f1_all"):
            if abs(original.aggregate_metrics[key] - payload["metrics"][key]) > 1e-10:
                raise ValueError(f"stored {key} does not match recomputation: {source}")
        by_domain = defaultdict(list)
        for trace in traces:
            by_domain[index[trace["sample_id"]].domain].append(trace["sample_id"])
        for domain, ids in by_domain.items():
            if ids != canonical[domain]:
                raise ValueError(
                    f"source sample order differs from fixed baseline: {source}, {domain}"
                )
        for size in (50, 25):
            selected = prefix_by_domain(traces, index, size)
            rows = to_evaluations(selected, index, registry.label_contracts)
            result = evaluate(rows)
            filename = source.name.replace("-100.json", f"-{size}.json")
            if filename == source.name:
                filename = source.stem + f"-{size}.json"
            target = out / method / experiment / filename
            subset = {
                "schema_version": "final_test_prefix_subset_v1",
                "source_file": source.relative_to(ROOT).as_posix(),
                "source_sha256": protected[source],
                "source_records": record_file.relative_to(ROOT).as_posix() if record_file else None,
                "source_records_sha256": protected.get(record_file),
                "source_samples_per_domain": 100,
                "samples_per_domain": size,
                "rule": "first_n_per_domain_in_original_test_order",
                "order_source": "test_manifest" if record_file else "source_trace_order",
                "selected_sample_ids": [t["sample_id"] for t in selected],
                "selected_sample_ids_digest": sample_id_digest([t["sample_id"] for t in selected]),
                "paired_with_static_100_baseline": True,
                "model_calls": 0,
                "budget_derivation": "sum_of_selected_trace_usage_excludes_training_and_failed_attempts",
            }
            document = {
                "mode": "test",
                "status": "complete",
                "method": method,
                "manifest_id": payload.get("manifest_id"),
                "skill_bank": payload.get("skill_bank"),
                "n_samples": len(selected),
                "n_traces": len(selected),
                "completed_samples": len(selected),
                "failed_samples": 0,
                "metrics_scope": "selected per-domain prefix only",
                "metrics": result.aggregate_metrics,
                "domain_metrics": result.domain_metrics,
                "label_schema_metrics": result.label_schema_metrics,
                "traces": selected,
                "evaluations": [r.model_dump() for r in rows],
                "budget": recompute_budget(selected),
                "subset": subset,
            }
            prepared.append((target, document))
            summary.append(
                {
                    "method": method,
                    "experiment": experiment,
                    "source": subset["source_file"],
                    "samples_per_domain": size,
                    "n": len(selected),
                    "metrics": result.aggregate_metrics,
                    "output": str(target.relative_to(out)),
                }
            )
            for domain, metrics in result.domain_metrics.items():
                report_rows.append((size, method, experiment, domain, metrics))
    # All input validation is complete before creating any output artifacts.
    for target, document in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    for source, digest in protected.items():
        if sha256(source) != digest:
            raise ValueError(f"source changed during recomputation: {source}")
    manifest = {
        "rule": "first_n_per_domain_in_original_test_order",
        "model_calls": 0,
        "source_files": len(sources),
        "derived_files": len(prepared),
        "protected_source_hashes_verified": len(protected),
        "results": summary,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Final-test 50 / 25: fixed per-domain prefixes",
        "",
        "No model calls. Original 100/domain results are unchanged. All subset IDs match the fixed static baseline prefixes.",
        "",
        "Accuracy and Macro-F1 are percentages. Prefix 25 is nested within prefix 50.",
        "",
        "| Per-domain N | Method | Experiment | Domain | Accuracy | Macro-F1 |",
        "|---:|---|---|---|---:|---:|",
    ]
    for size, method, experiment, domain, metrics in sorted(report_rows):
        lines.append(
            f"| {size} | {method} | {experiment} | {domain} | {metrics['accuracy_all'] * 100:.2f} | {metrics['macro_f1_all'] * 100:.2f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    paired = defaultdict(dict)
    for size, method, experiment, domain, metrics in report_rows:
        if experiment == "static-100-before-pr-review" or experiment.startswith(
            "docs-weibo21_evolve_fold_"
        ):
            paired[size, domain][method] = metrics
    comparison = [
        "# 无 evolve 与普通 evolve：固定前缀测试结果",
        "",
        "每个 domain 分别取原 100 条测试样本的前 50 / 25 条，模型调用为 0。普通 evolve 两折按领域合并展示，各领域仍使用对应折的已训练技能库预测。",
        "基线采用 static-100-before-pr-review；static 中对应记录的样本顺序和预测相同。所有源实验的完整列表见 manifest.json 与 summary.md。",
        "",
        "指标单位为 %，变化为普通 evolve 减去无 evolve，单位为百分点。",
        "",
        "| 每域样本数 | Domain | 无 evolve Acc | evolve Acc | Δ Acc | 无 evolve F1 | evolve F1 | Δ F1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for size in (50, 25):
        values = []
        for (n, domain), methods in sorted(paired.items()):
            if n != size:
                continue
            b, e = methods["no-evolve"], methods["evolve"]
            ba, ea, bf, ef = [
                x * 100
                for x in (
                    b["accuracy_all"],
                    e["accuracy_all"],
                    b["macro_f1_all"],
                    e["macro_f1_all"],
                )
            ]
            values.append((ba, ea, bf, ef))
            comparison.append(
                f"| {size} | {domain} | {ba:.2f} | {ea:.2f} | {ea - ba:+.2f} | {bf:.2f} | {ef:.2f} | {ef - bf:+.2f} |"
            )
        means = [sum(row[i] for row in values) / len(values) for i in range(4)]
        ba, ea, bf, ef = means
        comparison.append(
            f"| {size} | **九领域平均** | **{ba:.2f}** | **{ea:.2f}** | **{ea - ba:+.2f}** | **{bf:.2f}** | **{ef:.2f}** | **{ef - bf:+.2f}** |"
        )
    comparison += [
        "",
        "九领域平均 F1 为各领域 Macro-F1 的算术平均，不等于合并样本后重新计算的 Macro-F1。原 100 条结果保持不变。",
        "",
    ]
    (out / "comparison.md").write_text("\n".join(comparison), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(out),
                "sources": len(sources),
                "derived_results": len(prepared),
                "protected_files_verified": len(protected),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
