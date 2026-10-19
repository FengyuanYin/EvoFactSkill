"""Refresh offline, sample-matched experiment tables without model calls."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evofact.core.models import SampleEvaluation  # noqa: E402
from evofact.validation.evaluator import evaluate  # noqa: E402

BASELINE = ROOT / "outputs/final-test-subsets/20261017-172226-013948"
FEEDBACK = ROOT / "docs/feedback-evolve/20261016-183947-791-4bdc478a"
METHODS = ("no-evolve", "evolve", "feedback-evolve", "adversarial-evolve")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(rows):
    result = evaluate(
        [SampleEvaluation(**{**r, "allowed_labels": tuple(r["allowed_labels"])}) for r in rows]
    )
    return {"metrics": result.aggregate_metrics, "domain_metrics": result.domain_metrics}


def main():
    hashes = {}

    def source(path):
        hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)

    report = {
        "updated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "model_calls": 0,
        "sizes": {},
    }
    sections = {}
    for n in (25, 50):
        data = {"samples_per_domain": n, "folds": {}, "combined": {}, "missing": []}
        pooled = {m: [] for m in METHODS}
        for fold in (1, 2):
            ordinary_path = BASELINE / f"evolve/docs-weibo21_evolve_fold_{fold}/final-test-{n}.json"
            ordinary = source(ordinary_path)
            expected = {r["sample_id"]: r["gold"] for r in ordinary["evaluations"]}
            domains = set(ordinary["domain_metrics"])
            baseline = []
            for path in sorted(
                (BASELINE / "no-evolve/static-100-before-pr-review").glob(f"*-{n}.json")
            ):
                result = source(path)
                if set(result["domain_metrics"]) <= domains:
                    baseline.extend(result["evaluations"])
            entries = {
                "no-evolve": (baseline, metrics(baseline)),
                "evolve": (ordinary["evaluations"], ordinary),
            }
            paths = {
                "feedback-evolve": FEEDBACK / f"fold_{fold}/feedback-evolve/final-test-{n}.json",
                "adversarial-evolve": ROOT
                / f"docs/weibo21_adversarial_knowledge_fold_{fold}/adversarial-evolve/final-test-{n}.json",
            }
            for method, path in paths.items():
                if not path.exists():
                    data["missing"].append({"method": method, "fold": fold, "reason": "not_found"})
                    continue
                result = source(path)
                if result["status"] != "complete":
                    data["missing"].append(
                        {"method": method, "fold": fold, "reason": result["status"]}
                    )
                    continue
                records_path = path.with_suffix(".samples.jsonl")
                hashes[records_path] = hashlib.sha256(records_path.read_bytes()).hexdigest()
                records = [
                    json.loads(line)
                    for line in records_path.read_text(encoding="utf-8").splitlines()
                ]
                manifest = next(r for r in records if r["kind"] == "manifest")
                latest = {r["sample_id"]: r for r in records if r["kind"] == "sample"}
                assert all(latest[s]["status"] == "complete" for s in manifest["sample_ids"])
                rows = [latest[s]["evaluation"] for s in manifest["sample_ids"]]
                checked = metrics(rows)
                for key in ("accuracy_all", "macro_f1_all", "coverage"):
                    assert abs(checked["metrics"][key] - result["metrics"][key]) < 1e-12
                entries[method] = (rows, result)
            fold_data = {}
            for method, (rows, result) in entries.items():
                assert len(rows) == len(expected) == len(domains) * n
                assert {r["sample_id"]: r["gold"] for r in rows} == expected
                assert all(int(v["n"]) == n for v in result["domain_metrics"].values())
                fold_data[method] = {k: result[k] for k in ("metrics", "domain_metrics")}
                if "skill_bank" in result:
                    fold_data[method]["skill_bank"] = result["skill_bank"]
                pooled[method].extend(rows)
            data["folds"][str(fold)] = fold_data
        for method in METHODS:
            if all(method in data["folds"][str(f)] for f in (1, 2)):
                data["combined"][method] = metrics(pooled[method])
        lines = [
            f"## 每领域 {n} 条 final-test",
            "",
            "指标单位为 %。整体 Macro-F1 按对应折或两折合并样本重新计算，不是领域 F1 的平均值。",
            "",
            "| 方法 | Fold 1 Acc | Fold 1 F1 | Fold 2 Acc | Fold 2 F1 | 合并 Acc | 合并 F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method in METHODS:
            values = []
            for group in (data["folds"]["1"], data["folds"]["2"], data["combined"]):
                m = group.get(method, {}).get("metrics")
                values.extend(
                    [f"{m[k] * 100:.2f}" if m else "—" for k in ("accuracy_all", "macro_f1_all")]
                )
            lines.append(f"| {method} | " + " | ".join(values) + " |")
        for title, key in (("准确率", "accuracy_all"), ("Macro-F1", "macro_f1_all")):
            lines += [
                "",
                f"### 按领域：{title}",
                "",
                "| Fold | 领域 | 无 evolve | 普通 evolve | feedback-evolve | adversarial-evolve |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for fold in (1, 2):
                fold_data = data["folds"][str(fold)]
                for domain in sorted(fold_data["evolve"]["domain_metrics"]):
                    cells = []
                    for method in METHODS:
                        m = fold_data.get(method, {}).get("domain_metrics", {}).get(domain)
                        cells.append(f"{m[key] * 100:.2f}" if m else "—")
                    lines.append(f"| {fold} | {domain} | " + " | ".join(cells) + " |")
        if data["missing"]:
            lines += ["", "“—”表示未发现完整测试结果，不填作 0。meta-evolve 尚无已完成的测试结果。"]
        sections[n] = "\n".join(lines) + "\n"
        report["sizes"][str(n)] = data
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    report["sources"] = [
        {"path": p.relative_to(ROOT).as_posix(), "sha256": h} for p, h in hashes.items()
    ]
    preface = (
        "# 实验结果对比\n\n"
        f"更新时间：{report['updated_at']}。本表为本地重算，模型调用 0 次，原始测试和训练结果未改动。\n\n"
        "每领域固定取原 100 条测试名单的前 N 条，三组进化结果与无进化基线的样本 ID、真实标签已核对一致。"
        "25 条和 50 条分别统计。每种规模 Fold 1 为 5 个领域，Fold 2 为 4 个领域。\n\n"
        "历史基线与新实验分别运行，差异也包含模型与运行随机性。adversarial-evolve 为内源知识生成路线。\n\n"
    )
    (ROOT / "docs/experiment-results-comparison.md").write_text(
        preface + "\n".join(sections.values()), encoding="utf-8"
    )
    (ROOT / "docs/experiment-results-comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    previous = FEEDBACK / "final-test-25-summary"
    (previous / "comparison.md").write_text(preface + sections[25], encoding="utf-8")
    old = read(previous / "summary.json")
    for fold in ("1", "2"):
        old["folds"][fold].update(report["sizes"]["25"]["folds"][fold])
    old["combined"] = report["sizes"]["25"]["combined"]
    old["updated_at"] = report["updated_at"]
    old["sources"] = report["sources"]
    (previous / "summary.json").write_text(
        json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(sections[25])
    print(f"Verified {len(hashes)} source hashes; reports updated.")


if __name__ == "__main__":
    main()
