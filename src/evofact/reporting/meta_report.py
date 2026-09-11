from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from evofact.core.models import MetaEvolutionOutcome


def write_meta_report(path: Path, outcome: MetaEvolutionOutcome) -> dict[str, str]:
    """函数作用：输出 DEMSE 运行结果、迁移效用和门控决策报告。
    输入要求：`path`（Path）需符合函数签名约定；`outcome`（MetaEvolutionOutcome）需符合函数签名约定。
    输出：返回 `dict[str, str]` 类型结果；校验或下游调用失败时异常向上传递。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload = asdict(outcome)
    json_path, md_path, csv_path = (
        path / "demse_report.json",
        path / "demse_report.md",
        path / "demse_utilities.csv",
    )
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "candidate",
                "episodes",
                "mean_gain",
                "ci_low",
                "ci_high",
                "negative_transfer_rate",
                "worst_domain_drop",
                "decision",
            ]
        )
        decisions = {item.candidate_fingerprint: item.disposition for item in outcome.decisions}
        for key, utility in sorted(outcome.utilities.items()):
            writer.writerow(
                [
                    key,
                    utility.episode_count,
                    utility.mean_gain,
                    *utility.confidence_interval,
                    utility.negative_transfer_rate,
                    utility.worst_domain_drop,
                    decisions.get(key, ""),
                ]
            )
    lines = [
        "# DEMSE 域情景元门控报告",
        "",
        "> 该报告实现离散 SkillBank 空间元学习；底层模型未进行梯度更新。",
        "",
    ]
    if outcome.mock_results:
        lines += ["> 当前数值来自 MockBackend，仅用于机制与工程验收，不代表真实模型实验结果。", ""]
    lines += [
        "## 领域 Episode",
        "",
        "| Episode | Meta-train domains | Meta-test domains |",
        "|---|---|---|",
    ]
    lines += [
        f"| {e.episode_id} | {', '.join(e.meta_train_domains)} | {', '.join(e.meta_test_domains)} |"
        for e in outcome.episodes
    ]
    lines += [
        "",
        "## 元门控决策",
        "",
        "| Candidate | Decision | Mean gain | 95% CI | Negative transfer |",
        "|---|---:|---:|---:|---:|",
    ]
    for decision in outcome.decisions:
        utility = decision.utility
        lines.append(
            f"| {decision.candidate_fingerprint[:12]} | {decision.disposition} | {utility.mean_gain:.4f} | [{utility.confidence_interval[0]:.4f}, {utility.confidence_interval[1]:.4f}] | {utility.negative_transfer_rate:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path)}
