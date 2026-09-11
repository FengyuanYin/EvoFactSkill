import csv
import json
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev


def summarize_runs(metric_runs: list[dict[str, float]]) -> dict:
    """函数作用：负责当前模块中的 `summarize_runs` 处理，封装调用方需要复用的业务步骤。
    输入要求：`metric_runs`（list[dict[str, float]]）需符合函数签名约定。
    输出：返回 `dict` 类型结果；校验或下游调用失败时异常向上传递。"""
    keys = sorted(set().union(*(x.keys() for x in metric_runs))) if metric_runs else []
    summary = {}
    for key in keys:
        values = [float(x[key]) for x in metric_runs if key in x]
        sd = stdev(values) if len(values) > 1 else 0.0
        half = 1.96 * sd / sqrt(len(values)) if values else 0.0
        avg = mean(values) if values else 0.0
        summary[key] = {
            "mean": avg,
            "std": sd,
            "ci95": [avg - half, avg + half],
            "n_runs": len(values),
        }
    return summary


def skill_evolution_curve(events: list[dict]) -> list[dict]:
    """函数作用：负责当前模块中的 `skill_evolution_curve` 处理，封装调用方需要复用的业务步骤。
    输入要求：`events`（list[dict]）需符合函数签名约定。
    输出：返回 `list[dict]` 类型结果；校验或下游调用失败时异常向上传递。"""
    return [
        {
            "step": index + 1,
            "action": event.get("action"),
            "skill": event.get("name"),
            "status": event.get("status"),
            "timestamp": event.get("timestamp"),
        }
        for index, event in enumerate(events)
    ]


def write_report(path: Path, title: str, payload: dict) -> dict[str, str]:
    """函数作用：将一次实验结果序列化为 JSON、CSV 和 Markdown 报告。
    输入要求：`path`（Path）需符合函数签名约定；`title`（str）需符合函数签名约定；`payload`（dict）需符合函数签名约定。
    输出：返回 `dict[str, str]` 类型结果；校验或下游调用失败时异常向上传递。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "report.json"
    md_path = path / "report.md"
    csv_path = path / "metrics.csv"
    clean = json.loads(
        json.dumps(
            payload, default=lambda x: asdict(x) if hasattr(x, "__dataclass_fields__") else str(x)
        )
    )
    metrics = clean.get("metrics") or clean.get("evaluation", {}).get("aggregate_metrics", {})
    reasons = []
    if clean.get("completed") is False:
        reasons.append("run is incomplete")
    if metrics.get("n", 1) < 1:
        reasons.append("no evaluated samples")
    if clean.get("parse_failures", 0) > 0:
        reasons.append("model output parse failures occurred")
    clean["run_status"] = {"valid": not reasons, "reasons": reasons}
    json_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(sorted(metrics.items()))
    md_path.write_text(
        f"# {title}\n\n```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}
