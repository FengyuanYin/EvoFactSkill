import json
from pathlib import Path

from .meta_report import write_meta_report


def write_adversarial_report(path, outcome, audit, config):
    """函数作用：输出对抗生成审计、逐 episode 样本和对应 DEMSE 报告。
    输入要求：`path`（未显式标注）需符合函数签名约定；`outcome`（未显式标注）需符合函数签名约定；`audit`（未显式标注）需符合函数签名约定；`config`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    path = Path(path)
    paths = write_meta_report(path, outcome)
    # Each episode stays separate: never pool synthetic data across held-out domains.
    sample_paths = {}
    for identifier, episode in audit.items():
        target = path / f"samples-{identifier}.jsonl"
        target.write_text(
            "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in episode["samples"]),
            encoding="utf-8",
        )
        sample_paths[identifier] = str(target)
    payload = {
        "schema_version": "adversarial_report_v2",
        "run_id": outcome.run_id,
        "mock_results": outcome.mock_results,
        "proposer": config.proposer,
        "generator": "one-shot trace-conditioned LLM",
        "episodes": audit,
        "sample_files": sample_paths,
        "detector_reports": paths,
    }
    json_path = path / "adversarial_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Trace 驱动的单次 LLM 样本生成",
        "",
        "成功 trace：同类案例升难度；失败 trace：同类增样强化。样本预算由 LLM 分配。",
        "每个 episode 一次生成调用，另有一次独立标签审核（结构审核后无样本则跳过）。",
        "审核只验证给定证据的一致性，不代表开放世界的真实新闻鉴定。",
        "probe 仅作诊断；meta-test 只用于检测 skill 的外层门控。",
        "samples 按 episode 分文件导出，勿跨 episode 混用造成留出域泄漏。",
        "",
    ]
    if outcome.mock_results:
        lines += ["> 检测侧为 mock，仅验证机制，不是模型效果证据。", ""]
    lines += [
        "| Episode | 生成数量 | 审核通过 | 有效率 | 主系统错误率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for identifier, episode in audit.items():
        m = episode["metrics"]
        lines.append(
            f"| {identifier} | {m['generated']} | {m['accepted']} | {m['validity']:.3f} | {m['difficulty']:.3f} |"
        )
    md = path / "adversarial_report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        **paths,
        "adversarial_json": str(json_path),
        "adversarial_markdown": str(md),
        "sample_files": sample_paths,
    }
