import json
from datetime import datetime
from pathlib import Path

from evofact.core.models import Evidence, Sample

from .models import EvidenceFact, numeric


def load_samples(path):
    """函数作用：读取标准化 Sample JSONL，并补全旧适配器可能省略的证据字段。
    输入要求：`path`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        evidence = []
        for raw in data.pop("evidence", []):
            raw = dict(raw)
            if raw.get("published_at"):
                raw["published_at"] = datetime.fromisoformat(raw["published_at"])
            evidence.append(Evidence(**raw))
        if data.get("published_at"):
            data["published_at"] = datetime.fromisoformat(data["published_at"])
        rows.append(Sample(**data, evidence=tuple(evidence)))
    if not rows or len({s.sample_id for s in rows}) != len(rows):
        raise ValueError("normalized samples must be non-empty with unique IDs")
    return rows


def load_facts(path):
    """函数作用：读取并转换 `load_facts` 所表示的数据，供当前模块后续流程使用。
    输入要求：`path`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    rows = [
        EvidenceFact(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({f.fact_id for f in rows}) != len(rows):
        raise ValueError("facts must have unique IDs and cannot be empty")
    return rows


def validate_fact_links(samples, facts):
    """函数作用：校验 `validate_fact_links` 所表示的数据，供当前模块后续流程使用。
    输入要求：`samples`（未显式标注）需符合函数签名约定；`facts`（未显式标注）需符合函数签名约定。
    输出：返回 `None`；可能按函数职责更新状态、执行断言或产生外部副作用。"""
    by_id = {s.sample_id: s for s in samples}
    if len({f.fact_id for f in facts}) != len(facts):
        raise ValueError("duplicate fact IDs")
    for fact in facts:
        sample = by_id.get(fact.sample_id)
        if (
            sample is None
            or (sample.domain or sample.dataset) != fact.domain
            or sample.event_id != fact.event_id
        ):
            raise ValueError("fact sample/domain/event provenance mismatch")
        if not any(
            e.text == fact.evidence_text and e.source == fact.source for e in sample.evidence
        ):
            raise ValueError("structured fact is not linked to the sample evidence snapshot")


def fixture_adversarial_data():
    """函数作用：构造离线测试夹具 `fixture_adversarial_data` 所表示的数据，供当前模块后续流程使用。
    输入要求：无显式输入；若函数位于另一函数内部，则依赖已初始化的外层变量。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    samples, facts = [], []
    for domain in ("health", "finance", "science", "society", "environment", "outer_holdout"):
        for index in range(4):
            event = f"{domain}-event-{index}"
            fact = EvidenceFact(
                f"fact-{event}",
                f"sample-{event}",
                domain,
                event,
                f"fixture://{event}",
                f"模拟项目{domain}{index}",
                "登记数量",
                str(20 + index),
                "项",
                True,
            )
            facts.append(fact)
            evidence = (Evidence(fact.evidence_text, fact.source),)
            samples.append(
                Sample(
                    fact.sample_id,
                    "fixture",
                    fact.evidence_text,
                    "REAL",
                    domain,
                    event,
                    evidence=evidence,
                )
            )
            samples.append(
                Sample(
                    f"other-{event}",
                    "fixture",
                    f"{fact.entity}的{fact.attribute}为{numeric(99 + index)}项。",
                    "FAKE",
                    domain,
                    event,
                    evidence=evidence,
                )
            )
    return samples, facts
