import json
from datetime import datetime
from pathlib import Path

from evofact.core.models import Evidence, Sample

from .models import EvidenceFact, numeric


def load_samples(path):
    """Load normalized Sample JSONL including evidence omitted by some legacy adapters."""
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
    rows = [
        EvidenceFact(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({f.fact_id for f in rows}) != len(rows):
        raise ValueError("facts must have unique IDs and cannot be empty")
    return rows


def validate_fact_links(samples, facts):
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
