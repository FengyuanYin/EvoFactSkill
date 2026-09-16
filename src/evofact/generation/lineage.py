from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from evofact.core.models import Sample
from evofact.governance.data_policy import DataIsolationPolicy


def normalize_content(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class LineageGroup:
    lineage_id: str
    sample_ids: tuple[str, ...]
    domains: tuple[str, ...]


def group_real_samples(
    samples: list[Sample] | tuple[Sample, ...],
    *,
    policy: DataIsolationPolicy = DataIsolationPolicy(),
) -> tuple[LineageGroup, ...]:
    if any(sample.metadata.get("synthetic") for sample in samples):
        raise ValueError("real-only lineage grouping rejects generated samples")
    by_id = {sample.sample_id: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("duplicate sample IDs")
    parent = {sample.sample_id: sample.sample_id for sample in samples}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    owners: dict[tuple[str, str], str] = {}
    for sample in sorted(samples, key=lambda item: item.sample_id):
        explicit = sample.metadata.get("lineage_id") or sample.metadata.get("parent_sample_id")
        keys = [("content", normalize_content(sample.text))]
        if explicit:
            keys.append(("lineage", str(explicit)))
        if sample.event_id:
            keys.append(("event", sample.event_id))
        for evidence in sample.evidence:
            keys.append(("evidence", normalize_content(evidence.text)))
            if evidence.source and (
                policy.merge_common_sources or evidence.source in policy.controlled_sources
            ):
                keys.append(("source", evidence.source.casefold()))
        for key in keys:
            owner = owners.get(key)
            if owner is not None:
                union(sample.sample_id, owner)
            else:
                owners[key] = sample.sample_id
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[find(sample.sample_id)].append(sample)
    result = []
    for rows in grouped.values():
        ids = tuple(sorted(item.sample_id for item in rows))
        domains = tuple(sorted({item.domain or item.dataset for item in rows}))
        if len(domains) != 1:
            raise ValueError("lineage group crosses domains")
        digest = hashlib.sha256("\0".join(ids).encode()).hexdigest()[:20]
        result.append(LineageGroup(f"lineage-{digest}", ids, domains))
    return tuple(sorted(result, key=lambda item: item.lineage_id))
