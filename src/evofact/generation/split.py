"""Group-preserving construction/probe split inside meta-train only."""
import hashlib
from collections import defaultdict


def split_construction_probe(samples, facts, *, fraction: float, seed: str):
    by_id = {s.sample_id: s for s in samples}
    if len(by_id) != len(samples):
        raise ValueError("duplicate training sample IDs")
    facts = [f for f in facts if f.sample_id in by_id]
    parent = {s.sample_id: s.sample_id for s in samples}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for sample in samples:
        keys = [("text", " ".join(sample.text.casefold().split()))]
        if sample.event_id:
            keys.append(("event", sample.event_id))
        keys += [("evidence", " ".join(e.text.casefold().split())) for e in sample.evidence]
        keys += [("source", e.source) for e in sample.evidence if e.source]
        keys += [("source", f.source) for f in facts if f.sample_id == sample.sample_id]
        for key in keys:
            if key in owners:
                parent[find(sample.sample_id)] = find(owners[key])
            else:
                owners[key] = sample.sample_id
    groups = defaultdict(list)
    for sample in samples:
        groups[find(sample.sample_id)].append(sample)
    domains = defaultdict(list)
    for group in groups.values():
        names = {s.domain or s.dataset for s in group}
        if len(names) != 1:
            raise ValueError("evidence/event group crosses domains")
        domains[next(iter(names))].append(group)
    construction, probe = [], []
    for domain, grouped in sorted(domains.items()):
        if len(grouped) < 2:
            raise ValueError(f"domain {domain} requires at least two independent event/evidence groups")
        grouped.sort(key=lambda g: hashlib.sha256(f"{seed}:{min(s.sample_id for s in g)}".encode()).hexdigest())
        count = min(len(grouped) - 1, max(1, round(len(grouped) * fraction)))
        probe.extend(s for group in grouped[:count] for s in group)
        construction.extend(s for group in grouped[count:] for s in group)
    allowed = {s.sample_id for s in construction}
    selected = [f for f in facts if f.sample_id in allowed and f.verified is True]
    return sorted(construction, key=lambda s: s.sample_id), sorted(probe, key=lambda s: s.sample_id), selected
