from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from collections import defaultdict
from collections.abc import Sequence

from evofact.core.models import Sample, SkillSpec


def effective_domain(sample: Sample) -> str:
    domain = str(sample.domain or sample.dataset or "").strip()
    if not domain:
        raise ValueError(f"sample {sample.sample_id!r} has no domain or dataset")
    return domain


def domain_index(samples: Sequence[Sample]) -> dict[str, tuple[Sample, ...]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    seen: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample id: {sample.sample_id}")
        seen.add(sample.sample_id)
        groups[effective_domain(sample)].append(sample)
    return {name: tuple(sorted(rows, key=lambda row: row.sample_id)) for name, rows in sorted(groups.items())}


def data_fingerprint(samples: Sequence[Sample]) -> str:
    payload = [asdict(s) for s in sorted(samples, key=lambda s: s.sample_id)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def skillbank_fingerprint(skills: Sequence[SkillSpec]) -> str:
    payload = [asdict(s) for s in sorted(skills, key=lambda s: s.skill_id)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def split_source_and_final(samples: Sequence[Sample], final_test_domains: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    known = set(domain_index(samples))
    final = tuple(sorted(set(final_test_domains)))
    missing = set(final) - known
    if missing:
        raise ValueError(f"unknown final test domains: {sorted(missing)}")
    source = tuple(sorted(known - set(final)))
    if set(source) & set(final):
        raise ValueError("source and final test domains overlap")
    return source, final
