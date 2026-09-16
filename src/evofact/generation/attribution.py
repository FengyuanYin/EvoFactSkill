from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationAttribution:
    category: str
    count: int
    audit_ids: tuple[str, ...]


CATEGORIES = (
    "invalid",
    "evidence",
    "label",
    "duplicate",
    "diversity",
    "difficulty",
    "teaching",
    "cost",
    "leakage",
    "safety",
)


def attribute_generation(entries) -> tuple[GenerationAttribution, ...]:
    buckets: dict[str, list[str]] = {category: [] for category in CATEGORIES}
    for entry in entries:
        blob = " ".join(entry.reasons).casefold()
        matched = False
        for category in CATEGORIES:
            if category in blob:
                buckets[category].append(entry.audit_id)
                matched = True
        if not matched and not entry.accepted:
            buckets["invalid"].append(entry.audit_id)
    return tuple(
        GenerationAttribution(category, len(ids), tuple(ids))
        for category, ids in buckets.items()
        if ids
    )
