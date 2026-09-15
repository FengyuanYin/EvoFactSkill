from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

from evofact.core.models import Sample


def normalize_text(text: str) -> str:
    """Return the canonical text representation used for IDs and leakage checks."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def content_fingerprint(dataset: str, text: str) -> str:
    """Hash normalized content inside a dataset namespace."""
    normalized = normalize_text(text)
    return hashlib.sha256(f"{dataset.casefold()}\0{normalized}".encode("utf-8")).hexdigest()


def stable_sample_id(dataset: str, text: str) -> str:
    """Build an order-independent sample ID from normalized content."""
    return f"{dataset}:{content_fingerprint(dataset, text)[:20]}"


def deduplicate_samples(samples: Iterable[Sample]) -> tuple[Sample, ...]:
    """Globally deduplicate content and discard ambiguous domain/label groups.

    Identical samples in one domain keep a single deterministic representative.
    If identical content is assigned to multiple domains or labels, the complete
    group is removed so it cannot bridge a source and held-out domain.
    """
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[content_fingerprint(sample.dataset, sample.text)].append(sample)

    unique: list[Sample] = []
    for fingerprint in sorted(groups):
        group = groups[fingerprint]
        domains = {str(sample.domain or sample.dataset).strip().casefold() for sample in group}
        labels = {
            str(sample.label).strip().casefold() for sample in group if sample.label is not None
        }
        if len(domains) != 1 or len(labels) > 1:
            continue
        unique.append(
            min(
                group,
                key=lambda sample: (
                    sample.sample_id,
                    str(sample.metadata.get("source_split", "")),
                ),
            )
        )

    return tuple(sorted(unique, key=lambda sample: sample.sample_id))
