from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from evofact.core.models import Sample

from .domains import effective_domain


def _domain_quotas(
    capacities: dict[str, int], domains: Sequence[str], limit: int
) -> dict[str, int]:
    """Allocate one total limit as evenly as possible across configured domains."""
    ordered = tuple(dict.fromkeys(str(domain) for domain in domains))
    if not ordered:
        raise ValueError("balanced sampling requires at least one source domain")
    if limit < 0:
        raise ValueError("sample limit must be non-negative")
    missing = [domain for domain in ordered if capacities.get(domain, 0) == 0]
    if missing:
        raise ValueError(f"configured domains contain no samples: {missing}")
    target = min(limit, sum(capacities[domain] for domain in ordered))
    base, remainder = divmod(target, len(ordered))
    quotas = {
        domain: min(capacities[domain], base + (index < remainder))
        for index, domain in enumerate(ordered)
    }
    deficit = target - sum(quotas.values())
    while deficit:
        progressed = False
        for domain in ordered:
            if quotas[domain] >= capacities[domain]:
                continue
            quotas[domain] += 1
            deficit -= 1
            progressed = True
            if not deficit:
                break
        if not progressed:
            break
    return quotas


def balanced_domain_sample(
    samples: Sequence[Sample],
    domains: Sequence[str],
    limit: int,
    *,
    seed: int,
) -> tuple[list[Sample], dict]:
    """Select a reproducible, approximately equal number of samples per domain."""
    domain_order = tuple(dict.fromkeys(str(domain) for domain in domains))
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        domain = effective_domain(sample)
        if domain in domain_order:
            grouped[domain].append(sample)
    capacities = {domain: len(grouped[domain]) for domain in domain_order}
    if limit == 0:
        selected = [sample for sample in samples if effective_domain(sample) in domain_order]
        counts = {
            domain: sum(effective_domain(sample) == domain for sample in selected)
            for domain in domain_order
        }
    else:
        quotas = _domain_quotas(capacities, domain_order, limit)
        selected = []
        counts = {}
        for domain in domain_order:
            pool = sorted(grouped[domain], key=lambda sample: sample.sample_id)
            random.Random(f"{seed}:balanced-domain:{domain}").shuffle(pool)
            chosen = pool[: quotas[domain]]
            selected.extend(chosen)
            counts[domain] = len(chosen)
        random.Random(f"{seed}:balanced-domain:combined").shuffle(selected)
    digest = hashlib.sha256(
        json.dumps([sample.sample_id for sample in selected], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return selected, {
        "strategy": "balanced_by_domain",
        "requested_total": limit,
        "selected_total": len(selected),
        "available_by_domain": capacities,
        "selected_by_domain": counts,
        "sample_ids_digest": digest,
    }


def static_sample_ids(path: Path) -> list[str]:
    """Read ordered sample IDs from a prior result file or a plain sample list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = (
        payload.get("traces", payload.get("samples", [])) if isinstance(payload, dict) else payload
    )
    if not isinstance(rows, list):
        raise ValueError(f"static test file has no trace/sample list: {path}")
    sample_ids = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("sample_id", "")).strip():
            raise ValueError(f"static test row has no sample_id: {path}")
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in static test file: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    return sample_ids


def select_final_test_samples(
    samples: Sequence[Sample],
    final_domains: Sequence[str],
    *,
    dataset: str,
    samples_per_domain: int,
    seed: int,
    repository_root: Path,
    static_pattern: str | None = None,
    require_static: bool = False,
) -> tuple[list[Sample], dict]:
    """Select final-test samples, optionally locking membership to prior result files."""
    domains = tuple(dict.fromkeys(str(domain) for domain in final_domains))
    final_rows = [sample for sample in samples if effective_domain(sample) in domains]
    if not final_rows:
        raise ValueError("final-test domains contain no samples")
    if not static_pattern:
        selected = []
        selected_by_domain = {}
        for domain in domains:
            pool = sorted(
                (sample for sample in final_rows if effective_domain(sample) == domain),
                key=lambda sample: sample.sample_id,
            )
            random.Random(f"{seed}:final-test:{domain}").shuffle(pool)
            if samples_per_domain and len(pool) < samples_per_domain:
                raise ValueError(
                    f"final-test domain {domain!r} has {len(pool)} samples; "
                    f"at least {samples_per_domain} are required"
                )
            chosen = pool[:samples_per_domain] if samples_per_domain else pool
            selected.extend(chosen)
            selected_by_domain[domain] = len(chosen)
        digest = hashlib.sha256(
            json.dumps([sample.sample_id for sample in selected], ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
        return selected, {
            "strategy": "seeded_by_domain",
            "source": "dataset",
            "requested_per_domain": samples_per_domain,
            "selected_total": len(selected),
            "selected_by_domain": selected_by_domain,
            "sample_ids_digest": digest,
            "static_files": [],
        }

    index = {sample.sample_id: sample for sample in final_rows}
    ordered_by_domain: dict[str, list[Sample]] = {}
    available_by_domain: dict[str, int] = {}
    static_files = []
    for domain in domains:
        rendered = static_pattern.format(dataset=dataset, domain=domain)
        path = Path(rendered)
        if not path.is_absolute():
            path = repository_root / path
        if not path.is_file():
            if require_static:
                raise ValueError(f"required static final-test file does not exist: {path}")
            fallback = sorted(
                (sample for sample in final_rows if effective_domain(sample) == domain),
                key=lambda sample: sample.sample_id,
            )
            ordered_by_domain[domain] = fallback
            available_by_domain[domain] = len(fallback)
            continue
        ids = static_sample_ids(path)
        unknown = [sample_id for sample_id in ids if sample_id not in index]
        if unknown:
            raise ValueError(
                f"static final-test file does not match the loaded dataset: {path}; "
                f"unknown sample IDs include {unknown[:3]}"
            )
        wrong_domain = [
            sample_id for sample_id in ids if effective_domain(index[sample_id]) != domain
        ]
        if wrong_domain:
            raise ValueError(
                f"static final-test file contains samples outside domain {domain!r}: "
                f"{wrong_domain[:3]}"
            )
        ordered_by_domain[domain] = [index[sample_id] for sample_id in ids]
        available_by_domain[domain] = len(ids)
        static_files.append(str(path.resolve()))

    if samples_per_domain:
        undersized = {
            domain: available_by_domain[domain]
            for domain in domains
            if available_by_domain[domain] < samples_per_domain
        }
        if undersized:
            raise ValueError(
                "static final-test domains must each contain at least "
                f"{samples_per_domain} samples; undersized domains: {undersized}"
            )
    selected_by_domain = {
        domain: samples_per_domain or available_by_domain[domain] for domain in domains
    }
    selected = [
        sample
        for domain in domains
        for sample in ordered_by_domain[domain][: selected_by_domain[domain]]
    ]
    digest = hashlib.sha256(
        json.dumps([sample.sample_id for sample in selected], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return selected, {
        "strategy": "static_result_order",
        "source": "static" if static_files else "dataset",
        "requested_per_domain": samples_per_domain,
        "selected_total": len(selected),
        "available_by_domain": available_by_domain,
        "selected_by_domain": selected_by_domain,
        "sample_ids_digest": digest,
        "static_files": static_files,
    }
