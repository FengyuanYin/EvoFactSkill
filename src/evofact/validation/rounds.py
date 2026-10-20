"""Fixed, disjoint paired rounds for one immutable batch candidate."""

from collections import defaultdict

from .independent import independent_rows

AGGREGATION_POLICY = "fixed_paired_rounds_v1"
DEFAULT_VALIDATION_ROUNDS = 3


def discovery_rounds(rows, *, size, rounds):
    selected = list(rows[: size * rounds])
    if len(selected) != size * rounds:
        raise ValueError("insufficient discovery samples for fixed validation rounds")
    ids = [s.sample_id for s in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("validation rounds must contain distinct sample IDs")
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def independent_rounds(pool, batch, *, rounds, per_domain, registry, seed, cross_domain):
    if type(per_domain) is not int or per_domain < 2 or per_domain % 2:
        raise ValueError("multi-round validation requires even evaluation_samples_per_domain >= 2")
    # Sample the entire predetermined union once, never based on intermediate results.
    selected = independent_rows(
        pool,
        batch,
        cross_domain=cross_domain,
        per_domain=per_domain * rounds,
        registry=registry,
        seed=seed,
    )
    groups = defaultdict(list)
    for sample in selected:
        contract = registry.resolve_sample(sample)
        groups[sample.domain or sample.dataset, contract.normalize(sample.label)].append(sample)
    quota = per_domain // 2
    return [
        [
            sample.sample_id
            for group in groups.values()
            for sample in group[i * quota : (i + 1) * quota]
        ]
        for i in range(rounds)
    ]
