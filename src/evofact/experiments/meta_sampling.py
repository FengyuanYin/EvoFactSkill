"""Match ordinary evolve's samples and mix domains while reserving validation domains."""

import random
from collections import Counter, deque

from evofact.data.domains import effective_domain
from evofact.data.manifests import build_manifest
from evofact.data.sampling import balanced_domain_sample


def mixed_domain_batches(training, domains, batch_size, seed):
    pools = {d: deque(s for s in training if effective_domain(s) == d) for d in domains}
    domains = tuple(d for d in domains if pools[d])
    if len(domains) < 3 or batch_size < 2 or len(training) < 4:
        raise ValueError(
            "mixed cross-domain batches require at least three training domains, four samples, and batch_size >= 2"
        )
    sizes = [
        min(batch_size, len(training) - start) for start in range(0, len(training), batch_size)
    ]
    if len(sizes) == 1:
        sizes = [(len(training) + 1) // 2, len(training) // 2]
    if sizes[-1] == 1:
        sizes[-2] -= 1
        sizes[-1] += 1
    remaining = {d: len(pools[d]) for d in domains}
    allocation = [{d: 0 for d in domains} for _ in sizes]
    reserved = [domains[i % len(domains)] for i in range(len(sizes))]
    # Two distinct domains are mandatory in every batch. Other domains can
    # contribute through the flow allocation below, except the reserved domain.
    for i in range(len(sizes)):
        eligible = [d for d in domains if d != reserved[i] and remaining[d]]
        eligible.sort(key=lambda d: (-remaining[d], (domains.index(d) - i) % len(domains)))
        if len(eligible) < 2:
            raise ValueError(
                "training quotas cannot form mixed batches with a validation domain reserved"
            )
        for d in eligible[:2]:
            allocation[i][d] += 1
            remaining[d] -= 1

    # Integer max-flow allocates every remaining sample without changing quotas.
    source, sink = ("source",), ("sink",)
    capacity, adjacent = {}, {}

    def edge(a, b, amount):
        capacity[a, b] = amount
        capacity[b, a] = 0
        adjacent.setdefault(a, []).append(b)
        adjacent.setdefault(b, []).append(a)

    for d in domains:
        edge(source, ("domain", d), remaining[d])
        for i, size in enumerate(sizes):
            if d != reserved[i]:
                edge(("domain", d), ("batch", i), size)
    for i, size in enumerate(sizes):
        edge(("batch", i), sink, size - 2)
    expected = sum(remaining.values())
    delivered = 0
    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for target in adjacent.get(node, ()):
                if target not in parent and capacity[node, target] > 0:
                    parent[target] = node
                    queue.append(target)
        if sink not in parent:
            break
        amount, node = expected - delivered, sink
        while parent[node] is not None:
            previous = parent[node]
            amount = min(amount, capacity[previous, node])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            capacity[previous, node] -= amount
            capacity[node, previous] += amount
            node = previous
        delivered += amount
    if delivered != expected:
        raise ValueError("training quotas cannot fit the mixed cross-domain batch schedule")
    batches = []
    for i in range(len(sizes)):
        batch = []
        for d in domains:
            count = allocation[i][d] + capacity.get((("batch", i), ("domain", d)), 0)
            batch.extend(pools[d].popleft() for _ in range(count))
        random.Random(f"{seed}:mixed-meta-batch:{i}").shuffle(batch)
        batches.append(batch)
    return batches


def matched_meta_plan(
    samples, *, source_domains, final_domains, seed, batch_size, train_limit, label_contract_digest
):
    manifest = build_manifest(
        samples,
        seed=seed,
        train_domains=source_domains,
        final_test_domains=final_domains,
        label_contract_digest=label_contract_digest,
    )
    training_ids = set(manifest.train_ids)
    training = [s for s in samples if s.sample_id in training_ids]
    if train_limit is not None:
        training, audit = balanced_domain_sample(training, source_domains, train_limit, seed=seed)
    else:
        audit = {"strategy": "ordered", "selected_total": len(training)}
    batches = mixed_domain_batches(training, source_domains, batch_size, seed)
    # Validation may reuse source-domain training samples; only the current
    # batch's domains must be excluded when selecting each validation chunk.
    validation = [s for s in samples if effective_domain(s) in set(source_domains)]
    audit = {
        **audit,
        "training_layout": "matched_mixed_domain_v2",
        "sample_ids": [s.sample_id for s in training],
        "selected_by_domain": dict(Counter(effective_domain(s) for s in training)),
        "batch_sample_ids": [[s.sample_id for s in batch] for batch in batches],
        "batch_domains": [sorted({effective_domain(s) for s in batch}) for batch in batches],
        "manifest_id": manifest.manifest_id,
        "validation_sample_ids": [s.sample_id for s in validation],
        "validation_training_overlap_allowed": True,
        "protected_sample_ids": list(manifest.protected_validation_ids),
    }
    return training, batches, validation, audit
