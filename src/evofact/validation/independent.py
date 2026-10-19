"""Source-only independent validation, hidden from training and optimizer feedback."""

import json
import random
from collections import defaultdict
from dataclasses import asdict

from evofact.data.domains import effective_domain


class InferenceIncomplete(RuntimeError):
    """A guarded inference stage still has runtime failures after bounded retries."""


def overlaps(sample, protected):
    """Treat near-duplicate texts and shared provenance as the same held-out unit."""
    return OverlapIndex(protected).overlaps(sample)


class OverlapIndex:
    """Compile held-out tokens once for dataset-sized pool selection."""

    def __init__(self, samples):
        self.tokens = set()
        self.texts = []
        for sample in samples:
            self.add(sample)

    def add(self, sample):
        self.tokens.add(sample.sample_id.casefold())
        text = " ".join(sample.text.casefold().split())
        self.texts.append(text)
        if len(text) >= 12:
            self.tokens.add(text)
        if sample.event_id:
            self.tokens.add(sample.event_id.casefold())
        for key in ("lineage_id", "parent_sample_id"):
            if sample.metadata.get(key):
                self.tokens.add(str(sample.metadata[key]).casefold())
        for evidence in sample.evidence:
            text = " ".join(evidence.text.casefold().split())
            if len(text) >= 12:
                self.tokens.add(text)

    def overlaps(self, sample):
        blob = json.dumps(
            {
                "text": sample.text,
                "event_id": sample.event_id,
                "metadata": sample.metadata,
                "evidence": [asdict(e) for e in sample.evidence],
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).casefold()
        if any(t and t in blob for t in self.tokens):
            return True
        text = " ".join(sample.text.casefold().split())
        return len(text) >= 12 and any(text in other for other in self.texts)


def independent_pool(validation, training, final, *, registry, per_domain, seed):
    """Keep training IDs unchanged; reserve equal counts for every domain/label cell."""
    training_ids = {s.sample_id for s in training}
    groups = defaultdict(list)
    contracts = {}
    blocked = list(training) + list(final)
    blocked_index = OverlapIndex(blocked)
    for sample in validation:
        domain = effective_domain(sample)
        contract = registry.resolve_sample(sample)
        if len(contract.allowed_labels) != 2 or contract.positive_label is None:
            raise ValueError("independent recall protection requires a binary label contract")
        contracts[domain] = contract
        if sample.sample_id not in training_ids:
            groups[domain, contract.normalize(sample.label)].append(sample)
    pool = []
    quota = per_domain // 2
    for domain in sorted(contracts):
        for label in contracts[domain].allowed_labels:
            candidates = sorted(groups[domain, label], key=lambda s: s.sample_id)
            random.Random(f"{seed}:independent:{domain}:{label}").shuffle(candidates)
            chosen = []
            for sample in candidates:
                if not blocked_index.overlaps(sample):
                    chosen.append(sample)
                    blocked_index.add(sample)
                    if len(chosen) == quota:
                        break
            if len(chosen) != quota:
                raise ValueError(
                    f"independent validation lacks {quota} disjoint {domain}/{label} samples"
                )
            pool.extend(chosen)
    return pool, discovery_without_pool(validation, pool)


def discovery_without_pool(validation, pool):
    pool_ids = {s.sample_id for s in pool}
    index = OverlapIndex(pool)
    return [s for s in validation if s.sample_id not in pool_ids and not index.overlaps(s)]


def guard_failures(baseline, candidate, policy):
    """Protect pooled F1, positive recall and per-domain behavior on paired clean rows."""
    failures = []
    if (
        candidate.aggregate_metrics["macro_f1_all"] + 1e-12
        < baseline.aggregate_metrics["macro_f1_all"]
    ):
        failures.append("independent macro-F1 regressed")
    for domain in sorted(baseline.domain_metrics):
        drop = (
            baseline.domain_metrics[domain]["macro_f1_all"]
            - candidate.domain_metrics[domain]["macro_f1_all"]
        )
        if drop > policy.max_domain_f1_drop + 1e-12:
            failures.append(f"independent domain macro-F1 regressed: {domain}")
    for domain in (None, *sorted(baseline.domain_metrics)):

        def recall(result):
            rows = [r for r in result.per_sample if domain is None or str(r.domain) == domain]
            positives = [r for r in rows if r.gold == r.positive_label]
            if not positives:
                raise ValueError("independent validation has no positive-label support")
            return sum(r.predicted == r.gold for r in positives) / len(positives)

        if recall(baseline) - recall(candidate) > policy.max_positive_recall_drop + 1e-12:
            failures.append(f"independent positive-label recall regressed: {domain or 'all'}")
    return failures


def independent_rows(pool, batch, *, cross_domain, per_domain=None, registry=None, seed=42):
    domains = {effective_domain(s) for s in batch}
    rows = [s for s in pool if not cross_domain or effective_domain(s) not in domains]
    if not rows:
        raise ValueError("no independent cross-domain validation for a training batch")
    if per_domain is not None:
        if type(per_domain) is not int or per_domain < 2 or per_domain % 2:
            raise ValueError("independent evaluation requires an even per_domain >= 2")
        if registry is None:
            raise ValueError("independent evaluation sampling requires a label registry")
        groups, contracts = defaultdict(list), {}
        for sample in rows:
            domain = effective_domain(sample)
            contract = registry.resolve_sample(sample)
            contracts[domain] = contract
            groups[domain, contract.normalize(sample.label)].append(sample)
        selected = []
        for domain in sorted(contracts):
            for label in contracts[domain].allowed_labels:
                candidates = sorted(groups[domain, label], key=lambda s: s.sample_id)
                random.Random(f"{seed}:independent-evaluation:{domain}:{label}").shuffle(candidates)
                if len(candidates) < per_domain // 2:
                    raise ValueError(f"independent evaluation lacks {domain}/{label} samples")
                selected.extend(candidates[: per_domain // 2])
        rows = selected
    return rows
