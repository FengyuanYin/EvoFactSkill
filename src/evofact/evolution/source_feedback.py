"""Source-domain development feedback; final-test data never enter this view."""

from collections import defaultdict

from evofact.data.domains import effective_domain
from evofact.evolution.paired_feedback import paired_source_diagnostics
from evofact.evolution.training_feedback import build_training_supervision

SOURCE_FEEDBACK_POLICY = "all_source_validation_feedback_v1"
SOURCE_TRAINING_FEEDBACK_POLICY = """
Optimizer-only supervision covers current real training samples, verified synthetic
challenges, and evaluated source-domain validation samples. These source validation
samples are development data, not an independent holdout. Their explicit gold_label,
predicted_label, outcome and trace_id are authorized in
training_supervision.source_validation_feedback, with both baseline and candidate traces.
Diagnose mistakes, including REAL-to-FAKE regressions, and generalize the error mechanism.
Correct outcomes are success cases; runtime_failure is execution failure, not a business error.
All validation rounds finish on one frozen candidate before this feedback is released.
Never access final-test samples or labels. Never copy sample IDs, training texts or
case-specific answers into Agent instructions, or expose optimizer supervision to
detector inference through an edited package. Do not force unsupported edits.
""".strip()


def validate_source_feedback_manifest(expected, training, validation):
    """Training order is fixed; a candidate pool records membership, not shuffle order."""
    if expected.get("validation_feedback_policy") != SOURCE_FEEDBACK_POLICY:
        raise ValueError("source feedback manifest policy does not match")
    if expected["training_sample_ids"] != [s.sample_id for s in training]:
        raise ValueError("source feedback manifest training plan does not match")
    if sorted(expected["source_validation_sample_ids"]) != sorted(s.sample_id for s in validation):
        raise ValueError("source feedback manifest validation candidates do not match")


def balanced_source_rows(rows, *, size, rounds, registry):
    """Freeze a balanced ordering without reserving a separate protection pool."""
    groups = defaultdict(list)
    for sample in rows:
        contract = registry.resolve_sample(sample)
        groups[effective_domain(sample), contract.normalize(sample.label)].append(sample)
    groups = [iter(group) for _, group in sorted(groups.items())]
    ordered = []
    while groups and len(ordered) < size * rounds:
        remaining = []
        for group in groups:
            sample = next(group, None)
            if sample is not None:
                ordered.append(sample)
                remaining.append(group)
        groups = remaining
    return ordered[: size * rounds]


def source_validation_feedback(
    pending, sources, registry, *, batch, attempt, decision, paired_diagnostics=False
):
    """Authorize labels only for evaluated source rows after the frozen evaluation."""
    result = {"batch": batch, "attempt": attempt, "accepted": decision.accepted}
    for role in ("baseline", "candidate"):
        traces = pending[role]["traces"]
        result[role] = {
            "traces": traces,
            "supervision": build_training_supervision(
                traces,
                [(sample, "source_validation") for sample in sources],
                registry,
                allow_source_validation=True,
            ),
        }
    if paired_diagnostics:
        result["paired_comparison"] = paired_source_diagnostics(
            result["baseline"]["supervision"]["records"],
            result["candidate"]["supervision"]["records"],
        )
    return result


def source_recall_failures(baseline, candidate, *, max_drop):
    """Protect FAKE/positive recall on the same pooled development evaluation."""
    failures = []
    for domain in (None, *sorted(baseline.domain_metrics)):
        recalls = []
        for result in (baseline, candidate):
            rows = [
                r
                for r in result.per_sample
                if (domain is None or str(r.domain) == domain) and r.gold == r.positive_label
            ]
            if not rows:
                failures.append(
                    f"source validation has no positive-label support: {domain or 'all'}"
                )
                break
            recalls.append(sum(r.predicted == r.gold for r in rows) / len(rows))
        if len(recalls) == 2 and recalls[0] - recalls[1] > max_drop + 1e-12:
            failures.append(f"source positive-label recall regressed: {domain or 'all'}")
    return failures
