"""Optimizer-only supervision from the current training batch and verified challenges."""

import hashlib
import json
from collections import Counter, defaultdict

FEEDBACK_POLICY_VERSION = "training_supervision_no_change_v1"
NO_CHANGE_PATIENCE = 3

TRAINING_FEEDBACK_POLICY = """
Optimizer-only training supervision is supplied in training_supervision.
Its gold_label belongs only to a current real training sample or a verified
synthetic challenge; predicted_label is the detector's actual answer. Use wrong
outcomes and their trace_id to diagnose business errors even when every node
executed successfully. A correct outcome is a success case, not an error.
Current supervision takes precedence over historical claims that no gold labels
were available; repeated no_change reviews are not evidence that an error is correct.
runtime_failure is not a wrong business judgment: diagnose execution separately.
Explain which supported error mechanism an edit addresses and how it generalizes.
Use gold labels for optimizer diagnosis only. Never copy sample IDs, case-specific
answers or training texts into skills, and never reveal supervision to detector
inference, the generator or verifier through an edited package. This input does
not authorize access to independent validation or final-test labels.
Return no_change if the supervised errors still do not support a safe improvement;
do not invent edits or force all agents to change.
""".strip()


def build_training_supervision(traces, sources, registry, *, allow_source_validation=False):
    """Project trusted training sources, never arbitrary dataset/validation rows."""
    by_id = {}
    for sample, provenance in sources:
        allowed = {"real_training", "verified_challenge"}
        if allow_source_validation:
            allowed.add("source_validation")
        if provenance not in allowed:
            raise ValueError("optimizer supervision source is not training")
        if sample.sample_id in by_id:
            raise ValueError("duplicate optimizer supervision source")
        by_id[sample.sample_id] = (sample, provenance)
    records = []
    groups = defaultdict(Counter)
    seen = set()
    for trace in traces:
        sample_id = trace["sample_id"]
        if sample_id not in by_id or sample_id in seen:
            raise ValueError("optimizer trace must resolve to one authorized training source")
        seen.add(sample_id)
        sample, provenance = by_id[sample_id]
        contract = registry.resolve_sample(sample)
        gold = contract.normalize(sample.label)
        decision = trace.get("decision") or {}
        prediction = decision.get("label")
        runtime = (
            bool(trace.get("errors"))
            or decision.get("origin") == "runtime"
            or prediction not in contract.allowed_labels
        )
        outcome = "runtime_failure" if runtime else "correct" if prediction == gold else "wrong"
        records.append(
            {
                "sample_id": sample_id,
                "trace_id": trace["trace_id"],
                "provenance": provenance,
                "domain": sample.domain or sample.dataset,
                "gold_label": gold,
                "predicted_label": prediction,
                "outcome": outcome,
            }
        )
        groups[provenance][outcome] += 1
    return {
        "schema_version": "optimizer_training_supervision_v1",
        "summary": {key: dict(value) for key, value in groups.items()},
        "records": records,
    }


def information_fingerprint(bank_digest, traces, feedback, supervision):
    """Repeated self-reviews are not new evidence; changed traces/feedback are."""
    payload = {
        "bank": bank_digest,
        "traces": traces,
        "validation_feedback": feedback,
        "training_supervision": supervision,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def record_no_change(state, fingerprint, *, no_change):
    if not no_change:
        state.pop("optimizer_no_change", None)
        return False
    previous = state.get("optimizer_no_change", {})
    count = previous.get("count", 0) + 1 if previous.get("fingerprint") == fingerprint else 1
    state["optimizer_no_change"] = {"fingerprint": fingerprint, "count": count}
    return count >= NO_CHANGE_PATIENCE
