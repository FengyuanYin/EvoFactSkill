"""Current-batch supervised challenge allocation; no held-out data or label quotas."""

import hashlib
from collections import Counter, defaultdict

POLICY_VERSION = "error_driven_challenges_v1"
DIAGNOSTIC_POLICY = """
Error-driven challenge allocation overrides general autonomous budget allocation.
Each current example has diagnosis: gold_label, predicted_label and outcome.
wrong is a detector judgment error; runtime_failure is not an adversarial success.
generation_plan.slots is the fixed source/goal allocation for this call. Return
samples in slot order, using that slot's source_sample_id in the paired decision.
You may return a shorter prefix or no samples if you cannot justify labels.
Never invent extra JSON fields, fill a label quota, or weaken independent review.
For error_mechanism_extension, inspect the assigned failure_trace and formulate
a testable error-mechanism hypothesis. Vary wording, entities or circumstances
while preserving that mechanism and a justified target label; do not merely make
all failed cases harder. The hypothesis is diagnostic, not established causation.
For success_boundary_extension, probe harder relations, minimal contrasts and
underrepresented strategies. If no scored errors exist, all slots explore.
Explain the slot goal, its relevant trace and target-label basis in decision.reason.
Verified challenge observations are diagnostic examples, not authorized new source
IDs. Construct only from the assigned current training source.
Diagnosis gold_label describes the referenced trace, which may be a challenge;
decision.source_label must still equal the assigned real training source label.
Past wrong outcomes apply to their recorded detector bank. Use generation memory to avoid repeating
already solved transformations, but do not resample until an attack succeeds.
""".strip()


def policy_marker():
    return {
        "version": POLICY_VERSION,
        "prompt_digest": hashlib.sha256(DIAGNOSTIC_POLICY.encode()).hexdigest(),
    }


def diagnosis(trace, gold, allowed):
    decision = trace.get("decision") or {}
    predicted = decision.get("label")
    runtime = (
        bool(trace.get("errors")) or decision.get("origin") == "runtime" or predicted not in allowed
    )
    return {
        "gold_label": gold,
        "predicted_label": predicted,
        "outcome": "runtime_failure" if runtime else "correct" if predicted == gold else "wrong",
        "trace_id": trace.get("trace_id"),
    }


def enrich_request(request, rounds):
    """Use current sources and verified current-batch failures, rotating target coverage."""
    examples = request["examples"]
    by_id = {e["source_sample"]["sample_id"]: e for e in examples}
    targets = []
    observations = []
    for example in examples:
        source = example["source_sample"]
        result = diagnosis(
            example["trace"], source["label"], example["label_contract"]["allowed_labels"]
        )
        example["diagnosis"] = result
        if result["outcome"] == "wrong":
            targets.append(
                {
                    "source_sample_id": source["sample_id"],
                    "provenance": "real_training",
                    "diagnosis": result,
                    "failure_trace": example["trace"],
                }
            )
    for round_ in rounds:
        verified = {
            s["sample_id"]: s for s in (round_.get("verification") or {}).get("accepted", [])
        }
        decisions = {d["sample_id"]: d for d in (round_.get("response") or {}).get("decisions", [])}
        for trace in round_.get("traces") or []:
            sample = verified.get(trace["sample_id"])
            source_id = decisions.get(trace["sample_id"], {}).get("source_sample_id")
            if sample is None or source_id not in by_id:
                continue
            allowed = by_id[source_id]["label_contract"]["allowed_labels"]
            result = diagnosis(trace, sample["label"], allowed)
            observation = {
                "source_sample_id": source_id,
                "provenance": "verified_challenge",
                "diagnosis": result,
                "failure_trace": trace,
            }
            observations.append(observation)
            if result["outcome"] == "wrong":
                targets.append(observation)
    size = request["batch_size"]
    focus = (min(size - 1, max(1, 3 * size // 4)) if size > 1 else 1) if targets else 0
    exploratory = [e for e in examples if e["diagnosis"]["outcome"] == "correct"] or examples
    offset = sum(bool(r.get("complete")) for r in rounds)
    slots = []
    for i in range(size):
        if i < focus:
            target = targets[(offset * max(focus, 1) + i) % len(targets)]
            slots.append({"goal": "error_mechanism_extension", **target})
        else:
            example = exploratory[(offset + i - focus) % len(exploratory)]
            slots.append(
                {
                    "goal": "success_boundary_extension",
                    "source_sample_id": example["source_sample"]["sample_id"],
                    "diagnosis": example["diagnosis"],
                }
            )
    request["generation_plan"] = {
        "policy": POLICY_VERSION,
        "error_extension_slots": focus,
        "boundary_exploration_slots": size - focus,
        "slots": slots,
    }
    request["challenge_observations"] = observations
    return request


def challenge_statistics(rounds):
    """Keep generation validity and scored detector errors as separate denominators."""
    counts = Counter()
    labels = Counter()
    strategies = defaultdict(Counter)
    banks = defaultdict(Counter)
    for round_ in rounds:
        response = round_.get("response") or {}
        counts["proposed"] += len(response.get("samples", []))
        counts["structurally_valid"] += round_.get("structural_candidates", 0)
        verified = (round_.get("verification") or {}).get("accepted", [])
        counts["accepted"] += len(verified)
        labels.update(s["label"] for s in verified)
        by_id = {s["sample_id"]: s for s in verified}
        decisions = {d["sample_id"]: d for d in response.get("decisions", [])}
        for trace in round_.get("traces") or []:
            sample = by_id.get(trace["sample_id"])
            if sample is None:
                continue
            # Knowledge route currently uses the source contract's exact labels.
            allowed = next(
                (
                    e["label_contract"]["allowed_labels"]
                    for e in round_["request"]["examples"]
                    if e["source_sample"]["sample_id"]
                    == decisions[sample["sample_id"]]["source_sample_id"]
                ),
                (),
            )
            result = diagnosis(trace, sample["label"], allowed)["outcome"]
            counts[result] += 1
            strategies[decisions[sample["sample_id"]]["strategy"]][result] += 1
            banks[round_.get("detector_bank_digest", "unknown")][result] += 1

    def metrics(c):
        scored = c["correct"] + c["wrong"]
        return {
            **dict(c),
            **{key: c[key] for key in ("correct", "wrong", "runtime_failure")},
            "scored": scored,
            "attack_success_rate": c["wrong"] / scored if scored else None,
        }

    return {
        **metrics(counts),
        **{key: counts[key] for key in ("proposed", "structurally_valid", "accepted")},
        "accepted_labels": dict(labels),
        "structural_pass_rate": counts["structurally_valid"] / counts["proposed"]
        if counts["proposed"]
        else None,
        "verification_pass_rate": counts["accepted"] / counts["structurally_valid"]
        if counts["structurally_valid"]
        else None,
        "by_strategy": {k: metrics(v) for k, v in strategies.items()},
        "by_detector_bank": {k: metrics(v) for k, v in banks.items()},
    }
