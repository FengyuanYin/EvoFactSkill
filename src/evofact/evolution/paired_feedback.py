"""Deterministic comparisons of authorized, same-attempt source evaluations."""

from collections import Counter, defaultdict

PAIRED_FEEDBACK_POLICY = "paired_source_diagnostics_v1"
CATEGORIES = ("repaired", "regressed", "preserved", "unresolved", "runtime_failure")


def paired_source_diagnostics(baseline, candidate):
    """Join supervision records by ID, never by trace order or across attempts."""

    def index(records):
        result = {}
        for row in records:
            if row["sample_id"] in result:
                raise ValueError("duplicate paired feedback sample ID")
            if row["provenance"] != "source_validation":
                raise ValueError("paired feedback requires authorized source validation")
            if row["outcome"] not in {"correct", "wrong", "runtime_failure"}:
                raise ValueError("unknown paired feedback outcome")
            if row["outcome"] != "runtime_failure" and (
                (row["predicted_label"] == row["gold_label"]) != (row["outcome"] == "correct")
            ):
                raise ValueError("paired feedback prediction and outcome disagree")
            result[row["sample_id"]] = row
        return result

    before, after = index(baseline), index(candidate)
    if before.keys() != after.keys():
        raise ValueError("paired feedback sample sets do not match")
    records = []
    counts = Counter({category: 0 for category in CATEGORIES})
    by_label, by_domain = defaultdict(Counter), defaultdict(Counter)
    for sample_id in sorted(before):
        left, right = before[sample_id], after[sample_id]
        if any(left[key] != right[key] for key in ("gold_label", "domain", "provenance")):
            raise ValueError("paired feedback source metadata do not match")
        outcomes = (left["outcome"], right["outcome"])
        category = (
            "runtime_failure"
            if "runtime_failure" in outcomes
            else {
                ("wrong", "correct"): "repaired",
                ("correct", "wrong"): "regressed",
                ("correct", "correct"): "preserved",
                ("wrong", "wrong"): "unresolved",
            }[outcomes]
        )
        records.append(
            {
                "sample_id": sample_id,
                "domain": left["domain"],
                "gold_label": left["gold_label"],
                "change_type": category,
                **{
                    role: {key: row[key] for key in ("predicted_label", "outcome", "trace_id")}
                    for role, row in (("baseline", left), ("candidate", right))
                },
            }
        )
        counts[category] += 1
        by_label[left["gold_label"]][category] += 1
        by_domain[left["domain"]][category] += 1

    def summary(counter):
        return {
            **{category: counter[category] for category in CATEGORIES},
            "total": sum(counter.values()),
            "judgment_pairs": sum(counter.values()) - counter["runtime_failure"],
            "net_correct_change": counter["repaired"] - counter["regressed"],
        }

    return {
        "schema_version": PAIRED_FEEDBACK_POLICY,
        "summary": summary(counts),
        "by_gold_label": {key: summary(value) for key, value in sorted(by_label.items())},
        "by_domain": {key: summary(value) for key, value in sorted(by_domain.items())},
        "records": records,
    }
