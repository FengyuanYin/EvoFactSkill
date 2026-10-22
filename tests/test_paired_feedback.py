from copy import deepcopy

import pytest

from evofact.evolution.optimizer_memory import reduce_context
from evofact.evolution.paired_feedback import paired_source_diagnostics


def row(sample_id, gold, prediction, role, *, runtime=False):
    return {
        "sample_id": sample_id,
        "trace_id": f"{role}-{sample_id}",
        "gold_label": gold,
        "predicted_label": prediction,
        "domain": "science" if gold == "REAL" else "society",
        "provenance": "source_validation",
        "outcome": "runtime_failure" if runtime else "correct" if gold == prediction else "wrong",
    }


def fixture():
    cases = [
        ("repair", "FAKE", "REAL", "FAKE"),
        ("regress", "REAL", "REAL", "FAKE"),
        ("preserve", "REAL", "REAL", "REAL"),
        ("unresolved", "FAKE", "REAL", "REAL"),
    ]
    baseline = [row(sid, gold, old, "baseline") for sid, gold, old, new in cases]
    candidate = [row(sid, gold, new, "candidate") for sid, gold, old, new in cases]
    baseline += [row("runtime", "REAL", None, "baseline", runtime=True)]
    candidate += [row("runtime", "REAL", "REAL", "candidate")]
    return baseline, list(reversed(candidate))


def test_pairs_by_id_and_separates_runtime_from_judgment():
    baseline, candidate = fixture()
    original = deepcopy((baseline, candidate))
    result = paired_source_diagnostics(baseline, candidate)
    assert result["summary"] == {
        "repaired": 1,
        "regressed": 1,
        "preserved": 1,
        "unresolved": 1,
        "runtime_failure": 1,
        "total": 5,
        "judgment_pairs": 4,
        "net_correct_change": 0,
    }
    assert result["by_gold_label"]["REAL"]["regressed"] == 1
    assert result["by_domain"]["society"]["repaired"] == 1
    repaired = next(r for r in result["records"] if r["sample_id"] == "repair")
    assert repaired["baseline"]["trace_id"] == "baseline-repair"
    assert repaired["candidate"]["predicted_label"] == "FAKE"
    assert (baseline, candidate) == original


@pytest.mark.parametrize(
    "problem", ["duplicate", "missing", "gold", "domain", "outcome", "provenance"]
)
def test_inconsistent_pairs_are_rejected(problem):
    baseline, candidate = fixture()
    if problem == "duplicate":
        candidate.append(candidate[0])
    elif problem == "missing":
        candidate.pop()
    elif problem == "gold":
        candidate[0]["gold_label"] = "FAKE"
    elif problem == "domain":
        candidate[0]["domain"] = "other"
    elif problem == "outcome":
        candidate[0]["outcome"] = "wrong"
    else:
        candidate[0]["provenance"] = "final_test"
    with pytest.raises(ValueError):
        paired_source_diagnostics(baseline, candidate)


def test_window_reduction_preserves_pair_diagnostics_when_full_traces_are_removed():
    baseline, candidate = fixture()
    paired = paired_source_diagnostics(baseline, candidate)
    latest = {
        "paired_comparison": paired,
        "baseline": {
            "supervision": {"records": baseline},
            "traces": [{"trace_id": "baseline-preserve"}],
        },
        "candidate": {"supervision": {"records": candidate}, "traces": []},
    }
    context = {"training_supervision": {"source_validation_feedback": [latest]}}
    assert reduce_context(context)["kind"] == "source_validation_trace"
    assert latest["paired_comparison"] == paired
    assert len(paired["records"]) == 5
