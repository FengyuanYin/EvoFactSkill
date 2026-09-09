from collections import defaultdict

from evofact.core.models import SampleEvaluation

LABELS = ("REAL", "FAKE")


def compute_metrics(rows: list[SampleEvaluation]) -> dict[str, float]:
    n = len(rows)
    covered = [r for r in rows if r.predicted in LABELS]
    correct = sum(r.gold == r.predicted for r in rows)
    f1s = []
    for label in LABELS:
        tp = sum(r.gold == label and r.predicted == label for r in rows)
        fp = sum(r.gold != label and r.predicted == label for r in rows)
        fn = sum(r.gold == label and r.predicted != label for r in rows)
        p = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * p * rec / (p + rec) if p + rec else 0)
    covered_correct = sum(r.gold == r.predicted for r in covered)
    return {
        "n": float(n),
        "accuracy_all": correct / n if n else 0,
        "macro_f1_all": sum(f1s) / 2,
        "coverage": len(covered) / n if n else 0,
        "covered_accuracy": covered_correct / len(covered) if covered else 0,
        "selective_risk": 1 - covered_correct / len(covered) if covered else 1,
        "mean_cost": sum(r.cost for r in rows) / n if n else 0,
    }


def grouped_metrics(rows: list[SampleEvaluation], field: str) -> dict[str, dict[str, float]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, field) or "unknown")].append(row)
    return {k: compute_metrics(v) for k, v in sorted(groups.items())}
