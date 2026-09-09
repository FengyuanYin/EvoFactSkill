import math
import random

from evofact.core.models import SampleEvaluation, StatisticalTestResult


def paired_rows(baseline, candidate):
    old = {row.sample_id: row for row in baseline}
    new = {row.sample_id: row for row in candidate}
    if len(old) != len(baseline) or len(new) != len(candidate) or old.keys() != new.keys():
        raise ValueError("paired evaluation requires unique matching sample ids")
    if any(old[key].gold != new[key].gold for key in old):
        raise ValueError("paired evaluation gold labels differ")
    return [(old[key], new[key]) for key in sorted(old)]


def mcnemar(
    baseline: list[SampleEvaluation], candidate: list[SampleEvaluation], alpha: float = 0.05
) -> StatisticalTestResult:
    b = c = 0
    for old, row in paired_rows(baseline, candidate):
        old_ok = old.gold == old.predicted
        new_ok = row.gold == row.predicted
        if old_ok and not new_ok:
            b += 1
        elif new_ok and not old_ok:
            c += 1
    stat = max(0, abs(b - c) - 1) ** 2 / (b + c) if b + c else 0
    p = math.erfc(math.sqrt(stat / 2)) if b + c else 1
    return StatisticalTestResult("mcnemar", stat, p, p < alpha)


def paired_bootstrap(
    baseline: list[SampleEvaluation],
    candidate: list[SampleEvaluation],
    *,
    seed: int = 42,
    iterations: int = 1000,
) -> tuple[float, float]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    pairs = paired_rows(baseline, candidate)
    rng = random.Random(seed)
    deltas = []
    if not pairs:
        return (0, 0)
    for _ in range(iterations):
        chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(
            sum((n.gold == n.predicted) - (o.gold == o.predicted) for o, n in chosen) / len(chosen)
        )
    deltas.sort()
    return deltas[int(0.025 * iterations)], deltas[min(iterations - 1, int(0.975 * iterations))]
