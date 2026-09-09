from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean, stdev

from evofact.core.models import EpisodeEvaluation, TransferUtility


def _delta(result: EpisodeEvaluation, name: str) -> float:
    return float(result.candidate_result.aggregate_metrics.get(name, 0.0)) - float(
        result.baseline_result.aggregate_metrics.get(name, 0.0)
    )


def _bootstrap_ci(
    values: list[float], seed: int, confidence_level: float, iterations: int = 2000
) -> tuple[float, float]:
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    draws = sorted(
        mean(values[rng.randrange(len(values))] for _ in values) for _ in range(iterations)
    )
    tail = (1 - confidence_level) / 2
    return draws[int(tail * iterations)], draws[min(iterations - 1, int((1 - tail) * iterations))]


class CrossEpisodeAggregator:
    def __init__(
        self, seed: int = 42, confidence_level: float = 0.95, aggregate_across_episodes: bool = True
    ):
        self.seed = seed
        self.confidence_level = confidence_level
        self.aggregate_across_episodes = aggregate_across_episodes

    def aggregate(
        self, results: list[EpisodeEvaluation] | tuple[EpisodeEvaluation, ...]
    ) -> dict[str, TransferUtility]:
        groups: dict[str, list[EpisodeEvaluation]] = defaultdict(list)
        seen = set()
        for result in results:
            key = (result.episode_id, result.candidate_fingerprint)
            if key in seen:
                raise ValueError("duplicate candidate evaluation in episode")
            seen.add(key)
            groups[result.candidate_fingerprint].append(result)
        utilities = {}
        for fingerprint, rows in sorted(groups.items()):
            if not self.aggregate_across_episodes:
                rows = rows[:1]
            gains = [_delta(row, "macro_f1_all") for row in rows]
            domain_values: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                for domain, metrics in row.domain_deltas.items():
                    domain_values[domain].append(float(metrics.get("macro_f1_all", 0.0)))
            domain_gains = {
                domain: mean(values) for domain, values in sorted(domain_values.items())
            }
            coverage_delta = mean(_delta(row, "coverage") for row in rows)
            calibration_delta = mean(_delta(row, "ece") for row in rows)
            candidate_coverage = mean(
                float(row.candidate_result.aggregate_metrics.get("coverage", 0.0)) for row in rows
            )
            ratios = []
            for row in rows:
                old = float(row.baseline_result.aggregate_metrics.get("mean_cost", 0.0))
                new = float(row.candidate_result.aggregate_metrics.get("mean_cost", 0.0))
                ratios.append(new / old if old > 0 else (1.0 if new == 0 else float("inf")))
            utilities[fingerprint] = TransferUtility(
                fingerprint,
                len(rows),
                tuple(sorted(domain_gains)),
                mean(gains),
                stdev(gains) if len(gains) > 1 else 0.0,
                _bootstrap_ci(gains, self.seed, self.confidence_level),
                sum(value > 0 for value in gains) / len(gains),
                sum(value < 0 for value in gains) / len(gains),
                min(domain_gains.values(), default=0.0),
                candidate_coverage,
                coverage_delta,
                calibration_delta,
                mean(ratios),
                domain_gains,
            )
        return utilities
