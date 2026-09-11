from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean, stdev

from evofact.core.models import EpisodeEvaluation, TransferUtility


def _delta(result: EpisodeEvaluation, name: str) -> float:
    """函数作用：负责当前模块中的 `_delta` 处理，封装调用方需要复用的业务步骤。
    输入要求：`result`（EpisodeEvaluation）需符合函数签名约定；`name`（str）需符合函数签名约定。
    输出：返回 `float` 类型结果；校验或下游调用失败时异常向上传递。"""
    return float(result.candidate_result.aggregate_metrics.get(name, 0.0)) - float(
        result.baseline_result.aggregate_metrics.get(name, 0.0)
    )


def _bootstrap_ci(
    values: list[float], seed: int, confidence_level: float, iterations: int = 2000
) -> tuple[float, float]:
    """函数作用：负责当前模块中的 `_bootstrap_ci` 处理，封装调用方需要复用的业务步骤。
    输入要求：`values`（list[float]）需符合函数签名约定；`seed`（int）需符合函数签名约定；`confidence_level`（float）需符合函数签名约定；`iterations`（int，默认 `2000`）需符合函数签名约定。
    输出：返回 `tuple[float, float]` 类型结果；校验或下游调用失败时异常向上传递。"""
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
        """函数作用：创建并初始化 `CrossEpisodeAggregator` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `CrossEpisodeAggregator` 实例；`seed`（int，默认 `42`）需符合函数签名约定；`confidence_level`（float，默认 `0.95`）需符合函数签名约定；`aggregate_across_episodes`（bool，默认 `True`）需符合函数签名约定。
        输出：返回 `None`；初始化 `CrossEpisodeAggregator` 的实例状态，构造参数非法时可能抛出异常。"""
        self.seed = seed
        self.confidence_level = confidence_level
        self.aggregate_across_episodes = aggregate_across_episodes

    def aggregate(
        self, results: list[EpisodeEvaluation] | tuple[EpisodeEvaluation, ...]
    ) -> dict[str, TransferUtility]:
        """函数作用：跨 episode 聚合同一候选的迁移收益、风险、覆盖率和成本。
        输入要求：`self` 应为已初始化的 `CrossEpisodeAggregator` 实例；`results`（list[EpisodeEvaluation] | tuple[EpisodeEvaluation, ...]）需符合函数签名约定。
        输出：返回 `dict[str, TransferUtility]` 类型结果；校验或下游调用失败时异常向上传递。"""
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
