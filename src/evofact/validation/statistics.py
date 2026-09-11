"""
用于比较“基线系统”和“候选 Skill 系统”的效果，判断候选方案的准确率变化是否可靠，而不是由随机波动造成。
"""

import math
import random

from evofact.core.models import SampleEvaluation, StatisticalTestResult


def paired_rows(baseline, candidate):
    """函数作用：负责当前模块中的 `paired_rows` 处理，封装调用方需要复用的业务步骤。
    输入要求：`baseline`（未显式标注）需符合函数签名约定；`candidate`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
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
    """函数作用：负责当前模块中的 `mcnemar` 处理，封装调用方需要复用的业务步骤。
    输入要求：`baseline`（list[SampleEvaluation]）需符合函数签名约定；`candidate`（list[SampleEvaluation]）需符合函数签名约定；`alpha`（float，默认 `0.05`）需符合函数签名约定。
    输出：返回 `StatisticalTestResult` 类型结果；校验或下游调用失败时异常向上传递。"""
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
    """函数作用：负责当前模块中的 `paired_bootstrap` 处理，封装调用方需要复用的业务步骤。
    输入要求：`baseline`（list[SampleEvaluation]）需符合函数签名约定；`candidate`（list[SampleEvaluation]）需符合函数签名约定；`seed`（int，默认 `42`）需以关键字传入并符合签名约定；`iterations`（int，默认 `1000`）需以关键字传入并符合签名约定。
    输出：返回 `tuple[float, float]` 类型结果；校验或下游调用失败时异常向上传递。"""
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
