def utility(
    metrics: dict[str, float], *, cost_weight: float = 0.01, calibration_weight: float = 0.05
) -> float:
    """函数作用：负责当前模块中的 `utility` 处理，封装调用方需要复用的业务步骤。
    输入要求：`metrics`（dict[str, float]）需符合函数签名约定；`cost_weight`（float，默认 `0.01`）需以关键字传入并符合签名约定；`calibration_weight`（float，默认 `0.05`）需以关键字传入并符合签名约定。
    输出：返回 `float` 类型结果；校验或下游调用失败时异常向上传递。"""
    return (
        metrics.get("macro_f1_all", 0)
        + 0.2 * metrics.get("coverage", 0)
        - cost_weight * metrics.get("mean_cost", 0)
        - calibration_weight * metrics.get("ece", 0)
    )


def regression_failures(
    baseline_domains: dict[str, dict[str, float]],
    candidate_domains: dict[str, dict[str, float]],
    max_drop: float,
) -> tuple[str, ...]:
    """函数作用：负责当前模块中的 `regression_failures` 处理，封装调用方需要复用的业务步骤。
    输入要求：`baseline_domains`（dict[str, dict[str, float]]）需符合函数签名约定；`candidate_domains`（dict[str, dict[str, float]]）需符合函数签名约定；`max_drop`（float）需符合函数签名约定。
    输出：返回 `tuple[str, ...]` 类型结果；校验或下游调用失败时异常向上传递。"""
    failures = []
    for domain, old in baseline_domains.items():
        drop = old.get("macro_f1_all", 0) - candidate_domains.get(domain, {}).get("macro_f1_all", 0)
        if drop > max_drop:
            failures.append(f"protected domain {domain} dropped by {drop:.4f}")
    return tuple(failures)
