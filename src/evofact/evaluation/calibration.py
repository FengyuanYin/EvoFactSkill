from evofact.core.models import SampleEvaluation


def brier_score(rows: list[SampleEvaluation]) -> float:
    """函数作用：负责当前模块中的 `brier_score` 处理，封装调用方需要复用的业务步骤。
    输入要求：`rows`（list[SampleEvaluation]）需符合函数签名约定。
    输出：返回 `float` 类型结果；校验或下游调用失败时异常向上传递。"""
    if not rows:
        return 0
    values = []
    for r in rows:
        p = (
            0.5
            if r.predicted == "ABSTAIN"
            else (r.confidence if r.predicted == "FAKE" else 1 - r.confidence)
        )
        values.append((p - (1 if r.gold == "FAKE" else 0)) ** 2)
    return sum(values) / len(values)


def expected_calibration_error(rows: list[SampleEvaluation], bins: int = 10) -> float:
    """函数作用：负责当前模块中的 `expected_calibration_error` 处理，封装调用方需要复用的业务步骤。
    输入要求：`rows`（list[SampleEvaluation]）需符合函数签名约定；`bins`（int，默认 `10`）需符合函数签名约定。
    输出：返回 `float` 类型结果；校验或下游调用失败时异常向上传递。"""
    if not rows:
        return 0
    total = 0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        bucket = [
            r for r in rows if lo <= r.confidence < hi or (i == bins - 1 and r.confidence == 1)
        ]
        if bucket:
            acc = sum(r.gold == r.predicted for r in bucket) / len(bucket)
            conf = sum(r.confidence for r in bucket) / len(bucket)
            total += len(bucket) / len(rows) * abs(acc - conf)
    return total
