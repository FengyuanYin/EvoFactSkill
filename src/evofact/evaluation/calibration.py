from evofact.core.models import SampleEvaluation


def brier_score(rows: list[SampleEvaluation]) -> float:
    """评估二分类预测概率与真实标签的差距，值越低越好。

    将每行置信度换算为正类概率 p，再对真实正类取 y=1、负类取 y=0，
    计算 (p-y)^2 的样本均值。运行时弃权或非法预测按 p=0.5 计入。
    要求所有行使用同一个二分类标签契约且定义了正类；空输入返回 0。
    """
    if not rows:
        return 0
    signatures = {(row.label_schema_id, row.allowed_labels, row.positive_label) for row in rows}
    if len(signatures) != 1:
        raise ValueError("Brier score requires one label schema")
    _, allowed_labels, positive_label = next(iter(signatures))
    if len(allowed_labels) != 2 or positive_label is None:
        raise ValueError("Brier score requires a binary contract with positive_label")
    values = []
    for r in rows:
        if r.predicted not in allowed_labels or r.decision_origin == "runtime":
            p = 0.5
        else:
            p = r.confidence if r.predicted == positive_label else 1 - r.confidence
        values.append((p - (1 if r.gold == positive_label else 0)) ** 2)
    return sum(values) / len(values)


def expected_calibration_error(rows: list[SampleEvaluation], bins: int = 10) -> float:
    """衡量预测置信度与实际正确率是否一致，值越低表示校准越好。

    将置信度划入 bins 个等宽区间，计算每桶的平均置信度与正确率之差，
    再按桶内样本占比加权求和。弃权也按预测错误计入对应置信度桶；
    空输入返回 0。该指标衡量校准程度，不单独表示分类准确率。
    """
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
