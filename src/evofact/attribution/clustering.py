"""
根据每份报告包含的错误类型生成稳定的分组键，把具有相同错误模式的推理报告聚集在一起，便于后续批量分析和生成 Skill 改进方案。
"""

import hashlib
from collections import defaultdict

from evofact.core.models import AttributionReport


# 把一个报告内出现的所以错误键值使用+连接，按照键值进行聚类 (这样做精度存疑)
def cluster_reports(reports: list[AttributionReport]) -> dict[str, list[AttributionReport]]:
    """函数作用：将具有相同错误类型组合的归因报告放入同一个稳定分组。
    输入要求：`reports`（list[AttributionReport]）需符合函数签名约定。
    输出：返回 `dict[str, list[AttributionReport]]` 类型结果；校验或下游调用失败时异常向上传递。"""

    # defaultdict(list) 会在首次遇到某个分组时自动创建报告列表。
    groups = defaultdict(list)
    for report in reports:
        # 先排序再拼接错误类型，避免相同错误因排列顺序不同而被分到不同组。
        # 没有错误的报告统一使用 success 作为分组依据。
        key = "+".join(sorted(x.value for x in report.error_types)) or "success"

        # 将错误组合哈希为短且稳定的分组 ID，并把当前报告加入对应分组。
        groups[f"cluster-{hashlib.sha1(key.encode()).hexdigest()[:8]}"].append(report)

    # 返回普通 dict，隐藏内部使用 defaultdict 的实现细节。
    return dict(groups)
