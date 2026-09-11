OBJECTIVES = {"macro_f1_all": 1, "coverage": 1, "ece": -1, "mean_cost": -1}


def dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    """函数作用：负责当前模块中的 `dominates` 处理，封装调用方需要复用的业务步骤。
    输入要求：`left`（dict[str, float]）需符合函数签名约定；`right`（dict[str, float]）需符合函数签名约定。
    输出：返回 `bool` 类型结果；校验或下游调用失败时异常向上传递。"""
    weak = []
    strict = []
    for key, direction in OBJECTIVES.items():
        a = direction * left.get(key, 0)
        b = direction * right.get(key, 0)
        weak.append(a >= b)
        strict.append(a > b)
    return all(weak) and any(strict)


def pareto_front(candidates: list[dict[str, float]]) -> list[int]:
    """函数作用：负责当前模块中的 `pareto_front` 处理，封装调用方需要复用的业务步骤。
    输入要求：`candidates`（list[dict[str, float]]）需符合函数签名约定。
    输出：返回 `list[int]` 类型结果；校验或下游调用失败时异常向上传递。"""
    return [
        i
        for i, x in enumerate(candidates)
        if not any(j != i and dominates(y, x) for j, y in enumerate(candidates))
    ]
