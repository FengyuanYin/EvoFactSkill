from evofact.routing.strategies import STRATEGIES


async def run_ablations(runner, samples=None):
    """函数作用：负责当前模块中的 `run_ablations` 处理，封装调用方需要复用的业务步骤。
    输入要求：`runner`（未显式标注）需符合函数签名约定；`samples`（未显式标注，默认 `None`）需符合函数签名约定。
    输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    results = {}
    for arm in STRATEGIES:
        strategy = {
            "single-llm": "static",
            "prompt-only-evolution": "static",
            "no-discovery": "utility-aware",
            "no-negative-transfer": "utility-aware",
            "full": "utility-aware",
        }.get(arm, arm)
        traces, evaluation = await runner.run(samples=samples, strategy=strategy)
        results[arm] = {"n_traces": len(traces), "metrics": evaluation.aggregate_metrics}
    return results
