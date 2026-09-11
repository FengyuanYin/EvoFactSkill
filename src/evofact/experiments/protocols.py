from evofact.routing.strategies import STRATEGIES


class ExperimentProtocol:
    def __init__(self, manifest_id: str, arm: str):
        """函数作用：创建并初始化 `ExperimentProtocol` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `ExperimentProtocol` 实例；`manifest_id`（str）需符合函数签名约定；`arm`（str）需符合函数签名约定。
        输出：返回 `None`；初始化 `ExperimentProtocol` 的实例状态，构造参数非法时可能抛出异常。"""
        if arm not in STRATEGIES:
            raise ValueError(f"unknown arm: {arm}")
        self.manifest_id = manifest_id
        self.arm = arm


def all_protocols(manifest_id: str) -> list[ExperimentProtocol]:
    """函数作用：负责当前模块中的 `all_protocols` 处理，封装调用方需要复用的业务步骤。
    输入要求：`manifest_id`（str）需符合函数签名约定。
    输出：返回 `list[ExperimentProtocol]` 类型结果；校验或下游调用失败时异常向上传递。"""
    return [ExperimentProtocol(manifest_id, x) for x in STRATEGIES]
