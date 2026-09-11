from pathlib import Path

from .adapters.advfake import AdvFakeAdapter
from .adapters.amtcele import AMTCeleAdapter
from .adapters.livefact import LiveFactAdapter
from .adapters.weibo21 import Weibo21Adapter
from .base import DatasetAdapter, DatasetDiagnostic


class DataRegistry:
    def __init__(self) -> None:
        """函数作用：创建并初始化 `DataRegistry` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `DataRegistry` 实例；无其他显式输入。
        输出：返回 `None`；初始化 `DataRegistry` 的实例状态，构造参数非法时可能抛出异常。"""
        self._adapters: dict[str, DatasetAdapter] = {}
        for adapter in (Weibo21Adapter(), AMTCeleAdapter(), LiveFactAdapter(), AdvFakeAdapter()):
            self.register(adapter)

    def register(self, adapter: DatasetAdapter) -> None:
        """函数作用：负责`DataRegistry` 中的 `register` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `DataRegistry` 实例；`adapter`（DatasetAdapter）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责修改对象状态或持久化文件。"""
        if adapter.name in self._adapters:
            raise ValueError(f"duplicate dataset adapter: {adapter.name}")
        self._adapters[adapter.name] = adapter

    def names(self) -> tuple[str, ...]:
        """函数作用：负责`DataRegistry` 中的 `names` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `DataRegistry` 实例；无其他显式输入。
        输出：返回 `tuple[str, ...]` 类型结果；校验或下游调用失败时异常向上传递。"""
        return tuple(sorted(self._adapters))

    def inspect(self, roots: dict[str, Path]) -> list[DatasetDiagnostic]:
        """函数作用：负责`DataRegistry` 中的 `inspect` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `DataRegistry` 实例；`roots`（dict[str, Path]）需符合函数签名约定。
        输出：返回 `list[DatasetDiagnostic]` 类型结果；校验或下游调用失败时异常向上传递。"""
        return [
            self._adapters[n].discover(roots.get(n, Path("data/raw") / n)) for n in self.names()
        ]

    def load(self, name: str, root: Path):
        """函数作用：从配置的存储位置读取并标准化当前对象负责的数据。
        输入要求：`self` 应为已初始化的 `DataRegistry` 实例；`name`（str）需符合函数签名约定；`root`（Path）需符合函数签名约定。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        if name not in self._adapters:
            raise KeyError(name)
        return list(self._adapters[name].load(root))
