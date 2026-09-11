from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from evofact.core.models import Sample


@dataclass(frozen=True)
class DatasetDiagnostic:
    name: str
    available: bool
    files: tuple[str, ...] = ()
    message: str = ""


class DatasetAdapter(Protocol):
    name: str

    def discover(self, root: Path) -> DatasetDiagnostic:
        """函数作用：检查指定目录中是否存在当前适配器支持的数据文件。
        输入要求：`self` 应为已初始化的 `DatasetAdapter` 实例；`root`（Path）需符合函数签名约定。
        输出：返回 `DatasetDiagnostic` 类型结果；校验或下游调用失败时异常向上传递。"""
        ...

    def load(self, root: Path) -> Iterable[Sample]:
        """函数作用：从配置的存储位置读取并标准化当前对象负责的数据。
        输入要求：`self` 应为已初始化的 `DatasetAdapter` 实例；`root`（Path）需符合函数签名约定。
        输出：返回 `Iterable[Sample]` 类型结果；校验或下游调用失败时异常向上传递。"""
        ...
